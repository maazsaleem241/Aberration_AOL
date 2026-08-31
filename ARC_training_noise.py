"""ARC noise training script -- physics-informed composite loss + curriculum
training over dataset_b (reflection_matrix_dataset.h5).

Model architecture lives separately in ARC.py; this file only
covers data loading, the loss function, and the training loop.

Design decisions this script makes explicit (see conversation for the full
reasoning -- summarized here so the code is self-documenting):

1. RR^dagger and R^dagger R are computed ON THE FLY from R inside the dataset
   (never stored in the H5 file -- they're trivially derivable from R, so
   storing them would just double the file size for free).

2. R, RR^dagger, R^dagger R are each reshaped LOSSLESSLY into 1600 channels of
   40x40 (real then imag -> 3200 channels), matching the base papers'
   "1600x40x40" input convention. No resizing/interpolation anywhere.

3. Composite loss has four physically-distinct terms:
     - L_coherent : global coherent phasor correlation. Invariant to a
                    constant global phase offset (physically correct, since a
                    uniform phase shift across the whole pupil is unobservable
                    in the reconstructed intensity image).
     - L_local    : unbounded per-pixel (1 - cos_delta) mean. Provides strong
                    gradient signal early in training, when residual error is
                    still large and a saturating loss would under-drive it.
     - L_strehl   : bounded/saturating 1 - exp(-local_error) term, i.e. the
                    Marechal-approximation-style term. Its weight is RAMPED UP
                    over curriculum stages (see STRE HL_WEIGHT_SCHEDULE) rather
                    than fixed, since it matters more for fine convergence
                    late in training than for driving early progress.
     - L_toeplitz : penalizes the corrected matrix R_c = P_o^dagger R P_i^dagger
                    for deviating from true block-Toeplitz structure -- i.e.
                    entries sharing the same 2D spatial-frequency DIFFERENCE
                    (not the same raw 1D diagonal index, which would be wrong
                    for this 2D k-space problem) should be consistent. Uses the
                    exact same difference-index convention validated in
                    zernike.py's construct_scattering_matrix.

     - L_image   : differentiable SSIM loss (1 - SSIM), weighted 0.5.
                    The image is reconstructed from the model's corrected
                    reflection matrix using full 2D Δk-based CASS-style
                    accumulation rather than a single center column.
     - L_image_l1: small foreground-weighted L1 image loss, weighted 0.1.
                    Target foreground pixels receive additional weight so
                    crosses, bars, letters, beads, and USAF structures
                    contribute more strongly to image-domain supervision.

4. Curriculum training uses dataset_b's own difficulty_tags/split_types fields
   directly (no random splitting). Two SEPARATE validation views are used:
     - a stage-matched, cumulative validation subset (mirrors the active
       training stages) used ONLY to decide when to advance the curriculum
     - a fixed, full-spectrum validation subset (all val_* tags) used ONLY for
       honest overall progress reporting
   The dedicated test_* splits (test_standard, test_high_aberration_low_noise,
   test_high_noise_low_aberration, test_intensity_ratio) are NEVER touched by
   this script -- they're reserved for final held-out evaluation.
"""

from __future__ import annotations

import csv
import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.ARC import ARC, make_na_mask
from CASS_visualizer import reconstruct_cass, reconstruct_dc_column


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

H5_FILE = "/home/awais/Desktop/Maaz/Maaz Data/dataset_b/data/reflection_matrix_dataset.h5"
GRID_SIZE = 40
N_ELEMENTS = GRID_SIZE * GRID_SIZE  # 1600

MAX_EPOCHS = 400
BATCH_SIZE = 16
INITIAL_LR = 1e-4
WEIGHT_DECAY = 0
GRAD_CLIP = 1.0
NUM_WORKERS = 4
CHECKPOINT_INTERVAL = 2
MC_DROPOUT_PASSES = 5
# FIX (do not revert): after each curriculum transition, linearly ramp LR
# up to INITIAL_LR over this many epochs instead of an instant full-strength
# jump onto harder data. An instant reset caused NaN loss within 15 batches
# at a stage transition in the aberration curriculum run.
WARMUP_EPOCHS = 3
USE_AMP = torch.cuda.is_available()
DEVICE = "cuda" if USE_AMP else "cpu"

BASE_OUTPUT_DIR = "training/ARC_Training/noise/ARC_training_noise_2"
CHECKPOINT_DIR = os.path.join(BASE_OUTPUT_DIR, "checkpoints")
VIS_DIR = os.path.join(BASE_OUTPUT_DIR, "visuals")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "ARC_best.pth")
METRICS_CSV_PATH = os.path.join(CHECKPOINT_DIR, "training_metrics.csv")

# Pretrained aberration checkpoint to initialize from -- MUST exist before
# running this script. The noise curriculum is designed as a second phase
# on top of the aberration-curriculum model; starting from random weights
# is incorrect and will produce garbage results.
PRETRAINED_CHECKPOINT_PATH = (
    "/home/awais/Desktop/Maaz/Maaz Data/Publication work/training/ARC_Training/"
    "aberration/ARC_training_aberration_new_loss/checkpoints/ARC_best.pth"
)
METRICS_CSV_PATH = os.path.join(CHECKPOINT_DIR, "training_metrics.csv")

# Curriculum stages,

# Curriculum stages -- noise_low removed after empirical evidence showed a
# 34x train/val gap at that stage (train coherent=0.017, val=0.585), meaning
# the model memorized rather than generalized. Harder noise (noise_med,
# noise_high) actually produced smaller gaps, consistent with the hypothesis
# that stronger noise forces more abstract representations. Starting directly
# at noise_med where the aberration pretraining's NSR [0,4] coverage ends.
CURRICULUM_STAGES = ["noise_med", "noise_high"]

# The matching held-out VAL tag for each train stage -- used only to decide
# when to advance (see design note 4 above). Must stay in lockstep with
# dataset_generator.py's tagging.
STAGE_TO_VAL_TAG = {
    "noise_med":  "val_noise_med",
    "noise_high": "val_noise_high",
}

# For early stop incase the loss does not improve by 0.1%
PATIENCE_LIMIT = 8          # tightened from 15 -- data showed model plateauing early and memorizing
MIN_DELTA_PERCENT = 0.0005  # tightened from 0.001 -- 0.1% threshold was too coarse near converged loss

# LR scheduler: drop the learning rate when stage_val_loss plateaus WITHIN a stage 
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MIN_LR = 1e-5

# loss weights.
# Image-domain supervision:
#   - SSIM remains at the existing weight of 0.5.
#   - A small foreground-weighted L1 term is added at weight 0.1.
# The differentiable image reconstruction used by both terms is a full
# Δk-based CASS-style accumulation over the corrected reflection matrix.
LOSS_WEIGHTS = {
    "coherent": 1.0,
    "local": 1.0,
    "strehl": 0.5,
    "toeplitz": 0.1,
    "image": 0.5,
    "image_l1": 0.1,
}
def strehl_weight_scale_for_stage(stage_idx: int) -> float:
    """Ramp the Strehl (bounded/saturating) term's effective weight up as the
    curriculum advances -- see design note 3 above for why it shouldn't be at
    full weight from epoch 1."""
    return (stage_idx + 1) / len(CURRICULUM_STAGES)


# =============================================================================
# 2. DATASET: H5 adapter, lossless [3200,40,40] reshape, curriculum filtering
# =============================================================================


class ARCH5Dataset(Dataset):
    """Adapter over dataset_generator.py's reflection_matrix_dataset.h5.

    Returns, per sample:
        r, rrh, rhr        [3200, 40, 40] float32 -- model inputs
        target_output      [2, 40, 40]   float32 -- unit phasor (cos,sin) of phi_o
        target_input       [2, 40, 40]   float32 -- unit phasor (cos,sin) of phi_i
        truth               [4, 40, 40]   float32 -- concat of the above two
        R_real, R_imag      [1600, 1600] float32 -- the raw (rms-normalized) R,
                                          needed only for the Toeplitz loss
    """

    def __init__(
        self,
        h5_path: Union[str, os.PathLike],
        split: Optional[str] = None,
        difficulty_tags: Optional[Sequence[str]] = None,
        normalize_by_rms: bool = True,
        grid_size: int = GRID_SIZE,
    ):
        self.h5_path = Path(h5_path)
        self.grid_size = grid_size
        self.n_elements = grid_size * grid_size
        self.normalize_by_rms = normalize_by_rms

        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 dataset not found: {self.h5_path.resolve()}")

        with h5py.File(self.h5_path, "r") as f:
            splits = [s.decode("utf-8") if isinstance(s, bytes) else s for s in f["split_types"][:]]
            tags = [t.decode("utf-8") if isinstance(t, bytes) else t for t in f["difficulty_tags"][:]]

        self.indices = [
            i
            for i, (s, t) in enumerate(zip(splits, tags))
            if (split is None or s == split) and (difficulty_tags is None or t in difficulty_tags)
        ]
        if not self.indices:
            raise ValueError(
                f"No samples matched split={split!r}, difficulty_tags={difficulty_tags!r} in {self.h5_path}"
            )

        # Lazily opened per-worker (see earlier discussion: h5py.File handles
        # don't reliably survive being forked across DataLoader workers).
        self._file: Optional[h5py.File] = None

    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # FIX (do not revert): this used to compute RRh, RhR (two full 1600x1600
        # complex matmuls) and reshape all three matrices into [3200,40,40]
        # channels HERE, on a single CPU thread, for every sample, every epoch.
        # That was the ~10x data-loading bottleneck confirmed empirically during
        # training (data-wait time dominating GPU compute time). This method now
        # does the minimum possible: decompress R from the H5 file, optionally
        # rms-normalize, and build the phasor targets -- nothing else. RRh/RhR
        # and the channel reshape now happen batched on the GPU, in the training
        # loop (see build_model_inputs), where a matmul this size is cheap.
        file_idx = self.indices[idx]
        f = self._ensure_open()

        R_real = f["R_matrices"][file_idx, 0].astype(np.float32)
        R_imag = f["R_matrices"][file_idx, 1].astype(np.float32)

        if self.normalize_by_rms:
            rms_val = float(f["rms_scaling_factors"][file_idx])
            if rms_val > 0:
                R_real = R_real / rms_val
                R_imag = R_imag / rms_val

        phi_o = f["phi_o_maps"][file_idx].astype(np.float32)
        phi_i = f["phi_i_maps"][file_idx].astype(np.float32)
        target_output = np.stack([np.cos(phi_o), np.sin(phi_o)], axis=0).astype(np.float32)
        target_input = np.stack([np.cos(phi_i), np.sin(phi_i)], axis=0).astype(np.float32)
        truth = np.concatenate([target_output, target_input], axis=0)
        target_object = f["target_objects"][file_idx].astype(np.float32)

        return {
            "target_output": torch.from_numpy(target_output),
            "target_input": torch.from_numpy(target_input),
            "truth": torch.from_numpy(truth),
            "R_real": torch.from_numpy(R_real),
            "R_imag": torch.from_numpy(R_imag),
            "target_object": torch.from_numpy(target_object),
        }


def _make_dataloader(dataset: Dataset, shuffle: bool) -> DataLoader:
    # FIX (do not revert): the default 'fork' start method on Linux copies the
    # CUDA context that's already been initialized in the main process (model
    # was already moved to DEVICE='cuda' before any loader is built) into each
    # worker process. Forking after CUDA init is a well-known source of silent
    # hangs -- workers here never touch the GPU at all, so there's no downside
    # to spawning them fresh instead.
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        multiprocessing_context="spawn" if NUM_WORKERS > 0 else None,
        persistent_workers=NUM_WORKERS > 0,
    )


def build_curriculum_loader(active_stages: List[str]) -> DataLoader:
    dataset = ARCH5Dataset(H5_FILE, split="train", difficulty_tags=active_stages)
    return _make_dataloader(dataset, shuffle=True)


def build_stage_matched_val_loader(active_stages: List[str]) -> DataLoader:
    val_tags = [STAGE_TO_VAL_TAG[s] for s in active_stages]
    dataset = ARCH5Dataset(H5_FILE, split="val", difficulty_tags=val_tags)
    return _make_dataloader(dataset, shuffle=False)


def build_full_val_loader() -> DataLoader:
    # difficulty_tags=None -> every val_* tag, regardless of curriculum progress.
    # Reserved purely for honest overall-progress reporting; never used for the
    # advancement decision.
    dataset = ARCH5Dataset(H5_FILE, split="val", difficulty_tags=None)
    return _make_dataloader(dataset, shuffle=False)


# =============================================================================
# 3. PHYSICS-INFORMED COMPOSITE LOSS
# =============================================================================


class PhysicsInformedLoss(nn.Module):
    def __init__(self, grid_size: int = GRID_SIZE, weights: Optional[Dict[str, float]] = None):
        super().__init__()
        w = weights or LOSS_WEIGHTS
        self.grid_size = grid_size
        self.w_coherent = w["coherent"]
        self.w_local = w["local"]
        self.w_strehl = w["strehl"]
        self.w_toeplitz = w["toeplitz"]
        self.w_image = w["image"]
        self.w_image_l1 = w["image_l1"]

        self.grid_size = grid_size
        self.num_bins = grid_size * grid_size

        # Precompute the 2D spatial-frequency DIFFERENCE bin id for every
        # (row, col) pair of the flattened 1600x1600 matrix. This must match
        # EXACTLY the convention already validated in zernike.py's
        # construct_scattering_matrix (diff_v, diff_u mod grid_size) -- this is
        # what makes the Toeplitz loss check the correct 2D block-Toeplitz
        # structure, not a naive (and physically wrong, for this 2D problem)
        # 1D raw-diagonal grouping.
        u = torch.arange(grid_size)
        uu, vv = torch.meshgrid(u, u, indexing="ij")
        u_flat = uu.flatten()  # column component
        v_flat = vv.flatten()  # row component
        diff_u = (u_flat[:, None] - u_flat[None, :]) % grid_size
        diff_v = (v_flat[:, None] - v_flat[None, :]) % grid_size
        # IMPORTANT: The dataset generator indexes the object FFT as object_fft[diff_u, diff_v].
        # In row-major flattening, the matching Delta-k bin is diff_u * N + diff_v.
        # Swapping these terms transposes the Delta-k spectrum and rotates asymmetric objects.
        bin_id = (diff_u * grid_size + diff_v).long().flatten()
        self.register_buffer("bin_id", bin_id, persistent=False)

        # All Δk bins contain the same number of matrix entries for this
        # periodic 2D-difference convention. Keeping the counts explicit
        # makes the reconstruction definition unambiguous.
        bin_counts = torch.bincount(bin_id, minlength=self.num_bins).to(torch.float32)
        self.register_buffer("bin_counts", bin_counts, persistent=False)

    @staticmethod
    def _phasor_deltas(pred_phasor: Tensor, target_phasor: Tensor) -> Tuple[Tensor, Tensor]:
        # Both [B, 2, H, W] unit phasors: channel 0 = cos, channel 1 = sin.
        # cos(a-b) = cos(a)cos(b) + sin(a)sin(b) ; sin(a-b) = sin(a)cos(b) - cos(a)sin(b)
        # -- no atan2 anywhere in this loss, matching the model's own rule.
        pred_cos, pred_sin = pred_phasor[:, 0:1], pred_phasor[:, 1:2]
        targ_cos, targ_sin = target_phasor[:, 0:1], target_phasor[:, 1:2]
        cos_delta = pred_cos * targ_cos + pred_sin * targ_sin
        sin_delta = pred_sin * targ_cos - pred_cos * targ_sin
        return cos_delta, sin_delta

    @staticmethod
    def _masked_mean(x: Tensor, mask: Optional[Tensor]) -> Tensor:
        if mask is None:
            return x.mean()
        mask = mask.to(dtype=x.dtype, device=x.device)
        denom = mask.sum().clamp(min=1.0) * (x.numel() / mask.numel())
        return (x * mask).sum() / denom

    def _coherent_term(self, cos_delta: Tensor, sin_delta: Tensor, mask: Optional[Tensor]) -> Tensor:
        diff = torch.complex(cos_delta, sin_delta)  # [B,1,H,W]
        if mask is not None:
            mask_c = mask.to(dtype=diff.real.dtype, device=diff.device)
            diff = diff * mask_c
            denom = mask_c.sum().clamp(min=1.0)
            coherent_mean = diff.sum(dim=(-2, -1)).squeeze(-1) / denom
        else:
            coherent_mean = diff.mean(dim=(-2, -1)).squeeze(-1)
        return 1.0 - torch.abs(coherent_mean).mean()

    def _toeplitz_term(self, R_c: Tensor) -> Tensor:
        """R_c: complex tensor [B, 1600, 1600] -- the model's own corrected
        matrix. Penalizes deviation from block-Toeplitz structure: entries
        sharing the same true 2D frequency difference should be consistent."""
        b = R_c.shape[0]
        real = R_c.real.reshape(b, -1)
        imag = R_c.imag.reshape(b, -1)

        batch_offset = (torch.arange(b, device=R_c.device) * self.num_bins).unsqueeze(1)
        batched_bin_id = (self.bin_id.unsqueeze(0) + batch_offset).reshape(-1)
        total_bins = b * self.num_bins

        flat_real = real.reshape(-1)
        flat_imag = imag.reshape(-1)

        sum_real = torch.zeros(total_bins, device=R_c.device, dtype=flat_real.dtype)
        sum_imag = torch.zeros(total_bins, device=R_c.device, dtype=flat_imag.dtype)
        counts = torch.zeros(total_bins, device=R_c.device, dtype=flat_real.dtype)
        sum_real.index_add_(0, batched_bin_id, flat_real)
        sum_imag.index_add_(0, batched_bin_id, flat_imag)
        counts.index_add_(0, batched_bin_id, torch.ones_like(flat_real))
        counts = counts.clamp(min=1.0)

        mean_real = (sum_real / counts)[batched_bin_id]
        mean_imag = (sum_imag / counts)[batched_bin_id]
        deviation = (flat_real - mean_real) ** 2 + (flat_imag - mean_imag) ** 2
        return deviation.mean()


    def _reconstruct_from_R_c(self, R_c: Tensor, grid_size: int) -> Tensor:
        """Differentiable Δk-based CASS-style reconstruction.

        All corrected reflection-matrix entries are coherently accumulated
        into their corresponding 2D momentum-transfer bin using the same
        binning convention already used by the Toeplitz regularizer.
        The resulting object-spectrum estimate is reshaped to the 40x40
        Δk grid, inverse-shifted, inverse Fourier transformed, converted
        to intensity, and normalized per sample.
        """
        b = R_c.shape[0]

        # Flatten the corrected reflection matrix while preserving the same
        # row/column ordering used to construct self.bin_id.
        real = R_c.real.reshape(b, -1)
        imag = R_c.imag.reshape(b, -1)

        # Coherently accumulate all matrix entries sharing the same Δk.
        object_real = torch.zeros(
            b, self.num_bins, device=R_c.device, dtype=real.dtype
        )
        object_imag = torch.zeros(
            b, self.num_bins, device=R_c.device, dtype=imag.dtype
        )

        object_real.index_add_(1, self.bin_id, real)
        object_imag.index_add_(1, self.bin_id, imag)

        object_spectrum = torch.complex(object_real, object_imag)
        object_spectrum = object_spectrum.reshape(b, grid_size, grid_size)

        # Match the project reconstruction convention.
        shift = grid_size // 2
        object_spectrum = torch.roll(
            torch.roll(object_spectrum, shift, dims=-1),
            shift, dims=-2
        )

        spatial = torch.fft.ifft2(object_spectrum)
        intensity = spatial.real.square() + spatial.imag.square()

        # Normalize each reconstructed image to [0, 1] before SSIM/L1.
        mx = intensity.flatten(1).max(dim=1).values.clamp(min=1e-12)
        intensity = intensity / mx.view(b, 1, 1)

        return intensity.unsqueeze(1)

    @staticmethod
    def _ssim_loss(pred: Tensor, target: Tensor,
                   window_size: int = 7, C1: float = 0.01**2, C2: float = 0.03**2) -> Tensor:
        """Differentiable patch-based SSIM loss (1 - SSIM), both inputs in [0,1].
        Uses a uniform averaging kernel (simpler than Gaussian, avoids border
        artifacts at 40x40, and gradients are cleaner). Returns a scalar."""
        import torch.nn.functional as F
        kernel = torch.ones(1, 1, window_size, window_size,
                            device=pred.device, dtype=pred.dtype) / (window_size ** 2)
        pad = window_size // 2

        mu_x  = F.conv2d(pred,   kernel, padding=pad, groups=1)
        mu_y  = F.conv2d(target, kernel, padding=pad, groups=1)
        mu_xx = F.conv2d(pred   ** 2, kernel, padding=pad, groups=1)
        mu_yy = F.conv2d(target ** 2, kernel, padding=pad, groups=1)
        mu_xy = F.conv2d(pred * target, kernel, padding=pad, groups=1)

        sigma_x  = mu_xx - mu_x ** 2
        sigma_y  = mu_yy - mu_y ** 2
        sigma_xy = mu_xy - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        return 1.0 - ssim_map.mean()

    @staticmethod
    def _foreground_weighted_l1_loss(
        pred: Tensor,
        target: Tensor,
        foreground_boost: float = 3.0,
    ) -> Tensor:
        """Foreground-weighted L1 loss.

        Background pixels retain unit weight. Target foreground pixels receive
        additional weight so thin crosses, bars, letters, and USAF structures
        contribute more strongly than the large background area.
        """
        weights = 1.0 + foreground_boost * target
        return (weights * torch.abs(pred - target)).mean()

    def forward(
        self,
        pred_output_phasor: Tensor,
        target_output_phasor: Tensor,
        pred_input_phasor: Tensor,
        target_input_phasor: Tensor,
        R_complex: Tensor,
        mask: Optional[Tensor] = None,
        strehl_weight_scale: float = 1.0,
        target_object: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        cos_o, sin_o = self._phasor_deltas(pred_output_phasor, target_output_phasor)
        cos_i, sin_i = self._phasor_deltas(pred_input_phasor, target_input_phasor)

        l_coherent_o = self._coherent_term(cos_o, sin_o, mask)
        l_coherent_i = self._coherent_term(cos_i, sin_i, mask)
        l_coherent = 0.5 * (l_coherent_o + l_coherent_i)

        local_o = 1.0 - cos_o
        local_i = 1.0 - cos_i
        l_local_o = self._masked_mean(local_o, mask)
        l_local_i = self._masked_mean(local_i, mask)
        l_local = 0.5 * (l_local_o + l_local_i)

        l_strehl_o = 1.0 - torch.exp(-l_local_o)
        l_strehl_i = 1.0 - torch.exp(-l_local_i)
        l_strehl = 0.5 * (l_strehl_o + l_strehl_i)

        # Build the model's own corrected matrix R_c = P_o^dagger R P_i^dagger
        # from its PREDICTED phasors (not ground truth) -- this tests whether
        # the network's own correction actually produces Toeplitz-consistent
        # output, which is the whole physical point of the regularizer.
        b = pred_output_phasor.shape[0]
        phasor_o = torch.complex(pred_output_phasor[:, 0], pred_output_phasor[:, 1]).reshape(b, -1)
        phasor_i = torch.complex(pred_input_phasor[:, 0], pred_input_phasor[:, 1]).reshape(b, -1)
        R_c = torch.conj(phasor_o).unsqueeze(-1) * R_complex * torch.conj(phasor_i).unsqueeze(-2)
        l_toeplitz = self._toeplitz_term(R_c)

        # Image-domain supervision: reconstruct the corrected image from R_c
        # differentiably and compare against the ground-truth object with SSIM.
        # This is the term that directly penalizes "the corrected image looks
        # nothing like the object" -- exactly the failure mode for crosses and
        # USAF targets where phase accuracy alone is insufficient.
        if target_object is not None and (self.w_image > 0 or self.w_image_l1 > 0):
            # The image-domain supervision now uses a full Δk-based
            # reconstruction from the model's own corrected matrix.
            img_pred = self._reconstruct_from_R_c(R_c, self.grid_size)

            # target_object: [B, H, W] -> [B, 1, H, W]
            img_gt = target_object.unsqueeze(1).to(
                dtype=img_pred.dtype,
                device=img_pred.device,
            )

            l_image = self._ssim_loss(img_pred, img_gt)
            l_image_l1 = self._foreground_weighted_l1_loss(img_pred, img_gt)
        else:
            l_image = torch.zeros(
                (), device=R_complex.device, dtype=torch.float32
            )
            l_image_l1 = torch.zeros(
                (), device=R_complex.device, dtype=torch.float32
            )

        total = (
            self.w_coherent * l_coherent
            + self.w_local * l_local
            + strehl_weight_scale * self.w_strehl * l_strehl
            + self.w_toeplitz * l_toeplitz
            + self.w_image * l_image
            + self.w_image_l1 * l_image_l1
        )

        return total, {
            "coherent": l_coherent.item(),
            "local": l_local.item(),
            "strehl": l_strehl.item(),
            "toeplitz": l_toeplitz.item(),
            "image": l_image.item(),
            "image_l1": l_image_l1.item(),
        }


def enable_mc_dropout(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def renormalize_phasor(x: Tensor, eps: float = 1e-8) -> Tensor:
    mag = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + eps)
    return x / mag


def save_visualization(
    batch: Dict[str, Tensor],
    pred_output_phasor: Tensor,
    pred_input_phasor: Tensor,
    epoch: int,
    stage_name: str,
    save_dir: str = VIS_DIR,
    sample_idx: int = 0,
) -> None:
    """4x3 panel: row 0 = the three inputs (R, RR^dagger, R^dagger R log-amplitude,
    reconstructed from the raw matrix -- NOT an arbitrary channel slice of the
    reshaped [3200,40,40] tensor, since we already have the true 1600x1600 R
    available); row 1 = input aberration (target, predicted, residual); row 2 =
    output aberration (target, predicted, residual); row 3 = ground truth
    object, uncorrected reconstructed image, model-corrected reconstructed
    image.

    Residual computation: NEVER pred_angle - target_angle directly (breaks at
    the +-pi wraparound). Instead, derived from the phasor DIFFERENCE -- the
    exact same cos/sin identity PhysicsInformedLoss uses -- then atan2 of that,
    which is inherently wrap-safe.

    Row 3 reconstruction: the Uncorrected panel uses the plain dc-column
    method (CASS_visualizer.reconstruct_dc_column) -- reverted back from full
    CASS on purpose: CASS's Δk-bin averaging suppresses noise so strongly
    (~40x SNR improvement, verified numerically) that an "Uncorrected"
    panel built from it can look nearly clean even at noise_med/noise_high,
    which was confusing during training runs -- dc-column shows this
    sample's actual raw noise/aberration level honestly, with no
    reconstruction-side noise suppression at that point. The Corrected panel
    still uses full CASS (reconstruct_cass) -- unchanged, since that's the
    same reconstruction PhysicsInformedLoss._reconstruct_from_R_c uses
    internally for the SSIM/L1 image-domain loss terms, so this panel stays
    an honest "what does the network's current correction actually do to
    the image" check against what's actually being optimized. It uses the
    model's own predicted phasors to build R_c = P_o^dagger R P_i^dagger --
    the exact same construction PhysicsInformedLoss uses internally for the
    Toeplitz term. Verified numerically: feeding the TRUE aberration into
    this same formula recovers the clean object reconstruction exactly.
    """
    os.makedirs(save_dir, exist_ok=True)

    R_complex = torch.complex(batch["R_real"][sample_idx], batch["R_imag"][sample_idx]).cpu().numpy()
    RRh = R_complex @ R_complex.conj().T
    RhR = R_complex.conj().T @ R_complex

    log_amp_R = np.log10(np.abs(R_complex) + 1e-5)
    log_amp_RRh = np.log10(np.abs(RRh) + 1e-5)
    log_amp_RhR = np.log10(np.abs(RhR) + 1e-5)

    target_output_phasor = batch["target_output"][sample_idx].detach().cpu()
    target_input_phasor = batch["target_input"][sample_idx].detach().cpu()
    pred_o = pred_output_phasor[sample_idx].detach().cpu()
    pred_i = pred_input_phasor[sample_idx].detach().cpu()

    # atan2 ONLY here, at the visualization boundary -- never inside the model
    # or the loss, same rule established throughout the rest of the project.
    target_phi_o = torch.atan2(target_output_phasor[1], target_output_phasor[0]).numpy()
    target_phi_i = torch.atan2(target_input_phasor[1], target_input_phasor[0]).numpy()
    pred_phi_o = torch.atan2(pred_o[1], pred_o[0]).numpy()
    pred_phi_i = torch.atan2(pred_i[1], pred_i[0]).numpy()

    def wrap_safe_residual(pred_phasor: Tensor, target_phasor: Tensor) -> np.ndarray:
        pred_cos, pred_sin = pred_phasor[0], pred_phasor[1]
        targ_cos, targ_sin = target_phasor[0], target_phasor[1]
        cos_delta = pred_cos * targ_cos + pred_sin * targ_sin
        sin_delta = pred_sin * targ_cos - pred_cos * targ_sin
        return torch.atan2(sin_delta, cos_delta).numpy()

    residual_o = wrap_safe_residual(pred_o, target_output_phasor)
    residual_i = wrap_safe_residual(pred_i, target_input_phasor)

    # FIX (do not revert): outside the NA aperture, PhasorHead forces the
    # predicted phasor to exactly (0,0) -- an inherently degenerate input to
    # atan2, whose sign-of-zero-driven output was verified to flicker
    # unpredictably between 0, +pi, -pi across different pixels (the "blob"
    # artifacts). Rather than cropping the canvas or blanking to white/NaN
    # (which loses the visual context), explicitly set the display value to
    # 0 there -- the same "no aberration" convention already used everywhere
    # else, so it blends into the existing flat background instead of
    # standing out. Inside-aperture pixels are completely untouched (verified
    # numerically). Target arrays are already naturally 0 there (Zernike
    # modes are 0 outside the pupil by construction, an unambiguous atan2
    # input), so this is a no-op for them and only actually changes the
    # predicted/residual arrays -- applied uniformly for simplicity.
    na_mask_np = make_na_mask(size=GRID_SIZE, nasz=GRID_SIZE // 2, device="cpu").numpy().astype(bool).squeeze()

    def zero_outside_aperture(arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        arr[~na_mask_np] = 0.0
        return arr

    target_phi_o = zero_outside_aperture(target_phi_o)
    target_phi_i = zero_outside_aperture(target_phi_i)
    pred_phi_o = zero_outside_aperture(pred_phi_o)
    pred_phi_i = zero_outside_aperture(pred_phi_i)
    residual_o = zero_outside_aperture(residual_o)
    residual_i = zero_outside_aperture(residual_i)

    n = log_amp_R.shape[0]
    grid_size = int(round(n ** 0.5))
    # Same y-flipped extent convention established in dataset_generator.py, so
    # these matrix panels read consistently with the dataset verification figures.
    matrix_extent = [0, n, n, 0]

    # -------- ground truth / uncorrected / model-corrected images --------
    # Uncorrected: plain dc-column (reverted from CASS -- see docstring).
    uncorrected_image = reconstruct_dc_column(R_complex, grid_size)

    # Corrected: still full CASS -- unchanged, matches the loss (see docstring).
    phasor_o_complex = torch.complex(pred_o[0], pred_o[1]).flatten().numpy()
    phasor_i_complex = torch.complex(pred_i[0], pred_i[1]).flatten().numpy()
    R_c = np.conj(phasor_o_complex)[:, None] * R_complex * np.conj(phasor_i_complex)[None, :]
    corrected_image = reconstruct_cass(R_c, grid_size)

    ground_truth_object = batch["target_object"][sample_idx].detach().cpu().numpy()

    fig, axes = plt.subplots(4, 3, figsize=(16, 20))
    for row in axes:
        for ax in row:
            ax.set_xlabel("Pixels")
            ax.set_ylabel("Pixels")

    im0 = axes[0, 0].imshow(log_amp_R, cmap="magma", extent=matrix_extent, origin="upper")
    axes[0, 0].set_title("Input R -- log|R|")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(log_amp_RRh, cmap="magma", extent=matrix_extent, origin="upper")
    axes[0, 1].set_title(r"$RR^\dagger$ -- log amplitude")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(log_amp_RhR, cmap="magma", extent=matrix_extent, origin="upper")
    axes[0, 2].set_title(r"$R^\dagger R$ -- log amplitude")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # NOTE: jet is not a cyclic colormap (unlike twilight), so there will be a
    # visible color seam at the +-pi wrap boundary (dark red at +pi vs. dark
    # blue at -pi, rather than matching colors) -- using it anyway per request,
    # since it's the more familiar colormap; just worth knowing that seam is a
    # colormap artifact, not a data discontinuity.
    phase_kwargs = dict(cmap="jet", vmin=-1.0, vmax=1.0, interpolation="nearest")
    pi_ticks = [-1.0, 0.0, 1.0]
    pi_tick_labels = [r"$-\pi$", "0", r"$\pi$"]

    def _set_pi_colorbar(mappable, ax):
        cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cb.set_ticks(pi_ticks)
        cb.set_ticklabels(pi_tick_labels)
        cb.set_label("phase")
        return cb

    im3 = axes[1, 0].imshow(target_phi_i / np.pi, **phase_kwargs)
    axes[1, 0].set_title("Target Input Phase")
    _set_pi_colorbar(im3, axes[1, 0])

    im4 = axes[1, 1].imshow(pred_phi_i / np.pi, **phase_kwargs)
    axes[1, 1].set_title("Predicted Input Phase")
    _set_pi_colorbar(im4, axes[1, 1])

    im5 = axes[1, 2].imshow(residual_i / np.pi, cmap="jet", vmin=-1.0, vmax=1.0, interpolation="nearest")
    axes[1, 2].set_title("Residual Input Phase")
    _set_pi_colorbar(im5, axes[1, 2])

    im6 = axes[2, 0].imshow(target_phi_o / np.pi, **phase_kwargs)
    axes[2, 0].set_title("Target Output Phase")
    _set_pi_colorbar(im6, axes[2, 0])

    im7 = axes[2, 1].imshow(pred_phi_o / np.pi, **phase_kwargs)
    axes[2, 1].set_title("Predicted Output Phase")
    _set_pi_colorbar(im7, axes[2, 1])

    im8 = axes[2, 2].imshow(residual_o / np.pi, cmap="jet", vmin=-1.0, vmax=1.0, interpolation="nearest")
    axes[2, 2].set_title("Residual Output Phase")
    _set_pi_colorbar(im8, axes[2, 2])

    im9 = axes[3, 0].imshow(ground_truth_object, cmap="gray", interpolation="nearest")
    axes[3, 0].set_title("Ground Truth Object")
    fig.colorbar(im9, ax=axes[3, 0], fraction=0.046, pad=0.04)

    unc_norm = uncorrected_image / (uncorrected_image.max() + 1e-12)
    im10 = axes[3, 1].imshow(unc_norm, cmap="hot", vmin=0.0, vmax=1.0, interpolation="bicubic")
    axes[3, 1].set_title("Uncorrected Image")
    fig.colorbar(im10, ax=axes[3, 1], fraction=0.046, pad=0.04)

    corr_norm = corrected_image / (corrected_image.max() + 1e-12)
    im11 = axes[3, 2].imshow(corr_norm, cmap="hot", vmin=0.0, vmax=1.0, interpolation="bicubic")
    axes[3, 2].set_title("Model-Corrected Image (CASS)")
    fig.colorbar(im11, ax=axes[3, 2], fraction=0.046, pad=0.04)

    stage_tag = stage_name.replace(" ", "_")
    plt.suptitle(f"Epoch {epoch} -- Stage: {stage_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch:04d}_{stage_tag}.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# 4. TRAINING LOOP
# =============================================================================


def move_batch_to_device(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _reshape_to_channels_gpu(mat_complex: Tensor, transpose_first: bool, grid_size: int = GRID_SIZE) -> Tensor:
    """Batched GPU version of the old per-sample CPU/NumPy reshape:
    [B, N, N] complex -> [B, 2*N, grid_size, grid_size] float32 (real channels,
    then imag), N = grid_size*grid_size. Uses .reshape() (not .view()) since
    the transpose below produces a non-contiguous tensor and .real/.imag are
    views too -- .reshape() copies as needed, .view() would raise."""
    b = mat_complex.shape[0]
    n = grid_size * grid_size
    m = mat_complex.transpose(-2, -1) if transpose_first else mat_complex
    real = m.real.reshape(b, n, grid_size, grid_size)
    imag = m.imag.reshape(b, n, grid_size, grid_size)
    return torch.cat([real, imag], dim=1)


def build_model_inputs(batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute RRh, RhR, and all three [3200,40,40] model inputs from the raw
    R already resident on the GPU (batch must already be moved to device).

    FIX (do not revert): this used to happen per-sample, on CPU, inside
    Dataset.__getitem__ -- two full 1600x1600 complex matmuls per sample, every
    epoch, single-threaded, which was empirically confirmed to be the ~10x
    data-loading bottleneck (data-wait time dominating GPU compute time in
    training logs). Doing it here instead means: (1) it's batched across the
    whole batch at once rather than looped one sample at a time, and (2) it
    runs on the GPU, which is >2 orders of magnitude faster at large matmuls
    than a single CPU thread -- and the GPU was sitting idle waiting for this
    exact computation anyway. Also reduces host->device transfer, since only
    the raw R (not three already-expanded [3200,40,40] tensors) needs to cross
    PCIe.

    FIX (do not revert): RRh/RhR are each a SUM of 1600 products of R's
    (already RMS-normalized, order-~1) entries -- verified numerically their
    own characteristic magnitude is therefore much larger than R's (diagonal
    entries land around magnitude ~800 in a representative test, vs R's own
    RMS of ~0.7), even though R itself is well-scaled. Feeding values at that
    scale into the model's forward pass, which runs under AMP/autocast (fp16,
    max representable ~65504), is a genuine overflow risk before any weight
    multiplication or squaring even happens inside the network. A previous
    labmate's own script explicitly warns about exactly this failure mode
    ("amp with unstandardized RRt/RtR inputs can overflow because covariance
    values are large") and defends against it with per-sample standardization.
    Standardize each of RRh/RhR by its OWN per-sample RMS magnitude here
    (mirroring how R itself is already normalized by its own
    rms_scaling_factor at the dataset level) so both land at a moderate,
    AMP-safe scale before ever reaching the model. R_complex itself is
    deliberately left untouched and returned raw -- it's used elsewhere (the
    loss's Toeplitz term, image reconstruction, R_c construction) where the
    true, unscaled magnitude is exactly what's physically meaningful.

    Returns (r, rrh, rhr, R_complex).
    """
    R_complex = torch.complex(batch["R_real"], batch["R_imag"])  # [B, 1600, 1600]

    RRh = R_complex @ R_complex.conj().transpose(-2, -1)  # output aberration cancels
    RhR = R_complex.conj().transpose(-2, -1) @ R_complex  # input aberration cancels

    def _standardize_per_sample(mat: Tensor) -> Tensor:
        # RMS computed per-sample (per batch index), over all N*N entries --
        # same "divide by sqrt(mean(|.|^2))" convention already used for R's
        # own normalization by rms_scaling_factor.
        rms = torch.sqrt(torch.mean(mat.abs() ** 2, dim=(-2, -1), keepdim=True).clamp(min=1e-12))
        return mat / rms

    RRh = _standardize_per_sample(RRh)
    RhR = _standardize_per_sample(RhR)

    r = _reshape_to_channels_gpu(R_complex, transpose_first=True)
    rrh = _reshape_to_channels_gpu(RRh, transpose_first=False)
    rhr = _reshape_to_channels_gpu(RhR, transpose_first=False)
    return r, rrh, rhr, R_complex


def run_training() -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    model = ARC(input_channels=N_ELEMENTS * 2, normalize_output=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    # FIX (do not revert): noise curriculum MUST start from the aberration
    # checkpoint, not random weights. Crash loudly if it's missing so there
    # is never any ambiguity about whether the model was actually loaded.
    if not os.path.exists(PRETRAINED_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"\n\n{'='*70}\n"
            f"PRETRAINED CHECKPOINT NOT FOUND:\n  {PRETRAINED_CHECKPOINT_PATH}\n"
            f"Set PRETRAINED_CHECKPOINT_PATH at the top of this script to the\n"
            f"ARC_best.pth produced by the aberration curriculum training.\n"
            f"{'='*70}\n"
        )
    model.load_state_dict(torch.load(PRETRAINED_CHECKPOINT_PATH, map_location=DEVICE))
    print(f"\n{'='*70}")
    print(f"  Model successfully loaded from:")
    print(f"  {PRETRAINED_CHECKPOINT_PATH}")
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*70}\n")

    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    loss_fn = PhysicsInformedLoss(grid_size=GRID_SIZE).to(DEVICE)
    na_mask = make_na_mask(size=GRID_SIZE, nasz=GRID_SIZE // 2, device=DEVICE)

    # FIX (do not revert): reduces LR when stage_val_loss plateaus WITHIN a
    # stage, giving a stalled run a chance to keep improving via a lower LR
    # before patience gives up on the stage entirely. Recreated fresh at every
    # curriculum advancement (see below) -- its internal "best value seen" and
    # bad-epoch counter are meaningless across two different training
    # distributions, same reasoning as best_stage_val_loss.
    def make_scheduler(opt: optim.Optimizer) -> optim.lr_scheduler.ReduceLROnPlateau:
        return optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE, min_lr=LR_SCHEDULER_MIN_LR,
        )

    scheduler = make_scheduler(optimizer)

    full_val_loader = build_full_val_loader()

    current_stage_idx = 0
    active_stages = [CURRICULUM_STAGES[current_stage_idx]]
    epochs_into_stage = 0  # drives the warmup ramp; reset at each curriculum transition

    best_stage_val_loss = float("inf")
    patience_counter = 0

    csv_path = METRICS_CSV_PATH
    csv_columns = ["Epoch", "Stage", "LR", "Train_Loss"]
    for comp in ("Coherent", "Local", "Strehl", "Toeplitz", "Image", "ImageL1"):
        csv_columns.append(f"Train_{comp}")
    csv_columns.append("StageVal_Loss")
    for comp in ("Coherent", "Local", "Strehl", "Toeplitz", "Image", "ImageL1"):
        csv_columns.append(f"StageVal_{comp}")
    csv_columns.append("FullVal_Loss")
    for comp in ("Coherent", "Local", "Strehl", "Toeplitz", "Image", "ImageL1"):
        csv_columns.append(f"FullVal_{comp}")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_columns)

    print(f"--> Starting ARC curriculum training on noise on {DEVICE.upper()}...")

    train_loader: Optional[DataLoader] = None
    stage_val_loader: Optional[DataLoader] = None
    loaders_built_for_stages: Optional[List[str]] = None

    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_into_stage += 1
        if epochs_into_stage <= WARMUP_EPOCHS:
            warmup_lr = INITIAL_LR * (epochs_into_stage / WARMUP_EPOCHS)
            for _pg in optimizer.param_groups:
                _pg["lr"] = warmup_lr
            if epochs_into_stage == 1:
                print(f"  [WARMUP] Ramping LR over {WARMUP_EPOCHS} epochs -> target {INITIAL_LR:.2e}")

        strehl_scale = strehl_weight_scale_for_stage(current_stage_idx)
        current_lr = optimizer.param_groups[0]["lr"]

        # FIX (do not revert): this used to rebuild both DataLoaders (and, since
        # we use multiprocessing_context="spawn", respawn brand-new worker
        # processes) every single epoch, even when active_stages hadn't
        # changed since the last one -- silently defeating
        # persistent_workers=True. Only rebuild when the curriculum actually
        # advances; `active_stages` is compared against a snapshot copy taken
        # at the last rebuild, since it's mutated in place via .append() when
        # advancing.
        if loaders_built_for_stages != active_stages:
            t_build_start = time.time()
            train_loader = build_curriculum_loader(active_stages)
            stage_val_loader = build_stage_matched_val_loader(active_stages)
            loaders_built_for_stages = list(active_stages)

        # ---------------- train ----------------
        model.train()
        epoch_train_loss = 0.0
        train_component_sums = {
            "coherent": 0.0,
            "local": 0.0,
            "strehl": 0.0,
            "toeplitz": 0.0,
            "image": 0.0,
            "image_l1": 0.0,
        }
        n_train_batches = 0
        data_wait_time = 0.0
        compute_time = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [{active_stages[-1]}]")
        t_batch_start = time.time()
        for batch_idx, batch in enumerate(pbar):
            data_wait_time += time.time() - t_batch_start  # time DataLoader took to hand us this batch

            t_compute_start = time.time()
            batch = move_batch_to_device(batch, DEVICE)
            r, rrh, rhr, R_complex = build_model_inputs(batch)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                pred = model(r, rrh, rhr)
                loss, loss_parts = loss_fn(
                    pred["output_aberration"], batch["target_output"],
                    pred["input_aberration"], batch["target_input"],
                    R_complex, mask=na_mask, strehl_weight_scale=strehl_scale,
                    target_object=batch["target_object"].to(DEVICE),
                )

            if not torch.isfinite(loss):
                print(
                    f"  [WARNING] Non-finite loss at epoch {epoch}, batch {batch_idx}: "
                    f"{float(loss.detach().cpu())}. Skipping batch -- optimizer NOT stepped. "
                    "If this recurs frequently, reduce INITIAL_LR or increase WARMUP_EPOCHS."
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            compute_time += time.time() - t_compute_start

            epoch_train_loss += float(loss.detach().cpu())
            for k, v in loss_parts.items():
                train_component_sums[k] += v
            n_train_batches += 1

            # Running visibility: every 10 batches, show the data-wait vs
            # compute split so a data-loading bottleneck (e.g. compressed H5
            # reads + CPU matmuls for RRh/RhR) is immediately obvious rather
            # than looking like a generic hang.
            if batch_idx % 10 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{current_lr:.2e}",
                    data_s=f"{data_wait_time:.1f}",
                    compute_s=f"{compute_time:.1f}",
                    **{k: f"{v:.4f}" for k, v in loss_parts.items()},
                )

            t_batch_start = time.time()

        n_train_batches = max(n_train_batches, 1)
        avg_train_loss = epoch_train_loss / n_train_batches
        avg_train_components = {k: v / n_train_batches for k, v in train_component_sums.items()}

        # ---------------- stage-matched validation (advancement decision only) ----------------
        model.eval()
        enable_mc_dropout(model)
        stage_val_loss, stage_val_components = evaluate(model, stage_val_loader, loss_fn, na_mask, strehl_scale)

        # ---------------- full validation (reporting only, never gates advancement) ----------------
        full_val_loss, full_val_components = evaluate(model, full_val_loader, loss_fn, na_mask, strehl_scale)

        print(
            f"Epoch {epoch:03d} | Stage: {active_stages[-1]:>10s} | LR {current_lr:.2e} | "
            f"Train {avg_train_loss:.5f} | StageVal {stage_val_loss:.5f} | FullVal {full_val_loss:.5f}"
        )
        with open(csv_path, "a", newline="") as f:
            row = [epoch, active_stages[-1], current_lr, avg_train_loss]
            row += [avg_train_components[c] for c in ("coherent", "local", "strehl", "toeplitz", "image", "image_l1")]
            row += [stage_val_loss]
            row += [stage_val_components[c] for c in ("coherent", "local", "strehl", "toeplitz", "image", "image_l1")]
            row += [full_val_loss]
            row += [full_val_components[c] for c in ("coherent", "local", "strehl", "toeplitz", "image", "image_l1")]
            csv.writer(f).writerow(row)

        # FIX (do not revert): checkpoints used to only ever overwrite a single
        # fixed filename ("last_checkpoint.pt"), so only the most recent one
        # ever survived on disk -- unlike the visualizations, which already had
        # unique per-epoch, stage-tagged filenames. Now saves BOTH: the
        # overwritten "last_checkpoint.pt" (quick "resume from most recent"
        # without hunting for the highest epoch number) AND a uniquely-named,
        # stage-tagged checkpoint every CHECKPOINT_INTERVAL epochs, matching
        # the visualization's naming convention. Note this means the
        # checkpoints folder will grow substantially over a long run (this
        # model is ~28M params, ~111MB per checkpoint file) -- worth keeping an
        # eye on disk usage if MAX_EPOCHS is large.
        if epoch % CHECKPOINT_INTERVAL == 0:
            stage_tag = active_stages[-1].replace(" ", "_")
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt"))
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:04d}_{stage_tag}.pt"),
            )

            # FIX (do not revert): this used to pull next(iter(full_val_loader)).
            # full_val_loader has shuffle=False, and dataset_generator.py lays
            # the H5 file's "val" split out in a fixed block order
            # (val_baseline samples first, then val_noise_low, val_noise_med,
            # val_noise_high). With no shuffling, the very first batch of
            # full_val_loader is ALWAYS val_baseline (no-noise) samples,
            # regardless of what curriculum stage training is actually in --
            # so the visualization was silently showing a no-noise example the
            # entire time, even deep into noise_high. Built fresh here (rare
            # enough -- once every CHECKPOINT_INTERVAL epochs -- that a
            # throwaway loader is fine) scoped to ONLY the current stage's own
            # val tag, with shuffle=True so it isn't frozen on one fixed
            # sample either.
            vis_dataset = ARCH5Dataset(
                H5_FILE, split="val", difficulty_tags=[STAGE_TO_VAL_TAG[active_stages[-1]]],
            )
            vis_loader = DataLoader(vis_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            vis_batch = next(iter(vis_loader))
            vis_batch = move_batch_to_device(vis_batch, DEVICE)
            model.eval()
            with torch.no_grad():
                vis_r, vis_rrh, vis_rhr, _ = build_model_inputs(vis_batch)
                vis_pred = model(vis_r, vis_rrh, vis_rhr)
            save_visualization(
                vis_batch, vis_pred["output_aberration"], vis_pred["input_aberration"],
                epoch, stage_name=active_stages[-1],
            )

        # ---------------- LR scheduling (stage-matched val, resets on stage change) ----------------
        # Only let the plateau scheduler act once the warmup ramp has finished --
        # otherwise it could reduce LR mid-ramp based on early stage_val_loss readings.
        if epochs_into_stage > WARMUP_EPOCHS:
            scheduler.step(stage_val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < current_lr:
            print(f"  [LR REDUCED] {current_lr:.2e} -> {new_lr:.2e} (stage_val_loss plateaued)")

        # ---------------- patience / curriculum advancement (uses STAGE-matched val only) ----------------
        # FIX (do not revert): this used to be TWO separate mechanisms -- see
        # the comment above PATIENCE_LIMIT's definition for the full story on
        # why that let noise_low/noise_med get skipped down to 2 epochs each
        # while noise_high silently consumed 380 of 400 total epochs. Now a
        # single patience counter, naturally per-stage since it resets at every
        # transition: advance only after PATIENCE_LIMIT consecutive epochs
        # with no genuine (>MIN_DELTA_PERCENT relative) improvement.
        #
        # FIX #2 (do not revert): the very first epoch after ANY reset has
        # previous_best == inf. inf - finite is still inf, and inf / inf is
        # NaN -- and any comparison against NaN (including >=) is silently
        # False in Python, with no error or warning. That meant
        # best_stage_val_loss NEVER actually updated away from infinity, so
        # EVERY epoch in a stage computed the same NaN and silently counted as
        # "no improvement" -- patience_counter incremented unconditionally
        # every single epoch regardless of what the model was actually doing,
        # making PATIENCE_LIMIT behave exactly like a hard fixed-length stage
        # (advance at epoch 15 no matter what) rather than a genuine patience
        # counter. The fix: treat the first post-reset epoch as an
        # unconditional baseline -- record it, reset patience, checkpoint it --
        # WITHOUT running it through the percent_improvement division at all.
        # Only epochs 2+ within a stage (where previous_best is finite) do the
        # real relative-improvement comparison.
        previous_best = best_stage_val_loss

        if previous_best == float("inf"):
            best_stage_val_loss = stage_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience_counter = 0
        else:
            percent_improvement = (previous_best - stage_val_loss) / (previous_best + 1e-8)
            if percent_improvement >= MIN_DELTA_PERCENT:
                best_stage_val_loss = stage_val_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                patience_counter = 0
            else:
                patience_counter += 1

        if patience_counter >= PATIENCE_LIMIT:
            if current_stage_idx < len(CURRICULUM_STAGES) - 1:
                current_stage_idx += 1
                active_stages.append(CURRICULUM_STAGES[current_stage_idx])
                print(f"  [CURRICULUM ADVANCEMENT] -> '{active_stages[-1]}'")
                patience_counter = 0
                best_stage_val_loss = float("inf")  # new stage, new validation distribution
                # FIX (do not revert): resetting epochs_into_stage to 0 hands
                # control to the WARMUP_EPOCHS ramp at the top of the loop,
                # which reaches INITIAL_LR gradually over WARMUP_EPOCHS epochs
                # rather than in one instant jump. An instant reset straight onto
                # harder data was confirmed to cause NaN loss within 15 batches
                # in the aberration curriculum run.
                epochs_into_stage = 0
                scheduler = make_scheduler(optimizer)  # fresh plateau-tracking for the new stage
            else:
                # FIX (do not revert): previously nothing happened here at all --
                # training just ground through every remaining epoch regardless
                # of validation having plateaued, which is exactly what turned
                # into the 380-epoch overfitting run (train loss collapsed 10x
                # while stage_val_loss stayed flat the entire time). Stop for
                # real once the final stage has genuinely plateaued; the best
                # checkpoint by stage_val_loss is already saved separately.
                print(
                    f"  [EARLY STOP] No improvement for {PATIENCE_LIMIT} epochs on the "
                    f"final curriculum stage ('{active_stages[-1]}') -- stopping training "
                    f"at epoch {epoch}. Best checkpoint: {BEST_MODEL_PATH}"
                )
                break


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: PhysicsInformedLoss,
    na_mask: Tensor,
    strehl_scale: float,
) -> Tuple[float, Dict[str, float]]:
    total_loss = 0.0
    component_sums = {
        "coherent": 0.0,
        "local": 0.0,
        "strehl": 0.0,
        "toeplitz": 0.0,
        "image": 0.0,
        "image_l1": 0.0,
    }
    n_batches = 0
    for batch in loader:
        batch = move_batch_to_device(batch, DEVICE)
        r, rrh, rhr, R_complex = build_model_inputs(batch)

        mc_out, mc_in = [], []
        for _ in range(MC_DROPOUT_PASSES):
            pred = model(r, rrh, rhr)
            mc_out.append(pred["output_aberration"])
            mc_in.append(pred["input_aberration"])
        pred_output = renormalize_phasor(torch.stack(mc_out).mean(dim=0))
        pred_input = renormalize_phasor(torch.stack(mc_in).mean(dim=0))

        loss, loss_parts = loss_fn(
            pred_output, batch["target_output"],
            pred_input, batch["target_input"],
            R_complex, mask=na_mask, strehl_weight_scale=strehl_scale,
            target_object=batch["target_object"].to(DEVICE),
        )
        total_loss += float(loss.detach().cpu())
        for k, v in loss_parts.items():
            component_sums[k] += v
        n_batches += 1
    n_batches = max(n_batches, 1)
    avg_components = {k: v / n_batches for k, v in component_sums.items()}
    return total_loss / n_batches, avg_components


if __name__ == "__main__":
    run_training()