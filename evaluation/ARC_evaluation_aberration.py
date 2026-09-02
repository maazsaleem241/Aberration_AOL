"""evaluate_ARC_evaluation_aberration.py -- held-out test-set evaluation for a
ARC checkpoint trained on dataset_c's ABERRATION curriculum.

noise-to-signal-ratio replaced by aberration_multiplier as the
swept difficulty axis throughout -- same metrics, same math, same plots,
different axis.

Loads a frozen checkpoint and reports metrics on dataset_c's four dedicated
TEST tags (test_standard, test_extreme_aberration,
test_aberration_with_noise, test_aberration_sweep) -- splits never touched
by training, curriculum advancement, patience, or the LR scheduler. Purely
observational: never writes back to the checkpoint, the H5 file, or any
training state.

Imports everything from ARC_training_datasetc.py (path added below)
rather than duplicating it, so dataset reshaping / loss math / model-input
construction stay a single source of truth. Does NOT modify
ARC_training_datasetc.py in any way.

NOTE: requires dataset_c to have been generated with a dataset_generator.py
that saves the 'aberration_multipliers' H5 field (added alongside this
script) -- the aberration-sweep plot specifically reads it back per-sample.
If dataset_c was generated before that field existed, regenerate it first.

Metrics + outputs, per test tag (never pooled):
  1. Inference time -- TOTAL pipeline (host batch -> R_c) and MODEL-ONLY
     (after R/RRh/RhR already built -> R_c), batch=1, CUDA-synchronized,
     warmed up.
  2. Strehl ratio (true Marechal form) and RMS residual phase error, input
     and output, uncorrected vs corrected -- mean AND std, every metric.
  3. Image-domain quality: peak/mean signal enhancement + correlation vs
     ground truth, uncorrected vs corrected.
  4. Toeplitz consistency of R_c (needs no ground-truth phase at all).
  5. Strehl/RMS vs aberration-strength sweep (test_aberration_sweep).
  6. Breakdown by object_style, and a second breakdown cross-tabulated by
     object_style x aberration_multiplier (test_aberration_sweep only --
     the one tag with a clean, discrete axis shared across every style) to
     show whether some object types degrade faster than others.
  7. MC-dropout uncertainty calibration (bounded subset), all four test tags.
  8. Example visualization panels per tag: a plain example panel (ground
     truth / uncorrected / corrected images + phase grids) in EXAMPLES_DIR,
     and a comprehensive master panel (adds the raw R/RRh/RhR matrices, PSF
     images AND a PSF profile curve with FWHM, and a ground-truth/
     uncorrected/corrected object intensity line profile -- matching Hou
     et al., Laser Photonics Rev. 2026, 20, e01943, Figure 4j-m) in
     MASTER_DIR, both under the single top-level OUTPUT_DIR ("results").
     Both reuse the exact forward pass already computed for the bulk
     metrics, so the picture and the reported numbers for that sample
     are always consistent with each
     other. For test_aberration_sweep, exactly one example is saved per
     distinct aberration_multiplier value in ABERRATION_LEVELS (pre-scanned
     from the H5 metadata, not "first N samples encountered"). For every
     other tag, N_EXAMPLE_PANELS_PER_TAG (10) examples are saved, spread
     across the SSIM(corrected) performance range -- including the best and
     worst cases, not just the first N sequential samples in the file (see
     select_representative_ptrs).

"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PUB_WORK_DIR = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_TRAINING_DIR = os.path.normpath(os.path.join(_PUB_WORK_DIR, "training"))
for _p in (_PUB_WORK_DIR, _TRAINING_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ARC_training_aberration import (  # noqa: E402
    H5_FILE,
    GRID_SIZE,
    N_ELEMENTS,
    DEVICE,
    MC_DROPOUT_PASSES,
    ARCH5Dataset,
    PhysicsInformedLoss,
    build_model_inputs,
    move_batch_to_device,
    enable_mc_dropout,
    renormalize_phasor,
)
from models.ARC import ARC, make_na_mask  # noqa: E402
from CASS_visualizer import reconstruct_cass, reconstruct_dc_column  # noqa: E402


# =============================================================================
# Configuration
# =============================================================================

CHECKPOINT_PATH = (
    "/home/awais/Desktop/Maaz/Maaz Data/Publication work/"
    "training/ARC_Training/aberration/ARC_training_aberration_new_loss/checkpoints/ARC_best.pth"
)
# Single top-level "results" directory; master panels get their own
# subfolder since those are the main thing being iterated on right now --
# everything else (plots, CSV, the plainer example panels) can get sorted
# into their own subfolders later once it's clear which figures matter.
OUTPUT_DIR = "/home/awais/Desktop/Maaz/Maaz Data/Publication work/evaluation/ARC/aberration/results_new"
EXAMPLES_DIR = os.path.join(OUTPUT_DIR, "examples")
MASTER_DIR = os.path.join(OUTPUT_DIR, "master_matrices")

TEST_TAGS = [
    "test_standard",
    "test_extreme_aberration",
    "test_aberration_with_noise",
    "test_aberration_sweep",
]

# dataset_generator.py's test_aberration_sweep draws aberration_multiplier
# from exactly this set (np.random.choice([1,2,3,5,7,10])) -- used to
# guarantee one saved example panel per distinct level (see evaluate_tag),
# not a "hope we encounter all of them while iterating batches" approach.
ABERRATION_LEVELS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

BATCH_SIZE_METRICS = 16
BATCH_SIZE_LATENCY = 1
N_TIMING_WARMUP = 10
N_TIMING_RUNS = 100

MC_CALIBRATION_MAX_SAMPLES_PER_TAG = 50
# Non-sweep tags: N panels spread across the SSIM(corrected) performance
# range (best/median/worst included), not "first N sequential samples" --
# see select_representative_ptrs. test_aberration_sweep ignores this and
# always saves exactly one panel per ABERRATION_LEVELS value instead.
N_EXAMPLE_PANELS_PER_TAG = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASTER_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)


# =============================================================================
# Model / dataset loading
# =============================================================================


def load_model(checkpoint_path: str) -> torch.nn.Module:
    print(f"[setup] loading checkpoint: {checkpoint_path}")
    model = ARC(input_channels=N_ELEMENTS * 2, normalize_output=True).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def build_test_loader(tag: str, batch_size: int) -> DataLoader:
    dataset = ARCH5Dataset(H5_FILE, split="test", difficulty_tags=[tag])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def read_h5_fields(indices: List[int], fields: List[str]) -> Dict[str, list]:
    """Read per-sample H5 fields directly by row index, WITHOUT modifying
    ARCH5Dataset -- avoids touching the training script's shared class
    while a training run may be actively using the same file."""
    out: Dict[str, list] = {f: [] for f in fields}
    with h5py.File(H5_FILE, "r") as h5f:
        for idx in indices:
            for f in fields:
                v = h5f[f][idx]
                out[f].append(_decode(v) if isinstance(v, (bytes, np.bytes_)) else v)
    return out


# =============================================================================
# Core metric math
# =============================================================================


def phasor_deltas(pred_cos, pred_sin, targ_cos, targ_sin):
    cos_delta = pred_cos * targ_cos + pred_sin * targ_sin
    sin_delta = pred_sin * targ_cos - pred_cos * targ_sin
    return cos_delta, sin_delta


def strehl_and_rms(cos_delta: torch.Tensor, sin_delta: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    """True Marechal-form Strehl S = |E_Omega[e^{i*delta}]|^2, and RMS
    residual phase error (radians), NA-aperture-restricted. delta is
    wrap-safe by construction (a phasor difference, never a raw angle
    subtraction). Verified numerically: perfect prediction -> (1.0, 0.0);
    matches (1 - L_coherent)^2 from the training loss exactly; masking
    confirmed to fully ignore garbage values outside the aperture."""
    mask_f = mask.to(cos_delta.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    mean_re = (cos_delta * mask_f).sum() / denom
    mean_im = (sin_delta * mask_f).sum() / denom
    strehl = float(mean_re**2 + mean_im**2)
    residual_angle = torch.atan2(sin_delta, cos_delta)
    rms = float(torch.sqrt(((residual_angle**2) * mask_f).sum() / denom))
    return strehl, rms


# Uncorrected (raw R) reconstruction: plain dc-column, reverted on purpose
# from full CASS -- CASS's Delta-k averaging suppresses noise so strongly
# that an "uncorrected" panel/metric built from it can look/score far
# better than the sample's actual raw degradation, which was misleading.
# dc-column shows (and scores) the honest, un-averaged raw state.
# Corrected (R_c) reconstruction stays full CASS everywhere in this script
# -- reconstruct_cass, imported above -- since that's the exact
# reconstruction PhysicsInformedLoss._reconstruct_from_R_c uses internally
# for the SSIM/L1 image-domain loss terms, so every "corrected" panel and
# metric here stays consistent with what the model was actually optimized
# against. Call reconstruct_dc_column(...) / reconstruct_cass(...) directly
# at each call site rather than through a shared wrapper, so which method
# is in play is never ambiguous from the call site alone.


def compute_psf(cos_o: np.ndarray, sin_o: np.ndarray, cos_i: np.ndarray, sin_i: np.ndarray,
                 na_mask_np: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Point spread function for the given (output, input) phase-screen
    phasors, amplitude-restricted to the NA aperture (zero outside --
    representing a real finite-aperture system; NOT the same "phase=0,
    amplitude=1 outside" convention used elsewhere in this project for R's
    full-grid construction -- this convention is specific to giving a
    physically realistic, diffraction-limited PSF comparison).

    A point-source object's spectrum is exactly flat (all-ones) regardless
    of position -- construct_scattering_matrix(ones) trivially returns an
    all-ones matrix, since a constant object_fft returns the same value at
    every modulo-indexed lookup regardless of the row/col difference. That
    makes R_psf simply the outer product of the two aperture-masked
    phasors -- no need to call construct_scattering_matrix at all.

    Reconstructed via full CASS (reconstruct_cass) -- unchanged by the
    uncorrected-image revert above, since the PSF triplet (ideal/aberrated/
    corrected) is a separate diagnostic, not "the uncorrected image" panel.
    Re-centered with fftshift for display. Verified numerically against a
    real delta object positioned at the array center (matching how
    dataset_generator.py actually places real objects) that this fftshift is
    the correct, physically-equivalent re-centering, not a cosmetic hack --
    this re-centering is about where the object sits on the discrete spatial
    grid, independent of which k-space accumulation method reconstructed it;
    the un-shifted "corner" result only shows up for a point source
    implicitly positioned at spatial index (0,0) rather than array center,
    which is what fftshift corrects for here."""
    m = na_mask_np.astype(np.float64)
    P_o = ((cos_o + 1j * sin_o) * m).flatten()
    P_i = ((cos_i + 1j * sin_i) * m).flatten()
    R_psf = np.outer(P_o, P_i)
    psf = reconstruct_cass(R_psf, grid_size)
    return np.fft.fftshift(psf)


def find_peak_row(img: np.ndarray) -> int:
    """Row index containing img's single brightest pixel. Used to choose
    which row to profile, instead of blindly the geometric center row --
    guarantees the extracted line actually passes through the brightest,
    most informative part of the image, rather than potentially missing
    real structure that isn't centered (beads/cross/grating/letter/digit/
    usaf targets don't all have their content sitting exactly on the
    center row)."""
    return int(np.unravel_index(np.argmax(img), img.shape)[0])


def extract_profile_at_row(img: np.ndarray, row: int) -> np.ndarray:
    """1D cross-section through a SPECIFIC row -- the SAME row index is
    used across every image being compared (e.g. ideal/aberrated/corrected,
    or ground-truth/uncorrected/corrected), so the three curves are cut
    along one consistent line, not independently re-picked per image."""
    return img[row, :]


def compute_fwhm_1d(profile: np.ndarray) -> float:
    """FWHM (in pixels) of a 1D profile, found by linearly interpolating the
    half-max crossings on either side of the peak. Does NOT assume the
    profile is already peak-normalized -- normalizes internally. Returns
    NaN if the profile never actually drops to half its own peak on both
    sides (e.g. a flat or edge-clipped profile), rather than returning a
    misleading number."""
    profile = np.asarray(profile, dtype=np.float64)
    peak = profile.max()
    if peak <= 0:
        return float("nan")
    profile = profile / peak
    peak_idx = int(np.argmax(profile))
    half = 0.5

    left = None
    for i in range(peak_idx, 0, -1):
        if profile[i] >= half and profile[i - 1] < half:
            frac = (half - profile[i - 1]) / (profile[i] - profile[i - 1] + 1e-12)
            left = (i - 1) + frac
            break

    right = None
    for i in range(peak_idx, len(profile) - 1):
        if profile[i] >= half and profile[i + 1] < half:
            frac = (profile[i] - half) / (profile[i] - profile[i + 1] + 1e-12)
            right = i + frac
            break

    if left is None or right is None:
        return float("nan")
    return float(right - left)


def signal_enhancement(uncorrected_img: np.ndarray, corrected_img: np.ndarray) -> Tuple[float, float]:
    """Peak/mean ratio (Kang CLASS tutorial's own Fig. 6e metric). NOTE
    (verified empirically, not a bug): this is meaningful for point-like
    targets (beads) where correction concentrates energy into a sharp peak,
    but for EXTENDED structured objects (cross/grating/letter/usaf) a
    strongly aberrated image can look MORE peaked than a correctly
    reconstructed one, since aberration scrambles light into random
    speckle-like bright spots. Trust compute_ssim()/compute_psnr() as the
    object-style-independent quality metrics; interpret this one alongside
    the per-object-style breakdown."""
    def peak_over_mean(img):
        m = img.mean()
        return float(img.max() / m) if m > 0 else 0.0
    return peak_over_mean(uncorrected_img), peak_over_mean(corrected_img)


try:
    from skimage.metrics import structural_similarity as _sk_ssim
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


def compute_ssim(img: np.ndarray, ground_truth: np.ndarray) -> float:
    """True SSIM (Wang et al. 2004) between the reconstructed image and the
    ground-truth object, both already normalized to [0,1]. Requires
    scikit-image -- if this raises ImportError, run:
        pip install scikit-image --break-system-packages
    Verified numerically: SSIM=1.0 exactly for identical images, near-zero
    for a structured object vs. random noise."""
    if not _SKIMAGE_AVAILABLE:
        raise ImportError(
            "scikit-image is required for SSIM. Install with: "
            "pip install scikit-image --break-system-packages"
        )
    a = img.astype(np.float64)
    b = ground_truth.astype(np.float64)
    return float(_sk_ssim(a, b, data_range=1.0))


def compute_psnr(img: np.ndarray, ground_truth: np.ndarray) -> float:
    """PSNR in dB between the reconstructed image and ground truth, both
    normalized to [0,1] so MAX=1.0. Dependency-free. Capped at 100.0 dB for
    a near-perfect match to avoid a divide-by-zero/inf for MSE~0."""
    a = img.astype(np.float64)
    b = ground_truth.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def select_representative_ptrs(records: List[Dict], n_panels: int) -> set:
    """Pick n_panels sample ptrs spread across the SSIM(corrected)
    performance range -- not "first N sequential samples" (depends on file
    order and can accidentally show only very similar cases, all easy or
    all hard). Sorts every per-sample record in this tag by ssim_corr and
    takes n_panels indices evenly spaced across that sorted order
    (including the true best and worst), so the saved qualitative examples
    actually demonstrate the range of what the model does -- including
    failures -- not just a lucky/unlucky first-N slice. Used for every tag
    except test_aberration_sweep, which has its own one-per-level selection
    (see evaluate_tag) since it already has a natural, more informative
    axis to spread across."""
    if len(records) <= n_panels:
        return {r["ptr"] for r in records}
    sorted_records = sorted(records, key=lambda r: r["ssim_corr"])
    idxs = np.linspace(0, len(sorted_records) - 1, n_panels).round().astype(int)
    idxs = sorted(set(int(i) for i in idxs))
    return {sorted_records[i]["ptr"] for i in idxs}


# =============================================================================
# Inference timing
# =============================================================================


@torch.no_grad()
def measure_inference_time(model: torch.nn.Module, loader: DataLoader) -> Dict[str, float]:
    """Two numbers: (a) TOTAL pipeline (host batch -> device transfer ->
    R/RRh/RhR construction -> forward pass -> R_c), (b) MODEL-ONLY (from
    right after R/RRh/RhR are already built -> forward pass -> R_c). Batch=1
    (realistic live-frame latency), CUDA-synchronized, with warmup discarded
    first -- naive unsynchronized GPU timing gives misleadingly fast numbers,
    and the first few calls include one-time kernel compilation overhead."""
    it = iter(loader)

    def _one_pass(cpu_batch):
        use_cuda = DEVICE == "cuda"
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        batch = move_batch_to_device(cpu_batch, DEVICE)
        r, rrh, rhr, R_complex = build_model_inputs(batch)

        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()  # R, RRh, RhR constructed

        pred = model(r, rrh, rhr)
        b = pred["output_aberration"].shape[0]
        phasor_o = torch.complex(pred["output_aberration"][:, 0], pred["output_aberration"][:, 1]).reshape(b, -1)
        phasor_i = torch.complex(pred["input_aberration"][:, 0], pred["input_aberration"][:, 1]).reshape(b, -1)
        R_c = torch.conj(phasor_o).unsqueeze(-1) * R_complex * torch.conj(phasor_i).unsqueeze(-2)

        if use_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()  # predictions + correction (R_c) done

        return t0, t1, t2

    for _ in range(N_TIMING_WARMUP):
        try:
            cpu_batch = next(it)
        except StopIteration:
            it = iter(loader)
            cpu_batch = next(it)
        _one_pass(cpu_batch)

    total_times, model_only_times, construction_times = [], [], []
    for _ in range(N_TIMING_RUNS):
        try:
            cpu_batch = next(it)
        except StopIteration:
            it = iter(loader)
            cpu_batch = next(it)
        t0, t1, t2 = _one_pass(cpu_batch)
        total_times.append(t2 - t0)
        model_only_times.append(t2 - t1)
        construction_times.append(t1 - t0)

    total_times = np.array(total_times) * 1000.0
    model_only_times = np.array(model_only_times) * 1000.0
    construction_times = np.array(construction_times) * 1000.0

    return {
        "total_ms_mean": float(total_times.mean()), "total_ms_std": float(total_times.std()),
        "model_only_ms_mean": float(model_only_times.mean()), "model_only_ms_std": float(model_only_times.std()),
        "construction_ms_mean": float(construction_times.mean()), "construction_ms_std": float(construction_times.std()),
    }


# =============================================================================
# Example visualization panels
# =============================================================================


def save_example_panel(
    tag: str,
    sample_idx: int,
    target_o_phasor: torch.Tensor, pred_o_phasor: torch.Tensor,
    target_i_phasor: torch.Tensor, pred_i_phasor: torch.Tensor,
    uncorrected_img: np.ndarray, corrected_img: np.ndarray, ground_truth: np.ndarray,
    metrics: Dict[str, float],
    aberration_multiplier: float,
    intensity_ratio: float,
) -> None:
    """Clean 3x3 panel (no spare/text-only cells): row 0 = Input phase
    (target/predicted/residual); row 1 = Output phase (target/predicted/
    residual); row 2 = ground truth object, uncorrected image (dc-column,
    reverted -- see the reconstruction-method note above compute_psf),
    model-corrected image (full CASS). This sample's own Strehl/RMS numbers,
    aberration_multiplier, and NSR (intensity_ratio) are printed as a
    caption below the figure instead of a dedicated grid cell. atan2 only
    here, at the visualization boundary -- never inside the model or the
    loss, same rule used throughout the rest of the project."""
    target_phi_i = torch.atan2(target_i_phasor[1], target_i_phasor[0]).cpu().numpy()
    pred_phi_i = torch.atan2(pred_i_phasor[1], pred_i_phasor[0]).cpu().numpy()
    target_phi_o = torch.atan2(target_o_phasor[1], target_o_phasor[0]).cpu().numpy()
    pred_phi_o = torch.atan2(pred_o_phasor[1], pred_o_phasor[0]).cpu().numpy()

    # Wrap-safe residual: atan2 of the PHASOR difference, never a raw angle
    # subtraction -- identical construction used in the training script's
    # save_visualization.
    def wrap_safe_residual(pred_phasor: torch.Tensor, target_phasor: torch.Tensor) -> np.ndarray:
        pred_cos, pred_sin = pred_phasor[0], pred_phasor[1]
        targ_cos, targ_sin = target_phasor[0], target_phasor[1]
        cos_delta = pred_cos * targ_cos + pred_sin * targ_sin
        sin_delta = pred_sin * targ_cos - pred_cos * targ_sin
        return torch.atan2(sin_delta, cos_delta).cpu().numpy()

    residual_i = wrap_safe_residual(pred_i_phasor, target_i_phasor)
    residual_o = wrap_safe_residual(pred_o_phasor, target_o_phasor)

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

    target_phi_i = zero_outside_aperture(target_phi_i)
    pred_phi_i = zero_outside_aperture(pred_phi_i)
    target_phi_o = zero_outside_aperture(target_phi_o)
    pred_phi_o = zero_outside_aperture(pred_phi_o)
    residual_i = zero_outside_aperture(residual_i)
    residual_o = zero_outside_aperture(residual_o)

    fig, axes = plt.subplots(3, 3, figsize=(14, 13.5))
    phase_kwargs = dict(cmap="jet", vmin=-1.0, vmax=1.0, interpolation="nearest")
    pi_ticks = [-1.0, 0.0, 1.0]
    pi_tick_labels = [r"$-\pi$", "0", r"$\pi$"]

    def _set_pi_colorbar(mappable, ax):
        cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cb.set_ticks(pi_ticks)
        cb.set_ticklabels(pi_tick_labels)
        cb.set_label("phase")
        return cb

    for ax, data, title in [
        (axes[0, 0], target_phi_i, "Target Input Phase"),
        (axes[0, 1], pred_phi_i, "Predicted Input Phase"),
        (axes[0, 2], residual_i, "Residual Input Phase (wrap-safe)"),
        (axes[1, 0], target_phi_o, "Target Output Phase"),
        (axes[1, 1], pred_phi_o, "Predicted Output Phase"),
        (axes[1, 2], residual_o, "Residual Output Phase (wrap-safe)"),
    ]:
        im = ax.imshow(data / np.pi, **phase_kwargs)
        ax.set_title(title)
        _set_pi_colorbar(im, ax)

    im_gt = axes[2, 0].imshow(ground_truth, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[2, 0].set_title("Ground Truth Object")
    fig.colorbar(im_gt, ax=axes[2, 0], fraction=0.046, pad=0.04)

    unc_norm = uncorrected_img / (uncorrected_img.max() + 1e-12)
    im_unc = axes[2, 1].imshow(unc_norm, cmap="hot", vmin=0, vmax=1, interpolation="bicubic")
    axes[2, 1].set_title("Uncorrected Image")
    fig.colorbar(im_unc, ax=axes[2, 1], fraction=0.046, pad=0.04)

    corr_norm = corrected_img / (corrected_img.max() + 1e-12)
    im_corr = axes[2, 2].imshow(corr_norm, cmap="hot", vmin=0, vmax=1, interpolation="bicubic")
    axes[2, 2].set_title("Model-Corrected Image (CASS)")
    fig.colorbar(im_corr, ax=axes[2, 2], fraction=0.046, pad=0.04)

    fig.suptitle(f"{tag} -- sample {sample_idx}  (aberration={aberration_multiplier:.2f}, NSR={intensity_ratio:.2f})", fontsize=14, fontweight="bold")
    caption = (
        f"Strehl(out)={metrics['strehl_o']:.3f}   Strehl(in)={metrics['strehl_i']:.3f}   "
        f"RMS(out)={metrics['rms_o']:.3f} rad   RMS(in)={metrics['rms_i']:.3f} rad"
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=11, family="monospace")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    tag_dir = os.path.join(EXAMPLES_DIR, tag)
    os.makedirs(tag_dir, exist_ok=True)
    plt.savefig(os.path.join(tag_dir, f"sample_{sample_idx:05d}.png"), dpi=150)
    plt.close(fig)


def save_psf_panel(
    tag: str,
    sample_idx: int,
    target_o_np: np.ndarray, target_i_np: np.ndarray,
    residual_o_cos: np.ndarray, residual_o_sin: np.ndarray,
    residual_i_cos: np.ndarray, residual_i_sin: np.ndarray,
    na_mask_np: np.ndarray,
) -> None:
    """3-panel PSF comparison: (a) Truth -- the diffraction-limited PSF for
    this aperture with zero aberration, computable because we have the
    ground-truth object and thus know exactly what a perfect system would
    produce; (b) Aberrated -- the PSF under this sample's true, uncorrected
    aberration; (c) Corrected -- the PSF under the RESIDUAL aberration left
    after the model's predicted correction is applied (the same residual
    delta already used for the Strehl/RMS metrics). All three normalized to
    the SAME shared scale (the ideal PSF's own peak) so peak-intensity loss
    from aberration and recovery from correction are directly, visually
    comparable -- this is the same physical quantity Strehl ratio measures,
    just shown as an image instead of a single number.

    Currently UNUSED as a standalone file -- no longer called from
    evaluate_tag. save_master_panel now includes both the PSF images (its
    own Row 5, same construction as here) AND a PSF profile curve with FWHM
    (Row 6) in one place, so this separate file was redundant. Left here,
    still correct, in case a standalone PSF-only file is wanted again."""
    ones = np.ones((GRID_SIZE, GRID_SIZE))
    zeros = np.zeros((GRID_SIZE, GRID_SIZE))

    ideal_psf = compute_psf(ones, zeros, ones, zeros, na_mask_np)
    aberrated_psf = compute_psf(target_o_np[0], target_o_np[1], target_i_np[0], target_i_np[1], na_mask_np)
    corrected_psf = compute_psf(residual_o_cos, residual_o_sin, residual_i_cos, residual_i_sin, na_mask_np)

    shared_max = ideal_psf.max()  # normalize all three to the diffraction-limited peak

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, psf, title in [
        (axes[0], ideal_psf, "Truth (Diffraction-Limited PSF)"),
        (axes[1], aberrated_psf, "Aberrated PSF"),
        (axes[2], corrected_psf, "Corrected PSF"),
    ]:
        im = ax.imshow(psf / shared_max, cmap="hot", vmin=0, vmax=1, interpolation="bicubic")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{tag} -- sample {sample_idx} -- PSF Estimation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    tag_dir = os.path.join(EXAMPLES_DIR, tag)
    os.makedirs(tag_dir, exist_ok=True)
    plt.savefig(os.path.join(tag_dir, f"sample_{sample_idx:05d}_psf.png"), dpi=150)
    plt.close(fig)


def save_master_panel(
    tag: str,
    sample_idx: int,
    R_np: np.ndarray,
    target_o_phasor: torch.Tensor, pred_o_phasor: torch.Tensor,
    target_i_phasor: torch.Tensor, pred_i_phasor: torch.Tensor,
    uncorrected_img: np.ndarray, corrected_img: np.ndarray, ground_truth: np.ndarray,
    metrics: Dict[str, float],
    na_mask_np: np.ndarray,
    sample_info_lines: List[str],
) -> None:
    """Clean 7-row comprehensive per-sample panel (no spare/text-only cells
    -- sample info and metrics are captions instead), combining everything
    computed elsewhere into one reference figure:
      Row 1: raw matrix amplitudes log|R|, log|RR^dagger|, log|R^dagger R|.
      Row 2: Input phase -- target / predicted / residual.
      Row 3: Output phase -- target / predicted / residual.
      Row 4: Ground truth object / uncorrected image (dc-column, reverted --
             see the reconstruction-method note above compute_psf) /
             corrected image (full CASS).
      Row 5: PSF images -- Truth (diffraction-limited) / Aberrated /
             Corrected, same construction as save_psf_panel (no longer
             called separately -- restored here as its own row rather than
             a standalone file).
      Row 6: PSF profile curve -- the same three PSFs, overlaid on ONE plot
             (in addition to Row 5's images, not instead of them), each
             peak-normalized with its own FWHM labeled in the legend.
             Matches Hou et al. (Laser Photonics Rev. 2026, 20, e01943)
             Figure 4j,k. Cut along the row containing the ideal PSF's peak
             (find_peak_row) -- not blindly the geometric center row -- and
             that SAME row index is used for all three curves.
      Row 7: Object intensity line profile -- ground truth / uncorrected /
             corrected, overlaid on ONE plot, each peak-normalized. Matches
             the same paper's Figure 4l,m. Cut along the row containing the
             ground truth's own peak, again shared across all three curves.
    Sample info (tag, aberration/NSR, object style) prints as a subtitle;
    metrics (Strehl, RMS, SSIM, PSNR) print as a caption below the figure.
    No SSIM map and no CASS Delta-k spectrum panels -- removed on request,
    the numeric SSIM/PSNR in the caption already carries that information.
    Generated for every saved example (not the full test set -- the RR^dagger
    /R^dagger R matmuls are expensive enough that this is only for the
    already-curated showcase examples). save_example_panel still exists
    alongside this one for when a narrower figure is more appropriate (e.g.
    for the paper)."""
    matrix_extent = [0, N_ELEMENTS, N_ELEMENTS, 0]

    log_R = np.log10(np.abs(R_np) + 1e-5)
    RRh = R_np @ R_np.conj().T
    RhR = R_np.conj().T @ R_np
    log_RRh = np.log10(np.abs(RRh) + 1e-5)
    log_RhR = np.log10(np.abs(RhR) + 1e-5)

    target_phi_i = torch.atan2(target_i_phasor[1], target_i_phasor[0]).cpu().numpy()
    pred_phi_i = torch.atan2(pred_i_phasor[1], pred_i_phasor[0]).cpu().numpy()
    target_phi_o = torch.atan2(target_o_phasor[1], target_o_phasor[0]).cpu().numpy()
    pred_phi_o = torch.atan2(pred_o_phasor[1], pred_o_phasor[0]).cpu().numpy()

    def wrap_safe_residual(pred_phasor: torch.Tensor, target_phasor: torch.Tensor) -> np.ndarray:
        pred_cos, pred_sin = pred_phasor[0], pred_phasor[1]
        targ_cos, targ_sin = target_phasor[0], target_phasor[1]
        cos_delta = pred_cos * targ_cos + pred_sin * targ_sin
        sin_delta = pred_sin * targ_cos - pred_cos * targ_sin
        return torch.atan2(sin_delta, cos_delta).cpu().numpy()

    residual_i = wrap_safe_residual(pred_i_phasor, target_i_phasor)
    residual_o = wrap_safe_residual(pred_o_phasor, target_o_phasor)

    def zero_outside_aperture(arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        arr[~na_mask_np] = 0.0
        return arr

    target_phi_i = zero_outside_aperture(target_phi_i)
    pred_phi_i = zero_outside_aperture(pred_phi_i)
    target_phi_o = zero_outside_aperture(target_phi_o)
    pred_phi_o = zero_outside_aperture(pred_phi_o)
    residual_i = zero_outside_aperture(residual_i)
    residual_o = zero_outside_aperture(residual_o)

    cos_o_res = (pred_o_phasor[0] * target_o_phasor[0] + pred_o_phasor[1] * target_o_phasor[1]).cpu().numpy()
    sin_o_res = (pred_o_phasor[1] * target_o_phasor[0] - pred_o_phasor[0] * target_o_phasor[1]).cpu().numpy()
    cos_i_res = (pred_i_phasor[0] * target_i_phasor[0] + pred_i_phasor[1] * target_i_phasor[1]).cpu().numpy()
    sin_i_res = (pred_i_phasor[1] * target_i_phasor[0] - pred_i_phasor[0] * target_i_phasor[1]).cpu().numpy()
    target_o_np, target_i_np = target_o_phasor.cpu().numpy(), target_i_phasor.cpu().numpy()

    ones, zeros = np.ones((GRID_SIZE, GRID_SIZE)), np.zeros((GRID_SIZE, GRID_SIZE))
    ideal_psf = compute_psf(ones, zeros, ones, zeros, na_mask_np)
    aberrated_psf = compute_psf(target_o_np[0], target_o_np[1], target_i_np[0], target_i_np[1], na_mask_np)
    corrected_psf = compute_psf(cos_o_res, sin_o_res, cos_i_res, sin_i_res, na_mask_np)
    shared_psf_max = ideal_psf.max()

    fig = plt.figure(figsize=(14, 29))
    gs = fig.add_gridspec(
        7, 3, height_ratios=[1, 1, 1, 1, 1, 1, 1], hspace=0.55, wspace=0.35,
        top=0.95, bottom=0.025, left=0.06, right=0.97,
    )
    phase_kwargs = dict(cmap="jet", vmin=-1.0, vmax=1.0, interpolation="nearest")
    pi_ticks, pi_tick_labels = [-1.0, 0.0, 1.0], [r"$-\pi$", "0", r"$\pi$"]

    def _set_pi_colorbar(mappable, ax):
        cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cb.set_ticks(pi_ticks); cb.set_ticklabels(pi_tick_labels); cb.set_label("phase")
        return cb

    for col, (data, title) in enumerate([
        (log_R, "log|R|"),
        (log_RRh, r"log|$RR^\dagger$|"),
        (log_RhR, r"log|$R^\dagger R$|"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(data, cmap="magma", origin="upper", extent=matrix_extent)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for col, (data, title) in enumerate([
        (target_phi_i, "Target Input Phase"),
        (pred_phi_i, "Predicted Input Phase"),
        (residual_i, "Residual Input Phase (wrap-safe)"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(data / np.pi, **phase_kwargs)
        ax.set_title(title)
        _set_pi_colorbar(im, ax)

    for col, (data, title) in enumerate([
        (target_phi_o, "Target Output Phase"),
        (pred_phi_o, "Predicted Output Phase"),
        (residual_o, "Residual Output Phase (wrap-safe)"),
    ]):
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(data / np.pi, **phase_kwargs)
        ax.set_title(title)
        _set_pi_colorbar(im, ax)

    ax_gt = fig.add_subplot(gs[3, 0])
    im_gt = ax_gt.imshow(ground_truth, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_gt.set_title("Ground Truth Object")
    fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

    ax_unc = fig.add_subplot(gs[3, 1])
    unc_norm = uncorrected_img / (uncorrected_img.max() + 1e-12)
    im_unc = ax_unc.imshow(unc_norm, cmap="hot", vmin=0, vmax=1, interpolation="bicubic")
    ax_unc.set_title("Uncorrected Image")
    fig.colorbar(im_unc, ax=ax_unc, fraction=0.046, pad=0.04)

    ax_corr = fig.add_subplot(gs[3, 2])
    corr_norm = corrected_img / (corrected_img.max() + 1e-12)
    im_corr = ax_corr.imshow(corr_norm, cmap="hot", vmin=0, vmax=1, interpolation="bicubic")
    ax_corr.set_title("Model-Corrected Image (CASS)")
    fig.colorbar(im_corr, ax=ax_corr, fraction=0.046, pad=0.04)

    # ---- Row 5: PSF images (restored) ----
    for col, (psf, title) in enumerate([
        (ideal_psf, "Truth (Diffraction-Limited PSF)"),
        (aberrated_psf, "Aberrated PSF"),
        (corrected_psf, "Corrected PSF"),
    ]):
        ax = fig.add_subplot(gs[4, col])
        im = ax.imshow(psf / shared_psf_max, cmap="hot", vmin=0, vmax=1, interpolation="bicubic")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- Row 6: PSF profile curve (Fig. 4j,k style) ----
    # Cut along the row containing the IDEAL PSF's own peak (not blindly
    # the geometric center) -- the same row index is then used for all
    # three curves so they stay directly comparable.
    psf_row = find_peak_row(ideal_psf)
    ax_psf = fig.add_subplot(gs[5, :])
    for psf, label, color in [
        (ideal_psf, "Ideal", "#2ca02c"),
        (aberrated_psf, "Aberrated", "#d62728"),
        (corrected_psf, "Corrected", "#1f77b4"),
    ]:
        profile = extract_profile_at_row(psf, psf_row)
        profile_norm = profile / (profile.max() + 1e-12)
        fwhm_px = compute_fwhm_1d(profile)
        fwhm_label = f"{label} (FWHM={fwhm_px:.2f} px)" if not np.isnan(fwhm_px) else f"{label} (FWHM=N/A)"
        ax_psf.plot(profile_norm, label=fwhm_label, color=color, linewidth=2)
    ax_psf.set_title(f"PSF Profile (row {psf_row}, ideal PSF's peak) -- Ideal vs Aberrated vs Corrected")
    ax_psf.set_xlabel("Pixels"); ax_psf.set_ylabel("Norm. intensity")
    ax_psf.set_ylim(-0.05, 1.05)
    ax_psf.legend(loc="upper right", fontsize=9)
    ax_psf.grid(alpha=0.3)

    # ---- Row 7: object intensity line profile (Fig. 4l,m style) ----
    # Cut along the row containing the GROUND TRUTH's own peak -- again
    # shared across all three curves for a fair, consistent comparison.
    object_row = find_peak_row(ground_truth)
    ax_line = fig.add_subplot(gs[6, :])
    for img, label, color in [
        (ground_truth, "Ground truth", "#555555"),
        (uncorrected_img, "Uncorrected", "#d62728"),
        (corrected_img, "Corrected", "#1f77b4"),
    ]:
        profile = extract_profile_at_row(img, object_row)
        profile_norm = profile / (profile.max() + 1e-12)
        ax_line.plot(profile_norm, label=label, color=color, linewidth=2)
    ax_line.set_title(f"Object Intensity Profile (row {object_row}, ground truth's peak) -- Ground Truth vs Uncorrected vs Corrected")
    ax_line.set_xlabel("Pixels"); ax_line.set_ylabel("Norm. PA amplitude")
    ax_line.set_ylim(-0.05, 1.05)
    ax_line.legend(loc="upper right", fontsize=9)
    ax_line.grid(alpha=0.3)

    fig.suptitle(
        f"{tag} -- sample {sample_idx} -- Master Panel\n" + "   |   ".join(sample_info_lines),
        fontsize=15, fontweight="bold",
    )
    caption = (
        f"Strehl(out)={metrics['strehl_o']:.3f}   Strehl(in)={metrics['strehl_i']:.3f}   "
        f"RMS(out)={metrics['rms_o']:.3f} rad   RMS(in)={metrics['rms_i']:.3f} rad   "
        f"SSIM={metrics['ssim_corr']:.3f}   PSNR={metrics['psnr_corr']:.2f} dB"
    )
    fig.text(0.5, 0.005, caption, ha="center", fontsize=11, family="monospace")
    tag_dir = os.path.join(MASTER_DIR, tag)
    os.makedirs(tag_dir, exist_ok=True)
    plt.savefig(os.path.join(tag_dir, f"sample_{sample_idx:05d}_master.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# Bulk per-tag metric pass
# =============================================================================


@torch.no_grad()
def evaluate_tag(model: torch.nn.Module, tag: str, na_mask: torch.Tensor, loss_fn: PhysicsInformedLoss) -> Dict:
    """Single full pass over the tag computes every metric for every sample
    (as before) plus caches the SMALL per-sample arrays needed to redraw a
    panel later (phasors, 40x40 images, metrics -- a few KB each). It does
    NOT cache the raw R matrices (1600x1600 complex, ~20MB each) for every
    sample -- with potentially hundreds of samples per tag that would blow
    up memory for no reason, since only save_master_panel actually needs R.
    After the full pass, target sample ptrs are selected (one-per-level for
    test_aberration_sweep; best/median/worst-spread by SSIM for every other
    tag -- see select_representative_ptrs) and save_example_panel /
    save_psf_panel are called immediately from the cache (they don't need
    R at all). Only save_master_panel needs a second, TARGETED pass over
    the loader -- skips straight past any batch containing no target ptr,
    and stops as soon as the last target ptr has been covered -- to fetch
    just the handful of R matrices actually needed."""
    loader = build_test_loader(tag, BATCH_SIZE_METRICS)
    dataset: ARCH5Dataset = loader.dataset
    na_mask_np = na_mask[0, 0].cpu().numpy().astype(bool)  # for PSF panels; computed once, not per-sample

    # Per-sample values (not running sums) so both mean AND std are cheap
    # to compute afterward, for every metric this script tracks.
    values: Dict[str, List[float]] = {
        "strehl_o_unc": [], "strehl_o_corr": [], "strehl_i_unc": [], "strehl_i_corr": [],
        "rms_o_unc": [], "rms_o_corr": [], "rms_i_unc": [], "rms_i_corr": [],
        "peak_mean_unc": [], "peak_mean_corr": [],
        "ssim_unc": [], "ssim_corr": [], "psnr_unc": [], "psnr_corr": [],
        "toeplitz": [],
    }
    per_sample_records = []
    cache: Dict[int, Dict] = {}

    h5_indices = dataset.indices
    extra = read_h5_fields(h5_indices, ["intensity_ratios", "aberration_multipliers", "object_styles"])

    ptr = 0
    for batch in loader:
        b = batch["target_output"].shape[0]
        batch_gpu = move_batch_to_device(batch, DEVICE)
        r, rrh, rhr, R_complex = build_model_inputs(batch_gpu)
        pred = model(r, rrh, rhr)

        pred_o, pred_i = pred["output_aberration"], pred["input_aberration"]
        targ_o, targ_i = batch_gpu["target_output"], batch_gpu["target_input"]

        cos_o_corr, sin_o_corr = phasor_deltas(pred_o[:, 0:1], pred_o[:, 1:2], targ_o[:, 0:1], targ_o[:, 1:2])
        cos_i_corr, sin_i_corr = phasor_deltas(pred_i[:, 0:1], pred_i[:, 1:2], targ_i[:, 0:1], targ_i[:, 1:2])
        cos_o_unc, sin_o_unc = targ_o[:, 0:1], targ_o[:, 1:2]
        cos_i_unc, sin_i_unc = targ_i[:, 0:1], targ_i[:, 1:2]

        phasor_o_c = torch.complex(pred_o[:, 0], pred_o[:, 1]).reshape(b, -1)
        phasor_i_c = torch.complex(pred_i[:, 0], pred_i[:, 1]).reshape(b, -1)
        R_c = torch.conj(phasor_o_c).unsqueeze(-1) * R_complex * torch.conj(phasor_i_c).unsqueeze(-2)
        toeplitz_val = float(loss_fn._toeplitz_term(R_c))

        for j in range(b):
            m = na_mask
            s_o_c, r_o_c = strehl_and_rms(cos_o_corr[j], sin_o_corr[j], m[0, 0])
            s_i_c, r_i_c = strehl_and_rms(cos_i_corr[j], sin_i_corr[j], m[0, 0])
            s_o_u, r_o_u = strehl_and_rms(cos_o_unc[j], sin_o_unc[j], m[0, 0])
            s_i_u, r_i_u = strehl_and_rms(cos_i_unc[j], sin_i_unc[j], m[0, 0])

            values["strehl_o_unc"].append(s_o_u); values["strehl_o_corr"].append(s_o_c)
            values["strehl_i_unc"].append(s_i_u); values["strehl_i_corr"].append(s_i_c)
            values["rms_o_unc"].append(r_o_u); values["rms_o_corr"].append(r_o_c)
            values["rms_i_unc"].append(r_i_u); values["rms_i_corr"].append(r_i_c)

            R_np = R_complex[j].cpu().numpy()
            R_c_np = R_c[j].cpu().numpy()
            gt = batch["target_object"][j].numpy()
            # Uncorrected: dc-column (reverted). Corrected: full CASS
            # (matches the loss). See the note above compute_psf.
            unc_img = reconstruct_dc_column(R_np, GRID_SIZE)
            corr_img = reconstruct_cass(R_c_np, GRID_SIZE)
            pm_unc, pm_corr = signal_enhancement(unc_img, corr_img)
            values["peak_mean_unc"].append(pm_unc); values["peak_mean_corr"].append(pm_corr)
            ssim_corr_val = compute_ssim(corr_img, gt)
            psnr_corr_val = compute_psnr(corr_img, gt)
            ssim_unc_val = compute_ssim(unc_img, gt)
            psnr_unc_val = compute_psnr(unc_img, gt)
            values["ssim_unc"].append(ssim_unc_val); values["ssim_corr"].append(ssim_corr_val)
            values["psnr_unc"].append(psnr_unc_val); values["psnr_corr"].append(psnr_corr_val)
            values["toeplitz"].append(toeplitz_val)

            intensity_ratio_value = float(extra["intensity_ratios"][ptr])
            aberration_value = float(extra["aberration_multipliers"][ptr])
            object_style_value = extra["object_styles"][ptr]
            per_sample_records.append({
                "ptr": ptr,
                "strehl_o_corr": s_o_c, "strehl_i_corr": s_i_c,
                "rms_o_corr": r_o_c, "rms_i_corr": r_i_c,
                "ssim_corr": ssim_corr_val, "psnr_corr": psnr_corr_val,
                "intensity_ratio": intensity_ratio_value,
                "aberration_multiplier": aberration_value,
                "object_style": object_style_value,
            })

            cache[ptr] = {
                "targ_o": targ_o[j].detach().cpu(), "pred_o": pred_o[j].detach().cpu(),
                "targ_i": targ_i[j].detach().cpu(), "pred_i": pred_i[j].detach().cpu(),
                "unc_img": unc_img, "corr_img": corr_img, "gt": gt,
                "metrics": {"strehl_o": s_o_c, "strehl_i": s_i_c, "rms_o": r_o_c, "rms_i": r_i_c},
                "ssim_corr": ssim_corr_val, "psnr_corr": psnr_corr_val,
                "aberration_value": aberration_value, "intensity_ratio_value": intensity_ratio_value,
                "object_style": object_style_value,
            }

            ptr += 1

    n_samples = ptr
    means = {k: float(np.mean(v)) if v else 0.0 for k, v in values.items()}
    stds = {k: float(np.std(v)) if v else 0.0 for k, v in values.items()}

    # ---- select which samples get a saved example/psf/master panel ----
    if tag == "test_aberration_sweep":
        # FIX (do not revert): the previous approach saved a panel for the
        # first N DISTINCT aberration_multiplier values encountered while
        # iterating batches in file order (shuffle=False) -- this depends
        # entirely on which values happen to show up early in the file, and
        # was producing far fewer than the intended one-per-level (6)
        # panels in practice. Instead, pre-scan the already-loaded metadata
        # and pick exactly one sample index per ABERRATION_LEVELS value --
        # guaranteed one panel per level actually present in the data,
        # regardless of iteration order. A level with zero matching samples
        # anywhere in the tag is logged and skipped, not silently dropped.
        target_ptrs: set = set()
        aberration_arr = np.array(extra["aberration_multipliers"], dtype=float)
        for level in ABERRATION_LEVELS:
            matches = np.where(np.isclose(aberration_arr, level, atol=1e-6))[0]
            if len(matches) > 0:
                target_ptrs.add(int(matches[0]))
            else:
                print(f"    [warn] test_aberration_sweep: no sample found at aberration_multiplier={level:g}")
    else:
        target_ptrs = select_representative_ptrs(per_sample_records, N_EXAMPLE_PANELS_PER_TAG)

    # save_example_panel needs no R matrix -- draw it straight from the
    # cache, no second pass required. save_psf_panel is no longer called
    # here (superseded by save_master_panel's Row 5 PSF profile curve).
    for p in sorted(target_ptrs):
        c = cache[p]
        save_example_panel(
            tag, p,
            c["targ_o"], c["pred_o"], c["targ_i"], c["pred_i"],
            c["unc_img"], c["corr_img"], c["gt"],
            c["metrics"], c["aberration_value"], c["intensity_ratio_value"],
        )

    # save_master_panel needs R -- a second, targeted pass, skipping any
    # batch with no target ptr and stopping once the last one is covered.
    if target_ptrs:
        max_ptr = max(target_ptrs)
        ptr2 = 0
        for batch in loader:
            if ptr2 > max_ptr:
                break
            b = batch["target_output"].shape[0]
            batch_has_target = any(ptr2 <= p < ptr2 + b for p in target_ptrs)
            if not batch_has_target:
                ptr2 += b
                continue

            batch_gpu = move_batch_to_device(batch, DEVICE)
            r, rrh, rhr, R_complex = build_model_inputs(batch_gpu)
            pred = model(r, rrh, rhr)
            pred_o, pred_i = pred["output_aberration"], pred["input_aberration"]
            phasor_o_c = torch.complex(pred_o[:, 0], pred_o[:, 1]).reshape(b, -1)
            phasor_i_c = torch.complex(pred_i[:, 0], pred_i[:, 1]).reshape(b, -1)
            R_c = torch.conj(phasor_o_c).unsqueeze(-1) * R_complex * torch.conj(phasor_i_c).unsqueeze(-2)

            for j in range(b):
                p = ptr2 + j
                if p in target_ptrs:
                    c = cache[p]
                    R_np = R_complex[j].cpu().numpy()
                    save_master_panel(
                        tag, p, R_np,
                        c["targ_o"], c["pred_o"], c["targ_i"], c["pred_i"],
                        c["unc_img"], c["corr_img"], c["gt"],
                        {**c["metrics"], "ssim_corr": c["ssim_corr"], "psnr_corr": c["psnr_corr"]},
                        na_mask_np,
                        [f"Tag: {tag}", f"Aberration multiplier: {c['aberration_value']:.2f}",
                         f"Baseline NSR: {c['intensity_ratio_value']:.2f}", f"Object style: {c['object_style']}"],
                    )
            ptr2 += b

    return {"tag": tag, "n_samples": n_samples, "means": means, "stds": stds, "records": per_sample_records}


# =============================================================================
# MC-dropout calibration
# =============================================================================


@torch.no_grad()
def mc_dropout_calibration(model: torch.nn.Module, tag: str, na_mask: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    dataset = ARCH5Dataset(H5_FILE, split="test", difficulty_tags=[tag])
    n = min(MC_CALIBRATION_MAX_SAMPLES_PER_TAG, len(dataset))
    loader = DataLoader(torch.utils.data.Subset(dataset, list(range(n))), batch_size=8, shuffle=False, num_workers=0)

    model.eval()
    enable_mc_dropout(model)

    uncertainties, errors = [], []
    mask = na_mask[0, 0].bool()

    for batch in loader:
        batch_gpu = move_batch_to_device(batch, DEVICE)
        r, rrh, rhr, _ = build_model_inputs(batch_gpu)

        passes_o = []
        for _ in range(MC_DROPOUT_PASSES):
            pred = model(r, rrh, rhr)
            passes_o.append(pred["output_aberration"])
        stacked = torch.stack(passes_o)
        mean_pred = renormalize_phasor(stacked.mean(dim=0))
        std_pred = stacked.std(dim=0).mean(dim=1)

        targ = batch_gpu["target_output"]
        cos_d, sin_d = phasor_deltas(mean_pred[:, 0], mean_pred[:, 1], targ[:, 0], targ[:, 1])
        err = torch.abs(torch.atan2(sin_d, cos_d))

        for j in range(err.shape[0]):
            uncertainties.append(std_pred[j][mask].cpu().numpy())
            errors.append(err[j][mask].cpu().numpy())

    model.eval()
    return np.concatenate(uncertainties), np.concatenate(errors)


# =============================================================================
# Plotting
# =============================================================================


def plot_strehl_rms(results: List[Dict]) -> None:
    tags = [r["tag"] for r in results]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(tags))
    width = 0.35

    specs = [
        (axes[0, 0], "strehl_o_unc", "strehl_o_corr", "Output Strehl Ratio"),
        (axes[0, 1], "strehl_i_unc", "strehl_i_corr", "Input Strehl Ratio"),
        (axes[1, 0], "rms_o_unc", "rms_o_corr", "Output RMS Phase Error (rad)"),
        (axes[1, 1], "rms_i_unc", "rms_i_corr", "Input RMS Phase Error (rad)"),
    ]
    for ax, unc_key, corr_key, title in specs:
        unc_vals = [r["means"][unc_key] for r in results]
        corr_vals = [r["means"][corr_key] for r in results]
        ax.bar(x - width / 2, unc_vals, width, label="Uncorrected", color="#C0392B")
        ax.bar(x + width / 2, corr_vals, width, label="Corrected", color="#2E5FA3")
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=9)
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "strehl_rms_by_tag.png"), dpi=150)
    plt.close(fig)


def plot_image_quality(results: List[Dict]) -> None:
    tags = [r["tag"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(tags))
    width = 0.35

    unc_pm = [r["means"]["peak_mean_unc"] for r in results]
    corr_pm = [r["means"]["peak_mean_corr"] for r in results]
    axes[0].bar(x - width / 2, unc_pm, width, label="Uncorrected", color="#C0392B")
    axes[0].bar(x + width / 2, corr_pm, width, label="Corrected", color="#2E5FA3")
    axes[0].set_xticks(x); axes[0].set_xticklabels(tags, rotation=20, ha="right", fontsize=9)
    axes[0].set_title("Peak/Mean Intensity (signal enhancement)*"); axes[0].legend(fontsize=9); axes[0].grid(axis="y", alpha=0.3)

    # SSIM and PSNR are the primary ground-truth-anchored quality metrics --
    # unlike Strehl (which only measures phase agreement and can be badly
    # misleading in noisy regimes), these directly compare the reconstructed
    # image against the true object.
    unc_ssim = [r["means"]["ssim_unc"] for r in results]
    corr_ssim = [r["means"]["ssim_corr"] for r in results]
    axes[1].bar(x - width / 2, unc_ssim, width, label="Uncorrected", color="#C0392B")
    axes[1].bar(x + width / 2, corr_ssim, width, label="Corrected", color="#2E5FA3")
    axes[1].set_xticks(x); axes[1].set_xticklabels(tags, rotation=20, ha="right", fontsize=9)
    axes[1].set_title("SSIM vs Ground Truth"); axes[1].legend(fontsize=9); axes[1].grid(axis="y", alpha=0.3)

    unc_psnr = [r["means"]["psnr_unc"] for r in results]
    corr_psnr = [r["means"]["psnr_corr"] for r in results]
    axes[2].bar(x - width / 2, unc_psnr, width, label="Uncorrected", color="#C0392B")
    axes[2].bar(x + width / 2, corr_psnr, width, label="Corrected", color="#2E5FA3")
    axes[2].set_xticks(x); axes[2].set_xticklabels(tags, rotation=20, ha="right", fontsize=9)
    axes[2].set_title("PSNR (dB) vs Ground Truth"); axes[2].legend(fontsize=9); axes[2].grid(axis="y", alpha=0.3)

    fig.text(0.01, 0.01, "*peak/mean most meaningful for point-like (beads) objects -- see strehl_by_object_style.png", fontsize=8, color="#666666")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "image_quality_by_tag.png"), dpi=150)
    plt.close(fig)


def plot_toeplitz(results: List[Dict]) -> None:
    tags = [r["tag"] for r in results]
    vals = [r["means"]["toeplitz"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(tags, vals, color="#2E5FA3")
    ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=9)
    ax.set_title("Toeplitz Consistency of Corrected $R_c$ (lower = better)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "toeplitz_by_tag.png"), dpi=150)
    plt.close(fig)


def plot_aberration_sweep(sweep_result: Dict) -> None:
    """Aberration strength vs performance, expanded to 2x2: Strehl and RMS,
    each for input and output aberration, all as a function of
    test_aberration_sweep's deliberate aberration_multiplier sweep. Direct
    analog of the dataset_b/noise-curriculum script's plot_nsr_sweep."""
    records = sweep_result["records"]
    by_aberration: Dict[float, List[Dict]] = {}
    for rec in records:
        by_aberration.setdefault(rec["aberration_multiplier"], []).append(rec)
    aberration_values = sorted(by_aberration.keys())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    specs = [
        (axes[0, 0], "strehl_o_corr", "Output Strehl Ratio", "#2E5FA3"),
        (axes[0, 1], "strehl_i_corr", "Input Strehl Ratio", "#2E5FA3"),
        (axes[1, 0], "rms_o_corr", "Output RMS Phase Error (rad)", "#C0392B"),
        (axes[1, 1], "rms_i_corr", "Input RMS Phase Error (rad)", "#C0392B"),
    ]
    for ax, key, title, color in specs:
        means = [np.mean([r[key] for r in by_aberration[k]]) for k in aberration_values]
        stds = [np.std([r[key] for r in by_aberration[k]]) for k in aberration_values]
        ax.errorbar(aberration_values, means, yerr=stds, fmt="o-", color=color, capsize=4)
        ax.set_xlabel("Aberration multiplier")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs. Aberration Strength")
        ax.grid(alpha=0.3)

    plt.suptitle("Performance vs. Aberration Strength (test_aberration_sweep sweep)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "aberration_vs_performance.png"), dpi=150)
    plt.close(fig)


def plot_object_style_breakdown(results: List[Dict]) -> None:
    all_records = [rec for r in results for rec in r["records"]]
    by_style: Dict[str, List[float]] = {}
    for rec in all_records:
        by_style.setdefault(rec["object_style"], []).append(rec["strehl_o_corr"])

    styles = sorted(by_style.keys())
    means = [np.mean(by_style[s]) for s in styles]
    stds = [np.std(by_style[s]) for s in styles]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(styles, means, yerr=stds, capsize=4, color="#2E5FA3")
    ax.set_title("Output Strehl Ratio (corrected) by Object Style")
    ax.set_ylabel("Strehl Ratio"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "strehl_by_object_style.png"), dpi=150)
    plt.close(fig)


def plot_object_style_by_aberration(sweep_result: Dict) -> None:
    """Cross-tabulated breakdown: does one object type degrade faster than
    another as aberration increases? Heatmap of mean output Strehl ratio
    (corrected), object_style x aberration_multiplier, built from
    test_aberration_sweep's records -- the only tag with a clean, discrete,
    directly-comparable aberration axis shared across every object style
    (other tags draw aberration_multiplier from a continuous range, which
    wouldn't bin cleanly the same way)."""
    records = sweep_result["records"]
    styles = sorted(set(r["object_style"] for r in records))
    levels = sorted(set(r["aberration_multiplier"] for r in records))

    grid = np.full((len(styles), len(levels)), np.nan)
    counts = np.zeros((len(styles), len(levels)), dtype=int)
    for si, style in enumerate(styles):
        for li, level in enumerate(levels):
            vals = [
                r["strehl_o_corr"] for r in records
                if r["object_style"] == style and np.isclose(r["aberration_multiplier"], level, atol=1e-6)
            ]
            if vals:
                grid[si, li] = np.mean(vals)
                counts[si, li] = len(vals)

    fig, ax = plt.subplots(figsize=(2.0 * len(levels) + 3, 1.0 * len(styles) + 3))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"{l:g}" for l in levels])
    ax.set_yticks(range(len(styles))); ax.set_yticklabels(styles)
    ax.set_xlabel("Aberration multiplier")
    ax.set_ylabel("Object style")
    ax.set_title("Output Strehl Ratio (corrected): Object Style x Aberration Level\n(test_aberration_sweep)")
    for si in range(len(styles)):
        for li in range(len(levels)):
            if not np.isnan(grid[si, li]):
                label = f"{grid[si, li]:.2f}\n(n={counts[si, li]})"
                color = "white" if grid[si, li] < 0.5 else "black"
                ax.text(li, si, label, ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="Strehl Ratio", fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "strehl_by_object_style_and_aberration.png"), dpi=150)
    plt.close(fig)


def plot_mc_calibration(uncertainty: np.ndarray, error: np.ndarray, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(uncertainty, error, s=2, alpha=0.15, color="#4C6EBE")

    bins = np.linspace(uncertainty.min(), uncertainty.max(), 15)
    bin_idx = np.digitize(uncertainty, bins)
    bin_means_x, bin_means_y = [], []
    for i in range(1, len(bins)):
        sel = bin_idx == i
        if sel.sum() > 5:
            bin_means_x.append(uncertainty[sel].mean())
            bin_means_y.append(error[sel].mean())
    ax.plot(bin_means_x, bin_means_y, "o-", color="#D9534F", label="binned mean")

    ax.set_xlabel("MC-dropout predicted uncertainty (std)")
    ax.set_ylabel("Actual residual phase error (rad)")
    ax.set_title(f"Uncertainty Calibration -- {tag}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"mc_calibration_{tag}.png"), dpi=150)
    plt.close(fig)


def plot_timing(timing: Dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    labels = ["Total pipeline\n(host batch -> $R_c$)", "Model + correction only\n(after R, $RR^\\dagger$, $R^\\dagger R$ built)"]
    means = [timing["total_ms_mean"], timing["model_only_ms_mean"]]
    stds = [timing["total_ms_std"], timing["model_only_ms_std"]]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, means, xerr=stds, height=0.4, capsize=5, color=["#C0392B", "#2E5FA3"])
    for i, m in enumerate(means):
        ax.text(m + stds[i] + 0.5, i, f"{m:.2f} ms", va="center", fontsize=11)
    # FIX (do not revert): with no explicit xlim, matplotlib auto-scales to
    # the bar+errorbar extent only -- the text label placed just past that
    # (m + std + 0.5) could then sit right at or past the right edge of the
    # figure, getting clipped. Explicit headroom (30% beyond the longest
    # bar+errorbar) guarantees the label always has room.
    right = max(m + s for m, s in zip(means, stds))
    ax.set_xlim(0, right * 1.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Inference time (ms), batch=1")
    ax.set_title(f"Inference Latency (N={N_TIMING_RUNS} runs, {N_TIMING_WARMUP} warmup discarded)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "inference_timing.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    model = load_model(CHECKPOINT_PATH)
    na_mask = make_na_mask(size=GRID_SIZE, nasz=GRID_SIZE // 2, device=DEVICE)
    loss_fn = PhysicsInformedLoss(grid_size=GRID_SIZE).to(DEVICE)

    print("\n[1/4] Inference timing (batch=1)...")
    timing_loader = build_test_loader("test_standard", BATCH_SIZE_LATENCY)
    timing = measure_inference_time(model, timing_loader)
    print(
        f"  TOTAL pipeline:        {timing['total_ms_mean']:.3f} +/- {timing['total_ms_std']:.3f} ms\n"
        f"  Model+correction only: {timing['model_only_ms_mean']:.3f} +/- {timing['model_only_ms_std']:.3f} ms\n"
        f"  (R/RRh/RhR construction alone: {timing['construction_ms_mean']:.3f} +/- {timing['construction_ms_std']:.3f} ms)"
    )
    plot_timing(timing)

    print("\n[2/4] Per-tag metrics + example panels...")
    results = []
    for tag in TEST_TAGS:
        print(f"  evaluating {tag} ...")
        r = evaluate_tag(model, tag, na_mask, loss_fn)
        n_panels = 6 if tag == "test_aberration_sweep" else min(N_EXAMPLE_PANELS_PER_TAG, r["n_samples"])
        print(f"    n={r['n_samples']}  Strehl_o(corr)={r['means']['strehl_o_corr']:.4f} +/- {r['stds']['strehl_o_corr']:.4f}  "
              f"RMS_o(corr)={r['means']['rms_o_corr']:.4f} +/- {r['stds']['rms_o_corr']:.4f} rad  "
              f"({n_panels} example panels saved)")
        results.append(r)

    plot_strehl_rms(results)
    plot_image_quality(results)
    plot_toeplitz(results)
    plot_object_style_breakdown(results)

    sweep_result = next(r for r in results if r["tag"] == "test_aberration_sweep")
    plot_aberration_sweep(sweep_result)
    plot_object_style_by_aberration(sweep_result)

    print("\n[3/4] MC-dropout uncertainty calibration (bounded subset)...")
    for tag in TEST_TAGS:
        unc, err = mc_dropout_calibration(model, tag, na_mask)
        plot_mc_calibration(unc, err, tag)
        print(f"  {tag}: pooled correlation(uncertainty, error) = {np.corrcoef(unc, err)[0,1]:.3f}")

    print("\n[4/4] Writing summary CSV...")
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        metric_keys = list(results[0]["means"].keys())
        header = ["tag", "n_samples"]
        for k in metric_keys:
            header += [f"{k}_mean", f"{k}_std"]
        writer.writerow(header)
        for r in results:
            row = [r["tag"], r["n_samples"]]
            for k in metric_keys:
                row += [r["means"][k], r["stds"][k]]
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(["inference_timing_ms"])
        for k, v in timing.items():
            writer.writerow([k, v])

    print(f"\nDone. Plots -> {OUTPUT_DIR}\nExample panels -> {EXAMPLES_DIR}\nMaster panels -> {MASTER_DIR}\nSummary CSV -> {csv_path}")


if __name__ == "__main__":
    main()
