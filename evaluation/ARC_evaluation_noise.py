"""ARC_evaluation_noise.py -- held-out test-set evaluation for a trained
ARC checkpoint (noise curriculum).

Loads a frozen checkpoint and reports metrics on dataset_b's four dedicated
TEST tags (test_standard, test_high_aberration_low_noise,
test_high_noise_low_aberration, test_intensity_ratio) -- splits never touched
by training, curriculum advancement, patience, or the LR scheduler. Purely
observational: never writes back to the checkpoint, the H5 file, or any
training state.

Imports everything from ARC_training_noise.py (path added below) rather
than duplicating it, so dataset reshaping / loss math / model-input
construction stay a single source of truth. Does NOT modify
ARC_training_noise.py in any way.

Metrics + outputs, per test tag (never pooled):
  1. Inference time -- FULL PIPELINE (host batch -> R/RRh/RhR construction
     -> forward pass -> R_c -> CASS reconstruction to the final corrected
     image -- the complete framework's real, end-to-end working time),
     TOTAL model pipeline (host batch -> R_c, stopping short of
     reconstruction), and MODEL-ONLY (after R/RRh/RhR already built -> R_c),
     plus the construction-alone and reconstruction-alone times on their
     own so the full breakdown is visible. Batch=1, CUDA-synchronized,
     warmed up.
  2. Strehl ratio (true Marechal form) and RMS residual phase error, input
     and output, uncorrected vs corrected.
  3. Image-domain quality: peak/mean signal enhancement, SSIM, and PSNR vs
     ground truth, uncorrected vs corrected. SSIM/PSNR are computed on
     normalize_image()'d reconstructions -- FIX (do not revert): these used
     to be computed directly on raw-intensity-scale CASS/dc-column output
     while assuming data_range=1.0/MAX=1, which made SSIM meaningless
     (a visually-better reconstruction could score near zero) and gave
     negative PSNR whenever the raw MSE exceeded 1.0.
  4. Strehl/RMS vs noise-to-signal ratio sweep (test_intensity_ratio).
  5. Breakdown by object_style, and a second breakdown cross-tabulated by
     object_style x intensity_ratio (test_intensity_ratio only -- the one
     tag with a clean, discrete axis shared across every style) to show
     whether some object types degrade faster than others.
  6. Example visualization panels per tag, both saved into the same
     EXAMPLES_DIR/{tag}/ (no separate master-matrix directory -- dropped on
     request):
       Panel A (save_example_panel): inputs, prediction, and correction --
       target/predicted/residual phase grids (input and output) plus
       ground truth / uncorrected (dc-column) / corrected (CASS) images.
       Panel B (save_psf_and_profile_panel): PSF recovery (Truth/Aberrated/
       Corrected images, each with a subtle white crosshair marking the
       true expected center) plus a PSF profile curve with FWHM, plus the
       ground-truth/uncorrected/corrected object intensity profile -- both
       curves marked with a thin white line showing where the object
       actually is.
     Both panels reuse the exact forward pass already computed for the
     bulk metrics, so the picture and the reported numbers for that sample
     are always consistent with each other. For test_intensity_ratio,
     exactly one example is saved per distinct intensity_ratio value in
     NSR_LEVELS (pre-scanned from the H5 metadata, not "first N samples
     encountered"). For every other tag, N_EXAMPLE_PANELS_PER_TAG (10)
     examples are saved, spread across the SSIM(corrected) performance
     range -- including the best and worst cases, not just the first N
     sequential samples in the file (see select_representative_ptrs).

Usage (run from Publication work/):
    PYTHONPATH=. python3 evaluation/ARC_evaluation_noise.py
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

from ARC_training_noise import (  # noqa: E402
    H5_FILE,
    GRID_SIZE,
    N_ELEMENTS,
    DEVICE,
    ARCH5Dataset,
    build_model_inputs,
    move_batch_to_device,
)
from models.ARC import ARC, make_na_mask  # noqa: E402
from CASS_visualizer import reconstruct_cass, reconstruct_dc_column  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================

CHECKPOINT_PATH = (
    "/home/awais/Desktop/Maaz/Maaz Data/Publication work/"
    "training/ARC_Training/noise/ARC_trained_noise/checkpoints/ARC_best.pth"
)
# Single top-level "results" directory -- Panel A (save_example_panel) and
# Panel B (save_psf_and_profile_panel) both save into EXAMPLES_DIR/{tag}/,
# no separate master-matrix directory anymore (dropped on request).
OUTPUT_DIR = "/home/awais/Desktop/Maaz/Maaz Data/Publication work/evaluation/ARC/noise/results_ppt"
EXAMPLES_DIR = os.path.join(OUTPUT_DIR, "examples")

TEST_TAGS = [
    "test_standard",
    "test_high_aberration_low_noise",
    "test_high_noise_low_aberration",
    "test_intensity_ratio",
]

# dataset_generator.py's test_intensity_ratio draws intensity_ratio (NSR)
# from exactly this discrete set (np.random.choice([1,2,4,8,16])) -- used
# to guarantee one saved example panel per distinct level (see
# evaluate_tag), not a "hope we encounter all of them while iterating
# batches in file order" approach. Mirrors ABERRATION_LEVELS in
# ARC_evaluation_aberration.py.
NSR_LEVELS = [1.0, 2.0, 4.0, 8.0, 16.0]

BATCH_SIZE_METRICS = 16
BATCH_SIZE_LATENCY = 1
N_TIMING_WARMUP = 10
N_TIMING_RUNS = 100

# Non-sweep tags: N panels spread across the SSIM(corrected) performance
# range (best/median/worst included), not "first N sequential samples" --
# see select_representative_ptrs. test_intensity_ratio ignores this and
# always saves exactly one panel per NSR_LEVELS value instead.
N_EXAMPLE_PANELS_PER_TAG = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
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
# dc-column shows the honest, un-averaged raw state.
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


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Robust [0,1] normalization for a reconstructed image BEFORE any SSIM/
    PSNR comparison. FIX (do not revert): compute_ssim/compute_psnr used to
    be called with raw-intensity-scale CASS/dc-column reconstructions
    (never actually normalized first) while assuming data_range=1.0/MAX=1 --
    that made SSIM meaningless (a visually-better reconstruction could score
    near zero) and could give negative PSNR whenever the raw MSE exceeded
    1.0. Every caller must normalize_image() its reconstructed image before
    passing it to compute_ssim/compute_psnr; ground truth objects are
    already peak-normalized at generation time (dataset_generator.py) so
    they're passed through as-is. Subtracts the min first if it's negative
    (CASS output can dip slightly below zero) rather than assuming a
    strictly non-negative floor."""
    img = np.asarray(img, dtype=np.float64)
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mn < 0:
        img = img - mn
    if mx <= 0:
        return np.zeros_like(img)
    return img / (mx + 1e-12)


def compute_ssim(img_norm: np.ndarray, gt_norm: np.ndarray) -> float:
    """True SSIM (Wang et al. 2004) between an ALREADY-[0,1]-normalized
    reconstructed image (see normalize_image) and the ground-truth object
    (already normalized at generation time). Requires scikit-image -- if
    this raises ImportError, run:
        pip install scikit-image --break-system-packages
    Verified numerically: SSIM=1.0 exactly for identical images, near-zero
    for a structured object vs. random noise."""
    if not _SKIMAGE_AVAILABLE:
        raise ImportError(
            "scikit-image is required for SSIM. Install with: "
            "pip install scikit-image --break-system-packages"
        )
    return float(_sk_ssim(img_norm.astype(np.float64), gt_norm.astype(np.float64), data_range=1.0))


def compute_psnr(img_norm: np.ndarray, gt_norm: np.ndarray) -> float:
    """PSNR in dB between an ALREADY-[0,1]-normalized reconstructed image
    (see normalize_image) and the ground truth (already normalized at
    generation time), so MAX=1.0 is actually correct here. Dependency-free.
    Capped at 100.0 dB for a near-perfect match to avoid a divide-by-zero/
    inf for MSE~0."""
    mse = float(np.mean((img_norm.astype(np.float64) - gt_norm.astype(np.float64)) ** 2))
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
    except test_intensity_ratio, which has its own one-per-level selection
    (see evaluate_tag) since it already has a natural, more informative
    axis to spread across."""
    if len(records) <= n_panels:
        return {r["ptr"] for r in records}
    sorted_records = sorted(records, key=lambda r: r["ssim_corr"])
    idxs = np.linspace(0, len(sorted_records) - 1, n_panels).round().astype(int)
    idxs = sorted(set(int(i) for i in idxs))
    return {sorted_records[i]["ptr"] for i in idxs}


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


def crop_window_around_signal(reference_profile: np.ndarray, margin: int = 4) -> Tuple[int, int]:
    """Find the [lo, hi] pixel range around where `reference_profile` has
    non-negligible signal (>1% of its own peak), with a margin on each
    side -- used to crop profile-plot x-axes so a mostly-flat-zero stretch
    on either side of the actual target doesn't dominate the plot. Always
    measured against a REFERENCE curve (the ideal PSF, or the ground truth
    object) rather than the aberrated/corrected curves -- so the crop
    window is a fixed, fair window ("where the target actually is"), not
    something that shifts around depending on how badly a given sample is
    aberrated. Falls back to the full width if the reference is all zero."""
    reference_profile = np.asarray(reference_profile, dtype=np.float64)
    peak = reference_profile.max()
    if peak <= 0:
        return 0, len(reference_profile) - 1
    nonzero = np.where(reference_profile / peak > 0.01)[0]
    if len(nonzero) == 0:
        return 0, len(reference_profile) - 1
    lo = max(0, int(nonzero.min()) - margin)
    hi = min(len(reference_profile) - 1, int(nonzero.max()) + margin)
    return lo, hi


# =============================================================================
# Inference timing
# =============================================================================


@torch.no_grad()
def measure_inference_time(model: torch.nn.Module, loader: DataLoader) -> Dict[str, float]:
    """Three numbers: (a) FULL PIPELINE (host batch -> device transfer ->
    R/RRh/RhR construction -> forward pass -> R_c -> CASS reconstruction to
    the final corrected image) -- the complete framework's real-world,
    end-to-end working time, from raw input all the way to a viewable
    output image; (b) TOTAL model pipeline (host batch -> ... -> R_c, same
    as before, stopping short of reconstruction); (c) MODEL-ONLY (from
    right after R/RRh/RhR are already built -> forward pass -> R_c). Also
    reports the R/RRh/RhR CONSTRUCTION time and the RECONSTRUCTION (CASS)
    time on their own, so the full pipeline's breakdown is fully visible,
    not just its total. Batch=1 (realistic live-frame latency),
    CUDA-synchronized, with warmup discarded first -- naive unsynchronized
    GPU timing gives misleadingly fast numbers, and the first few calls
    include one-time kernel compilation overhead. The CASS reconstruction
    step itself is plain NumPy (not a CUDA op), so moving R_c to host with
    .cpu().numpy() is timed as part of that step -- a real cost the live
    system actually pays, not something to hide by excluding it."""
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

        # Full reconstruction: CASS on the corrected R_c, batch index 0
        # only (batch=1 here -- this is the realistic single-frame latency
        # case). Plain NumPy, not a CUDA op; .cpu().numpy() forces the host
        # transfer to actually complete before this segment's clock starts,
        # so no separate synchronize() call is needed here.
        R_c_np = R_c[0].cpu().numpy()
        reconstruct_cass(R_c_np, GRID_SIZE)
        t3 = time.perf_counter()  # full reconstruction to final image done

        return t0, t1, t2, t3

    for _ in range(N_TIMING_WARMUP):
        try:
            cpu_batch = next(it)
        except StopIteration:
            it = iter(loader)
            cpu_batch = next(it)
        _one_pass(cpu_batch)

    full_pipeline_times, total_times, model_only_times, construction_times, reconstruction_times = [], [], [], [], []
    for _ in range(N_TIMING_RUNS):
        try:
            cpu_batch = next(it)
        except StopIteration:
            it = iter(loader)
            cpu_batch = next(it)
        t0, t1, t2, t3 = _one_pass(cpu_batch)
        full_pipeline_times.append(t3 - t0)
        total_times.append(t2 - t0)
        model_only_times.append(t2 - t1)
        construction_times.append(t1 - t0)
        reconstruction_times.append(t3 - t2)

    full_pipeline_times = np.array(full_pipeline_times) * 1000.0
    total_times = np.array(total_times) * 1000.0
    model_only_times = np.array(model_only_times) * 1000.0
    construction_times = np.array(construction_times) * 1000.0
    reconstruction_times = np.array(reconstruction_times) * 1000.0

    # CLASS/DeepCLASS-style reporting convention: optics reconstruction
    # papers typically report throughput as frames/second (or seconds/
    # frame) for the complete raw-input -> final-image task, not raw
    # milliseconds -- full_pipeline_ms is exactly that same end-to-end task
    # (host batch -> R/RRh/RhR -> model -> R_c -> CASS reconstruction), so
    # this is a direct, apples-to-apples comparison point.
    full_pipeline_fps = 1000.0 / float(full_pipeline_times.mean())

    return {
        "full_pipeline_ms_mean": float(full_pipeline_times.mean()), "full_pipeline_ms_std": float(full_pipeline_times.std()),
        "full_pipeline_fps": full_pipeline_fps,
        "total_ms_mean": float(total_times.mean()), "total_ms_std": float(total_times.std()),
        "model_only_ms_mean": float(model_only_times.mean()), "model_only_ms_std": float(model_only_times.std()),
        "construction_ms_mean": float(construction_times.mean()), "construction_ms_std": float(construction_times.std()),
        "reconstruction_ms_mean": float(reconstruction_times.mean()), "reconstruction_ms_std": float(reconstruction_times.std()),
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
    intensity_ratio: float,
) -> None:
    """Clean 3x3 panel (no spare/text-only cells): row 0 = Input phase
    (target/predicted/residual); row 1 = Output phase (target/predicted/
    residual); row 2 = ground truth object, uncorrected image (dc-column,
    reverted -- see the reconstruction-method note above compute_psf),
    model-corrected image (full CASS). This sample's own Strehl/RMS numbers
    and NSR (intensity_ratio) are printed as a caption below the figure
    instead of a dedicated grid cell. atan2 only here, at the visualization
    boundary -- never inside the model or the loss, same rule used
    throughout the rest of the project."""
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

    fig.suptitle(f"{tag} -- sample {sample_idx}  (NSR={intensity_ratio:.2f})", fontsize=14, fontweight="bold")
    caption = (
        f"Strehl (avg in/out) = {metrics['strehl_avg']:.3f}       "
        f"RMS phase error (avg in/out) = {metrics['rms_avg']:.3f} rad"
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=11, family="monospace")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    tag_dir = os.path.join(EXAMPLES_DIR, tag)
    os.makedirs(tag_dir, exist_ok=True)
    plt.savefig(os.path.join(tag_dir, f"sample_{sample_idx:05d}.png"), dpi=150)
    plt.close(fig)


def save_psf_and_profile_panel(
    tag: str,
    sample_idx: int,
    target_o_phasor: torch.Tensor, pred_o_phasor: torch.Tensor,
    target_i_phasor: torch.Tensor, pred_i_phasor: torch.Tensor,
    uncorrected_img: np.ndarray, corrected_img: np.ndarray, ground_truth: np.ndarray,
    na_mask_np: np.ndarray,
    intensity_ratio: float,
) -> None:
    """Panel B: PSF recovery + object images + profiles, saved in the same
    EXAMPLES_DIR as Panel A (save_example_panel) -- no separate master-
    matrix directory anymore (dropped on request).
      Row 1: PSF images -- Truth (diffraction-limited) / Aberrated /
             Corrected. Displayed on a SQRT (gamma=0.5) color stretch, not
             plain linear [0,1] -- a severely aberrated PSF's peak can be a
             tiny fraction of the ideal PSF's peak (that's exactly what a
             low Strehl ratio means -- energy spread out, not concentrated),
             so on plain linear it could look like there was simply no PSF
             there at all. Sqrt lifts faint-but-real signal into visibility
             without the over-correction full LOG scale caused (log lifted
             the noise floor too, making every panel look like a diffuse
             glowing blob instead of a clean point spread function -- see
             conversation record). A single thin white horizontal line
             marks the row used for the profile cut below (Row 3) -- one
             line only, not a crosshair.
      Row 2: The same GT / Uncorrected / Corrected object images shown in
             Panel A, each with the SAME thin white horizontal line marking
             exactly which row Row 4's object intensity profile is cut
             from -- so the profile plot's line is traceable back to a
             specific place on the actual image, not an abstract row index.
      Row 3: PSF profile curve -- the same three PSFs, overlaid on one
             plot, peak-normalized with FWHM per curve, cut along the row
             containing the ideal PSF's peak (find_peak_row). X-axis
             cropped to a window around the ideal PSF's own extent (see
             crop_window_around_signal) rather than the full grid width, so
             the long flat-zero stretch on either side doesn't dominate.
      Row 4: Object intensity profile -- ground truth / uncorrected /
             corrected, overlaid on one plot, peak-normalized, cut along
             the row containing the ground truth's own peak
             (find_peak_row). X-axis cropped to a window around the ground
             truth object's own extent, same reasoning as Row 3."""
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

    psf_row = find_peak_row(ideal_psf)
    object_row = find_peak_row(ground_truth)

    fig = plt.figure(figsize=(14, 19.5))
    gs = fig.add_gridspec(
        4, 3, height_ratios=[1.15, 1.15, 1, 1], hspace=0.5, wspace=0.35,
        top=0.945, bottom=0.045, left=0.07, right=0.97,
    )

    # ---- Row 1: PSF images, sqrt stretch, single horizontal line ----
    # FIX (do not revert without checking first): reverted from a full LOG
    # color scale back to a gentler SQUARE-ROOT (gamma=0.5) stretch. Log
    # scale was originally added because a severely aberrated PSF's peak
    # can be a tiny fraction of the ideal PSF's peak (a low Strehl ratio
    # literally means "energy spread out, not concentrated") -- on a plain
    # LINEAR shared scale that made faint-but-real aberrated PSFs look
    # completely absent. But full log over-corrects: it lifts the noise
    # floor into visibility too, giving every panel a diffuse, smeared
    # "glowing" appearance even where the true signal is genuinely small,
    # not just faint. Sqrt is the standard astronomy-imaging middle ground
    # -- it still meaningfully boosts faint real signal above pure linear
    # (so a genuinely low-Strehl PSF stays visible), without stretching the
    # noise floor as aggressively as log does, so panels look like clean,
    # sharp point-spread functions again instead of smeared blobs.
    for col, (psf, title) in enumerate([
        (ideal_psf, "Truth (Diffraction-Limited PSF)"),
        (aberrated_psf, "Aberrated PSF"),
        (corrected_psf, "Corrected PSF"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        data = np.sqrt(np.clip(psf / shared_psf_max, 0.0, 1.0))
        im = ax.imshow(data, cmap="hot", vmin=0.0, vmax=1.0, interpolation="bicubic")
        ax.axhline(psf_row, color="white", linewidth=0.7, alpha=0.6)
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized intensity")

    # ---- Row 2: GT / Uncorrected / Corrected images, same line marker ----
    for col, (img, title, cmap, vmax_img) in enumerate([
        (ground_truth, "Ground Truth Object", "gray", 1.0),
        (uncorrected_img / (uncorrected_img.max() + 1e-12), "Uncorrected Image", "hot", 1.0),
        (corrected_img / (corrected_img.max() + 1e-12), "Corrected Image", "hot", 1.0),
    ]):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax_img, interpolation="bicubic" if col else "nearest")
        ax.axhline(object_row, color="white", linewidth=0.9, alpha=0.7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- Row 3: PSF profile curve, cropped x-axis ----
    ax_psf = fig.add_subplot(gs[2, :])
    ax_psf.set_facecolor("#f0f0f0")
    ideal_profile = extract_profile_at_row(ideal_psf, psf_row)
    ideal_profile_norm = ideal_profile / (ideal_profile.max() + 1e-12)
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
    psf_lo, psf_hi = crop_window_around_signal(ideal_profile_norm)
    ax_psf.set_xlim(psf_lo, psf_hi)
    ax_psf.set_title(f"PSF Profile (row {psf_row}, ideal PSF's peak) -- Ideal vs Aberrated vs Corrected")
    ax_psf.set_xlabel("Pixels"); ax_psf.set_ylabel("Norm. intensity")
    ax_psf.set_ylim(-0.05, 1.05)
    ax_psf.legend(loc="upper right", fontsize=9)
    ax_psf.grid(alpha=0.3)

    # ---- Row 4: object intensity line profile, cropped x-axis ----
    ax_line = fig.add_subplot(gs[3, :])
    ax_line.set_facecolor("#f0f0f0")
    gt_profile = extract_profile_at_row(ground_truth, object_row)
    gt_profile_norm = gt_profile / (gt_profile.max() + 1e-12)
    for img, label, color in [
        (ground_truth, "Ground truth", "#555555"),
        (uncorrected_img, "Uncorrected", "#d62728"),
        (corrected_img, "Corrected", "#1f77b4"),
    ]:
        profile = extract_profile_at_row(img, object_row)
        profile_norm = profile / (profile.max() + 1e-12)
        ax_line.plot(profile_norm, label=label, color=color, linewidth=2)
    obj_lo, obj_hi = crop_window_around_signal(gt_profile_norm)
    ax_line.set_xlim(obj_lo, obj_hi)
    ax_line.set_title(f"Object Intensity Profile (row {object_row}) -- Ground Truth vs Uncorrected vs Corrected")
    ax_line.set_xlabel("Pixels"); ax_line.set_ylabel("Norm. PA amplitude")
    ax_line.set_ylim(-0.05, 1.05)
    ax_line.legend(loc="upper right", fontsize=9)
    ax_line.grid(alpha=0.3)

    fig.suptitle(f"{tag} -- sample {sample_idx} -- PSF & Profile  (NSR={intensity_ratio:.2f})", fontsize=14, fontweight="bold")
    tag_dir = os.path.join(EXAMPLES_DIR, tag)
    os.makedirs(tag_dir, exist_ok=True)
    plt.savefig(os.path.join(tag_dir, f"sample_{sample_idx:05d}_psf_profile.png"), dpi=150)
    plt.close(fig)




# =============================================================================
# Bulk per-tag metric pass
# =============================================================================


@torch.no_grad()
def evaluate_tag(model: torch.nn.Module, tag: str, na_mask: torch.Tensor) -> Dict:
    """Single full pass over the tag computes every metric for every sample
    plus caches the SMALL per-sample arrays needed to redraw a panel later
    (phasors, 40x40 images, metrics -- a few KB each). No raw R matrices are
    cached (1600x1600 complex, ~20MB each would blow up memory for no
    reason) -- and with the master matrix panel removed, NEITHER saved
    panel (Panel A: save_example_panel, Panel B: save_psf_and_profile_panel)
    needs R at all, so there's no second pass over the loader anymore
    either. After the full pass, target sample ptrs are selected (one-per-
    level for test_intensity_ratio; best/median/worst-spread by SSIM for
    every other tag -- see select_representative_ptrs) and both panels are
    drawn straight from the cache."""
    loader = build_test_loader(tag, BATCH_SIZE_METRICS)
    dataset: ARCH5Dataset = loader.dataset
    na_mask_np = na_mask[0, 0].cpu().numpy().astype(bool)  # for PSF panels; computed once, not per-sample

    # Per-sample values (not running sums) so both mean AND std are cheap
    # to compute afterward, for every metric this script tracks. Input and
    # output Strehl/RMS are still computed separately below (the physics
    # genuinely has two numbers), but per request, everywhere this gets
    # REPORTED (captions, plots, CSV headline columns) uses the AVERAGE of
    # the two -- input and output track each other closely in practice, and
    # one number per metric is easier to read than two. The raw in/out
    # values are still kept in `values`/CSV for anyone who wants to dig in,
    # just not shown on their own in any plot or panel caption anymore.
    values: Dict[str, List[float]] = {
        "strehl_o_unc": [], "strehl_o_corr": [], "strehl_i_unc": [], "strehl_i_corr": [],
        "rms_o_unc": [], "rms_o_corr": [], "rms_i_unc": [], "rms_i_corr": [],
        "strehl_avg_unc": [], "strehl_avg_corr": [], "rms_avg_unc": [], "rms_avg_corr": [],
        "peak_mean_unc": [], "peak_mean_corr": [],
        "ssim_unc": [], "ssim_corr": [], "psnr_unc": [], "psnr_corr": [],
    }
    per_sample_records = []
    cache: Dict[int, Dict] = {}

    h5_indices = dataset.indices
    extra = read_h5_fields(h5_indices, ["intensity_ratios", "object_styles"])

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

            strehl_avg_u, strehl_avg_c = (s_o_u + s_i_u) / 2.0, (s_o_c + s_i_c) / 2.0
            rms_avg_u, rms_avg_c = (r_o_u + r_i_u) / 2.0, (r_o_c + r_i_c) / 2.0
            values["strehl_avg_unc"].append(strehl_avg_u); values["strehl_avg_corr"].append(strehl_avg_c)
            values["rms_avg_unc"].append(rms_avg_u); values["rms_avg_corr"].append(rms_avg_c)

            R_np = R_complex[j].cpu().numpy()
            R_c_np = R_c[j].cpu().numpy()
            gt = batch["target_object"][j].numpy()
            # Uncorrected: dc-column. Corrected: full CASS (matches the
            # loss). See the note above compute_psf.
            unc_img = reconstruct_dc_column(R_np, GRID_SIZE)
            corr_img = reconstruct_cass(R_c_np, GRID_SIZE)
            pm_unc, pm_corr = signal_enhancement(unc_img, corr_img)
            values["peak_mean_unc"].append(pm_unc); values["peak_mean_corr"].append(pm_corr)
            # FIX (do not revert): normalize_image() BEFORE SSIM/PSNR --
            # see normalize_image's docstring. Raw-scale reconstructions
            # passed to data_range=1.0 SSIM / MAX=1 PSNR gave meaningless
            # near-zero SSIM and negative PSNR before this fix.
            unc_img_norm = normalize_image(unc_img)
            corr_img_norm = normalize_image(corr_img)
            ssim_corr_val = compute_ssim(corr_img_norm, gt)
            psnr_corr_val = compute_psnr(corr_img_norm, gt)
            ssim_unc_val = compute_ssim(unc_img_norm, gt)
            psnr_unc_val = compute_psnr(unc_img_norm, gt)
            values["ssim_unc"].append(ssim_unc_val); values["ssim_corr"].append(ssim_corr_val)
            values["psnr_unc"].append(psnr_unc_val); values["psnr_corr"].append(psnr_corr_val)

            nsr_value = float(extra["intensity_ratios"][ptr])
            object_style_value = extra["object_styles"][ptr]
            per_sample_records.append({
                "ptr": ptr,
                "strehl_avg_corr": strehl_avg_c, "rms_avg_corr": rms_avg_c,
                "ssim_corr": ssim_corr_val, "psnr_corr": psnr_corr_val,
                "intensity_ratio": nsr_value,
                "object_style": object_style_value,
            })

            cache[ptr] = {
                "targ_o": targ_o[j].detach().cpu(), "pred_o": pred_o[j].detach().cpu(),
                "targ_i": targ_i[j].detach().cpu(), "pred_i": pred_i[j].detach().cpu(),
                "unc_img": unc_img, "corr_img": corr_img, "gt": gt,
                "metrics": {"strehl_avg": strehl_avg_c, "rms_avg": rms_avg_c},
                "nsr_value": nsr_value, "object_style": object_style_value,
            }

            ptr += 1

    n_samples = ptr
    means = {k: float(np.mean(v)) if v else 0.0 for k, v in values.items()}
    stds = {k: float(np.std(v)) if v else 0.0 for k, v in values.items()}

    # ---- select which samples get a saved example/PSF panel ----
    if tag == "test_intensity_ratio":
        # FIX (do not revert): the previous approach saved a panel for the
        # first N DISTINCT intensity_ratio values encountered while
        # iterating batches in file order (shuffle=False) -- this depends
        # entirely on which values happen to show up early in the file, and
        # was producing far fewer than the intended one-per-level panels in
        # practice. Instead, pre-scan the already-loaded metadata and pick
        # exactly one sample index per NSR_LEVELS value -- guaranteed one
        # panel per level actually present in the data, regardless of
        # iteration order. A level with zero matching samples anywhere in
        # the tag is logged and skipped, not silently dropped.
        target_ptrs: set = set()
        nsr_arr = np.array(extra["intensity_ratios"], dtype=float)
        for level in NSR_LEVELS:
            matches = np.where(np.isclose(nsr_arr, level, atol=1e-6))[0]
            if len(matches) > 0:
                target_ptrs.add(int(matches[0]))
            else:
                print(f"    [warn] test_intensity_ratio: no sample found at intensity_ratio={level:g}")
    else:
        target_ptrs = select_representative_ptrs(per_sample_records, N_EXAMPLE_PANELS_PER_TAG)

    # Neither Panel A nor Panel B needs R -- both draw straight from the
    # cache, no second loader pass required.
    for p in sorted(target_ptrs):
        c = cache[p]
        save_example_panel(
            tag, p,
            c["targ_o"], c["pred_o"], c["targ_i"], c["pred_i"],
            c["unc_img"], c["corr_img"], c["gt"],
            c["metrics"], c["nsr_value"],
        )
        save_psf_and_profile_panel(
            tag, p,
            c["targ_o"], c["pred_o"], c["targ_i"], c["pred_i"],
            c["unc_img"], c["corr_img"], c["gt"],
            na_mask_np, c["nsr_value"],
        )

    return {
        "tag": tag, "n_samples": n_samples, "means": means, "stds": stds,
        "records": per_sample_records, "n_panels_saved": len(target_ptrs),
    }


# =============================================================================
# Plotting
# =============================================================================


def plot_strehl_rms(results: List[Dict]) -> None:
    tags = [r["tag"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(tags))
    width = 0.35

    specs = [
        (axes[0], "strehl_avg_unc", "strehl_avg_corr", "Strehl Ratio"),
        (axes[1], "rms_avg_unc", "rms_avg_corr", "RMS Phase Error (rad)"),
    ]
    for ax, unc_key, corr_key, ylabel in specs:
        unc_vals = [r["means"][unc_key] for r in results]
        corr_vals = [r["means"][corr_key] for r in results]
        ax.bar(x - width / 2, unc_vals, width, label="Uncorrected", color="#C0392B")
        ax.bar(x + width / 2, corr_vals, width, label="Corrected", color="#2E5FA3")
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
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
    axes[0].set_ylabel("Peak/Mean Intensity Ratio"); axes[0].legend(fontsize=9); axes[0].grid(axis="y", alpha=0.3)

    # SSIM and PSNR are the primary ground-truth-anchored quality metrics --
    # unlike Strehl (which only measures phase agreement and can be badly
    # misleading in noisy regimes, verified: test_high_noise_low_aberration
    # had the HIGHEST Strehl but the WORST image quality of any tag), these
    # directly compare the reconstructed image against the true object.
    unc_ssim = [r["means"]["ssim_unc"] for r in results]
    corr_ssim = [r["means"]["ssim_corr"] for r in results]
    axes[1].bar(x - width / 2, unc_ssim, width, label="Uncorrected", color="#C0392B")
    axes[1].bar(x + width / 2, corr_ssim, width, label="Corrected", color="#2E5FA3")
    axes[1].set_xticks(x); axes[1].set_xticklabels(tags, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("SSIM"); axes[1].legend(fontsize=9); axes[1].grid(axis="y", alpha=0.3)

    unc_psnr = [r["means"]["psnr_unc"] for r in results]
    corr_psnr = [r["means"]["psnr_corr"] for r in results]
    axes[2].bar(x - width / 2, unc_psnr, width, label="Uncorrected", color="#C0392B")
    axes[2].bar(x + width / 2, corr_psnr, width, label="Corrected", color="#2E5FA3")
    axes[2].set_xticks(x); axes[2].set_xticklabels(tags, rotation=20, ha="right", fontsize=9)
    axes[2].set_ylabel("PSNR (dB)"); axes[2].legend(fontsize=9); axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "image_quality_by_tag.png"), dpi=150)
    plt.close(fig)



def plot_nsr_sweep(nsr_result: Dict) -> None:
    """Noise intensity (noise-to-signal ratio) vs performance: Strehl and
    RMS (each averaged over input/output -- see evaluate_tag's note), as a
    function of test_intensity_ratio's deliberate NSR sweep."""
    records = nsr_result["records"]
    by_ratio: Dict[float, List[Dict]] = {}
    for rec in records:
        by_ratio.setdefault(rec["intensity_ratio"], []).append(rec)
    ratios = sorted(by_ratio.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    specs = [
        (axes[0], "strehl_avg_corr", "Strehl Ratio", "#2E5FA3"),
        (axes[1], "rms_avg_corr", "RMS Phase Error (rad)", "#C0392B"),
    ]
    for ax, key, ylabel, color in specs:
        means = [np.mean([r[key] for r in by_ratio[k]]) for k in ratios]
        stds = [np.std([r[key] for r in by_ratio[k]]) for k in ratios]
        ax.errorbar(ratios, means, yerr=stds, fmt="o-", color=color, capsize=4)
        ax.set_xlabel("Noise-to-Signal (Intensity) Ratio")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "noise_vs_performance.png"), dpi=150)
    plt.close(fig)


def plot_object_style_breakdown(results: List[Dict]) -> None:
    all_records = [rec for r in results for rec in r["records"]]
    by_style: Dict[str, List[float]] = {}
    for rec in all_records:
        by_style.setdefault(rec["object_style"], []).append(rec["strehl_avg_corr"])

    styles = sorted(by_style.keys())
    means = [np.mean(by_style[s]) for s in styles]
    stds = [np.std(by_style[s]) for s in styles]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(styles, means, yerr=stds, capsize=4, color="#2E5FA3")
    ax.set_title("Strehl Ratio (avg in/out, corrected) by Object Style")
    ax.set_ylabel("Strehl Ratio"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "strehl_by_object_style.png"), dpi=150)
    plt.close(fig)


def plot_object_style_by_nsr(nsr_result: Dict) -> None:
    """Cross-tabulated breakdown: does one object type degrade faster than
    another as noise increases? Heatmap of mean Strehl ratio (avg in/out,
    corrected), object_style x intensity_ratio (NSR), built from
    test_intensity_ratio's records -- the only tag with a clean, discrete,
    directly-comparable NSR axis shared across every object style (other
    tags draw intensity_ratio from a continuous range, which wouldn't bin
    cleanly the same way)."""
    records = nsr_result["records"]
    styles = sorted(set(r["object_style"] for r in records))
    levels = sorted(set(r["intensity_ratio"] for r in records))

    grid = np.full((len(styles), len(levels)), np.nan)
    counts = np.zeros((len(styles), len(levels)), dtype=int)
    for si, style in enumerate(styles):
        for li, level in enumerate(levels):
            vals = [
                r["strehl_avg_corr"] for r in records
                if r["object_style"] == style and np.isclose(r["intensity_ratio"], level, atol=1e-6)
            ]
            if vals:
                grid[si, li] = np.mean(vals)
                counts[si, li] = len(vals)

    fig, ax = plt.subplots(figsize=(2.0 * len(levels) + 3, 1.0 * len(styles) + 3))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"{l:g}" for l in levels])
    ax.set_yticks(range(len(styles))); ax.set_yticklabels(styles)
    ax.set_xlabel("Intensity ratio (NSR)")
    ax.set_ylabel("Object style")
    ax.set_title("Output Strehl Ratio (corrected): Object Style x NSR Level\n(test_intensity_ratio)")
    for si in range(len(styles)):
        for li in range(len(levels)):
            if not np.isnan(grid[si, li]):
                label = f"{grid[si, li]:.2f}\n(n={counts[si, li]})"
                color = "white" if grid[si, li] < 0.5 else "black"
                ax.text(li, si, label, ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="Strehl Ratio", fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "strehl_by_object_style_and_nsr.png"), dpi=150)
    plt.close(fig)



def plot_timing(timing: Dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    labels = [
        "Full pipeline\n(host batch -> reconstructed image)",
        "Total model pipeline\n(host batch -> $R_c$)",
        "Model + correction only\n(after R, $RR^\\dagger$, $R^\\dagger R$ built)",
    ]
    means = [timing["full_pipeline_ms_mean"], timing["total_ms_mean"], timing["model_only_ms_mean"]]
    stds = [timing["full_pipeline_ms_std"], timing["total_ms_std"], timing["model_only_ms_std"]]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, means, xerr=stds, height=0.4, capsize=5, color=["#7B241C", "#C0392B", "#2E5FA3"])
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
    ax.set_title(
        f"Inference Latency (N={N_TIMING_RUNS} runs, {N_TIMING_WARMUP} warmup discarded)\n"
        f"Full pipeline throughput: {timing['full_pipeline_fps']:.2f} FPS (vs. CLASS/DeepCLASS)",
        fontsize=11,
    )
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

    print("\n[1/4] Inference timing (batch=1)...")
    timing_loader = build_test_loader("test_standard", BATCH_SIZE_LATENCY)
    timing = measure_inference_time(model, timing_loader)
    print(
        f"  FULL pipeline (-> reconstructed image): {timing['full_pipeline_ms_mean']:.3f} +/- {timing['full_pipeline_ms_std']:.3f} ms"
        f"  ({timing['full_pipeline_fps']:.2f} FPS -- for direct comparison against CLASS/DeepCLASS throughput)\n"
        f"  Total model pipeline (-> R_c):          {timing['total_ms_mean']:.3f} +/- {timing['total_ms_std']:.3f} ms\n"
        f"  Model+correction only:                  {timing['model_only_ms_mean']:.3f} +/- {timing['model_only_ms_std']:.3f} ms\n"
        f"  (R/RRh/RhR construction alone:  {timing['construction_ms_mean']:.3f} +/- {timing['construction_ms_std']:.3f} ms)\n"
        f"  (CASS reconstruction alone:     {timing['reconstruction_ms_mean']:.3f} +/- {timing['reconstruction_ms_std']:.3f} ms)"
    )
    plot_timing(timing)

    print("\n[2/4] Per-tag metrics + example panels...")
    results = []
    for tag in TEST_TAGS:
        print(f"  evaluating {tag} ...")
        r = evaluate_tag(model, tag, na_mask)
        n_panels = r["n_panels_saved"]
        print(f"    n={r['n_samples']}  Strehl(avg,corr)={r['means']['strehl_avg_corr']:.4f} +/- {r['stds']['strehl_avg_corr']:.4f}  "
              f"RMS(avg,corr)={r['means']['rms_avg_corr']:.4f} +/- {r['stds']['rms_avg_corr']:.4f} rad  "
              f"({n_panels} panel A + panel B pairs saved)")
        results.append(r)

    plot_strehl_rms(results)
    plot_image_quality(results)
    plot_object_style_breakdown(results)

    nsr_result = next(r for r in results if r["tag"] == "test_intensity_ratio")
    plot_nsr_sweep(nsr_result)
    plot_object_style_by_nsr(nsr_result)

    print("\n[3/3] Writing summary CSV...")
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

    print(f"\nDone. Plots -> {OUTPUT_DIR}\nPanel A + Panel B examples -> {EXAMPLES_DIR}\nSummary CSV -> {csv_path}")


if __name__ == "__main__":
    main()