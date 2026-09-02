"""Shared, non-differentiable NumPy reconstruction helpers for visualization
and evaluation scripts across the CLEAR / ARC pipeline.

Why this module exists
-----------------------
`PhysicsInformedLoss._reconstruct_from_R_c` (defined identically in both
ARC_training_aberration.py and ARC_training_noise.py) is the DIFFERENTIABLE
torch implementation of the full 2D Delta-k CASS-style reconstruction used to
drive the SSIM and foreground-weighted L1 loss terms during training. That
implementation stays exactly where it is -- it is part of the loss, not the
visualizer, and this module does not touch it.

Every plotting/evaluation script in the project (both training scripts'
save_visualization(), and both evaluation scripts' example/master/PSF panels
and SSIM/PSNR metric computation) previously reconstructed images with a much
weaker "dc-column" method: pull the single on-axis illumination column of the
matrix, reshape, ifft2, intensity. That throws away almost all of the signal
in R and is NOT what the paper claims or what the network is actually
optimized against.

This module provides one canonical NumPy port of the CASS reconstruction
(`reconstruct_cass`), matching the training loss's bin_id / Delta-k
convention exactly, for every script that needs a reconstructed image but
isn't inside an autograd graph. Import this instead of reimplementing the
accumulation locally -- that duplication is exactly the kind of convention
drift that has caused bugs elsewhere in this project (see zernike.py's
off=0/stride=2 fix, and the linear-vs-wraparound Delta-k fix).

The old dc-column method is kept here too, as `reconstruct_dc_column`, purely
as an optional side-by-side comparison (e.g. a supplementary figure showing
why CASS is used) -- it is not used by default anywhere in the pipeline.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np


# =============================================================================
# Delta-k bin precompute (matches PhysicsInformedLoss.__init__ exactly)
# =============================================================================


@lru_cache(maxsize=8)
def precompute_bin_id(grid_size: int) -> Tuple[np.ndarray, int]:
    """Precompute the 2D spatial-frequency DIFFERENCE bin id for every
    (row, col) pair of the flattened (grid_size**2, grid_size**2) matrix.

    Identical convention to PhysicsInformedLoss.__init__'s bin_id (torch) and
    to zernike.py's construct_scattering_matrix (diff_v, diff_u mod
    grid_size) -- do not change this independently of those two.

    Cached per grid_size since it depends only on grid_size, not on any
    particular sample -- avoid recomputing it on every reconstruction call in
    a tight evaluation loop.

    Returns:
        bin_id:   int64 array, shape (grid_size**4,), flattened.
        num_bins: grid_size ** 2.
    """
    u = np.arange(grid_size)
    uu, vv = np.meshgrid(u, u, indexing="ij")
    u_flat = uu.flatten()  # column component
    v_flat = vv.flatten()  # row component
    diff_u = (u_flat[:, None] - u_flat[None, :]) % grid_size
    diff_v = (v_flat[:, None] - v_flat[None, :]) % grid_size
    bin_id = (diff_v * grid_size + diff_u).astype(np.int64).flatten()
    num_bins = grid_size * grid_size
    return bin_id, num_bins


# =============================================================================
# CASS reconstruction (primary method -- matches the paper and the loss)
# =============================================================================


def compute_cass_spectrum(mat_complex: np.ndarray, grid_size: int) -> np.ndarray:
    """The intermediate Delta-k-binned object spectrum -- i.e. everything
    reconstruct_cass does UP TO but NOT INCLUDING the final ifft2/intensity
    step. This is its own function (rather than inlined) specifically so it
    can be plotted as a standalone "CASS spectrum, pre-IFFT" panel, matching
    the CLEAR framework figure's "CASS-based reconstruction -> 2D IFFT ->
    Corrected image" pipeline: this function is that first box, reconstruct_
    cass's own ifft2 call is the second box.

    All reflection-matrix entries are coherently accumulated into their
    corresponding momentum-transfer (Delta-k) bin, reshaped to the object
    spectrum, and re-centered (roll) -- identical accumulation/convention to
    PhysicsInformedLoss._reconstruct_from_R_c, just stopped one step earlier.

    Args:
        mat_complex: complex matrix, shape (grid_size**2, grid_size**2).
        grid_size:   pupil grid size (40 throughout this project).

    Returns:
        Complex object spectrum, shape (grid_size, grid_size), NOT yet
        inverse-Fourier-transformed.
    """
    bin_id, num_bins = precompute_bin_id(grid_size)

    real = mat_complex.real.reshape(-1)
    imag = mat_complex.imag.reshape(-1)

    object_real = np.zeros(num_bins, dtype=np.float64)
    object_imag = np.zeros(num_bins, dtype=np.float64)
    np.add.at(object_real, bin_id, real)
    np.add.at(object_imag, bin_id, imag)

    object_spectrum = (object_real + 1j * object_imag).reshape(grid_size, grid_size)

    shift = grid_size // 2
    object_spectrum = np.roll(np.roll(object_spectrum, shift, axis=-1), shift, axis=-2)

    return object_spectrum


def reconstruct_cass(mat_complex: np.ndarray, grid_size: int, normalize: bool = False) -> np.ndarray:
    """Full 2D Delta-k-based CASS-style reconstruction.

    All reflection-matrix entries are coherently accumulated into their
    corresponding momentum-transfer (Delta-k) bin, then reshaped to the
    object spectrum, inverse-shifted, inverse Fourier transformed, and
    converted to intensity. This is the NumPy, non-differentiable twin of
    PhysicsInformedLoss._reconstruct_from_R_c -- same accumulation, same
    roll-based re-centering, same convention. Use this for any raw R (the
    "distorted" / uncorrected view) or any model-predicted R_c (the
    "corrected" / final view); the same function handles both.

    Unchanged signature/behavior from before compute_cass_spectrum and
    reconstruct_cass_with_spectrum existed -- every existing call site keeps
    working exactly as-is. If you also want the intermediate spectrum (e.g.
    for a "CASS spectrum, pre-IFFT" panel), use reconstruct_cass_with_
    spectrum instead so the accumulation isn't done twice.

    Args:
        mat_complex: complex matrix, shape (grid_size**2, grid_size**2).
        grid_size:   pupil grid size (40 throughout this project).
        normalize:   if True, divide by the per-image max (matching the
                     internal normalization the training loss applies for
                     loss-scale stability). Default False, since every
                     existing plotting script already does its own
                     img / img.max() normalization at display time -- leave
                     that in place rather than double-normalizing.

    Returns:
        Real intensity image, shape (grid_size, grid_size).
    """
    object_spectrum = compute_cass_spectrum(mat_complex, grid_size)
    spatial = np.fft.ifft2(object_spectrum)
    intensity = np.abs(spatial) ** 2

    if normalize:
        mx = intensity.max()
        if mx > 1e-12:
            intensity = intensity / mx

    return intensity


def reconstruct_cass_with_spectrum(
    mat_complex: np.ndarray, grid_size: int, normalize: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Same reconstruction as reconstruct_cass, but also returns the
    intermediate Delta-k spectrum (pre-IFFT) alongside the final intensity
    image, computing the shared accumulation step only once. Use this
    wherever a panel wants to show both the "CASS spectrum" box and the
    "Corrected image" box from the CLEAR framework figure.

    Returns:
        (object_spectrum, intensity) -- complex (grid_size, grid_size)
        spectrum, and real (grid_size, grid_size) intensity image.
    """
    object_spectrum = compute_cass_spectrum(mat_complex, grid_size)
    spatial = np.fft.ifft2(object_spectrum)
    intensity = np.abs(spatial) ** 2

    if normalize:
        mx = intensity.max()
        if mx > 1e-12:
            intensity = intensity / mx

    return object_spectrum, intensity


def spectrum_log_amplitude(object_spectrum: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """log10(|spectrum| + eps) for display -- same convention already used
    for the log_amp_R / log_amp_RRh / log_amp_RhR panels elsewhere in this
    project, applied here to the intermediate CASS Delta-k spectrum so it
    plots on a comparable dynamic range."""
    return np.log10(np.abs(object_spectrum) + eps)


# =============================================================================
# dc-column reconstruction -- kept ONLY as an optional side-by-side reference.
# Not used by default anywhere in the pipeline; the paper reports CASS.
# =============================================================================


def reconstruct_dc_column(mat_complex: np.ndarray, grid_size: int) -> np.ndarray:
    """Legacy single-column reconstruction: pull the on-axis illumination
    column, reshape, ifft2, intensity. Throws away almost all of the signal
    in R compared to reconstruct_cass -- retained here only so a
    dc-column-vs-CASS comparison figure can be made if ever useful, not for
    any figure or metric reported in the paper.
    """
    dc_idx = (grid_size // 2) * grid_size + (grid_size // 2)
    col = mat_complex[:, dc_idx].reshape(grid_size, grid_size)
    spatial = np.fft.ifft2(np.fft.ifftshift(col))
    return np.abs(spatial) ** 2


def reconstruct_comparison(mat_complex: np.ndarray, grid_size: int) -> dict:
    """Convenience helper for an explicit side-by-side comparison figure.
    Returns {'cass': ..., 'dc_column': ...}, both un-normalized intensity
    images. Not used in any default panel -- call this explicitly only when
    you want to show the two methods against each other.
    """
    return {
        "cass": reconstruct_cass(mat_complex, grid_size),
        "dc_column": reconstruct_dc_column(mat_complex, grid_size),
    }