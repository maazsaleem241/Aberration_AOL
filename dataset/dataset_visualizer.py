"""dataset_visualizer.py -- regenerate dataset_generator.py's verification
plots from an ALREADY-GENERATED reflection_matrix_dataset.h5, without
re-running the (multi-hour) physics simulation at all.

Everything the plots need was already persisted per-sample when the dataset
was generated: R_matrices, phi_i_maps, phi_o_maps, target_objects,
object_styles, intensity_ratios, rms_scaling_factors, difficulty_tags,
split_types. This script only reads and plots -- it never touches the
Zernike engine, the object generator, or the USAF chart.

FIX (do not revert): the Input/Output Phase Aberration panels (phi_i, phi_o)
used to plot the RAW, UNWRAPPED phase directly. Since phi_i/phi_o are built
as a sum of many Zernike modes, their raw magnitude isn't bounded to
[-pi, pi] at all -- it can span several multiples of pi (samples have shown
up to +-6pi), with the color range scaled dynamically to match. This is
inconsistent with the "Reflection Matrix Phase angle R" panel right next to
it, which uses np.angle(R) -- inherently wrapped to [-pi, pi], since R only
ever depends on phase through exp(i*phi) (adding any multiple of 2*pi to phi
changes nothing about R). The unwrapped magnitude isn't something R -- or
anything downstream that only ever sees R -- can actually be sensitive to,
so wrapping loses no physically meaningful information. This script wraps
phi_i/phi_o consistently before plotting and uses a fixed vmin=-pi, vmax=+pi
range instead of the previous dynamic multi-pi scaling.

Usage examples:
    # Render specific sample indices
    python dataset_visualizer.py --h5-file /path/to/data/reflection_matrix_dataset.h5 --indices 0-99,9000-9099

    # Render 20 random samples from a specific difficulty tag
    python dataset_visualizer.py --h5-file /path/to/data/reflection_matrix_dataset.h5 --tag noise_high --n-samples 20

    # Only the combined master-matrix figure, skip the 6 individual panel exports (faster for many samples)
    python dataset_visualizer.py --h5-file /path/to/data/reflection_matrix_dataset.h5 --tag no_noise --n-samples 20 --master-only
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Optional

import h5py
import matplotlib
matplotlib.use("Agg")  # headless: this script only ever saves figures, never shows a window
import matplotlib.pyplot as plt
import numpy as np

GRID_SIZE = 40
NUM_ELEMENTS = GRID_SIZE * GRID_SIZE


def wrap_phase(phi: np.ndarray) -> np.ndarray:
    """Fold raw (possibly many-multiples-of-pi) phase into the physically
    meaningful, R-observable range [-pi, pi]."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def apply_plot_styling(ax, im, data, style: str = "general", vmin: float = 0.0, vmax: float = 1.0, pad: float = 0.04) -> None:
    ax.set_aspect("equal")

    if style == "phase":
        if np.allclose(vmin, 0) and np.allclose(vmax, 0):
            ticks = [0.0]
            tick_labels = ["0"]
        else:
            low_bound = int(np.floor(vmin / np.pi))
            high_bound = int(np.ceil(vmax / np.pi))

            ticks = [low_bound * np.pi, 0.0, high_bound * np.pi]
            ticks = sorted(set(ticks))

            tick_labels = []
            for val in ticks:
                k = int(np.round(val / np.pi))
                if k == 0:
                    tick_labels.append("0")
                elif k == 1:
                    tick_labels.append(r"$\pi$")
                elif k == -1:
                    tick_labels.append(r"$-\pi$")
                else:
                    tick_labels.append(f"${k}\\pi$")

        cbar = plt.colorbar(im, ax=ax, ticks=ticks, pad=pad)
        cbar.ax.set_yticklabels(tick_labels)
    else:
        mid = (vmin + vmax) / 2.0
        ticks = [vmin, mid, vmax] if vmax > vmin else [vmin]
        cbar = plt.colorbar(im, ax=ax, ticks=ticks, pad=pad)
        cbar.ax.set_yticklabels([f"{v:.2f}" for v in ticks])


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def render_dataset_plots(
    h5_path: str,
    output_dir: str,
    indices_to_render: List[int],
    dataset_name: str,
    master_only: bool = False,
) -> None:
    master_root = os.path.join(output_dir, "master_matrix")
    indiv_root = os.path.join(output_dir, "individual_panels")

    with h5py.File(h5_path, "r") as h5f:
        for idx in indices_to_render:
            R = h5f["R_matrices"][idx, 0] + 1j * h5f["R_matrices"][idx, 1]
            phi_i_raw = h5f["phi_i_maps"][idx]
            phi_o_raw = h5f["phi_o_maps"][idx]
            target_obj = h5f["target_objects"][idx]
            target_style = _decode(h5f["object_styles"][idx])
            tag = _decode(h5f["difficulty_tags"][idx])
            split = _decode(h5f["split_types"][idx])
            intensity_ratio = float(h5f["intensity_ratios"][idx])
            aberration_multiplier = float(h5f["aberration_multipliers"][idx])
            rms_val = float(h5f["rms_scaling_factors"][idx])

            # FIX (do not revert): wrap here, once, before any plotting or
            # vmin/vmax computation touches these arrays -- see module
            # docstring for why the previous unwrapped, dynamically-scaled
            # (up to +-6pi) representation was inconsistent with the rest of
            # the project's phase displays.
            phi_i = wrap_phase(phi_i_raw)
            phi_o = wrap_phase(phi_o_raw)

            master_dir = os.path.join(master_root, tag)
            sample_panel_dir = os.path.join(indiv_root, tag, f"sample_{idx:05d}")
            os.makedirs(master_dir, exist_ok=True)
            if not master_only:
                os.makedirs(sample_panel_dir, exist_ok=True)

            dc_idx = (GRID_SIZE // 2) * GRID_SIZE + (GRID_SIZE // 2)
            degraded_fourier = np.reshape(R[:, dc_idx], (GRID_SIZE, GRID_SIZE))
            degraded_spatial = np.fft.ifft2(np.fft.ifftshift(degraded_fourier))
            # No .T here -- verified correct orientation without one; see
            # dataset_generator.py's own history for why (a stale .T left
            # over from before an earlier indexing bug was fixed would
            # otherwise silently re-introduce a rotated reconstruction).
            degraded_field = np.abs(degraded_spatial) ** 2
            max_deg = np.max(degraded_field)
            norm_degraded = degraded_field / max_deg if max_deg > 0 else np.zeros_like(degraded_field, dtype=np.float32)

            freq_extent = [-1, 1, -1, 1]
            matrix_extent_display = [0, NUM_ELEMENTS, NUM_ELEMENTS, 0]

            # FIX (do not revert): fixed +-pi range now, matching every other
            # phase display in the project, instead of the old dynamic
            # multi-pi scaling based on the raw unwrapped magnitude.
            global_vmin_phase, global_vmax_phase = -np.pi, np.pi

            fig = plt.figure(figsize=(22, 15))
            gs = fig.add_gridspec(3, 4, wspace=0.35, hspace=0.35)
            fig.suptitle(f"Master Verification Matrix - {dataset_name.upper()} | #{idx:05d} | [{tag.upper()}]", fontsize=18, fontweight="bold")

            ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2]); ax4 = fig.add_subplot(gs[0, 3])
            ax5 = fig.add_subplot(gs[1, 0]); ax6 = fig.add_subplot(gs[1, 1]); ax7 = fig.add_subplot(gs[2, 0]); ax8 = fig.add_subplot(gs[2, 1])
            ax_meta = fig.add_subplot(gs[1:, 2:].subgridspec(3, 1, height_ratios=[0.18, 0.64, 0.18])[1])

            im1 = ax1.imshow(phi_i, cmap="jet", extent=freq_extent, interpolation="nearest", vmin=global_vmin_phase, vmax=global_vmax_phase)
            ax1.set_title("Input Phase Aberration", fontsize=12)
            ax1.set_xlabel("Normalized Pupil X"); ax1.set_ylabel("Normalized Pupil Y")
            ax1.set_xticks([-1.0, 0.0, 1.0]); ax1.set_yticks([-1.0, 0.0, 1.0])
            apply_plot_styling(ax1, im1, phi_i, style="phase", vmin=global_vmin_phase, vmax=global_vmax_phase)

            im2 = ax2.imshow(phi_o, cmap="jet", extent=freq_extent, interpolation="nearest", vmin=global_vmin_phase, vmax=global_vmax_phase)
            ax2.set_title("Output Phase Aberration", fontsize=12)
            ax2.set_xlabel("Normalized Pupil X"); ax2.set_ylabel("Normalized Pupil Y")
            ax2.set_xticks([-1.0, 0.0, 1.0]); ax2.set_yticks([-1.0, 0.0, 1.0])
            apply_plot_styling(ax2, im2, phi_o, style="phase", vmin=global_vmin_phase, vmax=global_vmax_phase)

            im3 = ax3.imshow(target_obj, cmap="gray", extent=freq_extent, interpolation="nearest")
            ax3.set_title("Original Target Object (Ground Truth)", fontsize=12)
            ax3.set_xlabel("Normalized Spatial X"); ax3.set_ylabel("Normalized Spatial Y")
            ax3.set_xticks([-1.0, 0.0, 1.0]); ax3.set_yticks([-1.0, 0.0, 1.0])
            apply_plot_styling(ax3, im3, target_obj, style="general", vmin=0.0, vmax=1.0)

            im4 = ax4.imshow(norm_degraded, cmap="hot", extent=freq_extent, interpolation="bicubic", vmin=0.0, vmax=1.0)
            ax4.set_title("Uncorrected Image Intensity", fontsize=12)
            ax4.set_xlabel("Normalized Spatial X"); ax4.set_ylabel("Normalized Spatial Y")
            ax4.set_xticks([-1.0, 0.0, 1.0]); ax4.set_yticks([-1.0, 0.0, 1.0])
            apply_plot_styling(ax4, im4, norm_degraded, style="general", vmin=0.0, vmax=1.0, pad=0.05)

            log_amp = np.log10(np.abs(R) + 1e-5)
            im5 = ax5.imshow(log_amp, cmap="magma", extent=matrix_extent_display, origin="upper")
            ax5.set_title("Reflection Matrix Amplitude |R|", fontsize=12)
            ax5.set_xlabel("Input Channel Element Index"); ax5.set_ylabel("Output Channel Element Index")
            ax5.set_xticks([0, NUM_ELEMENTS // 2, NUM_ELEMENTS]); ax5.set_yticks([0, NUM_ELEMENTS // 2, NUM_ELEMENTS])
            apply_plot_styling(ax5, im5, log_amp, style="general", vmin=np.min(log_amp), vmax=np.max(log_amp), pad=0.08)

            phase_ang = np.angle(R)  # already inherently wrapped -- np.angle always returns the principal value
            im6 = ax6.imshow(phase_ang, cmap="twilight", extent=matrix_extent_display, origin="upper", vmin=-np.pi, vmax=np.pi)
            ax6.set_title("Reflection Matrix Phase angle R", fontsize=12)
            ax6.set_xlabel("Input Channel Element Index"); ax6.set_ylabel("Output Channel Element Index")
            ax6.set_xticks([0, NUM_ELEMENTS // 2, NUM_ELEMENTS]); ax6.set_yticks([0, NUM_ELEMENTS // 2, NUM_ELEMENTS])
            apply_plot_styling(ax6, im6, phase_ang, style="phase", vmin=-np.pi, vmax=np.pi, pad=0.08)

            ax7.plot(phi_i[GRID_SIZE // 2, :], label="Input Aberration Center Cut", color="b", lw=2)
            ax7.plot(phi_o[GRID_SIZE // 2, :], label="Output Aberration Center Cut", color="r", lw=2)
            ax7.set_title("Mid-Line Spatial Profiles (wrapped)", fontsize=12)
            ax7.set_xlabel("Pupil Grid Position Index"); ax7.set_ylabel("Phase Variation (rad)")
            ax7.set_ylim(-np.pi - 0.2, np.pi + 0.2)
            ax7.legend(); ax7.grid(True)

            ax8.hist(phi_i.flatten(), bins=20, alpha=0.5, label="Input Aberration Spectrum", color="blue")
            ax8.hist(phi_o.flatten(), bins=20, alpha=0.5, label="Output Aberration Spectrum", color="red")
            ax8.set_title("Phase Spatial Histograms (wrapped)", fontsize=12)
            ax8.set_xlabel("Phase Value Amplitude (rad)"); ax8.set_ylabel("Total Normalized Counts")
            ax8.legend()

            ax_meta.axis("on")
            ax_meta.set_facecolor("#0d1117")
            ax_meta.get_xaxis().set_visible(False); ax_meta.get_yaxis().set_visible(False)
            for spine in ax_meta.spines.values():
                spine.set_color("#30363d"); spine.set_linewidth(2)

            meta_text = (
                f"    Meta Data\n------------------------------------------------------------------\n\n"
                f"  \u2022 Target Folder Name     :  {dataset_name}\n"
                f"  \u2022 Selected Dataset Split :  {split.upper()}\n"
                f"  \u2022 Structural Object Type :  {target_style.upper()}\n"
                f"  \u2022 Dynamic Physics Tag    :  {tag.upper()}\n"
                f"  \u2022 Intensity Ratio        :  {intensity_ratio:.4f}\n"
                f"  \u2022 Aberration Factor      :  {aberration_multiplier:.4f}\n"
                f"  \u2022 RMS Scaling Factor     :  {rms_val:.4e}\n"
                f"  \u2022 Total Array Boundaries :  {GRID_SIZE} x {GRID_SIZE} Pixels"
            )
            ax_meta.text(0.08, 0.5, meta_text, transform=ax_meta.transAxes, fontsize=14, color="#c9d1d9", family="monospace", ha="left", va="center", fontweight="bold")

            plt.savefig(os.path.join(master_dir, f"sample_{idx:05d}_visualized.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

            if not master_only:
                plots = [
                    ("fig_01_input_phase.png", phi_i, "jet", "spatial", "Input Phase Aberration (phi_i, wrapped)", "phase", global_vmin_phase, global_vmax_phase, "nearest"),
                    ("fig_02_output_phase.png", phi_o, "jet", "spatial", "Output Phase Aberration (phi_o, wrapped)", "phase", global_vmin_phase, global_vmax_phase, "nearest"),
                    ("fig_03a_original_object.png", target_obj, "gray", "spatial", "Original Target Object", "general", 0.0, 1.0, "nearest"),
                    ("fig_04_uncorrected_intensity.png", norm_degraded, "hot", "spatial", "Uncorrected Image Plane Intensity", "general", 0.0, 1.0, "bicubic"),
                    ("fig_05_matrix_amplitude.png", log_amp, "magma", "matrix", "Reflection Matrix Amplitude |R|", "general", np.min(log_amp), np.max(log_amp), None),
                    ("fig_06_matrix_phase.png", phase_ang, "twilight", "matrix", "Reflection Matrix Phase angle R", "phase", -np.pi, np.pi, None),
                ]
                for fname, data, cmap, ptype, title, style, vmin, vmax, interp in plots:
                    plt.figure(figsize=(7, 6))
                    this_extent = freq_extent if ptype == "spatial" else matrix_extent_display
                    im = plt.imshow(data, cmap=cmap, extent=this_extent, interpolation=interp, origin="upper", vmin=vmin, vmax=vmax)
                    plt.title(title, fontsize=13, fontweight="bold")
                    apply_plot_styling(plt.gca(), im, data, style=style, vmin=vmin, vmax=vmax)
                    plt.tight_layout()
                    plt.savefig(os.path.join(sample_panel_dir, fname), dpi=200)
                    plt.close()

            print(f"  rendered sample {idx:05d} [{tag}]")


def mode_a_indices(total_samples: int) -> List[int]:
    """Exact replica of dataset_generator.py's mode='A' viz_triggers: the
    first 100 and last 100 samples of the whole dataset (mode A has only one
    difficulty_tag, 'output_only_aberration', spanning the entire dataset --
    there's no per-stage split to iterate over)."""
    first = list(range(0, min(100, total_samples)))
    last = list(range(max(0, total_samples - 100), total_samples))
    return sorted(set(first + last))


def mode_b_indices(total_samples: int) -> List[int]:
    """Exact replica of dataset_generator.py's mode='B' viz_triggers: for
    EACH of the 12 difficulty-tag sub-intervals (the literal same
    sub_mode_intervals list from the generator), the first 100 and last 100
    samples of that interval. Each interval boundary corresponds exactly to
    one difficulty_tag value, so render_dataset_plots' existing per-tag
    folder organization (master_matrix/<tag>/, individual_panels/<tag>/)
    automatically sorts these into their own folders with no extra logic
    needed here."""
    sub_mode_intervals = [
        (0, 1500), (1500, 3000), (3000, 5500), (5500, 9000),
        (9000, 9500), (9500, 10000), (10000, 10500), (10500, 11000),
        (11000, 12000), (12000, 13000), (13000, 14000), (14000, 15000),
    ]
    indices: set = set()
    for start, end in sub_mode_intervals:
        if end <= total_samples:
            indices.update(range(start, start + 100))
            indices.update(range(end - 100, end))
    return sorted(indices)


def mode_c_indices(total_samples: int) -> List[int]:
    """Exact replica of dataset_generator.py's mode='C' (aberration
    curriculum) sub-intervals: the first 100 and last 100 samples of each of
    its 12 difficulty-tag sub-intervals. The boundaries here are currently
    identical to mode_b_indices' -- dataset_c mirrors dataset_b's exact
    sample-count structure by design (aberration_low/med/high/extreme in
    place of no_noise/noise_low/noise_med/noise_high, etc.) -- but kept as a
    separate function so mode C's boundaries can be changed independently of
    mode B's later without silently affecting both."""
    sub_mode_intervals = [
        (0, 1500), (1500, 3000), (3000, 5500), (5500, 9000),
        (9000, 9500), (9500, 10000), (10000, 10500), (10500, 11000),
        (11000, 12000), (12000, 13000), (13000, 14000), (14000, 15000),
    ]
    indices: set = set()
    for start, end in sub_mode_intervals:
        if end <= total_samples:
            indices.update(range(start, start + 100))
            indices.update(range(end - 100, end))
    return sorted(indices)


def _total_samples(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as h5f:
        return h5f["R_matrices"].shape[0]


def _parse_indices(spec: str) -> List[int]:
    """Parse '0-99,9000-9099,150' style range/list specs into a sorted list
    of unique integer indices."""
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            indices.update(range(lo, hi + 1))
        else:
            indices.add(int(part))
    return sorted(indices)


def visualize_dataset(
    base_path: str,
    dataset_name: str,
    tag: Optional[str] = None,
    indices: Optional[List[int]] = None,
    n_samples: int = 20,
    seed: int = 0,
    master_only: bool = False,
) -> None:
    """Convenience wrapper matching dataset_generator.py's own
    (root_dir, dataset_name) call pattern -- resolves h5_path/output_dir the
    same way dataset_generator.py itself does (base_path/dataset_name/data/
    reflection_matrix_dataset.h5), so this is a drop-in visualize-only
    counterpart to execute_dataset_production(). Provide either `tag`
    (renders `n_samples` random samples matching that difficulty_tag) or
    `indices` (an explicit list of sample indices) -- not both.
    """
    h5_path = os.path.join(base_path, dataset_name, "data", "reflection_matrix_dataset.h5")
    output_dir = os.path.join(base_path, dataset_name, "visualization")

    if indices is not None:
        idx_list = indices
    elif tag is not None:
        rng = np.random.RandomState(seed)
        with h5py.File(h5_path, "r") as h5f:
            tags = np.array([_decode(t) for t in h5f["difficulty_tags"][:]])
        matching = np.where(tags == tag)[0]
        if len(matching) == 0:
            raise ValueError(f"No samples found with difficulty_tag='{tag}' in {dataset_name}. Available tags: {sorted(set(tags))}")
        n = min(n_samples, len(matching))
        idx_list = sorted(rng.choice(matching, size=n, replace=False).tolist())
    else:
        raise ValueError("Provide either tag= or indices=")

    print(f"=== Visualizing {dataset_name} ===")
    print(f"H5 file:     {h5_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Rendering {len(idx_list)} sample(s): {idx_list[:10]}{' ...' if len(idx_list) > 10 else ''}")

    render_dataset_plots(h5_path, output_dir, idx_list, dataset_name, master_only=master_only)
    print(f"Done with {dataset_name}.\n")


def main_cli() -> None:
    """Flexible argparse entry point for one-off / ad-hoc use, e.g.:
        python dataset_visualizer.py --h5-file /path/to/reflection_matrix_dataset.h5 --tag noise_high --n-samples 20
    Not used by the __main__ block below (that uses visualize_dataset()
    directly, mirroring dataset_generator.py's own two-line style) -- call
    this yourself if you want CLI-argument-driven behavior instead.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5-file", type=str, required=True, help="Path to reflection_matrix_dataset.h5")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to write the visualization/ folder (default: sibling of the H5 file's parent 'data' dir, matching dataset_generator.py's own layout)")
    parser.add_argument("--dataset-name", type=str, default=None, help="Label used in figure titles (default: inferred from the H5 file's grandparent directory name)")
    parser.add_argument("--indices", type=str, default=None, help="Explicit indices/ranges to render, e.g. '0-99,9000-9099,150'")
    parser.add_argument("--tag", type=str, default=None, help="Render N random samples matching this difficulty_tag (use with --n-samples) instead of --indices")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of random samples to draw when using --tag")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --tag sampling")
    parser.add_argument("--master-only", action="store_true", help="Only render the combined master-matrix figure, skip the 6 individual panel exports (faster for many samples)")
    args = parser.parse_args()

    h5_path = os.path.abspath(args.h5_file)
    data_dir = os.path.dirname(h5_path)
    dataset_root = os.path.dirname(data_dir)

    output_dir = args.output_dir or os.path.join(dataset_root, "visualization")
    dataset_name = args.dataset_name or os.path.basename(dataset_root)

    if args.indices:
        indices = _parse_indices(args.indices)
    elif args.tag:
        rng = np.random.RandomState(args.seed)
        with h5py.File(h5_path, "r") as h5f:
            tags = np.array([_decode(t) for t in h5f["difficulty_tags"][:]])
        matching = np.where(tags == args.tag)[0]
        if len(matching) == 0:
            raise ValueError(f"No samples found with difficulty_tag='{args.tag}'. Available tags: {sorted(set(tags))}")
        n = min(args.n_samples, len(matching))
        indices = sorted(rng.choice(matching, size=n, replace=False).tolist())
    else:
        raise ValueError("Provide either --indices or --tag/--n-samples")

    print(f"H5 file:     {h5_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Rendering {len(indices)} sample(s): {indices[:10]}{' ...' if len(indices) > 10 else ''}")

    render_dataset_plots(h5_path, output_dir, indices, dataset_name, master_only=args.master_only)
    print("Done.")


if __name__ == "__main__":
    # Exact replica of dataset_generator.py's own viz_triggers logic (see
    # mode_a_indices/mode_b_indices/mode_c_indices above) -- first 100 and
    # last 100 samples, per difficulty-tag interval for dataset_b/dataset_c,
    # whole-dataset for dataset_a. Comment out whichever dataset you don't
    # need visualized right now.
    BASE_PATH = "/home/awais/Desktop/Maaz/Maaz Data"

   # h5_a = os.path.join(BASE_PATH, "dataset_a", "data", "reflection_matrix_dataset.h5")
   # visualize_dataset(BASE_PATH, "dataset_a", indices=mode_a_indices(_total_samples(h5_a)))

    #h5_b = os.path.join(BASE_PATH, "dataset_b", "data", "reflection_matrix_dataset.h5")
    #visualize_dataset(BASE_PATH, "dataset_b", indices=mode_b_indices(_total_samples(h5_b)))

    h5_c = os.path.join(BASE_PATH, "dataset_c", "data", "reflection_matrix_dataset.h5")
    visualize_dataset(BASE_PATH, "dataset_c", indices=mode_c_indices(_total_samples(h5_c)))
