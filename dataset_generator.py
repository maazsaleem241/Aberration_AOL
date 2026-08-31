import os
import json
import h5py
import numpy as np
import time  
from scipy.io import loadmat
from scipy.ndimage import zoom
from typing import List, Tuple
import backup.zernike_backup as z

GRID_SIZE = 40          
NUM_ELEMENTS = GRID_SIZE * GRID_SIZE  

# ==========================================
# DYNAMIC TARGET GENERATOR (OBJECT DIVERSITY)
# ==========================================


def generate_diverse_target(
    grid_size: int = GRID_SIZE,
    usaf_matrix: np.ndarray = None,
) -> Tuple[np.ndarray, str]:
    """
    Generate a randomized target that is fully contained inside a common
    interior bounding box.

    Design:
      - All object classes share the same safe spatial region.
      - No object can touch the 40x40 image boundary or extend into a corner.
      - The default object box is 20x20, matching the reduced scale used by
        the existing cross/grating/letter generators.
      - Object placement is randomized inside the image while the full object
        remains inside the safe box.
      - Beads are filled circular disks rather than cross-shaped pixels.
      - Crosses are thin single-pixel arms rather than thick crosses.
      - USAF crops are resized to the same reduced scale before placement.
    """
    obj = np.zeros((grid_size, grid_size), dtype=np.float32)

    # ------------------------------------------------------------------
    # Global safe bounding box
    # ------------------------------------------------------------------
    # Keep the same general "half-field" scale already used by the other
    # structured objects, while reserving a margin around every target.
    box_size = min(20, grid_size - 4)
    margin = max(1, (grid_size - box_size) // 2)

    # The box is [top, left] : [top + box_size, left + box_size].
    # For 40x40 this is 10:30 in both dimensions.
    box_top = np.random.randint(margin, grid_size - margin - box_size + 1)
    box_left = np.random.randint(margin, grid_size - margin - box_size + 1)
    box_bottom = box_top + box_size
    box_right = box_left + box_size

    # ------------------------------------------------------------------
    # Helper: paste a local object into the global safe box
    # ------------------------------------------------------------------
    def paste_local(local: np.ndarray, top: int, left: int) -> None:
        """Paste a local object while guaranteeing it stays inside the box."""
        h, w = local.shape
        if h > box_size or w > box_size:
            raise ValueError(
                f"Local object {(h, w)} exceeds safe box {(box_size, box_size)}"
            )

        max_top = box_bottom - h
        max_left = box_right - w
        top = int(np.clip(top, box_top, max_top))
        left = int(np.clip(left, box_left, max_left))

        obj[top:top + h, left:left + w] = np.maximum(
            obj[top:top + h, left:left + w],
            local.astype(np.float32),
        )

    style_options = ["beads", "cross", "grating", "letter"]
    if usaf_matrix is not None:
        style_options.append("usaf")

    style = np.random.choice(style_options)

    # ------------------------------------------------------------------
    # Beads: filled circular disks, not cross-like plus signs
    # ------------------------------------------------------------------
    if style == "beads":
        num_beads = np.random.randint(3, 7)

        # Track placed bead centres so we can enforce a minimum gap.
        placed_centres = []  # list of (cy, cx, radius)

        for _ in range(num_beads):
            radius = np.random.choice([1, 1, 2])
            diameter = 2 * radius + 1
            local = np.zeros((diameter, diameter), dtype=np.float32)
            yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            disk = (xx * xx + yy * yy) <= radius * radius
            local[disk] = 1.0

            # FIX (do not revert): try up to 20 positions; only accept one
            # that keeps this bead at least (2*radius + 2) pixels away from
            # every already-placed bead centre. This prevents beads from
            # touching or overlapping, which made them look like cross-shaped
            # blobs in training samples.
            placed = False
            for _attempt in range(20):
                top = np.random.randint(
                    box_top,
                    max(box_top + 1, box_bottom - diameter + 1),
                )
                left = np.random.randint(
                    box_left,
                    max(box_left + 1, box_right - diameter + 1),
                )
                cy, cx = top + radius, left + radius
                min_sep = 2 * radius + 6  # generous gap so beads never look like merged blobs
                if all(
                    ((cy - pc[0]) ** 2 + (cx - pc[1]) ** 2) >= (min_sep + pc[2]) ** 2
                    for pc in placed_centres
                ):
                    placed_centres.append((cy, cx, radius))
                    paste_local(local, top, left)
                    placed = True
                    break

            if not placed:
                # No valid non-overlapping position found after 20 attempts;
                # skip this bead rather than place a touching one.
                pass

    # ------------------------------------------------------------------
    # Cross: single thin cross, one-pixel thickness
    # ------------------------------------------------------------------
    elif style == "cross":
        local = np.zeros((11, 11), dtype=np.float32)

        # One-pixel-thick cross with randomized arm extent.
        half_arm = np.random.randint(3, 6)
        center = local.shape[0] // 2

        local[
            center - half_arm:center + half_arm + 1,
            center,
        ] = 1.0
        local[
            center,
            center - half_arm:center + half_arm + 1,
        ] = 1.0

        # Trim to the actual active region so placement is not constrained by
        # unused zero padding.
        rows, cols = np.where(local > 0)
        local = local[
            rows.min():rows.max() + 1,
            cols.min():cols.max() + 1,
        ]

        top = np.random.randint(
            box_top,
            max(box_top + 1, box_bottom - local.shape[0] + 1),
        )
        left = np.random.randint(
            box_left,
            max(box_left + 1, box_right - local.shape[1] + 1),
        )
        paste_local(local, top, left)

    # ------------------------------------------------------------------
    # Grating: thin bars, fully contained inside the common safe box
    # ------------------------------------------------------------------
    elif style == "grating":
        orientation = np.random.choice(["vertical", "horizontal"])
        num_slits = np.random.randint(3, 5)
        spacing = np.random.randint(2, 4)
        thick = 1

        # Keep the pattern comfortably inside the 20x20 box.
        pattern_size = min(
            box_size,
            max(10, num_slits * spacing + 5),
        )
        local = np.zeros((pattern_size, pattern_size), dtype=np.float32)

        start = np.random.randint(
            2,
            max(3, pattern_size - num_slits * spacing - 1),
        )

        for i in range(num_slits):
            pos = start + i * spacing
            if pos < pattern_size - 1:
                if orientation == "vertical":
                    local[3:pattern_size - 3, pos:pos + thick] = 1.0
                else:
                    local[pos:pos + thick, 3:pattern_size - 3] = 1.0

        rows, cols = np.where(local > 0)
        local = local[
            rows.min():rows.max() + 1,
            cols.min():cols.max() + 1,
        ]

        top = np.random.randint(
            box_top,
            max(box_top + 1, box_bottom - local.shape[0] + 1),
        )
        left = np.random.randint(
            box_left,
            max(box_left + 1, box_right - local.shape[1] + 1),
        )
        paste_local(local, top, left)

    # ------------------------------------------------------------------
    # Letters: retained reduced-size bitmap approach, now safely bounded
    # ------------------------------------------------------------------
    elif style == "letter":
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        letter_type = np.random.choice(list(alphabet))

        font_bitmaps = {
            "A": ["01110", "10001", "11111", "10001", "10001"],
            "B": ["11110", "10001", "11110", "10001", "11110"],
            "C": ["01111", "10000", "10000", "10000", "01111"],
            "D": ["11110", "10001", "10001", "10001", "11110"],
            "E": ["11111", "10000", "11110", "10000", "11111"],
            "F": ["11111", "10000", "11110", "10000", "10000"],
            "G": ["01111", "10000", "10111", "10001", "01111"],
            "H": ["10001", "10001", "11111", "10001", "10001"],
            "I": ["01110", "00100", "00100", "00100", "01110"],
            "J": ["00111", "00010", "00010", "10010", "01100"],
            "K": ["10001", "10010", "11100", "10010", "10001"],
            "L": ["10000", "10000", "10000", "10000", "11111"],
            "M": ["10001", "11011", "10101", "10001", "10001"],
            "N": ["10001", "11001", "10101", "10011", "10001"],
            "O": ["01110", "10001", "10001", "10001", "01110"],
            "P": ["11110", "10001", "11110", "10000", "10000"],
            "Q": ["01110", "10001", "10001", "10011", "01111"],
            "R": ["11110", "10001", "11110", "10010", "10001"],
            "S": ["01111", "10000", "01110", "00001", "11110"],
            "T": ["11111", "00100", "00100", "00100", "00100"],
            "U": ["10001", "10001", "10001", "10001", "01110"],
            "V": ["10001", "10001", "10001", "01010", "00100"],
            "W": ["10001", "10001", "10101", "11011", "10001"],
            "X": ["10001", "01010", "00100", "01010", "10001"],
            "Y": ["10001", "01010", "00100", "00100", "00100"],
            "Z": ["11111", "00010", "00100", "01000", "11111"],
            "0": ["01110", "10011", "10101", "11001", "01110"],
            "1": ["00100", "01100", "00100", "00100", "01110"],
            "2": ["01110", "10001", "00110", "01000", "11111"],
            "3": ["11110", "00001", "00110", "00001", "11110"],
            "4": ["00110", "01010", "10010", "11111", "00010"],
            "5": ["11111", "10000", "11110", "00001", "11110"],
            "6": ["01110", "10000", "11110", "10001", "01110"],
            "7": ["11111", "00001", "00010", "00100", "00100"],
            "8": ["01110", "10001", "01110", "10001", "01110"],
            "9": ["01110", "10001", "01111", "00001", "01110"],
        }

        bitmap = font_bitmaps[letter_type]

        # Reduced-scale letter: 5x5 cells mapped to ~7-10 pixels.
        letter_h = np.random.randint(7, 11)
        letter_w = np.random.randint(7, 11)
        local = np.zeros((letter_h, letter_w), dtype=np.float32)

        row_edges = np.linspace(0, letter_h, 6).astype(int)
        col_edges = np.linspace(0, letter_w, 6).astype(int)

        for r_idx in range(5):
            for c_idx in range(5):
                if bitmap[r_idx][c_idx] == "1":
                    local[
                        row_edges[r_idx]:row_edges[r_idx + 1],
                        col_edges[c_idx]:col_edges[c_idx + 1],
                    ] = 1.0

        top = np.random.randint(
            box_top,
            max(box_top + 1, box_bottom - local.shape[0] + 1),
        )
        left = np.random.randint(
            box_left,
            max(box_left + 1, box_right - local.shape[1] + 1),
        )
        paste_local(local, top, left)

    # ------------------------------------------------------------------
    # USAF: crop, reduce to the same approximate object scale, then place
    # inside the common safe bounding box.
    # ------------------------------------------------------------------
    elif style == "usaf" and usaf_matrix is not None:
        h, w = usaf_matrix.shape
        min_dim = min(h, w)

        # FIX (do not revert): only take small crops (10-30% of chart width)
        # so a handful of clearly-readable bars fill the box after resize,
        # rather than the entire chart compressed into 20px.
        min_crop = max(grid_size, int(0.05 * min_dim))
        max_crop = int(0.85 * min_dim)

        crop = None
        for _attempt in range(10):
            crop_size = np.random.randint(min_crop, max_crop + 1)
            top0 = np.random.randint(0, h - crop_size + 1)
            left0 = np.random.randint(0, w - crop_size + 1)

            candidate = usaf_matrix[
                top0:top0 + crop_size,
                left0:left0 + crop_size,
            ]

            if (
                candidate.max() > candidate.min()
                and candidate.std() > 0.05 * usaf_matrix.std()
            ):
                crop = candidate
                break

        if crop is None:
            crop = candidate

        usaf_size = np.random.randint(
            max(12, box_size - 4),
            box_size + 1,
        )

        zoom_factors = (
            usaf_size / crop.shape[0],
            usaf_size / crop.shape[1],
        )

        # Nearest-neighbor keeps the resolution-target bars crisp and avoids
        # introducing interpolated gray levels.
        local = zoom(crop, zoom_factors, order=0).astype(np.float32)

        # Safety clamp in case rounding creates a one-pixel overshoot.
        local = local[:box_size, :box_size]

        top = np.random.randint(
            box_top,
            max(box_top + 1, box_bottom - local.shape[0] + 1),
        )
        left = np.random.randint(
            box_left,
            max(box_left + 1, box_right - local.shape[1] + 1),
        )
        paste_local(local, top, left)

    # Normalize each target to unit peak intensity.
    peak = np.max(obj)
    if peak > 0:
        obj /= peak

    # Final hard guarantee: zero everything outside the bounding box.
    # This protects against accidental boundary writes from future object
    # modifications.
    outside = np.ones_like(obj, dtype=bool)
    outside[box_top:box_bottom, box_left:box_right] = False
    obj[outside] = 0.0

    return obj, style

# ==========================================
# BATCHED DATASET PRODUCTION ENGINE
# ==========================================
def execute_dataset_production(root_dir, dataset_name, total_samples, mode):
    dataset_root = os.path.join(root_dir, dataset_name)
    data_dir = os.path.join(dataset_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    usaf_path = "/home/awais/Desktop/Maaz/Maaz Data/Publication work/USAF.mat"
    mat = loadmat(usaf_path)
    data_keys = [k for k in mat.keys() if not k.startswith("__")]
    usaf_matrix = np.abs(np.squeeze(mat[data_keys[0]])).astype(np.float32)
    if usaf_matrix.ndim == 3:
        usaf_matrix = usaf_matrix[:, :, 0]
    print(f"--> Loaded USAF.mat  shape={usaf_matrix.shape}")

    engine = z.ZernikeGenerator()
    hdf5_file_path = os.path.join(data_dir, "reflection_matrix_dataset.h5")
    
    # FIX (do not revert): this used to be THE fixed mode list (always n=0..9,
    # ~55 modes, identical every sample), which is why aberrations looked
    # "stale" -- every sample was a different weighted combination of the same
    # fixed shape vocabulary. `full_mode_catalogue` is now just the maximum
    # available catalogue; the actual per-sample active subset (and thus the
    # effective max order) is drawn randomly for every sample further below,
    # mirroring the base paper's "max Zernike order randomly selected between
    # 10 and 60" recipe.
    full_mode_catalogue = []
    for n in range(10):
        for m in range(-n, n + 1):
            if (n - m) % 2 == 0:
                full_mode_catalogue.append((n, m))
    max_available_modes = len(full_mode_catalogue)
    sample_logs_list = []
    
    with h5py.File(hdf5_file_path, 'w') as h5f:
        h5f.create_dataset('R_matrices', shape=(total_samples, 2, NUM_ELEMENTS, NUM_ELEMENTS), dtype=np.float32, chunks=(1, 2, NUM_ELEMENTS, NUM_ELEMENTS), compression="gzip", compression_opts=4)
        h5f.create_dataset('phi_i_maps', shape=(total_samples, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        h5f.create_dataset('phi_o_maps', shape=(total_samples, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        h5f.create_dataset('target_objects', shape=(total_samples, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        h5f.create_dataset('object_styles', shape=(total_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
        h5f.create_dataset('intensity_ratios', shape=(total_samples,), dtype=np.float32)
        h5f.create_dataset('aberration_multipliers', shape=(total_samples,), dtype=np.float32)
        h5f.create_dataset('rms_scaling_factors', shape=(total_samples,), dtype=np.float32)  # <-- Added RMS Scaling Factors Dataset
        h5f.create_dataset('difficulty_tags', shape=(total_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
        h5f.create_dataset('split_types', shape=(total_samples,), dtype=h5py.string_dtype(encoding='utf-8'))

        print(f"\n--> Instantiating '{dataset_name}' with {total_samples} samples [Mode: {mode}]...")
        start_time = time.time()
        
        for idx in range(total_samples):
            intensity_ratio = 0.0
            aberration_multiplier = 2.0
            
            if mode == "A":
                if idx < 10000: split_type = "train"
                elif idx < 12500: split_type = "val"
                else: split_type = "test"
                mode_name = "output_only_aberration"
            elif mode == "B":
                if idx < 1500:
                    intensity_ratio = 0.0; aberration_multiplier = 1.5; split_type = "train"; mode_name = "no noise"
                elif idx < 3000:
                    intensity_ratio = np.random.uniform(1.0, 4.0); aberration_multiplier = np.random.uniform(2.0, 3.0); split_type = "train"; mode_name = "noise_low"
                elif idx < 5500:
                    intensity_ratio = np.random.uniform(4.0, 8.0); aberration_multiplier = np.random.uniform(2.0, 3.0); split_type = "train"; mode_name = "noise_med"
                elif idx < 9000:
                    intensity_ratio = np.random.uniform(8.0, 16.0); aberration_multiplier = np.random.uniform(2.0, 3.0); split_type = "train"; mode_name = "noise_high"
                elif idx < 9500:
                    intensity_ratio = 0.0; aberration_multiplier = 1.5; split_type = "val"; mode_name = "val_baseline"
                elif idx < 10000:
                    intensity_ratio = np.random.uniform(1.0, 4.0); aberration_multiplier = np.random.uniform(2.0, 3.0); split_type = "val"; mode_name = "val_noise_low"
                elif idx < 10500:
                    intensity_ratio = np.random.uniform(4.0, 8.0); aberration_multiplier = np.random.uniform(2.0, 3.0); split_type = "val"; mode_name = "val_noise_med"
                elif idx < 11000:
                    intensity_ratio = np.random.uniform(8.0, 16.0); aberration_multiplier = np.random.uniform(2.0, 3.0); split_type = "val"; mode_name = "val_noise_high"
                elif idx < 12000:
                    intensity_ratio = np.random.uniform(0.1, 0.4); aberration_multiplier = np.random.uniform(2.0, 3.0); split_type = "test"; mode_name = "test_standard"
                elif idx < 13000:
                    intensity_ratio = np.random.uniform(0.2, 1.0); aberration_multiplier = 5.0; split_type = "test"; mode_name = "test_high_aberration_low_noise"
                elif idx < 14000:
                    intensity_ratio = np.random.uniform(6.0,16.0); aberration_multiplier = 0.5; split_type = "test"; mode_name = "test_high_noise_low_aberration"
                else:
                    intensity_ratio = np.random.choice([1.0, 2.0, 3.0, 8.0, 10.0, 12.0, 14.0, 16.0]); aberration_multiplier = 2.0; split_type = "test"; mode_name = "test_intensity_ratio"

            elif mode == "C":
                # Aberration curriculum -- direct mirror of mode "B"'s structure,
                # but with aberration_multiplier as the ramped axis instead of
                # intensity_ratio.
                #
                # FIX (do not revert): train/val stages and test_standard now
                # draw a randomized BASELINE intensity_ratio in [0, 4) per
                # sample -- not curriculum-ramped, just present -- so the model
                # gets at least some exposure to realistic sensor noise
                # throughout the aberration curriculum instead of learning
                # against a totally sterile, noise-free assumption.
                # test_extreme_aberration and test_aberration_sweep are
                # deliberately left at exactly 0.0 to preserve a clean,
                # uncontaminated read on aberration difficulty alone --
                # test_aberration_with_noise already covers the deliberate
                # combined-stress case separately.
                #
                # Note the training range here (up to 8.0) deliberately exceeds
                # mode B's curriculum max of 3.0 -- mode B only ever saw
                # aberration_multiplier=5.0 in one TEST tag, never during
                # actual training.
                if idx < 1500:
                    aberration_multiplier = np.random.uniform(0.5, 1.5); intensity_ratio = np.random.uniform(0.0, 4.0); split_type = "train"; mode_name = "aberration_low"
                elif idx < 3000:
                    aberration_multiplier = np.random.uniform(1.5, 3.0); intensity_ratio = np.random.uniform(0.0, 4.0); split_type = "train"; mode_name = "aberration_med"
                elif idx < 5500:
                    aberration_multiplier = np.random.uniform(3.0, 5.0); intensity_ratio = np.random.uniform(0.0, 4.0); split_type = "train"; mode_name = "aberration_high"
                elif idx < 9000:
                    aberration_multiplier = np.random.uniform(5.0, 8.0); intensity_ratio = np.random.uniform(0.0, 4.0); split_type = "train"; mode_name = "aberration_extreme"
                elif idx < 9500:
                    aberration_multiplier = np.random.uniform(0.5, 1.5); intensity_ratio = np.random.uniform(0.0, 4.0); split_type = "val"; mode_name = "val_aberration_low"
                elif idx < 10000:
                    aberration_multiplier = np.random.uniform(1.5, 3.0); intensity_ratio = np.random.uniform(0.0, 4.0); split_type = "val"; mode_name = "val_aberration_med"
                elif idx < 10500:
                    aberration_multiplier = np.random.uniform(3.0, 5.0); intensity_ratio = np.random.uniform(0.0, 4.0); split_type = "val"; mode_name = "val_aberration_high"
                elif idx < 11000:
                    aberration_multiplier = np.random.uniform(5.0, 8.0); intensity_ratio = np.random.uniform(0.0, 4.0); split_type = "val"; mode_name = "val_aberration_extreme"
                elif idx < 12000:
                    aberration_multiplier = np.random.uniform(2.0, 3.0); intensity_ratio = np.random.uniform(0.0, 4.0); split_type = "test"; mode_name = "test_standard"
                elif idx < 13000:
                    # Genuine extrapolation check: beyond the training curriculum's
                    # own max (8.0), never seen during training at all. Kept
                    # noise-free on purpose -- see note above.
                    aberration_multiplier = np.random.uniform(8.0, 12.0); intensity_ratio = 0.0; split_type = "test"; mode_name = "test_extreme_aberration"
                elif idx < 14000:
                    # Combined-difficulty stress test -- real deployment has both
                    # strong aberration AND noise at once, never tested together
                    # elsewhere in this mode.
                    aberration_multiplier = np.random.uniform(3.0, 6.0); intensity_ratio = np.random.uniform(1.0, 4.0); split_type = "test"; mode_name = "test_aberration_with_noise"
                else:
                    # Direct analog of mode B's test_intensity_ratio sweep, for
                    # the aberration axis instead of noise -- gives the same kind
                    # of "performance vs. difficulty" headline plot.
                    aberration_multiplier = np.random.choice([1.0, 2.0, 3.0, 5.0, 7.0, 10.0]); intensity_ratio = 0.0; split_type = "test"; mode_name = "test_aberration_sweep"

            mock_object, target_style = generate_diverse_target(GRID_SIZE, usaf_matrix)
            # FIX (do not revert): pass the RAW SPATIAL-DOMAIN object directly.
            # construct_scattering_matrix now does the zero-padding and the
            # (raw, unshifted) fft2 internally -- do NOT pre-FFT mock_object
            # here, or it gets double-transformed and the scattering matrix
            # is completely invalid (produces pure-noise R matrices with no
            # object structure).
            O_matrix = engine.construct_scattering_matrix(mock_object).astype(np.complex64)

            # FIX (do not revert): draw a random number of ACTIVE modes per sample
            # (effective max Zernike order) instead of always using the full fixed
            # catalogue -- this is what gives samples genuinely different amounts of
            # high-spatial-frequency structure, not just different overall RMS
            # strength. Also randomize the damping decay rate per sample so the
            # balance between low- and high-order content varies too, instead of a
            # fixed exponential taper applied identically every time. Output and
            # input aberrations get independently randomized draws, matching how
            # their coefficients are already drawn independently.
            num_active_o = np.random.randint(10, max_available_modes + 1)
            modes_o = full_mode_catalogue[:num_active_o]
            decay_rate_o = np.random.uniform(0.5, 3.0)
            damping_o = np.exp(-np.linspace(0, decay_rate_o, num_active_o))
            coeffs_o = np.random.normal(0, 0.5, num_active_o) * damping_o * aberration_multiplier
            phi_o = engine.generate_phase_from_coefficients(coeffs_o, modes_o)
            
            if mode == "A":
                phi_i = np.zeros((GRID_SIZE, GRID_SIZE))  
            else:
                num_active_i = np.random.randint(10, max_available_modes + 1)
                modes_i = full_mode_catalogue[:num_active_i]
                decay_rate_i = np.random.uniform(0.5, 3.0)
                damping_i = np.exp(-np.linspace(0, decay_rate_i, num_active_i))
                coeffs_i = np.random.normal(0, 0.5, num_active_i) * damping_i * aberration_multiplier
                phi_i = engine.generate_phase_from_coefficients(coeffs_i, modes_i)
            
            P_i_diag = np.exp(1j * phi_i.flatten()).astype(np.complex64)
            P_o_diag = np.exp(1j * phi_o.flatten()).astype(np.complex64)
            R_clean = (P_o_diag[:, None] * O_matrix * P_i_diag[None, :]).astype(np.complex64)
            
            # --- Intensity-Based Noise Calculation ---
            if intensity_ratio > 0.0:
                P_s = np.mean(np.abs(R_clean)**2)
                sigma = np.sqrt((intensity_ratio * P_s) / 2)
                noise = (np.random.normal(0, sigma, R_clean.shape) + 1j * np.random.normal(0, sigma, R_clean.shape)).astype(np.complex64)
                R_noisy = R_clean + noise
            else:
                R_noisy = R_clean
            
            # --- Calculate Sample-Wise RMS Normalization Factor ---
            rms_val = float(np.sqrt(np.mean(np.abs(R_noisy) ** 2)))
            
            h5f['R_matrices'][idx, 0] = np.real(R_noisy)
            h5f['R_matrices'][idx, 1] = np.imag(R_noisy)
            h5f['phi_i_maps'][idx] = phi_i
            h5f['phi_o_maps'][idx] = phi_o
            h5f['target_objects'][idx] = mock_object
            h5f['object_styles'][idx] = target_style
            h5f['intensity_ratios'][idx] = intensity_ratio
            h5f['aberration_multipliers'][idx] = aberration_multiplier
            h5f['rms_scaling_factors'][idx] = rms_val  # <-- Saved to HDF5
            h5f['difficulty_tags'][idx] = mode_name
            h5f['split_types'][idx] = split_type
            
            sample_logs_list.append({
                "sample_id": idx, "split_type": split_type, "difficulty_mode": mode_name, "structural_style": target_style,
                "intensity_ratio": float(intensity_ratio), "rms_scaling_factor": rms_val,
                "input_phase_rms": float(np.sqrt(np.mean(phi_i**2))), "output_phase_rms": float(np.sqrt(np.mean(phi_o**2))),
                "num_active_modes_o": int(num_active_o), "decay_rate_o": float(decay_rate_o),
                "num_active_modes_i": (int(num_active_i) if mode != "A" else 0),
                "decay_rate_i": (float(decay_rate_i) if mode != "A" else 0.0)
            })

            current_count = idx + 1
            if current_count % 500 == 0 or current_count == total_samples:
                elapsed_time = time.time() - start_time
                avg_speed = elapsed_time / current_count
                rem_min, rem_sec = divmod(int(avg_speed * (total_samples - current_count)), 60)
                elap_min, elap_sec = divmod(int(elapsed_time), 60)
                print(f"   >> [{dataset_name.upper()}] Generated {current_count}/{total_samples} samples ({(current_count/total_samples)*100:.1f}%) | Elapsed: {elap_min}m {elap_sec}s | Est. Remaining: {rem_min}m {rem_sec}s")

    with open(os.path.join(dataset_root, f"{dataset_name}_log.json"), "w") as jf:
        # FIX: 'total_zernike_modes_applied' used to reference a single fixed
        # num_modes value that no longer exists -- the active mode count is now
        # randomized per sample (see sample_logs_list entries for the per-sample
        # values). Report the size of the full catalogue modes are drawn from instead.
        json.dump({"dataset_name": dataset_name, "mode_type": mode, "grid_size": f"{GRID_SIZE} x {GRID_SIZE}", "total_samples": total_samples, "max_available_zernike_modes": max_available_modes, "data_file_path": "data/reflection_matrix_dataset.h5", "samples": sample_logs_list}, jf, indent=2)
    print(f"--> Dataset '{dataset_name}' compiled successfully.")

# ==========================================
# RUN ENGINE ENTRY GATE
# ==========================================
if __name__ == "__main__":

    BASE_PATH = "/home/awais/Desktop/Maaz/Maaz Data"
    #execute_dataset_production(root_dir=BASE_PATH, dataset_name="dataset_a", total_samples=15000, mode="A")
    execute_dataset_production(root_dir=BASE_PATH, dataset_name="dataset_b", total_samples=15000, mode="B")
   # execute_dataset_production(root_dir=BASE_PATH, dataset_name="dataset_c", total_samples=15000, mode="C")