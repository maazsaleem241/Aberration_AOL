import os
import json
import h5py
import numpy as np
import time  
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.io import loadmat
from scipy.ndimage import zoom
from typing import List, Tuple
import zernike as z

GRID_SIZE = 40          
NUM_ELEMENTS = GRID_SIZE * GRID_SIZE  

# ==========================================
# DYNAMIC TARGET GENERATOR (OBJECT DIVERSITY)
# ==========================================
def generate_diverse_target(grid_size: int = GRID_SIZE, usaf_matrix: np.ndarray = None) -> Tuple[np.ndarray, str]:
    """
    Generates a unique, randomized structural object to prevent neural network 
    memorization and simulate realistic computational imaging benchmarks.
    """
    obj = np.zeros((grid_size, grid_size), dtype=np.float32)
    style_options = ["beads", "cross", "grating", "letter"]
    if usaf_matrix is not None:
        style_options.append("usaf")
        
    style = np.random.choice(style_options)
    
    if style == "beads":
        num_beads = np.random.randint(3, 7)
        for _ in range(num_beads):
            r, c = np.random.randint(5, grid_size-5, size=2)
            obj[r, c] = 1.0
            obj[r-1:r+2, c] = np.maximum(obj[r-1:r+2, c], 0.6)
            obj[r, c-1:c+2] = np.maximum(obj[r, c-1:c+2], 0.6)
            
    elif style == "cross":
        num_crosses = np.random.randint(1, 3)
        for _ in range(num_crosses):
            cr, cc = np.random.randint(10, grid_size-10, size=2)
            length = np.random.randint(6, 12)
            thick = np.random.randint(1, 3)
            obj[cr - length:cr + length, cc - thick:cc + thick] = 1.0
            obj[cr - thick:cr + thick, cc - length:cc + length] = 1.0
            
    elif style == "grating":
        orientation = np.random.choice(["vertical", "horizontal"])
        num_slits = np.random.randint(3, 5)
        spacing = np.random.randint(4, 7)
        thick = np.random.randint(1, 3)
        start = np.random.randint(4, 10)
        
        for i in range(num_slits):
            pos = start + i * spacing
            if pos < grid_size - 2:
                if orientation == "vertical":
                    obj[8:32, pos:pos+thick] = 1.0
                else:
                    obj[pos:pos+thick, 8:32] = 1.0
                    
    elif style == "letter":
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letter_type = np.random.choice(list(alphabet))
        
        h_start, h_end = np.random.randint(8, 14), np.random.randint(26, 32)
        v_start, v_end = np.random.randint(8, 14), np.random.randint(26, 32)
        
        font_bitmaps = {
            "A": ["01110", "10001", "11111", "10001", "10001"], "B": ["11110", "10001", "11110", "10001", "11110"],
            "C": ["01111", "10000", "10000", "10000", "01111"], "D": ["11110", "10001", "10001", "10001", "11110"],
            "E": ["11111", "10000", "11110", "10000", "11111"], "F": ["11111", "10000", "11110", "10000", "10000"],
            "G": ["01111", "10000", "10111", "10001", "01111"], "H": ["10001", "10001", "11111", "10001", "10001"],
            "I": ["01110", "00100", "00100", "00100", "01110"], "J": ["00111", "00010", "00010", "10010", "01100"],
            "K": ["10001", "10010", "11100", "10010", "10001"], "L": ["10000", "10000", "10000", "10000", "11111"],
            "M": ["10001", "11011", "10101", "10001", "10001"], "N": ["10001", "11001", "10101", "10011", "10001"],
            "O": ["01110", "10001", "10001", "10001", "01110"], "P": ["11110", "10001", "11110", "10000", "10000"],
            "Q": ["01110", "10001", "10001", "10011", "01111"], "R": ["11110", "10001", "11110", "10010", "10001"],
            "S": ["01111", "10000", "01110", "00001", "11110"], "T": ["11111", "00100", "00100", "00100", "00100"],
            "U": ["10001", "10001", "10001", "10001", "01110"], "V": ["10001", "10001", "10001", "01010", "00100"],
            "W": ["10001", "10001", "10101", "11011", "10001"], "X": ["10001", "01010", "00100", "01010", "10001"],
            "Y": ["10001", "01010", "00100", "00100", "00100"], "Z": ["11111", "00010", "00100", "01000", "11111"]
        }
        bitmap = font_bitmaps[letter_type]
        row_edges = np.linspace(v_start, v_end, 6).astype(int)
        col_edges = np.linspace(h_start, h_end, 6).astype(int)
        
        for r_idx in range(5):
            for c_idx in range(5):
                if bitmap[r_idx][c_idx] == "1":
                    obj[row_edges[r_idx]:row_edges[r_idx + 1], col_edges[c_idx]:col_edges[c_idx + 1]] = 1.0
            
    elif style == "usaf" and usaf_matrix is not None:
        h, w = usaf_matrix.shape
        zoom_factors = (grid_size / h, grid_size / w)
        obj = zoom(usaf_matrix, zoom_factors, order=3).astype(np.float32)

    if np.max(obj) > 0:
        obj /= np.max(obj)
    return obj, style

# ==========================================
# BATCHED DATASET PRODUCTION ENGINE
# ==========================================
def execute_dataset_production(root_dir, dataset_name, total_samples, mode):
    dataset_root = os.path.join(root_dir, dataset_name)
    data_dir = os.path.join(dataset_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    usaf_matrix = None
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    usaf_path = os.path.join(current_script_dir, "USAF.mat")

    if os.path.exists(usaf_path):
        try:
            mat = loadmat(usaf_path)
            data_keys = [k for k in mat.keys() if not k.startswith('__')]
            if data_keys:
                usaf_matrix = np.abs(np.squeeze(mat[data_keys[0]]))
                if usaf_matrix.ndim == 3:
                    usaf_matrix = usaf_matrix[:, :, 0]
        except Exception as e:
            print(f"Warning: Could not load USAF.mat: {e}")
    
    engine = z.ZernikeGenerator()
    hdf5_file_path = os.path.join(data_dir, "reflection_matrix_dataset.h5")
    
    modes_list = []
    for n in range(10):
        for m in range(-n, n + 1):
            if (n - m) % 2 == 0:
                modes_list.append((n, m))
    num_modes = len(modes_list) 
    sample_logs_list = []
    
    viz_triggers = {}
    if mode == "A":
        viz_triggers[99] = list(range(0, 100))
        viz_triggers[total_samples - 1] = list(range(total_samples - 100, total_samples))
    elif mode == "B":
        sub_mode_intervals = [
            (0, 1500), (1500, 3000), (3000, 5500), (5500, 9000),
            (9000, 9500), (9500, 10000), (10000, 10500), (10500, 11000),
            (11000, 12000), (12000, 13000), (13000, 14000), (14000, 15000)
        ]
        for start, end in sub_mode_intervals:
            if end <= total_samples:
                viz_triggers[start + 99] = list(range(start, start + 100))
                viz_triggers[end - 1] = list(range(end - 100, end))
    
    with h5py.File(hdf5_file_path, 'w') as h5f:
        h5f.create_dataset('R_matrices', shape=(total_samples, 2, NUM_ELEMENTS, NUM_ELEMENTS), dtype=np.float32, chunks=(1, 2, NUM_ELEMENTS, NUM_ELEMENTS), compression="gzip", compression_opts=4)
        # <-- Added O_matrices Dataset: persists the scattering matrix O per sample
        # (real/imag, same layout/compression as R_matrices) so the physics-informed
        # reconstruction loss (R_hat = P_o . O . P_i) can be computed at train time
        # without needing to recompute O from target_objects on the fly.
        h5f.create_dataset('O_matrices', shape=(total_samples, 2, NUM_ELEMENTS, NUM_ELEMENTS), dtype=np.float32, chunks=(1, 2, NUM_ELEMENTS, NUM_ELEMENTS), compression="gzip", compression_opts=4)
        h5f.create_dataset('phi_i_maps', shape=(total_samples, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        h5f.create_dataset('phi_o_maps', shape=(total_samples, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        h5f.create_dataset('target_objects', shape=(total_samples, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        h5f.create_dataset('object_styles', shape=(total_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
        h5f.create_dataset('intensity_ratios', shape=(total_samples,), dtype=np.float32)
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
                    intensity_ratio = np.random.choice([1.0, 2.0, 4.0, 8.0, 16.0]); aberration_multiplier = 2.0; split_type = "test"; mode_name = "test_intensity_ratio"

            mock_object, target_style = generate_diverse_target(GRID_SIZE, usaf_matrix)
            object_fft = fftshift(fft2(mock_object))
            O_matrix = engine.construct_scattering_matrix(object_fft).astype(np.complex64)
            
            frequency_damping = np.exp(-np.linspace(0, 1.5, num_modes))
            coeffs_o = np.random.normal(0, 0.5, num_modes) * frequency_damping * aberration_multiplier
            phi_o = engine.generate_phase_from_coefficients(coeffs_o, modes_list)
            
            if mode == "A":
                phi_i = np.zeros((GRID_SIZE, GRID_SIZE))  
            else:
                coeffs_i = np.random.normal(0, 0.5, num_modes) * frequency_damping * aberration_multiplier
                phi_i = engine.generate_phase_from_coefficients(coeffs_i, modes_list)
            
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
            h5f['O_matrices'][idx, 0] = np.real(O_matrix)  # <-- Saved to HDF5
            h5f['O_matrices'][idx, 1] = np.imag(O_matrix)  # <-- Saved to HDF5
            h5f['phi_i_maps'][idx] = phi_i
            h5f['phi_o_maps'][idx] = phi_o
            h5f['target_objects'][idx] = mock_object
            h5f['object_styles'][idx] = target_style
            h5f['intensity_ratios'][idx] = intensity_ratio
            h5f['rms_scaling_factors'][idx] = rms_val  # <-- Saved to HDF5
            h5f['difficulty_tags'][idx] = mode_name
            h5f['split_types'][idx] = split_type
            
            sample_logs_list.append({
                "sample_id": idx, "split_type": split_type, "difficulty_mode": mode_name, "structural_style": target_style,
                "intensity_ratio": float(intensity_ratio), "rms_scaling_factor": rms_val,
                "input_phase_rms": float(np.sqrt(np.mean(phi_i**2))), "output_phase_rms": float(np.sqrt(np.mean(phi_o**2)))
            })

            if idx in viz_triggers:
                h5f.flush()  
                print(f"\n   >> [INTERLEAVED AUDIT] Visualizing boundary slice for '{mode_name}'...")
                render_optimized_plots(root_dir, dataset_name, viz_triggers[idx], h5f_passed=h5f)

            current_count = idx + 1
            if current_count % 500 == 0 or current_count == total_samples:
                elapsed_time = time.time() - start_time
                avg_speed = elapsed_time / current_count
                rem_min, rem_sec = divmod(int(avg_speed * (total_samples - current_count)), 60)
                elap_min, elap_sec = divmod(int(elapsed_time), 60)
                print(f"   >> [{dataset_name.upper()}] Generated {current_count}/{total_samples} samples ({(current_count/total_samples)*100:.1f}%) | Elapsed: {elap_min}m {elap_sec}s | Est. Remaining: {rem_min}m {rem_sec}s")

    with open(os.path.join(dataset_root, f"{dataset_name}_log.json"), "w") as jf:
        json.dump({"dataset_name": dataset_name, "mode_type": mode, "grid_size": f"{GRID_SIZE} x {GRID_SIZE}", "total_samples": total_samples, "total_zernike_modes_applied": num_modes, "data_file_path": "data/reflection_matrix_dataset.h5", "samples": sample_logs_list}, jf, indent=2)
    print(f"--> Dataset '{dataset_name}' compiled successfully.")

# ==========================================
# OPTIMIZED VISUALIZATION ENGINE (3-TICK PHASE KEY)
# ==========================================
def apply_plot_styling(ax, im, data, style='general', vmin=0.0, vmax=1.0, pad=0.04):
    ax.set_aspect('equal')
    
    if style == 'phase':
        if np.allclose(vmin, 0) and np.allclose(vmax, 0):
            ticks = [0.0]
            tick_labels = ['0']
        else:
            low_bound = int(np.floor(vmin / np.pi))
            high_bound = int(np.ceil(vmax / np.pi))
            
            ticks = [low_bound * np.pi, 0.0, high_bound * np.pi]
            ticks = sorted(list(set(ticks)))  
            
            tick_labels = []
            for val in ticks:
                k = int(np.round(val / np.pi))
                if k == 0: 
                    tick_labels.append('0')
                elif k == 1: 
                    tick_labels.append(r'$\pi$')
                elif k == -1: 
                    tick_labels.append(r'$-\pi$')
                else: 
                    tick_labels.append(f'${k}\pi$')
                        
        cbar = plt.colorbar(im, ax=ax, ticks=ticks, pad=pad)
        cbar.ax.set_yticklabels(tick_labels)
    else:
        mid = (vmin + vmax) / 2.0
        ticks = [vmin, mid, vmax] if vmax > vmin else [vmin]
        cbar = plt.colorbar(im, ax=ax, ticks=ticks, pad=pad)
        cbar.ax.set_yticklabels([f"{v:.2f}" for v in ticks])

def render_optimized_plots(root_dir, dataset_name, indices_to_render, h5f_passed=None):
    dataset_root = os.path.join(root_dir, dataset_name)
    data_dir = os.path.join(dataset_root, "data")
    vis_dir = os.path.join(dataset_root, "visualization")
    hdf5_file_path = os.path.join(data_dir, "reflection_matrix_dataset.h5")
    
    def process_render_loop(h5f):
        for idx in indices_to_render:
            R = h5f['R_matrices'][idx, 0] + 1j * h5f['R_matrices'][idx, 1]
            phi_i = h5f['phi_i_maps'][idx]
            phi_o = h5f['phi_o_maps'][idx]
            target_obj = h5f['target_objects'][idx]
            target_style = h5f['object_styles'][idx].decode('utf-8') if isinstance(h5f['object_styles'][idx], bytes) else h5f['object_styles'][idx]
            tag = h5f['difficulty_tags'][idx].decode('utf-8') if isinstance(h5f['difficulty_tags'][idx], bytes) else h5f['difficulty_tags'][idx]
            split = h5f['split_types'][idx].decode('utf-8') if isinstance(h5f['split_types'][idx], bytes) else h5f['split_types'][idx]
            intensity_ratio = h5f['intensity_ratios'][idx]
            rms_val = h5f['rms_scaling_factors'][idx]
            
            master_dir = os.path.join(vis_dir, "master_matrix", tag)
            indiv_base_dir = os.path.join(vis_dir, "individual_panels", tag)
            sample_panel_dir = os.path.join(indiv_base_dir, f"sample_{idx:05d}")
            os.makedirs(master_dir, exist_ok=True)
            os.makedirs(sample_panel_dir, exist_ok=True)
            
            obj_fft = fftshift(fft2(target_obj))
            dc_idx = (GRID_SIZE // 2) * GRID_SIZE + (GRID_SIZE // 2)
            degraded_fourier = np.reshape(R[:, dc_idx], (GRID_SIZE, GRID_SIZE))
            degraded_spatial = np.fft.ifft2(np.fft.ifftshift(degraded_fourier))
            degraded_field = np.abs(degraded_spatial.T) ** 2
            max_deg = np.max(degraded_field)
            norm_degraded = degraded_field / max_deg if max_deg > 0 else np.zeros_like(degraded_field, dtype=np.float32)

            freq_extent = [-1, 1, -1, 1]
            matrix_extent = [0, NUM_ELEMENTS, 0, NUM_ELEMENTS]
            
            max_absolute_phase = max(np.max(np.abs(phi_i)), np.max(np.abs(phi_o)))
            max_pi_integer = max(1, int(np.ceil(max_absolute_phase / np.pi)))
            global_vmin_phase, global_vmax_phase = -max_pi_integer * np.pi, max_pi_integer * np.pi
            
            fig = plt.figure(figsize=(22, 15))
            gs = fig.add_gridspec(3, 4, wspace=0.35, hspace=0.35)
            fig.suptitle(f"Master Verification Matrix - {dataset_name.upper()} | #{idx:05d} | [{tag.upper()}]", fontsize=18, fontweight='bold')
            
            ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2]); ax4 = fig.add_subplot(gs[0, 3])
            ax5 = fig.add_subplot(gs[1, 0]); ax6 = fig.add_subplot(gs[1, 1]); ax7 = fig.add_subplot(gs[2, 0]); ax8 = fig.add_subplot(gs[2, 1])
            ax_meta = fig.add_subplot(gs[1:, 2:].subgridspec(3, 1, height_ratios=[0.18, 0.64, 0.18])[1])
            
            im1 = ax1.imshow(phi_i, cmap='jet', extent=freq_extent, interpolation='bicubic', vmin=global_vmin_phase, vmax=global_vmax_phase)
            ax1.set_title("Input Phase Aberration (phi_i)", fontsize=12)
            ax1.set_xlabel("Normalized Pupil X"); ax1.set_ylabel("Normalized Pupil Y")
            ax1.set_xticks([-1.0, 0.0, 1.0]); ax1.set_yticks([-1.0, 0.0, 1.0])
            apply_plot_styling(ax1, im1, phi_i, style='phase', vmin=global_vmin_phase, vmax=global_vmax_phase)

            im2 = ax2.imshow(phi_o, cmap='jet', extent=freq_extent, interpolation='bicubic', vmin=global_vmin_phase, vmax=global_vmax_phase)
            ax2.set_title("Output Phase Aberration (phi_o)", fontsize=12)
            ax2.set_xlabel("Normalized Pupil X"); ax2.set_ylabel("Normalized Pupil Y")
            ax2.set_xticks([-1.0, 0.0, 1.0]); ax2.set_yticks([-1.0, 0.0, 1.0])
            apply_plot_styling(ax2, im2, phi_o, style='phase', vmin=global_vmin_phase, vmax=global_vmax_phase)

            im3 = ax3.imshow(target_obj, cmap='gray', extent=freq_extent, interpolation='nearest')
            ax3.set_title("Original Target Object (Ground Truth)", fontsize=12)
            ax3.set_xlabel("Normalized Spatial X"); ax3.set_ylabel("Normalized Spatial Y")
            ax3.set_xticks([-1.0, 0.0, 1.0]); ax3.set_yticks([-1.0, 0.0, 1.0])
            apply_plot_styling(ax3, im3, target_obj, style='general', vmin=0.0, vmax=1.0)

            im4 = ax4.imshow(norm_degraded, cmap='hot', extent=freq_extent, interpolation='bicubic', vmin=0.0, vmax=1.0)
            ax4.set_title("Uncorrected Image Intensity", fontsize=12)
            ax4.set_xlabel("Normalized Spatial X"); ax4.set_ylabel("Normalized Spatial Y")
            ax4.set_xticks([-1.0, 0.0, 1.0]); ax4.set_yticks([-1.0, 0.0, 1.0])
            apply_plot_styling(ax4, im4, norm_degraded, style='general', vmin=0.0, vmax=1.0, pad=0.05)

            log_amp = np.log10(np.abs(R) + 1e-5)
            im5 = ax5.imshow(log_amp, cmap='magma', extent=matrix_extent, origin='lower')
            ax5.set_title("Reflection Matrix Amplitude |R|", fontsize=12)
            ax5.set_xlabel("Input Channel Element Index"); ax5.set_ylabel("Output Channel Element Index")
            ax5.set_xticks([0, NUM_ELEMENTS // 2, NUM_ELEMENTS]); ax5.set_yticks([0, NUM_ELEMENTS // 2, NUM_ELEMENTS])
            apply_plot_styling(ax5, im5, log_amp, style='general', vmin=np.min(log_amp), vmax=np.max(log_amp), pad=0.08)

            phase_ang = np.angle(R)
            im6 = ax6.imshow(phase_ang, cmap='twilight', extent=matrix_extent, origin='lower', vmin=-np.pi, vmax=np.pi)
            ax6.set_title("Reflection Matrix Phase angle R", fontsize=12)
            ax6.set_xlabel("Input Channel Element Index"); ax6.set_ylabel("Output Channel Element Index")
            ax6.set_xticks([0, NUM_ELEMENTS // 2, NUM_ELEMENTS]); ax6.set_yticks([0, NUM_ELEMENTS // 2, NUM_ELEMENTS])
            apply_plot_styling(ax6, im6, phase_ang, style='phase', vmin=-np.pi, vmax=np.pi, pad=0.08)

            ax7.plot(phi_i[GRID_SIZE//2, :], label='Input Aberration Center Cut', color='b', lw=2)
            ax7.plot(phi_o[GRID_SIZE//2, :], label='Output Aberration Center Cut', color='r', lw=2)
            ax7.set_title("Mid-Line Spatial Profiles", fontsize=12)
            ax7.set_xlabel("Pupil Grid Position Index"); ax7.set_ylabel("Phase Variation (rad)")
            ax7.legend(); ax7.grid(True)

            ax8.hist(phi_i.flatten(), bins=20, alpha=0.5, label='Input Aberration Spectrum', color='blue')
            ax8.hist(phi_o.flatten(), bins=20, alpha=0.5, label='Output Aberration Spectrum', color='red')
            ax8.set_title("Phase Spatial Histograms", fontsize=12)
            ax8.set_xlabel("Phase Value Amplitude (rad)"); ax8.set_ylabel("Total Normalized Counts")
            ax8.legend()

            ax_meta.axis('on')
            ax_meta.set_facecolor('#0d1117')
            ax_meta.get_xaxis().set_visible(False); ax_meta.get_yaxis().set_visible(False)
            for spine in ax_meta.spines.values(): spine.set_color('#30363d'); spine.set_linewidth(2)
            
            meta_text = (
                f"    Meta Data\n------------------------------------------------------------------\n\n"
                f"  • Target Folder Name     :  {dataset_name}\n"
                f"  • Selected Dataset Split :  {split.upper()}\n"
                f"  • Structural Object Type :  {target_style.upper()}\n"
                f"  • Dynamic Physics Tag    :  {tag.upper()}\n"
                f"  • Intensity Ratio        :  {intensity_ratio:.4f}\n"
                f"  • RMS Scaling Factor     :  {rms_val:.4e}\n"
                f"  • Total Array Boundaries :  {GRID_SIZE} x {GRID_SIZE} Pixels"
            )
            ax_meta.text(0.08, 0.5, meta_text, transform=ax_meta.transAxes, fontsize=14, color='#c9d1d9', family='monospace', ha='left', va='center', fontweight='bold')
            
            plt.savefig(os.path.join(master_dir, f"sample_{idx:05d}_visualized.png"), dpi=150, bbox_inches='tight')
            plt.close()

            plots = [
                ("fig_01_input_phase.png", phi_i, "jet", "spatial", "Input Phase Aberration (phi_i)", "phase", global_vmin_phase, global_vmax_phase, "bicubic"),
                ("fig_02_output_phase.png", phi_o, "jet", "spatial", "Output Phase Aberration (phi_o)", "phase", global_vmin_phase, global_vmax_phase, "bicubic"),
                ("fig_03a_original_object.png", target_obj, "gray", "spatial", "Original Target Object", "general", 0.0, 1.0, "nearest"),
                ("fig_04_uncorrected_intensity.png", norm_degraded, "hot", "spatial", "Uncorrected Image Plane Intensity", "general", 0.0, 1.0, "bicubic"),
                ("fig_05_matrix_amplitude.png", log_amp, "magma", "matrix", "Reflection Matrix Amplitude |R|", "general", np.min(log_amp), np.max(log_amp), None),
                ("fig_06_matrix_phase.png", phase_ang, "twilight", "matrix", "Reflection Matrix Phase angle R", "phase", -np.pi, np.pi, None)
            ]

            for fname, data, cmap, ptype, title, style, vmin, vmax, interp in plots:
                plt.figure(figsize=(7, 6))
                im = plt.imshow(data, cmap=cmap, extent=(freq_extent if ptype=="spatial" else matrix_extent), interpolation=interp, origin=('upper' if ptype=="spatial" else 'lower'), vmin=vmin, vmax=vmax)
                plt.title(title, fontsize=13, fontweight='bold')
                apply_plot_styling(plt.gca(), im, data, style=style, vmin=vmin, vmax=vmax)
                plt.tight_layout(); plt.savefig(os.path.join(sample_panel_dir, fname), dpi=200); plt.close()

    if h5f_passed is not None: process_render_loop(h5f_passed)
    else:
        with h5py.File(hdf5_file_path, 'r') as h5f: process_render_loop(h5f)

# ==========================================
# RUN ENGINE ENTRY GATE
# ==========================================
if __name__ == "__main__":

    BASE_PATH = "/home/awais/Desktop/Maaz/Maaz Data"
    execute_dataset_production(root_dir=BASE_PATH, dataset_name="dataset_a", total_samples=15000, mode="A")
    execute_dataset_production(root_dir=BASE_PATH, dataset_name="dataset_b", total_samples=15000, mode="B")