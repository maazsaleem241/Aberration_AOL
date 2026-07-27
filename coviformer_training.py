import os
import csv
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compute_ssim
from models.CoviFormer import CoviFormer

# =============================================================================
# 1. GLOBAL CONFIGURATION & HYPERPARAMETERS
# =============================================================================
H5_FILE = "/home/awais/Desktop/Maaz/Maaz Data/dataset_b/data/reflection_matrix_dataset.h5"
MAX_EPOCHS = 400
BATCH_SIZE = 16
INITIAL_LR = 1e-5
GRAD_CLIP = 1.5
CHECKPOINT_INTERVAL = 2
NUM_WORKERS = 4
MC_DROPOUT_PASSES = 5
USE_AMP = cuda_available = torch.cuda.is_available()
DEVICE = 'cuda' if cuda_available else 'cpu'

CURRICULUM_STAGES = ["no noise", "noise_low", "noise_med", "noise_high"]

# Dedicated root directory and path structures for CoviFormer outputs
BASE_OUTPUT_DIR = "coviformer_training"
CHECKPOINT_DIR = os.path.join(BASE_OUTPUT_DIR, "checkpoints")
VIS_DIR = os.path.join(BASE_OUTPUT_DIR, "visuals")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "coviformer_best.pth")
METRICS_CSV_PATH = os.path.join(CHECKPOINT_DIR, "training_metrics.csv")

# =============================================================================
# 2. HELPER: VISUALIZATION
# =============================================================================
def save_visualization(input_tensor, rrt_tensor, rtr_tensor, target_phi_o, target_phi_i, pred_phi_o, pred_phi_i, epoch, save_dir=VIS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    
    def apply_blur(tensor, kernel_size=7, sigma=1.5):
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        x = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device) - kernel_size // 2
        y = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device) - kernel_size // 2
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
        kernel = (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)
        padding = kernel_size // 2
        return F.conv2d(tensor, kernel, padding=padding).squeeze()

    inp_r = input_tensor[0, 0].cpu()
    rrt_sample = torch.log10(torch.abs(rrt_tensor[0, 0] + 1j * rrt_tensor[0, 1]) + 1e-5).cpu().numpy()
    rtr_sample = torch.log10(torch.abs(rtr_tensor[0, 0] + 1j * rtr_tensor[0, 1]) + 1e-5).cpu().numpy()
    
    tgt_o = target_phi_o[0].squeeze().cpu()
    tgt_i = target_phi_i[0].squeeze().cpu()
    pre_o = pred_phi_o[0].squeeze().cpu()
    pre_i = pred_phi_i[0].squeeze().cpu()

    # Scale phase values by 1/pi to represent them in terms of pi units
    tgt_o_smooth = apply_blur(tgt_o).numpy() / np.pi
    tgt_i_smooth = apply_blur(tgt_i).numpy() / np.pi
    pre_o_smooth = apply_blur(pre_o).numpy() / np.pi
    pre_i_smooth = apply_blur(pre_i).numpy() / np.pi

    # Residuals also scaled in terms of pi units
    res_o = np.abs(tgt_o_smooth - pre_o_smooth)
    res_i = np.abs(tgt_i_smooth - pre_i_smooth)

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    for row in axes:
        for ax in row:
            ax.set_xlabel("Pixels")
            ax.set_ylabel("Pixels")

    im0 = axes[0, 0].imshow(inp_r.numpy(), cmap='gray')
    axes[0, 0].set_title("Input R (Real Channel)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(rrt_sample, cmap='magma')
    axes[0, 1].set_title("Input Covariance R*R^t (Log-Amp)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(rtr_sample, cmap='magma')
    axes[0, 2].set_title("Input Covariance R^t*R (Log-Amp)")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im3 = axes[1, 0].imshow(tgt_i_smooth, cmap='jet')
    axes[1, 0].set_title("Target Input Phase (Smoothed)")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04).set_label(r"[$\pi$ rad]")

    im4 = axes[1, 1].imshow(pre_i_smooth, cmap='jet')
    axes[1, 1].set_title("Predicted Input Phase (Smoothed)")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04).set_label(r"[$\pi$ rad]")

    im5 = axes[1, 2].imshow(res_i, cmap='jet')
    axes[1, 2].set_title("Input Phase Residual")
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04).set_label(r"[$\pi$ rad]")

    im6 = axes[2, 0].imshow(tgt_o_smooth, cmap='jet')
    axes[2, 0].set_title("Target Output Phase (Smoothed)")
    fig.colorbar(im6, ax=axes[2, 0], fraction=0.046, pad=0.04).set_label(r"[$\pi$ rad]")

    im7 = axes[2, 1].imshow(pre_o_smooth, cmap='jet')
    axes[2, 1].set_title("Predicted Output Phase (Smoothed)")
    fig.colorbar(im7, ax=axes[2, 1], fraction=0.046, pad=0.04).set_label(r"[$\pi$ rad]")

    im8 = axes[2, 2].imshow(res_o, cmap='jet')
    axes[2, 2].set_title("Output Phase Residual")
    fig.colorbar(im8, ax=axes[2, 2], fraction=0.046, pad=0.04).set_label(r"[$\pi$ rad]")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"), dpi=150)
    plt.close()

# =============================================================================
# 3. COMPLETE COMPOSITE LOSS FUNCTION ($L_{total}$)
# =============================================================================
class CompleteCompositeLoss(nn.Module):
    def __init__(self, weights=[1.0, 0.5, 0.2, 0.2]):
        super().__init__()
        self.l1, self.l2, self.l3, self.l4 = weights

    @staticmethod
    def _phasor_deltas(pred_phasor: torch.Tensor, target_angle: torch.Tensor):
        # pred_phasor: (B, 2, H, W) -- channel 0 = cos(phi_pred), channel 1 = sin(phi_pred)
        # target_angle: (B, 1, H, W) angle map, as stored in the dataset.
        # cos(a-b) = cos(a)cos(b) + sin(a)sin(b) ; sin(a-b) = sin(a)cos(b) - cos(a)sin(b)
        # -- no atan2 anywhere in this loss, same rule the model itself follows.
        pred_cos = pred_phasor[:, 0:1, :, :]
        pred_sin = pred_phasor[:, 1:2, :, :]
        target_cos = torch.cos(target_angle)
        target_sin = torch.sin(target_angle)
        cos_delta = pred_cos * target_cos + pred_sin * target_sin
        sin_delta = pred_sin * target_cos - pred_cos * target_sin
        return cos_delta, sin_delta

    def forward(self, pred_phasor_o, target_phi_o, pred_phasor_i, target_phi_i, R_complex, rrt, rtr, smoothness_regularizer):
        # FIX: pred_phasor_i / target_phi_i were previously accepted as arguments but
        # never actually used -- the input-aberration branch had no ground-truth
        # supervision at all, only the unsupervised smoothness term below, which is
        # happy with a flat/collapsed prediction. Both branches now get identical
        # treatment.
        cos_delta_o, sin_delta_o = self._phasor_deltas(pred_phasor_o, target_phi_o)
        cos_delta_i, sin_delta_i = self._phasor_deltas(pred_phasor_i, target_phi_i)

        diff_o = torch.complex(cos_delta_o, sin_delta_o)
        diff_i = torch.complex(cos_delta_i, sin_delta_i)

        l_phasor_o = 1.0 - torch.abs(torch.mean(diff_o))
        l_phasor_i = 1.0 - torch.abs(torch.mean(diff_i))
        l_phasor = 0.5 * (l_phasor_o + l_phasor_i)

        # sigma_sq proxy: (1 - cos_delta) is a bounded, wrap-safe stand-in for angular
        # variance, computed separately per branch then averaged.
        sigma_sq_o = torch.mean(1.0 - cos_delta_o)
        sigma_sq_i = torch.mean(1.0 - cos_delta_i)
        l_strehl_o = 1.0 - torch.exp(-sigma_sq_o)
        l_strehl_i = 1.0 - torch.exp(-sigma_sq_i)
        l_strehl = 0.5 * (l_strehl_o + l_strehl_i)

        l_toeplitz = smoothness_regularizer

        l_cov_consistency = torch.tensor(0.0, device=R_complex.device)

        total_loss = (
            self.l1 * l_phasor + 
            self.l2 * l_strehl + 
            self.l3 * l_toeplitz + 
            self.l4 * l_cov_consistency
        )
        return total_loss, {
            "phasor": l_phasor.item(),
            "phasor_o": l_phasor_o.item(),
            "phasor_i": l_phasor_i.item(),
            "strehl": l_strehl.item(),
            "strehl_o": l_strehl_o.item(),
            "strehl_i": l_strehl_i.item(),
            "toeplitz": l_toeplitz.item(),
            "cov_consistency": l_cov_consistency.item()
        }

# =============================================================================
# 4. DYNAMIC CURRICULUM DATASET (WITH 40x40 INTERPOLATION)
# =============================================================================
class CurriculumReflectionDataset(Dataset):
    def __init__(self, h5_path, active_stages):
        self.file = h5py.File(h5_path, 'r')
        tags = [t.decode('utf-8') if isinstance(t, bytes) else t for t in self.file['difficulty_tags'][:]]
        splits = [s.decode('utf-8') if isinstance(s, bytes) else s for s in self.file['split_types'][:]]
        
        self.indices = [
            i for i, (t, s) in enumerate(zip(tags, splits)) 
            if t in active_stages and s == "train"
        ]
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        R_real = self.file['R_matrices'][file_idx, 0]
        R_imag = self.file['R_matrices'][file_idx, 1]
        rms_val = self.file['rms_scaling_factors'][file_idx]  # <-- Added: load stored per-sample RMS factor
        input_data = np.stack([R_real, R_imag], axis=0).astype(np.float32) / rms_val  # <-- Added: normalize before interpolation
        
        # Interpolate from 1600x1600 to 40x40 to prevent CUDA OOM and match model specs
        tensor_in = torch.from_numpy(input_data).unsqueeze(0) # [1, 2, 1600, 1600]
        tensor_in = F.interpolate(tensor_in, size=(40, 40), mode='bilinear', align_corners=False).squeeze(0) # [2, 40, 40]

        target_phi_o = self.file['phi_o_maps'][file_idx].astype(np.float32)
        target_phi_i = self.file['phi_i_maps'][file_idx].astype(np.float32)
        target_obj = self.file['target_objects'][file_idx].astype(np.float32)
        
        return (
            tensor_in, 
            torch.from_numpy(target_phi_o).unsqueeze(0), 
            torch.from_numpy(target_phi_i).unsqueeze(0),
            torch.from_numpy(target_obj)
        )

def get_validation_loader(h5_path):
    with h5py.File(h5_path, 'r') as f:
        splits = [s.decode('utf-8') if isinstance(s, bytes) else s for s in f['split_types'][:]]
        val_indices = [i for i, s in enumerate(splits) if s == "val"]
    
    class ValDataset(Dataset):
        def __init__(self, path, indices):
            self.file = h5py.File(path, 'r')
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            f_idx = self.indices[idx]
            R_real = self.file['R_matrices'][f_idx, 0]
            R_imag = self.file['R_matrices'][f_idx, 1]
            rms_val = self.file['rms_scaling_factors'][f_idx]  # <-- Added: load stored per-sample RMS factor
            input_data = np.stack([R_real, R_imag], axis=0).astype(np.float32) / rms_val  # <-- Added: normalize before interpolation
            
            tensor_in = torch.from_numpy(input_data).unsqueeze(0)
            tensor_in = F.interpolate(tensor_in, size=(40, 40), mode='bilinear', align_corners=False).squeeze(0)

            target_phi_o = self.file['phi_o_maps'][f_idx].astype(np.float32)
            target_phi_i = self.file['phi_i_maps'][f_idx].astype(np.float32)
            target_obj = self.file['target_objects'][f_idx].astype(np.float32)
            return tensor_in, torch.from_numpy(target_phi_o).unsqueeze(0), torch.from_numpy(target_phi_i).unsqueeze(0), torch.from_numpy(target_obj)

    return DataLoader(ValDataset(h5_path, val_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()

# =============================================================================
# 5. TRAINING SCRIPT EXECUTION
# =============================================================================
if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    
    model = CoviFormer().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-4)
    scaler = GradScaler('cuda', enabled=USE_AMP)
    loss_fn = CompleteCompositeLoss().to(DEVICE)
    
    val_loader = get_validation_loader(H5_FILE)

    current_stage_idx = 0
    active_stages = [CURRICULUM_STAGES[current_stage_idx]]
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5
    # FIX: was 0.10 (10%) -- a single ordinary epoch's relative val-loss improvement
    # is almost always below 10%, so this condition was satisfied on nearly every
    # epoch once epoch > 15, regardless of the patience_counter. That's why the
    # curriculum raced through "noise_low" and "noise_med" in one epoch each in the
    # last run. Lowered to require a genuine plateau (<0.1% relative improvement).
    min_delta_percent = 0.001

    csv_path = METRICS_CSV_PATH
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Stage', 'Train_Loss', 'Val_Loss', 'Val_SSIM'])

    print(f"\n--> Starting CoviFormer Curriculum Training Pipeline on {DEVICE.upper()}...")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_dataset = CurriculumReflectionDataset(H5_FILE, active_stages)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        
        model.train()
        epoch_train_loss = 0.0

        for input_batch, target_phi_o, target_phi_i, _ in tqdm(train_loader, desc=f"Epoch {epoch} [Stage: {active_stages[-1]}]"):
            input_batch = input_batch.to(DEVICE)
            target_phi_o = target_phi_o.to(DEVICE)
            target_phi_i = target_phi_i.to(DEVICE)
            
            # Complex conversion & 2-channel real conversion for covariance streams
            R_complex = torch.complex(input_batch[:, 0], input_batch[:, 1])
            rrt = torch.matmul(R_complex, torch.conj(R_complex).transpose(-2, -1))
            rtr = torch.matmul(torch.conj(R_complex).transpose(-2, -1), R_complex)

            rrt_real = torch.stack([rrt.real, rrt.imag], dim=1)
            rtr_real = torch.stack([rtr.real, rtr.imag], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=USE_AMP):
                preds = model(input_batch, rrt_real, rtr_real)
                loss, _ = loss_fn(
                    preds["output_aberration"], target_phi_o,
                    preds["input_aberration"], target_phi_i,
                    R_complex, rrt, rtr,
                    preds["smoothness_regularizer"]
                )

            scaler.scale(loss).backward()
            if GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)

        model.eval()
        enable_mc_dropout(model)
        
        val_loss = 0.0
        ssim_scores = []
        last_val_batch = None

        with torch.no_grad():
            for val_input, val_targ_o, val_targ_i, val_targ_obj in val_loader:
                val_input = val_input.to(DEVICE)
                val_targ_o = val_targ_o.to(DEVICE)
                val_targ_i = val_targ_i.to(DEVICE)
                
                R_complex_val = torch.complex(val_input[:, 0], val_input[:, 1])
                rrt_val = torch.matmul(R_complex_val, torch.conj(R_complex_val).transpose(-2, -1))
                rtr_val = torch.matmul(torch.conj(R_complex_val).transpose(-2, -1), R_complex_val)

                rrt_val_real = torch.stack([rrt_val.real, rrt_val.imag], dim=1)
                rtr_val_real = torch.stack([rtr_val.real, rtr_val.imag], dim=1)

                mc_preds_o = []
                mc_preds_i = []
                for _ in range(MC_DROPOUT_PASSES):
                    v_preds = model(val_input, rrt_val_real, rtr_val_real)
                    mc_preds_o.append(v_preds["output_aberration"])
                    mc_preds_i.append(v_preds["input_aberration"])

                # Averaging (cos, sin) phasors across MC-dropout passes is the correct
                # way to do circular averaging (this actually fixes a latent bug the old
                # angle-averaging had: naively averaging angles like 179 deg and -179 deg
                # gives ~0 deg instead of ~180 deg; averaging phasors avoids that). The
                # mean of several unit vectors isn't itself unit-length, so renormalize.
                def _renormalize_phasor(x, eps=1e-8):
                    mag = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + eps)
                    return x / mag

                pred_phasor_o_mean = _renormalize_phasor(torch.stack(mc_preds_o).mean(dim=0))
                pred_phasor_i_mean = _renormalize_phasor(torch.stack(mc_preds_i).mean(dim=0))
                v_preds["output_aberration"] = pred_phasor_o_mean
                v_preds["input_aberration"] = pred_phasor_i_mean

                v_loss, _ = loss_fn(
                    v_preds["output_aberration"], val_targ_o,
                    v_preds["input_aberration"], val_targ_i,
                    R_complex_val, rrt_val, rtr_val,
                    v_preds["smoothness_regularizer"]
                )
                val_loss += v_loss.item()

                # Convert phasor -> angle only here, at the SSIM/visualization boundary
                # (never inside the model or the loss) -- this is the one place atan2
                # is acceptable, per the same rule PhasorHead now follows.
                pred_phi_o_mean = torch.atan2(pred_phasor_o_mean[:, 1:2, :, :], pred_phasor_o_mean[:, 0:1, :, :])
                pred_phi_i_mean = torch.atan2(pred_phasor_i_mean[:, 1:2, :, :], pred_phasor_i_mean[:, 0:1, :, :])

                pred_np = pred_phi_o_mean.squeeze().cpu().numpy()
                targ_np = val_targ_o.squeeze().cpu().numpy()
                if pred_np.ndim == 2:
                    score = compute_ssim(targ_np, pred_np, data_range=pred_np.max() - pred_np.min() if pred_np.max() != pred_np.min() else 1.0)
                    ssim_scores.append(score)

                last_val_batch = (val_input, rrt_val_real, rtr_val_real, val_targ_o, val_targ_i, pred_phi_o_mean, pred_phi_i_mean)

        avg_val_loss = val_loss / len(val_loader)
        avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0

        print(f"Epoch {epoch:03d} | Stage: {active_stages[-1]} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val SSIM: {avg_ssim:.4f}")

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, active_stages[-1], avg_train_loss, avg_val_loss, avg_ssim])

        if epoch % CHECKPOINT_INTERVAL == 0 or avg_val_loss < best_val_loss:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                patience_counter = 0
            else:
                patience_counter += 1

            if last_val_batch is not None and (epoch % CHECKPOINT_INTERVAL == 0 or epoch == 1):
                v_inp, v_rrt, v_rtr, v_t_o, v_t_i, v_p_o, v_p_i = last_val_batch
                save_visualization(v_inp, v_rrt, v_rtr, v_t_o, v_t_i, v_p_o, v_p_i, epoch-1)

        percent_improvement = (best_val_loss - avg_val_loss) / (best_val_loss + 1e-8)
        
        if patience_counter >= patience_limit or (percent_improvement < min_delta_percent and epoch > 15):
            if current_stage_idx < len(CURRICULUM_STAGES) - 1:
                current_stage_idx += 1
                active_stages.append(CURRICULUM_STAGES[current_stage_idx])
                print(f"\n[CURRICULUM ADVANCEMENT] Graduating to next stage tier: '{active_stages[-1]}'")
                patience_counter = 0