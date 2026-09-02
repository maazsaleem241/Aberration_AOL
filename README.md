# CLEAR / ARC — Covariance-Aware Physics-Guided Aberration Correction for Reflection Matrix Microscopy

## Abstract

Reflection matrix microscopy (RMM) recovers high-resolution images through
scattering media by measuring a full angle-resolved reflection matrix and
computationally correcting the aberrations it contains. Existing correction
methods such as CLASS iteratively cross-correlate the matrix against itself
and converge over many passes, which is accurate but slow. **ARC**
(Aberrated Reflection-matrix Corrector) is a 27.7M-parameter dual-head
SwinV2-UNet that predicts **both** the input- and output-side aberration
phase directly from the raw reflection matrix in a **single forward pass**
— unlike prior deep-learning approaches, which predict only one side per
call and still require repeated, iterative re-inference to recover both.
Correction itself is closed-form physics, not learned: given the predicted
phasors, $R_c = \hat P_o^\dagger R \hat P_i^\dagger$ collapses the measured
matrix back toward the object's own Fourier-domain scattering matrix,
exactly as the underlying forward model, $R = P_o\,O\,P_i^T$, predicts.
ARC is trained entirely on physically simulated data — a multi-scale object
curriculum (beads, letters, digits, crosses, gratings) combined with a
randomized Zernike aberration model (order 10–60, numerically stable at
every order) and a noise-injection model matching real sensor statistics —
and is validated against real experimental and simulated RMM data with
known ground truth.

---

## Repository structure

```
dataset/
    USAF.mat                    # USAF resolution-target crops, used as one training object style
    zernike.py                  # Zernike aberration basis (numerically stable up to order 60)
    dataset_generator.py        # Generates the two training/eval HDF5 datasets
    dataset_visualizer.py       # Renders sample panels from a generated .h5 for sanity-checking

training/
    ARC_training_noise.py       # Trains ARC on the noise curriculum (dataset_b)
    ARC_training_aberration.py  # Trains ARC on the aberration curriculum (dataset_a)
    CASS_visualizer.py          # Reconstruction utilities (dc-column, full CASS) used during training

evaluation/
    ARC_evaluation_noise.py       # Full metric suite + example panels, noise curriculum
    ARC_evaluation_aberration.py  # Full metric suite + example panels, aberration curriculum
    CASS_visualizer.py             # Same reconstruction utilities as above

ARC.py       # Model architecture (dual-head SwinV2-UNet)
README.md
```

---

## Requirements

```
python >= 3.9
torch
numpy
scipy
h5py
matplotlib
scikit-image
```

Install with:

```bash
pip install torch numpy scipy h5py matplotlib scikit-image
```

---

## Important: fixing the `models.ARC` import

`ARC_training_noise.py`, `ARC_training_aberration.py`, `ARC_evaluation_noise.py`,
and `ARC_evaluation_aberration.py` all import the model as:

```python
from models.ARC import ARC, make_na_mask
```

This repo keeps `ARC.py` at the **top level**, not inside a `models/`
subfolder — that import line reflects the original developer's local
layout, which you will not have after cloning. **Change it to:**

```python
from ARC import ARC, make_na_mask
```

in each of the four files above before running anything (each file already
has a comment marking exactly where to make this change).

---

## Usage

### 1. Generate the datasets

Open `dataset/dataset_generator.py` and check the bottom of the file:

```python
if __name__ == "__main__":
    BASE_PATH = "/your/desired/output/path"
    execute_dataset_production(root_dir=BASE_PATH, dataset_name="dataset_a", total_samples=15000, mode="C")
    execute_dataset_production(root_dir=BASE_PATH, dataset_name="dataset_b", total_samples=15000, mode="B")
```

- **`dataset_a`** (mode `"C"`) is the **aberration curriculum**: `aberration_low → med → high → extreme`, plus matched validation and test splits (including a discrete aberration sweep for headline plots).
- **`dataset_b`** (mode `"B"`) is the **noise curriculum**: `no noise → noise_low → noise_med → noise_high`, plus matched validation/test splits.

Update `BASE_PATH`, then run:

```bash
python3 dataset_generator.py
```

**Disk space warning**: each 15,000-sample dataset is dominated almost
entirely by its `R_matrices` array (full 1600×1600 complex reflectance
per sample) — roughly **~280 GB per dataset**, even after compression
(this data doesn't compress well; gzip only recovers a few percent).
Make sure `BASE_PATH` points somewhere with sufficient free space before
running a full generation.

Sanity-check what got generated with:

```bash
python3 dataset_visualizer.py --h5-file /path/to/dataset_a/data/reflection_matrix_dataset.h5
```

### 2. Train

Edit `H5_FILE` and the output/checkpoint paths near the top of each
training script to match where you generated your datasets, then:

```bash
python3 training/ARC_training_noise.py         # trains on dataset_b
python3 training/ARC_training_aberration.py     # trains on dataset_a
```

### 3. Evaluate

Edit `CHECKPOINT_PATH` (pointing at a checkpoint produced above),
`H5_FILE`, and `OUTPUT_DIR` near the top of each evaluation script, then:

```bash
python3 evaluation/ARC_evaluation_noise.py
python3 evaluation/ARC_evaluation_aberration.py
```

Each produces a full metrics CSV (Strehl ratio, RMS phase error, SSIM,
PSNR, inference timing) plus example reconstruction panels, per difficulty
tag, in `OUTPUT_DIR`.

---

## Metric conventions

- **Uncorrected** reconstructions use a single-column (dc-column) reconstruction of the raw, uncorrected reflection matrix.
- **Corrected** reconstructions use the full 2D $\Delta k$ CASS accumulation of the model-corrected matrix $R_c$.
- **Strehl ratio** and **RMS phase error** are computed from the coherent phasor mean of the wrap-safe residual between predicted and true aberration phase, reported as the average of the input- and output-side values.
- **SSIM** / **PSNR** compare the reconstructed image directly against the ground-truth object.
