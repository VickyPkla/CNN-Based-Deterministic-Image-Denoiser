# Deep Deterministic Denoising Network

[![PyPI version](https://img.shields.io/pypi/v/your-package-name.svg)](https://pypi.org/project/your-package-name) <!-- optional placeholder -->
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/your-repo/your-project/actions) <!-- placeholder -->
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> A lightweight, PyTorch‑based implementation of a deterministic model for image denoising using a conditional U-Net.

**Table of Contents**
1. [What the Project Does](#what-the-project-does)
2. [Why It’s Useful](#why-its-useful)
3. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Training](#training)
   - [Evaluation / Inference](#evaluation--inference)
4. [Support & Documentation](#support--documentation)
5. [Maintainers & Contributing](#maintainers--contributing)

---

## What the Project Does

This repository contains the code for a **Deep Deterministic Denoising Network**.
A conditional U‑Net architecture processes noisy grayscale images through a diffusion-like
forward and reverse process to recover clean signals. The model is trained on paired clean/noisy
datasets and can be used for both training new models and evaluating/denoising image sets.

The implementation includes:

- `unet.py`: Conditional U‑Net model with time embeddings and self‑attention blocks.
- `train.py`: Training loop with custom dataset, noise scheduling, checkpoints, and plotting.
- `test.py`: Evaluation script computing SSIM, RMSE, PSNR and saving denoised outputs.
- `loss.py`: Custom Charbonnier/SSIM composite loss functions.

## Why It’s Useful

This project is useful for researchers and engineers working on:

- Image denoising tasks where deterministic models are preferred.
- Prototyping custom U‑Net architectures with time conditioning for conditional generation.
- Learning about training loops involving noise scheduling without Monte Carlo sampling.

Key benefits:

- ✅ Lightweight and self‑contained PyTorch code with minimal dependencies.
- ✅ Supports checkpointing/resuming and multi‑GPU training via `DataParallel`.
- ✅ Evaluation script provides common image quality metrics and exports results.

## Getting Started

### Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/) (CUDA version if GPU training is desired)
- `torchvision`, `torchmetrics`, `scikit-image`, `tqdm`, `matplotlib`, `Pillow`

### Installation

```bash
# clone the repo
git clone https://github.com/your-repo/your-project.git
cd "CNN Based Deterministic Image Denoiser"

# create a venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# install dependencies
pip install torch torchvision torchmetrics scikit-image tqdm matplotlib pillow
```

> You can also add other utilities such as `torchsummary` or `tensorboard` as needed.

### Training

Prepare your training data in the following structure:

```
train/
├── gt/        # clean ground‑truth images (grayscale PNG/JPEG)
└── Mixed/     # corresponding noisy images (same filenames)
```

Then run the training script:

```bash
python train.py
```

The script will:

1. Load `CustomDataset` with normalization.
2. Add noise to inputs based on a linear schedule.
3. Train the `ConditionalUNet` for 1 000 epochs (configurable).
4. Save checkpoints under `unet checkpoints/` including a `best_model.pth`.

Adjust hyperparameters such as `batch_size`, `lr`, or number of timesteps directly in `train.py`.

### Evaluation / Inference

Place your test images under:

```
test/clean/   # clean targets for metric computation
test/noisy/   # noisy inputs to denoise
```

Then run:

```bash
python test.py
```

Metrics (SSIM, RMSE, PSNR) are averaged and displayed, and denoised images are saved to
`Outputs/`.
You can modify `num_timesteps` or other settings in `test.py` as needed.

#### Using a Pretrained Model

If you have a saved model (`.pth` checkpoint) you can copy it to
`unet checkpoints/best_model.pth` and the evaluation script will load it automatically.
