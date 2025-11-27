# Snake World Modeling

A DIAMOND-style pixel-space EDM (Elucidating the Design Space of Diffusion-Based Generative Models) implementation for world modeling in the Snake game environment.

## Project Structure

```
snake-world-modeling/
├── src/                    # Core source code modules
│   ├── models/            # Model definitions
│   │   └── pixel_edm.py   # Pixel-space EDM UNet and components
│   ├── data/              # Dataset classes
│   │   └── pixel_dataset.py
│   └── utils/             # Utility functions
│       └── ema.py         # Exponential Moving Average
│
├── scripts/               # Executable scripts
│   ├── train_pixel_edm.py      # Main training script
│   ├── play_pixel_edm.py       # Interactive inference script
│   ├── data_gen.py             # Data generation (Snake game)
│   └── check_metadata_stats.py # Dataset statistics
│
├── archive/               # Archived experimental code
│   ├── experiments/       # Old training scripts (v1, v2, VAE/diffusion)
│   └── data/              # Old data versions (v2-v4)
│
├── data_v5/               # Latest dataset (metadata only, images gitignored)
│   ├── images/            # Training images (not in git)
│   └── metadata.csv        # Episode metadata
│
├── output/                # Training outputs (gitignored)
│   └── pixel_edm/         # Model checkpoints and logs
│
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Features

- **Pixel-space EDM**: Works directly in pixel space (64x64x3), no VAE needed
- **Fast inference**: Only 3 denoising steps required
- **Action conditioning**: Uses Adaptive Group Normalization for strong action conditioning
- **Real data initialization**: Loads real game frames to avoid OOD issues
- **Weighted sampling**: Balanced training on rare events (death, eating)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Training Data

```bash
python scripts/data_gen.py --output data_v5/ --count 200000
```

### 2. Check Dataset Statistics

```bash
python scripts/check_metadata_stats.py --metadata data_v5/metadata.csv
```

### 3. Train Model

```bash
python scripts/train_pixel_edm.py \
    --img_dir data_v5/images \
    --metadata data_v5/metadata.csv \
    --output_dir output/pixel_edm \
    --epochs 40 \
    --batch_size 512
```

### 4. Play/Inference

```bash
python scripts/play_pixel_edm.py \
    --model_path output/pixel_edm/best_model/model.pt \
    --data_dir data_v5/images \
    --cfg_scale 2.0 \
    --steps 3
```

**Controls:**
- `WASD` or Arrow Keys - Move snake
- `R` - Reset with new random seed from data
- `ESC` - Quit

## Architecture

The model uses a DIAMOND-style architecture:

- **UNet backbone**: Encoder-decoder with skip connections
- **Adaptive GroupNorm**: Conditions normalization on action + noise level
- **Self-attention**: Global reasoning at bottleneck (8x8 resolution)
- **EDM formulation**: Preconditioned network for stable training with few steps
- **Frame stacking**: 4 previous frames concatenated channel-wise (12 channels)

### Model Configuration

- Input: `(B, 15, 64, 64)` - noisy frame (3) + context (12)
- Output: `(B, 3, 64, 64)` - denoised frame
- Base dimensions: 128 channels, 512 condition dimension
- Denoising steps: 3 (Euler method)

## Training Details

- **Loss**: Weighted MSE with EDM loss weighting
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: Cosine annealing
- **Mixed precision**: BF16 AMP
- **EMA**: Exponential moving average (decay=0.9999)
- **Weighted sampling**: 5x weight for death/eating events
- **CFG dropout**: 30% for classifier-free guidance

### Output Structure

```
output/pixel_edm/
├── best_model/
│   └── model.pt           # Best model checkpoint
├── checkpoints/
│   └── model_ep*.pt       # Periodic checkpoints
└── training_log.csv       # Training metrics
```

## Notes

- **Data**: Only `data_v5/` is the current dataset. Older versions (v2-v4) are archived.
- **Archive**: Old experimental code (VAE/diffusion approaches, v1/v2 scripts) is in `archive/` for reference.
- **Images**: Training images are gitignored due to size. Only metadata CSVs are tracked.
