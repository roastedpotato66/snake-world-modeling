# Archive Directory

This directory contains archived experimental code and old data versions that are no longer actively used.

## Structure

### `experiments/`
Contains old training and inference scripts:

- **`train_pixel_edm_v1.py`** - Original pixel EDM implementation
- **`train_pixel_edm_v2.py`** - V2 with is_eating support (now in `scripts/train_pixel_edm.py`)
- **`play_pixel_edm_v1.py`** - Original play script
- **`play_pixel_edm_v2.py`** - V2 with real data loading (now in `scripts/play_pixel_edm.py`)
- **`vae_diffusion/`** - All VAE and diffusion-based approaches (not used in final model)
  - `train_vae*.py` - VAE training scripts
  - `train_diffusion_v*.py` - Various diffusion training attempts
  - `play*.py` - Inference scripts for VAE/diffusion models
  - `diagnose_pipeline.py` - Diagnostic tools

### `data/`
Old data versions:
- `data_v2/` - Second data generation run
- `data_v3/` - Third data generation run  
- `data_v4/` - Fourth data generation run

**Note**: `data_v5/` is kept in root as it's the latest dataset.

### `diagnostic_output/`
Old diagnostic visualizations from VAE/diffusion experiments.

## Current Active Code

The current active implementation is:
- **Models**: `src/models/pixel_edm.py`
- **Training**: `scripts/train_pixel_edm.py`
- **Inference**: `scripts/play_pixel_edm.py`
- **Data**: `data_v5/`

All other code in this archive can be deleted if you want to clean up further, but it's kept for reference.

