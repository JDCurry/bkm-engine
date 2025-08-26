# Engine Variants

## unified_bkm_engine.py
- Full float64 precision throughout
- Maximum numerical accuracy
- ~10GB memory for 256³ grid
- Recommended for: validation studies, paper figures, small grids

## mixed_precision_engine.py
- Float32 for velocity fields and FFT operations
- Float64 for diagnostics, energy/enstrophy calculations, and checkpoints
- ~5GB memory for 256³ grid (50% reduction)
- 1.8-2x faster than full float64
- Recommended for: production runs, large grids, parameter sweeps

## Installation

### Requirements
```bash
# Core dependencies
pip install numpy scipy matplotlib h5py

# For GPU acceleration (recommended)
pip install cupy-cuda11x  # Adjust CUDA version as needed

# Optional for analysis

pip install jupyter pandas
