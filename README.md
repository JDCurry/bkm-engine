# The BKM Engine

A GPU-accelerated pseudo-spectral solver for probing classical blow-up scenarios in 3D incompressible Navier-Stokes equations.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The BKM Engine implements a high-fidelity numerical framework for investigating potential finite-time singularities in 3D incompressible fluid flows. Key features include:

- **GPU Acceleration**: Full CUDA support via CuPy for grids up to 512³
- **Beale-Kato-Majda Monitoring**: Continuous tracking of BKM integral I(t)
- **Vorticity-Strain Alignment**: Real-time geometric diagnostics
- **Adaptive Timestepping**: CFL-based control with diagnostic guards
- **Machine-Precision Incompressibility**: Spectral projection maintaining ∇·u < 10⁻¹⁴
- **Comprehensive Test Suite**: Five canonical benchmarks with standardized data export

## Paper

Curry, J.D. (2025). "The BKM Engine: A GPU-Accelerated Pseudo-Spectral Solver for Classical Blow-Up Scenarios in 3D Incompressible Navier-Stokes." [Pending Publication]


## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- 16GB+ RAM for 256³ simulations
- 32GB+ GPU memory for 512³ simulations

### Quick Install
```bash
git clone https://github.com/JDCurry/bkm-engine.git
cd bkm-engine
pip install -r requirements.txt
python setup.py install
