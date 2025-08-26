# Educational Demonstration

This directory contains educational materials for understanding the BKM Engine concepts. These are **teaching tools**, not production code.

## Important Distinction

| Aspect | Educational (This Folder) | Production (Main Engine) |
|--------|---------------------------|--------------------------|
| **Purpose** | Teaching concepts | Research simulations |
| **Implementation** | Simplified CPU code | GPU-optimized engine |
| **Grid Size** | 64³ (limited) | Up to 512³ |
| **Reynolds Number** | ~300 | Up to 1600+ |
| **Performance** | ~1 step/sec | 100+ steps/sec |
| **Use Case** | Learning/Teaching | Paper reproduction |

## Contents

### bkm_monitoring_demo.ipynb
An interactive Jupyter notebook demonstrating:
- BKM integral computation and monitoring
- Vorticity-strain alignment observation
- The ρ (rho) guard for numerical stability
- Adaptive timestep control
- Why alignment remains naturally bounded

**Key Finding**: The notebook shows that vorticity-strain alignment naturally remains bounded without artificial enforcement, validating the observational approach used in the production engine.

## Running the Notebook

### Prerequisites
```bash
pip install jupyter numpy scipy matplotlib
```

### Launch Notebook
### From this directory

```
jupyter notebook bkm_monitoring_demo.ipynb
```

### JupyterLab

```
jupyter lab bkm_monitoring_demo.ipynb
```
