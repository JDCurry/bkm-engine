#!/usr/bin/env python3
"""
quick_start.py
==============
Minimal example to verify BKM Engine installation and basic functionality.

This runs a small Taylor-Green vortex simulation and reports key metrics.
Takes ~30 seconds on GPU, ~5 minutes on CPU.
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path to import engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from engine.unified_bkm_engine import CUDABKMSolver
except ImportError:
    print("Error: Could not import BKM Engine.")
    print("Please ensure the engine is properly installed.")
    sys.exit(1)


def main():
    print("="*70)
    print("BKM ENGINE - QUICK START VERIFICATION")
    print("="*70)
    
    # Small grid for quick test
    N = 128
    Re = 800
    t_final = 2.0
    
    print(f"\nConfiguration:")
    print(f"  Grid: {N}³")
    print(f"  Reynolds number: {Re}")
    print(f"  Simulation time: {t_final}")
    
    # Create solver
    print("\nInitializing solver...")
    try:
        solver = CUDABKMSolver(
            grid_size=N,
            reynolds_number=Re,
            dt=0.001,
            adapt_dt=True,
            CFL_target=0.25,
            use_dealiasing=True
        )
        print(f"  ✓ Solver initialized")
        print(f"  ✓ GPU: {'ENABLED' if solver.use_gpu else 'DISABLED (CPU fallback)'}")
    except Exception as e:
        print(f"  ✗ Failed to initialize solver: {e}")
        return False
    
    # Initialize Taylor-Green vortex
    print("\nSetting up Taylor-Green vortex...")
    L = 2 * np.pi
    xp = solver.xp  # Use cupy if available, numpy otherwise
    
    x = xp.linspace(0, L, N, endpoint=False)
    y = xp.linspace(0, L, N, endpoint=False)
    z = xp.linspace(0, L, N, endpoint=False)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    solver.u = xp.sin(X) * xp.cos(Y) * xp.cos(Z)
    solver.v = -xp.cos(X) * xp.sin(Y) * xp.cos(Z)
    solver.w = xp.zeros_like(solver.u)
    
    # Initial diagnostics
    E0 = solver.compute_energy()
    print(f"  ✓ Initial energy: {E0:.6f}")
    
    # Check divergence
    div_max, div_l2 = solver.compute_divergence_metrics()
    print(f"  ✓ Initial divergence: max={div_max:.2e}, L2={div_l2:.2e}")
    
    # Run simulation
    print(f"\nRunning simulation to t={t_final}...")
    print("-"*50)
    
    step = 0
    t_start = time.time()
    
    while solver.current_time < t_final and step < 10000:
        # Evolve
        metrics = solver.evolve_one_timestep(use_rk2=True)
        step += 1
        
        # Progress update
        if step % 100 == 0:
            progress = (solver.current_time / t_final) * 100
            print(f"  Step {step:4d}: t={solver.current_time:.3f} ({progress:.1f}%), "
                  f"E={metrics['energy']:.4f}, ω_max={metrics['vorticity']:.2f}")
    
    elapsed = time.time() - t_start
    
    # Final diagnostics
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    
    E_final = solver.compute_energy()
    div_max_final, _ = solver.compute_divergence_metrics()
    
    print(f"\nPerformance:")
    print(f"  Total steps: {step}")
    print(f"  Wall time: {elapsed:.1f} seconds")
    print(f"  Steps/second: {step/elapsed:.1f}")
    
    print(f"\nPhysics Verification:")
    print(f"  Energy conservation: {(E_final/E0 - 1)*100:.2f}%")
    print(f"  Final divergence: {div_max_final:.2e}")
    print(f"  Final time: {solver.current_time:.4f}")
    
    # Success criteria
    energy_error = abs(E_final/E0 - 1)
    success = (energy_error < 0.1 and div_max_final < 1e-10)
    
    if success:
        print(f"\n✓ SUCCESS: BKM Engine is working correctly!")
        if solver.use_gpu:
            print(f"  GPU acceleration is active")
        else:
            print(f"  Running on CPU (install CuPy for GPU support)")
    else:
        print(f"\n✗ WARNING: Large errors detected")
        print(f"  Please check your installation")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)