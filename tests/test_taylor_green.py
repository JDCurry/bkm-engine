#!/usr/bin/env python3
"""
test_taylor_green.py
====================
Taylor-Green vortex test for BKM solver validation.

This test simulates the Taylor-Green vortex, a canonical problem for
studying vortex dynamics and energy cascade in 3D turbulence.

Reference:
    Taylor, G. I., & Green, A. E. (1937). Mechanism of the production 
    of small eddies from large ones. Proc. R. Soc. Lond. A, 158(895).
"""

import numpy as np
import time
import os
import sys
import argparse
from datetime import datetime

# Import unified components
from unified_bkm_engine import CUDABKMSolver
from test_base import UnifiedCheckpointer, compute_comprehensive_diagnostics


def initialize_taylor_green(solver, amplitude=1.0):
    """
    Initialize Taylor-Green vortex.
    
    Parameters:
    -----------
    solver : CUDABKMSolver
        Solver instance
    amplitude : float
        Initial amplitude of velocity field
    
    Returns:
    --------
    E0 : float
        Initial energy
    W0 : float
        Initial maximum vorticity
    Z0 : float
        Initial enstrophy
    """
    xp = solver.xp
    
    # Create coordinate arrays
    x = xp.linspace(0, solver.Lx, solver.nx, endpoint=False)
    y = xp.linspace(0, solver.Ly, solver.ny, endpoint=False)
    z = xp.linspace(0, solver.Lz, solver.nz, endpoint=False)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    # Standard Taylor-Green initial condition
    solver.u = amplitude * xp.sin(X) * xp.cos(Y) * xp.cos(Z)
    solver.v = -amplitude * xp.cos(X) * xp.sin(Y) * xp.cos(Z)
    solver.w = xp.zeros_like(solver.u)
    
    # Ensure divergence-free
    solver.u, solver.v, solver.w = solver.project_div_free(solver.u, solver.v, solver.w)
    
    # Compute initial diagnostics
    E0 = solver.compute_energy()
    W0 = solver.compute_max_vorticity()
    Z0 = solver.compute_enstrophy()
    
    return E0, W0, Z0


def run_test(args):
    """
    Run Taylor-Green test with standardized data export.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    
    print("\n" + "="*70)
    print("TAYLOR-GREEN VORTEX TEST")
    print("="*70)
    print(f"Configuration:")
    print(f"  Grid: {args.N}^3")
    print(f"  Reynolds number: {args.Re}")
    print(f"  Target time: {args.time}")
    print(f"  Save fields: {args.save_fields}")
    print(f"  Save interval: {args.save_every} steps")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"taylor_green_N{args.N}_Re{args.Re}_{timestamp}"
    
    # Initialize checkpointer
    params = {
        'Re': args.Re,
        'nu': 1.0/args.Re,
        'grid_size': args.N,
        'target_time': args.time,
        'amplitude': args.amplitude,
        'dt_initial': 2e-3
    }
    
    checkpointer = UnifiedCheckpointer(
        output_dir=output_dir,
        test_name='taylor_green',
        grid_shape=(args.N, args.N, args.N),
        params=params
    )
    
    # Create solver
    print("\nInitializing solver...")
    solver = CUDABKMSolver(
        grid_size=args.N,
        reynolds_number=args.Re,
        dt=2e-3,
        adapt_dt=True,
        CFL_target=0.4,
        use_dealiasing=True,
        projection_tol=1e-9,
        extra_projection_iters=2,
        divergence_threshold=1e-6
    )
    
    print(f"  GPU: {'ENABLED' if solver.use_gpu else 'DISABLED'}")
    print(f"  Viscosity: {solver.viscosity:.6e}")
    
    # Initialize flow
    print("\nInitializing Taylor-Green vortex...")
    E0, W0, Z0 = initialize_taylor_green(solver, args.amplitude)
    print(f"  Initial energy: {E0:.6f}")
    print(f"  Initial max vorticity: {W0:.3f}")
    print(f"  Initial enstrophy: {Z0:.6f}")
    
    # Check initial divergence
    div_max, div_l2 = solver.compute_divergence_metrics()
    print(f"  Initial divergence: max={div_max:.3e}, L2={div_l2:.3e}")
    
    # Evolution loop
    print(f"\nStarting evolution...")
    print("-" * 60)
    
    step = 0
    start_time = time.time()
    prev_energy = E0
    prev_time = 0.0
    
    # Track maximum values for summary
    max_vorticity = W0
    max_enstrophy = Z0
    
    while solver.current_time < args.time and step < args.max_steps:
        step += 1
        
        # Adaptive timestep
        if solver.adapt_dt:
            dt_new, cfl, rho, limiter, _ = solver.choose_dt()
            solver.dt = dt_new
        
        # Evolve one timestep
        result = solver.evolve_one_timestep(use_rk2=True)
        
        # Compute comprehensive diagnostics
        diag = compute_comprehensive_diagnostics(solver, prev_energy, prev_time)
        diag['step'] = step
        diag['time'] = solver.current_time
        diag['dt'] = solver.dt
        diag['cfl'] = getattr(solver, 'last_cfl', cfl if 'cfl' in locals() else 0)
        
        # Track maxima
        if diag['vorticity_max'] > max_vorticity:
            max_vorticity = diag['vorticity_max']
        if diag['enstrophy'] > max_enstrophy:
            max_enstrophy = diag['enstrophy']
        
        # Save time series
        checkpointer.append_time_series(diag)
        
        # Save checkpoint
        if args.save_fields and (step % args.save_every == 0 or step == 1):
            filename = checkpointer.save_checkpoint(step, solver)
            if step == 1 or step % 1000 == 0:
                print(f"  Checkpoint saved: {os.path.basename(filename)}")
        
        # Save spectrum
        if step % args.spectrum_every == 0:
            filename = checkpointer.save_spectrum(step, solver)
            if step % 1000 == 0:
                print(f"  Spectrum saved: {os.path.basename(filename)}")
        
        # Progress reporting
        if step % 100 == 0:
            elapsed = time.time() - start_time
            progress = (solver.current_time / args.time) * 100
            print(f"Step {step:5d} ({progress:5.1f}%):")
            print(f"  t={solver.current_time:.4f}, dt={solver.dt:.3e}")
            print(f"  E/E0={diag['energy']/E0:.6f}, max_vort={diag['vorticity_max']:.2f}")
            print(f"  Re_lambda={diag['Re_lambda']:.1f}, k_max*eta={diag['k_max_eta']:.2f}")
            if elapsed > 0:
                print(f"  Performance: {step/elapsed:.1f} steps/s")
        
        # Update for next iteration
        prev_energy = diag['energy']
        prev_time = solver.current_time
    
    # Final checkpoint
    if args.save_fields:
        filename = checkpointer.save_checkpoint(step, solver)
        print(f"\nFinal checkpoint: {os.path.basename(filename)}")
    
    # Close files
    checkpointer.close()
    
    # Summary
    elapsed_total = time.time() - start_time
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"Total runtime: {elapsed_total:.1f} seconds ({elapsed_total/60:.1f} minutes)")
    print(f"Total steps: {step}")
    print(f"Final time: {solver.current_time:.4f} / {args.time}")
    print(f"Average rate: {step/elapsed_total:.1f} steps/s")
    
    print(f"\nKey results:")
    print(f"  Maximum vorticity: {max_vorticity:.2f} (at Re={args.Re})")
    print(f"  Maximum enstrophy: {max_enstrophy:.3f}")
    print(f"  Final energy: E/E0 = {diag['energy']/E0:.6f}")
    print(f"  Final Re_lambda: {diag['Re_lambda']:.1f}")
    
    print(f"\nResults saved to: {output_dir}/")
    print(f"  Time series: time_series.h5")
    print(f"  Diagnostics: diagnostics.csv")
    if args.save_fields:
        print(f"  Checkpoints: {step//args.save_every + 1} files")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Taylor-Green vortex test with comprehensive data export',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Grid and physics
    parser.add_argument('--N', type=int, default=256,
                       help='Grid resolution (N x N x N)')
    parser.add_argument('--Re', type=float, default=1600,
                       help='Reynolds number')
    parser.add_argument('--amplitude', type=float, default=1.0,
                       help='Initial velocity amplitude')
    
    # Time integration
    parser.add_argument('--time', type=float, default=20.0,
                       help='Target simulation time')
    parser.add_argument('--max-steps', type=int, default=100000,
                       help='Maximum number of steps')
    
    # Output control
    parser.add_argument('--save-fields', action='store_true',
                       help='Save full 3D velocity and vorticity fields')
    parser.add_argument('--save-every', type=int, default=100,
                       help='Steps between field checkpoints')
    parser.add_argument('--spectrum-every', type=int, default=500,
                       help='Steps between spectrum saves')
    
    args = parser.parse_args()
    
    success = run_test(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()