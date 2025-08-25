#!/usr/bin/env python3
"""
test_kida_pelz.py
=================
Kida-Pelz vortex test for BKM solver validation.

This test simulates the Kida-Pelz vortex configuration, consisting of
two orthogonal vortex tubes that interact and develop complex dynamics.

Reference:
    Kida, S., & Pelz, R. B. (1985). Collision of two vortex rings.
    Journal of Fluid Mechanics, 230, 583-646.
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


def initialize_kida_pelz(solver, amplitude=1.3):
    """
    Initialize Kida-Pelz vortex configuration.
    
    Parameters:
    -----------
    solver : CUDABKMSolver
        Solver instance
    amplitude : float
        Vortex strength parameter
    
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
    nx, ny, nz = solver.nx, solver.ny, solver.nz
    
    # Create coordinate arrays
    x = xp.linspace(0, solver.Lx, nx, endpoint=False)
    y = xp.linspace(0, solver.Ly, ny, endpoint=False)
    z = xp.linspace(0, solver.Lz, nz, endpoint=False)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    # Two orthogonal vortex tubes
    r1_sq = (X - np.pi)**2 + (Y - np.pi)**2
    r2_sq = (Y - np.pi)**2 + (Z - np.pi)**2
    sigma = 0.8
    
    # Vorticity distributions
    vort1 = amplitude * xp.exp(-r1_sq / (2*sigma**2))
    vort2 = amplitude * xp.exp(-r2_sq / (2*sigma**2))
    
    # Velocity field from vorticity (simplified Biot-Savart)
    eps = 0.1  # Regularization parameter
    solver.u = -vort1 * (Y - np.pi) / xp.sqrt(r1_sq + eps**2)
    solver.v = vort1 * (X - np.pi) / xp.sqrt(r1_sq + eps**2) + \
               vort2 * (Z - np.pi) / xp.sqrt(r2_sq + eps**2)
    solver.w = -vort2 * (Y - np.pi) / xp.sqrt(r2_sq + eps**2)
    
    # Convert to float64 for accuracy
    solver.u = solver.u.astype(xp.float64)
    solver.v = solver.v.astype(xp.float64)
    solver.w = solver.w.astype(xp.float64)
    
    # Project to ensure divergence-free
    solver.u, solver.v, solver.w = solver.project_div_free(solver.u, solver.v, solver.w, 3)
    
    # Compute initial diagnostics
    E0 = solver.compute_energy()
    W0 = solver.compute_max_vorticity()
    Z0 = solver.compute_enstrophy()
    
    return E0, W0, Z0


def run_test(args):
    """
    Run Kida-Pelz test with standardized data export.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    
    print("\n" + "="*70)
    print("KIDA-PELZ VORTEX TEST")
    print("="*70)
    print(f"Configuration:")
    print(f"  Grid: {args.N}^3")
    print(f"  Reynolds number: {args.Re}")
    print(f"  Target time: {args.time}")
    print(f"  Amplitude: {args.amplitude}")
    print(f"  Save fields: {args.save_fields}")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"kida_pelz_N{args.N}_Re{args.Re}_{timestamp}"
    
    # Initialize checkpointer
    params = {
        'Re': args.Re,
        'nu': 1.0/args.Re,
        'grid_size': args.N,
        'target_time': args.time,
        'amplitude': args.amplitude,
        'sigma': 0.8,
        'eps': 0.1
    }
    
    checkpointer = UnifiedCheckpointer(
        output_dir=output_dir,
        test_name='kida_pelz',
        grid_shape=(args.N, args.N, args.N),
        params=params
    )
    
    # Create solver with appropriate settings for Kida-Pelz
    print("\nInitializing solver...")
    
    # Scale timestep with grid size
    dt_init = 1e-3 * (256.0 / args.N)
    dt_max = 3e-3 * (256.0 / args.N)
    
    solver = CUDABKMSolver(
        grid_size=args.N,
        reynolds_number=args.Re,
        dt=dt_init,
        adapt_dt=True,
        CFL_target=0.35,
        use_dealiasing=True,
        dt_min=1e-5,
        dt_max=dt_max,
        dt_max_startup=dt_max,
        startup_steps=100,
        growth_cap_startup=1.1,
        projection_tol=5e-10,
        extra_projection_iters=2,
        divergence_threshold=1e-6
    )
    
    print(f"  GPU: {'ENABLED' if solver.use_gpu else 'DISABLED'}")
    print(f"  Viscosity: {solver.viscosity:.6e}")
    print(f"  Initial dt: {dt_init:.3e}")
    
    # Initialize flow
    print("\nInitializing Kida-Pelz vortices...")
    E0, W0, Z0 = initialize_kida_pelz(solver, args.amplitude)
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
    
    # Track interaction metrics
    peak_vorticity = W0
    peak_enstrophy = Z0
    interaction_detected = False
    interaction_time = None
    
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
        
        # Detect vortex interaction (significant increase in vorticity)
        if diag['vorticity_max'] > peak_vorticity:
            peak_vorticity = diag['vorticity_max']
            if peak_vorticity > 1.2 * W0 and not interaction_detected:
                interaction_detected = True
                interaction_time = solver.current_time
                print(f"\n  VORTEX INTERACTION DETECTED at t={interaction_time:.3f}")
                print(f"  Peak vorticity: {peak_vorticity:.2f} ({peak_vorticity/W0:.2f}x initial)\n")
        
        if diag['enstrophy'] > peak_enstrophy:
            peak_enstrophy = diag['enstrophy']
        
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
        
        # Progress reporting
        if step % 100 == 0:
            elapsed = time.time() - start_time
            progress = (solver.current_time / args.time) * 100
            print(f"Step {step:5d} ({progress:5.1f}%):")
            print(f"  t={solver.current_time:.4f}, dt={solver.dt:.3e}")
            print(f"  E/E0={diag['energy']/E0:.6f}, max_vort={diag['vorticity_max']:.2f}")
            print(f"  Re_lambda={diag['Re_lambda']:.1f}, k_max*eta={diag['k_max_eta']:.2f}")
            print(f"  Budget error: {diag['budget_relative']:.1%}")
        
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
    
    print(f"\nKey results:")
    print(f"  Peak vorticity: {peak_vorticity:.2f} ({peak_vorticity/W0:.2f}x initial)")
    print(f"  Peak enstrophy: {peak_enstrophy:.3f} ({peak_enstrophy/Z0:.2f}x initial)")
    if interaction_detected:
        print(f"  Interaction time: t={interaction_time:.3f}")
    print(f"  Final energy: E/E0 = {diag['energy']/E0:.6f}")
    print(f"  Final Re_lambda: {diag['Re_lambda']:.1f}")
    
    print(f"\nResults saved to: {output_dir}/")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Kida-Pelz vortex test with comprehensive data export',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Grid and physics
    parser.add_argument('--N', type=int, default=256,
                       help='Grid resolution (N x N x N)')
    parser.add_argument('--Re', type=float, default=600,
                       help='Reynolds number')
    parser.add_argument('--amplitude', type=float, default=1.3,
                       help='Vortex amplitude')
    
    # Time integration
    parser.add_argument('--time', type=float, default=0.2,
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