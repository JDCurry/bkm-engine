#!/usr/bin/env python3
"""
test_antiparallel_vortex.py
============================
Anti-parallel vortex tube interaction test for BKM solver validation.

This test simulates two parallel vortex tubes with opposite circulation,
studying their interaction, reconnection, and subsequent dynamics.

Reference:
    Melander, M. V., & Hussain, F. (1989). Cross-linking of two
    antiparallel vortex tubes. Physics of Fluids A, 1(4), 633-636.
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


def initialize_antiparallel_tubes(solver, separation=0.60, perturbation=0.20, Gamma=2.0):
    """
    Initialize anti-parallel vortex tubes.
    
    Parameters:
    -----------
    solver : CUDABKMSolver
        Solver instance
    separation : float
        Separation between tube centers
    perturbation : float
        Perturbation amplitude
    Gamma : float
        Circulation strength
    
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
    N = solver.nx
    L = 2 * np.pi
    
    # Create coordinate arrays
    x = xp.linspace(0, L, N, endpoint=False)
    y = xp.linspace(0, L, N, endpoint=False)
    z = xp.linspace(0, L, N, endpoint=False)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    # Core radius
    rc = 0.35
    
    # Helical perturbation
    z_perturb = perturbation * rc * xp.sin(X)
    
    # Axial velocity modulation
    axial_mod = 1.0 + 0.10 * xp.sin(X) + 0.06 * xp.sin(2*X)
    
    # Vortex centers
    y0 = L/2 + separation/2
    y1 = L/2 - separation/2
    
    # Distance from vortex cores
    r0 = xp.sqrt((Y - y0)**2 + (Z - (L/2 + z_perturb))**2)
    r1 = xp.sqrt((Y - y1)**2 + (Z - (L/2 - z_perturb))**2)
    
    # Vorticity distribution (Gaussian cores)
    prefactor = Gamma / (np.pi * rc**2)
    omega_x = prefactor * (xp.exp(-(r0/rc)**2) - xp.exp(-(r1/rc)**2))
    omega_x *= axial_mod
    
    # Add 3D perturbations
    omega_y = 0.03 * omega_x * xp.sin(2*Y)
    omega_z = 0.03 * omega_x * xp.cos(2*Z)
    
    # Solve for velocity via Biot-Savart in Fourier space
    k = xp.fft.fftfreq(N, L/N) * 2*np.pi
    KX, KY, KZ = xp.meshgrid(k, k, k, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0,0,0] = 1.0
    
    # Transform vorticity to Fourier space
    omega_x_hat = xp.fft.fftn(omega_x)
    omega_y_hat = xp.fft.fftn(omega_y)
    omega_z_hat = xp.fft.fftn(omega_z)
    
    # Biot-Savart law
    i = 1j
    u_hat = i * (KY * omega_z_hat - KZ * omega_y_hat) / K2
    v_hat = i * (KZ * omega_x_hat - KX * omega_z_hat) / K2
    w_hat = i * (KX * omega_y_hat - KY * omega_x_hat) / K2
    
    # Zero mean flow
    u_hat[0,0,0] = 0
    v_hat[0,0,0] = 0
    w_hat[0,0,0] = 0
    
    # Transform back to physical space
    solver.u = xp.fft.ifftn(u_hat).real
    solver.v = xp.fft.ifftn(v_hat).real
    solver.w = xp.fft.ifftn(w_hat).real
    
    # Ensure divergence-free
    solver.u, solver.v, solver.w = solver.project_div_free(solver.u, solver.v, solver.w)
    
    # Compute initial diagnostics
    E0 = solver.compute_energy()
    W0 = solver.compute_max_vorticity()
    Z0 = solver.compute_enstrophy()
    
    return E0, W0, Z0


def run_test(args):
    """
    Run anti-parallel vortex test with standardized data export.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    
    print("\n" + "="*70)
    print("ANTI-PARALLEL VORTEX TUBE TEST")
    print("="*70)
    print(f"Configuration:")
    print(f"  Grid: {args.N}^3")
    print(f"  Reynolds number: {args.Re}")
    print(f"  Target time: {args.time}")
    print(f"  Separation/rc: {args.separation/0.35:.2f}")
    print(f"  Perturbation: {args.perturbation}")
    print(f"  Save fields: {args.save_fields}")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"antiparallel_N{args.N}_Re{args.Re}_{timestamp}"
    
    # Initialize checkpointer
    params = {
        'Re': args.Re,
        'nu': 1.0/args.Re,
        'grid_size': args.N,
        'target_time': args.time,
        'separation': args.separation,
        'perturbation': args.perturbation,
        'Gamma': args.Gamma,
        'rc': 0.35
    }
    
    checkpointer = UnifiedCheckpointer(
        output_dir=output_dir,
        test_name='antiparallel_vortex',
        grid_shape=(args.N, args.N, args.N),
        params=params
    )
    
    # Create solver
    print("\nInitializing solver...")
    
    # Scale timestep with grid size
    dt_init = 2e-3 * (256.0 / args.N)
    dt_max = 3e-3 * (256.0 / args.N)
    
    solver = CUDABKMSolver(
        grid_size=args.N,
        reynolds_number=args.Re,
        dt=dt_init,
        adapt_dt=True,
        CFL_target=0.25,
        use_dealiasing=True,
        dt_min=1e-4 * (256.0 / args.N),
        dt_max=dt_max,
        dt_max_startup=dt_max,
        startup_steps=100,
        growth_cap_startup=1.5,
        projection_tol=1e-9,
        extra_projection_iters=2,
        divergence_threshold=1e-6
    )
    
    print(f"  GPU: {'ENABLED' if solver.use_gpu else 'DISABLED'}")
    print(f"  Viscosity: {solver.viscosity:.6e}")
    
    # Initialize flow
    print("\nInitializing anti-parallel vortex tubes...")
    E0, W0, Z0 = initialize_antiparallel_tubes(
        solver,
        separation=args.separation,
        perturbation=args.perturbation,
        Gamma=args.Gamma
    )
    
    print(f"  Initial energy: {E0:.6f}")
    print(f"  Initial max vorticity: {W0:.3f}")
    print(f"  Initial enstrophy: {Z0:.6f}")
    
    # Check initial helicity (should be near zero)
    diag_init = compute_comprehensive_diagnostics(solver)
    print(f"  Initial helicity: {diag_init['helicity']:.3e} (should be ~0)")
    
    # Evolution loop
    print(f"\nStarting evolution...")
    print("-" * 60)
    
    step = 0
    start_time = time.time()
    prev_energy = E0
    prev_time = 0.0
    
    # Track reconnection
    peak_vorticity = W0
    peak_enstrophy = Z0
    reconnection_detected = False
    reconnection_time = None
    
    # Checkpoint times for vortex reconnection
    checkpoint_times = [0.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    next_checkpoint_idx = 1 if 0.0 not in checkpoint_times else 0
    
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
        
        # Detect reconnection
        if diag['vorticity_max'] > peak_vorticity:
            peak_vorticity = diag['vorticity_max']
            if peak_vorticity > 1.15 * W0 and not reconnection_detected:
                reconnection_detected = True
                reconnection_time = solver.current_time
                print(f"\n  RECONNECTION DETECTED at t={reconnection_time:.3f}")
                print(f"  Vorticity: {W0:.2f} -> {peak_vorticity:.2f} ({peak_vorticity/W0:.2f}x)\n")
        
        if diag['enstrophy'] > peak_enstrophy:
            peak_enstrophy = diag['enstrophy']
        
        # Save time series
        checkpointer.append_time_series(diag)
        
        # Save checkpoint at regular intervals
        if args.save_fields and (step % args.save_every == 0 or step == 1):
            filename = checkpointer.save_checkpoint(step, solver)
        
        # Save at specific times for reconnection study
        if args.save_fields and next_checkpoint_idx < len(checkpoint_times):
            if solver.current_time >= checkpoint_times[next_checkpoint_idx]:
                filename = checkpointer.save_checkpoint(step, solver)
                print(f"  Checkpoint at t={checkpoint_times[next_checkpoint_idx]:.1f}: "
                      f"{os.path.basename(filename)}")
                next_checkpoint_idx += 1
        
        # Progress reporting
        if step % 100 == 0:
            elapsed = time.time() - start_time
            progress = (solver.current_time / args.time) * 100
            print(f"Step {step:5d} ({progress:5.1f}%):")
            print(f"  t={solver.current_time:.4f}, dt={solver.dt:.3e}")
            print(f"  Vorticity: {diag['vorticity_max']:.2f} ({diag['vorticity_max']/W0:.2f}x)")
            print(f"  E/E0={diag['energy']/E0:.6f}, helicity={diag['helicity']:.2e}")
            print(f"  Re_lambda={diag['Re_lambda']:.1f}, k_max*eta={diag['k_max_eta']:.2f}")
        
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
    if reconnection_detected:
        print(f"  RECONNECTION at t={reconnection_time:.3f}")
    print(f"  Peak vorticity: {peak_vorticity:.2f} ({peak_vorticity/W0:.2f}x initial)")
    print(f"  Peak enstrophy: {peak_enstrophy:.4f} ({peak_enstrophy/Z0:.2f}x initial)")
    print(f"  Final energy: E/E0 = {diag['energy']/E0:.6f}")
    print(f"  Final Re_lambda: {diag['Re_lambda']:.1f}")
    
    print(f"\nResults saved to: {output_dir}/")
    
    # Summary for paper
    if reconnection_detected:
        sep_over_rc = args.separation / 0.35
        print(f"\nSUMMARY: Anti-parallel vortex tubes with sep/rc={sep_over_rc:.1f} "
              f"at Re={args.Re} underwent reconnection at t={reconnection_time:.2f}, "
              f"with {(peak_vorticity/W0-1)*100:.0f}% vorticity increase.")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Anti-parallel vortex tube test with comprehensive data export',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Grid and physics
    parser.add_argument('--N', type=int, default=256,
                       help='Grid resolution (N x N x N)')
    parser.add_argument('--Re', type=float, default=1000,
                       help='Reynolds number')
    
    # Vortex parameters
    parser.add_argument('--separation', type=float, default=0.60,
                       help='Separation between tube centers')
    parser.add_argument('--perturbation', type=float, default=0.20,
                       help='Perturbation amplitude')
    parser.add_argument('--Gamma', type=float, default=2.0,
                       help='Circulation strength')
    
    # Time integration
    parser.add_argument('--time', type=float, default=10.0,
                       help='Target simulation time')
    parser.add_argument('--max-steps', type=int, default=100000,
                       help='Maximum number of steps')
    
    # Output control
    parser.add_argument('--save-fields', action='store_true',
                       help='Save full 3D velocity and vorticity fields')
    parser.add_argument('--save-every', type=int, default=100,
                       help='Steps between field checkpoints')
    
    args = parser.parse_args()
    
    success = run_test(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()