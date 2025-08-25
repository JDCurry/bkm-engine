#!/usr/bin/env python3
"""
test_isotropic_decay.py
=======================
Isotropic turbulence decay test for BKM solver validation.

This test simulates freely decaying isotropic turbulence from
well-resolved initial conditions, tracking energy cascade,
spectrum evolution, and turbulence statistics.

Reference:
    Comte-Bellot, G., & Corrsin, S. (1971). Simple Eulerian time
    correlation of full-and narrow-band velocity signals in
    grid-generated, 'isotropic' turbulence. JFM, 48(2), 273-337.
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


def generate_isotropic_field(solver, k_peak=3, E_target=0.15, k_cutoff=10):
    """
    Generate clean isotropic turbulence initial conditions.
    
    Parameters:
    -----------
    solver : CUDABKMSolver
        Solver instance
    k_peak : float
        Peak wavenumber of initial spectrum
    E_target : float
        Target kinetic energy
    k_cutoff : float
        Cutoff wavenumber (no energy above this)
    
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
    L = 2*np.pi
    
    # Wavenumber grid
    k = xp.fft.fftfreq(N, L/N) * 2*np.pi
    kx, ky, kz = xp.meshgrid(k, k, k, indexing='ij')
    k_mag = xp.sqrt(kx**2 + ky**2 + kz**2)
    
    # Initialize spectrum with band limit
    E_k = xp.zeros_like(k_mag)
    mask = (k_mag > 0) & (k_mag <= k_cutoff)
    E_k[mask] = (k_mag[mask]**4) * xp.exp(-2.0*(k_mag[mask]/k_peak)**2)
    
    # Random phases
    phase_u = xp.random.uniform(-np.pi, np.pi, (N, N, N))
    phase_v = xp.random.uniform(-np.pi, np.pi, (N, N, N))
    phase_w = xp.random.uniform(-np.pi, np.pi, (N, N, N))
    
    # Create velocity field in Fourier space
    u_hat = xp.sqrt(E_k) * xp.exp(1j * phase_u)
    v_hat = xp.sqrt(E_k) * xp.exp(1j * phase_v)
    w_hat = xp.sqrt(E_k) * xp.exp(1j * phase_w)
    
    # Enforce reality condition
    u_hat[0,0,0] = 0
    v_hat[0,0,0] = 0
    w_hat[0,0,0] = 0
    
    # Project to divergence-free
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1.0
    
    k_dot_u = kx*u_hat + ky*v_hat + kz*w_hat
    u_hat -= kx * k_dot_u / k2
    v_hat -= ky * k_dot_u / k2
    w_hat -= kz * k_dot_u / k2
    
    # Transform to physical space
    solver.u = xp.fft.ifftn(u_hat).real
    solver.v = xp.fft.ifftn(v_hat).real
    solver.w = xp.fft.ifftn(w_hat).real
    
    # Rescale to target energy
    current_energy = 0.5 * xp.mean(solver.u**2 + solver.v**2 + solver.w**2)
    if current_energy > 0:
        scale = xp.sqrt(E_target / current_energy)
        solver.u *= scale
        solver.v *= scale
        solver.w *= scale
    
    # Remove mean
    solver.u -= solver.u.mean()
    solver.v -= solver.v.mean()
    solver.w -= solver.w.mean()
    
    # Compute initial diagnostics
    E0 = solver.compute_energy()
    W0 = solver.compute_max_vorticity()
    Z0 = solver.compute_enstrophy()
    
    return E0, W0, Z0


def run_test(args):
    """
    Run isotropic decay test with standardized data export.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    
    print("\n" + "="*70)
    print("ISOTROPIC TURBULENCE DECAY TEST")
    print("="*70)
    print(f"Configuration:")
    print(f"  Grid: {args.N}^3")
    print(f"  Reynolds number: {args.Re}")
    print(f"  Target time: {args.time}")
    print(f"  Initial spectrum: k_peak={args.k_peak}, k_cutoff={args.k_cutoff}")
    print(f"  Save fields: {args.save_fields}")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"isotropic_decay_N{args.N}_Re{args.Re}_{timestamp}"
    
    # Initialize checkpointer
    params = {
        'Re': args.Re,
        'nu': 1.0/args.Re,
        'grid_size': args.N,
        'target_time': args.time,
        'E_target': args.E_target,
        'k_peak': args.k_peak,
        'k_cutoff': args.k_cutoff
    }
    
    checkpointer = UnifiedCheckpointer(
        output_dir=output_dir,
        test_name='isotropic_decay',
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
        dt_min=1e-4,
        dt_max=3e-3,
        projection_tol=1e-9,
        extra_projection_iters=1,
        divergence_threshold=1e-6
    )
    
    print(f"  GPU: {'ENABLED' if solver.use_gpu else 'DISABLED'}")
    print(f"  Viscosity: {solver.viscosity:.6e}")
    
    # Initialize flow
    print("\nGenerating isotropic turbulence...")
    E0, W0, Z0 = generate_isotropic_field(
        solver, 
        k_peak=args.k_peak,
        E_target=args.E_target,
        k_cutoff=args.k_cutoff
    )
    
    print(f"  Initial energy: {E0:.6f}")
    print(f"  Initial max vorticity: {W0:.3f}")
    print(f"  Initial enstrophy: {Z0:.6f}")
    
    # Check initial resolution
    initial_eps = 2.0 * solver.viscosity * Z0
    if initial_eps > 0:
        eta0 = (solver.viscosity**3 / initial_eps)**0.25
        k_max_eta0 = (args.N/3.0) * eta0
    else:
        k_max_eta0 = np.inf
    
    print(f"  Initial k_max*eta: {k_max_eta0:.2f}")
    if k_max_eta0 < 1.0:
        print(f"  WARNING: Under-resolved initially")
    
    # Evolution loop
    print(f"\nStarting evolution...")
    print("-" * 60)
    
    step = 0
    start_time = time.time()
    prev_energy = E0
    prev_time = 0.0
    
    # Schedule for spectrum saves (key times for decay)
    spectrum_times = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 12.0, 15.0]
    spectrum_times = [t for t in spectrum_times if t <= args.time]
    next_spectrum_idx = 0
    
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
        
        # Save time series
        checkpointer.append_time_series(diag)
        
        # Save checkpoint
        if args.save_fields and (step % args.save_every == 0 or step == 1):
            filename = checkpointer.save_checkpoint(step, solver)
            if step == 1 or step % 1000 == 0:
                print(f"  Checkpoint saved: {os.path.basename(filename)}")
        
        # Save spectrum at scheduled times
        if next_spectrum_idx < len(spectrum_times):
            if solver.current_time >= spectrum_times[next_spectrum_idx]:
                filename = checkpointer.save_spectrum(step, solver)
                print(f"  Spectrum saved at t={spectrum_times[next_spectrum_idx]:.1f}")
                next_spectrum_idx += 1
        
        # Progress reporting
        if step % 100 == 0:
            elapsed = time.time() - start_time
            progress = (solver.current_time / args.time) * 100
            print(f"Step {step:5d} ({progress:5.1f}%):")
            print(f"  t={solver.current_time:.4f}, dt={solver.dt:.3e}")
            print(f"  E/E0={diag['energy']/E0:.6f}, dissipation={diag['dissipation']:.4f}")
            print(f"  Re_lambda={diag['Re_lambda']:.1f}, k_max*eta={diag['k_max_eta']:.2f}")
            
            # Warn if budget error is large
            if diag['budget_relative'] > 0.1:
                print(f"  WARNING: Budget error = {diag['budget_relative']:.1%}")
        
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
    final_energy_ratio = diag['energy'] / E0
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"Total runtime: {elapsed_total:.1f} seconds ({elapsed_total/60:.1f} minutes)")
    print(f"Total steps: {step}")
    print(f"Final time: {solver.current_time:.4f} / {args.time}")
    
    print(f"\nKey results:")
    print(f"  Energy decay: E/E0 = {final_energy_ratio:.4f}")
    print(f"  Final Re_lambda: {diag['Re_lambda']:.1f}")
    print(f"  Final spectrum slope: {diag['spectrum_slope']:.2f}")
    print(f"  Resolution: k_max*eta = {diag['k_max_eta']:.2f}")
    
    print(f"\nResults saved to: {output_dir}/")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Isotropic turbulence decay test with comprehensive data export',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Grid and physics
    parser.add_argument('--N', type=int, default=256,
                       help='Grid resolution (N x N x N)')
    parser.add_argument('--Re', type=float, default=1000,
                       help='Reynolds number')
    
    # Initial conditions
    parser.add_argument('--E-target', type=float, default=0.15,
                       help='Target initial energy')
    parser.add_argument('--k-peak', type=float, default=3.0,
                       help='Peak wavenumber of initial spectrum')
    parser.add_argument('--k-cutoff', type=float, default=10.0,
                       help='Cutoff wavenumber')
    
    # Time integration
    parser.add_argument('--time', type=float, default=15.0,
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