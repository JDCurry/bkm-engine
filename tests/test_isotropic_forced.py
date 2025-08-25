#!/usr/bin/env python3
"""
test_isotropic_forced.py
========================
Forced isotropic turbulence test for BKM solver validation.

This test simulates forced isotropic turbulence using a fraction-controlled
PI controller that maintains spectral shape while allowing natural energy decay.

Reference:
    Lundgren, T. S. (2003). Linearly forced isotropic turbulence.
    Annual Research Briefs, Center for Turbulence Research, 461-473.
"""

import numpy as np
import time
import os
import sys
import argparse
from datetime import datetime

# Import unified components
from unified_bkm_engine import CUDABKMSolver
from test_base import (
    UnifiedCheckpointer, 
    compute_comprehensive_diagnostics,
    generate_isotropic_field,
    compute_budget_residual,
    get_spectrum_save_schedule,
    FractionController,
    apply_spectral_forcing
)


def run_test(args):
    """
    Run forced isotropic turbulence test with standardized data export.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    
    print("\n" + "="*70)
    print("FORCED ISOTROPIC TURBULENCE TEST")
    print("="*70)
    print(f"Configuration:")
    print(f"  Grid: {args.N}^3")
    print(f"  Reynolds number: {args.Re}")
    print(f"  Target time: {args.time}")
    print(f"  Forcing variant: {args.variant}")
    print(f"  Save fields: {args.save_fields}")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"isotropic_forced_N{args.N}_Re{args.Re}_{timestamp}"
    
    # Initialize checkpointer
    params = {
        'Re': args.Re,
        'nu': 1.0/args.Re,
        'grid_size': args.N,
        'target_time': args.time,
        'forcing_variant': args.variant,
        'k_min': args.k_min,
        'k_max': args.k_max,
        'f_band': args.f_band
    }
    
    checkpointer = UnifiedCheckpointer(
        output_dir=output_dir,
        test_name='isotropic_forced',
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
    if args.N == 128:
        generate_isotropic_field(solver, k_peak=4, E_target=0.08, k_cutoff=15)
    else:
        generate_isotropic_field(solver, k_peak=6, E_target=0.1, k_cutoff=20)
    
    E0 = solver.compute_energy()
    W0 = solver.compute_max_vorticity()
    Z0 = solver.compute_enstrophy()
    
    print(f"  Initial energy: {E0:.6f}")
    print(f"  Initial max vorticity: {W0:.3f}")
    print(f"  Initial enstrophy: {Z0:.6f}")
    
    # Setup forcing controller based on variant
    if args.variant == 'A':
        controller = FractionController(
            f_band=args.f_band,
            c_fracP=2.0,
            c_fracI=0.2,
            alpha_max=0.25
        )
        apply_sponge = False
        print(f"\nForcing configuration (Variant A - fraction only):")
    elif args.variant == 'B':
        controller = FractionController(
            f_band=args.f_band,
            c_fracP=2.0,
            c_fracI=0.2,
            alpha_max=0.20
        )
        apply_sponge = False
        print(f"\nForcing configuration (Variant B - with cap):")
    else:  # variant 'C'
        controller = FractionController(
            f_band=args.f_band,
            c_fracP=2.0,
            c_fracI=0.2,
            alpha_max=0.15
        )
        apply_sponge = True
        print(f"\nForcing configuration (Variant C - with sponge):")
    
    print(f"  Forcing band: k in [{args.k_min}, {args.k_max}]")
    print(f"  Target fraction: {args.f_band:.0%}")
    print(f"  PI gains: Kp={controller.c_fracP:.1f}, Ki={controller.c_fracI:.2f}")
    
    # Evolution loop
    print(f"\nStarting evolution...")
    print("-" * 60)
    
    step = 0
    start_time = time.time()
    prev_energy = E0
    prev_time = 0.0
    
    # Forcing parameters
    dt_force = 0.05  # Apply forcing every 0.05 time units
    next_force_time = dt_force
    tau_ramp = 1.0  # Ramp forcing over first 1.0 time units
    
    # Track forcing state
    current_forcing = {
        'alpha': 0.0,
        'E_band': 0.0,
        'f_error': 0.0,
        'P_inj': 0.0,
        'band_fraction': 0.0
    }
    
    # Schedule spectrum saves
    spectrum_times = get_spectrum_save_schedule(args.time, 'forced')
    next_spectrum_idx = 0
    
    while solver.current_time < args.time and step < args.max_steps:
        step += 1
        
        # Adaptive timestep
        if solver.adapt_dt:
            dt_new, cfl, rho, limiter, _ = solver.choose_dt()
            solver.dt = dt_new
        
        # Evolve one timestep
        result = solver.evolve_one_timestep(use_rk2=True)
        
        # Apply forcing at intervals
        if solver.current_time >= next_force_time:
            ramp_factor = min(1.0, solver.current_time / tau_ramp)
            
            alpha, E_band, f_error, E_band_target, eps_smooth, f_current, P_inj = \
                apply_spectral_forcing(
                    solver, controller, 
                    k_min=args.k_min, k_max=args.k_max,
                    dt_force=dt_force, 
                    apply_sponge=apply_sponge, 
                    ramp_factor=ramp_factor
                )
            
            current_forcing = {
                'alpha': alpha,
                'E_band': E_band,
                'f_error': f_error,
                'P_inj': P_inj,
                'band_fraction': f_current
            }
            
            next_force_time += dt_force
        
        # Compute comprehensive diagnostics
        diag = compute_comprehensive_diagnostics(solver, prev_energy, prev_time)
        diag['step'] = step
        diag['time'] = solver.current_time
        diag['dt'] = solver.dt
        diag['cfl'] = getattr(solver, 'last_cfl', cfl if 'cfl' in locals() else 0)
        
        # Add forcing metrics
        diag['forcing_alpha'] = current_forcing['alpha']
        diag['forcing_P_inj'] = current_forcing['P_inj']
        diag['band_fraction'] = current_forcing['band_fraction']
        
        # Correct budget for forcing
        residual, relative_error = compute_budget_residual(
            (diag['energy'] - prev_energy) / (solver.current_time - prev_time) if solver.current_time > prev_time else 0,
            diag['dissipation'],
            current_forcing['P_inj']
        )
        diag['budget_residual'] = residual
        diag['budget_relative'] = relative_error
        
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
            print(f"  E={diag['energy']:.4f}, epsilon={diag['dissipation']:.4f}")
            print(f"  Band fraction: {current_forcing['band_fraction']:.1%} (target: {args.f_band:.0%})")
            print(f"  Forcing: alpha={current_forcing['alpha']:+.6f}, P_inj={current_forcing['P_inj']:.6f}")
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
    print(f"  Energy: E/E0 = {diag['energy']/E0:.4f}")
    print(f"  Final Re_lambda: {diag['Re_lambda']:.1f}")
    print(f"  Final band fraction: {current_forcing['band_fraction']:.1%}")
    print(f"  Final spectrum slope: {diag.get('spectrum_slope', -5/3):.2f}")
    
    print(f"\nResults saved to: {output_dir}/")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Forced isotropic turbulence test with comprehensive data export',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Grid and physics
    parser.add_argument('--N', type=int, default=256,
                       help='Grid resolution (N x N x N)')
    parser.add_argument('--Re', type=float, default=2000,
                       help='Reynolds number')
    
    # Forcing parameters
    parser.add_argument('--variant', type=str, default='A',
                       choices=['A', 'B', 'C'],
                       help='Forcing variant: A=fraction only, B=with cap, C=with sponge')
    parser.add_argument('--k-min', type=float, default=2,
                       help='Minimum forcing wavenumber')
    parser.add_argument('--k-max', type=float, default=5,
                       help='Maximum forcing wavenumber')
    parser.add_argument('--f-band', type=float, default=0.20,
                       help='Target fraction of energy in forcing band')
    
    # Time integration
    parser.add_argument('--time', type=float, default=40.0,
                       help='Target simulation time')
    parser.add_argument('--max-steps', type=int, default=100000,
                       help='Maximum number of steps')
    
    # Output control
    parser.add_argument('--save-fields', action='store_true',
                       help='Save full 3D velocity and vorticity fields')
    parser.add_argument('--save-every', type=int, default=100,
                       help='Steps between field checkpoints')
    
    # Quick test mode
    parser.add_argument('--quick', action='store_true',
                       help='Quick 5-second validation test')
    
    args = parser.parse_args()
    
    # Override for quick test
    if args.quick:
        args.time = 5.0
        print("Quick test mode: 5-second validation")
    
    success = run_test(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()