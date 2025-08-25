#!/usr/bin/env python3
"""
test_kelvin_helmholtz.py
=========================
Kelvin-Helmholtz shear layer instability test for BKM solver validation.

This test simulates the development of Kelvin-Helmholtz instability
in a shear layer, demonstrating the solver's ability to capture
instability growth and transition to turbulence.

Reference:
    Brown, G. L., & Roshko, A. (1974). On density effects and large
    structure in turbulent mixing layers. JFM, 64(4), 775-816.
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


def initialize_shear_layer(solver, thickness=0.03, amplitude=0.05, U0=3.0):
    """
    Initialize Kelvin-Helmholtz shear layer.
    
    Parameters:
    -----------
    solver : CUDABKMSolver
        Solver instance
    thickness : float
        Shear layer thickness (fraction of domain)
    amplitude : float
        Perturbation amplitude
    U0 : float
        Velocity difference across layer
    
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
    Lx, Ly = solver.Lx, solver.Ly
    
    # Create 2D coordinates (extend to 3D)
    x = xp.linspace(0, Lx, nx, endpoint=False)
    y = xp.linspace(0, Ly, ny, endpoint=False)
    X, Y = xp.meshgrid(x, y, indexing='ij')
    
    # Base shear profile (tanh)
    delta = thickness * Ly
    y0 = Ly / 2.0
    U_base = U0 * xp.tanh((Y - y0) / delta)
    
    # Envelope for perturbation
    envelope = xp.exp(-((Y - y0)**2) / (2.0 * delta**2))
    
    # Single-mode perturbation (m=2 fastest growing)
    m = 2
    kx = 2.0 * xp.pi * m / Lx
    phi = xp.random.uniform(0, 2*xp.pi)
    A = amplitude * U0
    psi = A * envelope * xp.sin(kx * X + phi)
    
    # Divergence-free perturbation from streamfunction
    dy = Ly / ny
    dx = Lx / nx
    u_p = xp.gradient(psi, dy, axis=1)
    v_p = -xp.gradient(psi, dx, axis=0)
    
    # Compose velocity field
    u = U_base + u_p
    v = v_p
    w = xp.zeros_like(u)
    
    # Extend to 3D
    if nz > 1:
        u = xp.tile(u[:, :, xp.newaxis], (1, 1, nz))
        v = xp.tile(v[:, :, xp.newaxis], (1, 1, nz))
        w = xp.tile(w[:, :, xp.newaxis], (1, 1, nz))
        
        # Add small 3D perturbation
        z = xp.linspace(0, solver.Lz, nz, endpoint=False)
        Z = z[xp.newaxis, xp.newaxis, :]
        w = w + 0.1 * amplitude * U0 * envelope[:, :, xp.newaxis] * \
            xp.sin(2*xp.pi*Z/solver.Lz)
    else:
        u = u[:, :, xp.newaxis]
        v = v[:, :, xp.newaxis]
        w = w[:, :, xp.newaxis]
    
    # Convert to float64 and project
    solver.u = u.astype(xp.float64)
    solver.v = v.astype(xp.float64)
    solver.w = w.astype(xp.float64)
    
    solver.u, solver.v, solver.w = solver.project_div_free(solver.u, solver.v, solver.w)
    
    # Compute initial diagnostics
    E0 = solver.compute_energy()
    W0 = solver.compute_max_vorticity()
    Z0 = solver.compute_enstrophy()
    
    return E0, W0, Z0


def detect_kh_phase(vort_ratio, v_rms_ratio, current_phase="initial"):
    """
    Detect Kelvin-Helmholtz instability phase.
    
    Phases: initial -> growth -> rollup -> breakdown -> mixing
    """
    if current_phase == "initial":
        if vort_ratio > 1.1 or v_rms_ratio > 1.5:
            return "growth"
    elif current_phase == "growth":
        if vort_ratio > 2.0 or v_rms_ratio > 3.0:
            return "rollup"
    elif current_phase == "rollup":
        if vort_ratio > 3.0:
            return "breakdown"
    elif current_phase == "breakdown":
        if vort_ratio < 2.5:  # Vorticity starts to decrease
            return "mixing"
    
    return current_phase


def run_test(args):
    """
    Run Kelvin-Helmholtz test with standardized data export.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    
    print("\n" + "="*70)
    print("KELVIN-HELMHOLTZ SHEAR LAYER TEST")
    print("="*70)
    print(f"Configuration:")
    print(f"  Grid: {args.N}^3")
    print(f"  Reynolds number: {args.Re}")
    print(f"  Target time: {args.time}")
    print(f"  Shear layer thickness: {args.thickness}")
    print(f"  Perturbation amplitude: {args.amplitude}")
    print(f"  Save fields: {args.save_fields}")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"kelvin_helmholtz_N{args.N}_Re{args.Re}_{timestamp}"
    
    # Initialize checkpointer
    params = {
        'Re': args.Re,
        'nu': 1.0/args.Re,
        'grid_size': args.N,
        'target_time': args.time,
        'thickness': args.thickness,
        'amplitude': args.amplitude,
        'U0': args.velocity
    }
    
    checkpointer = UnifiedCheckpointer(
        output_dir=output_dir,
        test_name='kelvin_helmholtz',
        grid_shape=(args.N, args.N, args.N),
        params=params
    )
    
    # Create solver
    print("\nInitializing solver...")
    
    # KH-specific settings
    if args.N == 128:
        dt_init = 1.6e-3
        dt_max = 2.0e-3
    elif args.N == 256:
        dt_init = 1.2e-3
        dt_max = 1.6e-3
    else:
        dt_init = 1.0e-3
        dt_max = 1.5e-3
    
    solver = CUDABKMSolver(
        grid_size=args.N,
        reynolds_number=args.Re,
        dt=dt_init,
        adapt_dt=True,
        CFL_target=0.25,
        use_dealiasing=True,
        dt_min=1e-4,
        dt_max=dt_max,
        dt_max_startup=dt_max,
        startup_steps=20,
        growth_cap_startup=1.5,
        projection_tol=1e-9,
        extra_projection_iters=1,
        divergence_threshold=1e-6
    )
    
    print(f"  GPU: {'ENABLED' if solver.use_gpu else 'DISABLED'}")
    print(f"  Viscosity: {solver.viscosity:.6e}")
    
    # Initialize flow
    print("\nInitializing shear layer...")
    E0, W0, Z0 = initialize_shear_layer(
        solver,
        thickness=args.thickness,
        amplitude=args.amplitude,
        U0=args.velocity
    )
    
    print(f"  Initial energy: {E0:.6f}")
    print(f"  Initial max vorticity: {W0:.3f}")
    print(f"  Initial enstrophy: {Z0:.6f}")
    
    # Initial v_rms (cross-stream velocity)
    xp = solver.xp
    v_rms_0 = float(xp.sqrt(xp.mean(solver.v**2)).get() if solver.use_gpu 
                   else xp.sqrt(xp.mean(solver.v**2)))
    print(f"  Initial v_rms: {v_rms_0:.6f}")
    
    # Evolution loop
    print(f"\nStarting evolution...")
    print("-" * 60)
    
    step = 0
    start_time = time.time()
    prev_energy = E0
    prev_time = 0.0
    
    # Track KH development
    current_phase = "initial"
    phase_transitions = []
    max_v_rms = v_rms_0
    max_vorticity = W0
    
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
        
        # KH-specific: track v_rms growth
        v_rms = diag['v_rms']
        v_rms_ratio = v_rms / v_rms_0 if v_rms_0 > 0 else 1.0
        vort_ratio = diag['vorticity_max'] / W0 if W0 > 0 else 1.0
        
        if v_rms > max_v_rms:
            max_v_rms = v_rms
        if diag['vorticity_max'] > max_vorticity:
            max_vorticity = diag['vorticity_max']
        
        # Phase detection
        new_phase = detect_kh_phase(vort_ratio, v_rms_ratio, current_phase)
        if new_phase != current_phase:
            phase_transitions.append((solver.current_time, current_phase, new_phase))
            print(f"\n  PHASE TRANSITION: {current_phase} -> {new_phase} at t={solver.current_time:.3f}")
            print(f"  v_rms ratio: {v_rms_ratio:.2f}, vort ratio: {vort_ratio:.2f}\n")
            current_phase = new_phase
        
        # Save time series
        checkpointer.append_time_series(diag)
        
        # Save checkpoint
        if args.save_fields and (step % args.save_every == 0 or step == 1):
            filename = checkpointer.save_checkpoint(step, solver)
            if step == 1 or (step % 1000 == 0):
                print(f"  Checkpoint: {os.path.basename(filename)}")
        
        # Save spectrum during rollup phase
        if current_phase in ["rollup", "breakdown"] and step % 500 == 0:
            filename = checkpointer.save_spectrum(step, solver)
        
        # Progress reporting
        if step % 100 == 0:
            elapsed = time.time() - start_time
            progress = (solver.current_time / args.time) * 100
            print(f"Step {step:5d} ({progress:5.1f}%) - Phase: {current_phase}")
            print(f"  t={solver.current_time:.4f}, dt={solver.dt:.3e}")
            print(f"  v_rms growth: {v_rms_ratio:.2f}x, vort: {vort_ratio:.2f}x")
            print(f"  E/E0={diag['energy']/E0:.6f}, Re_lambda={diag['Re_lambda']:.1f}")
        
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
    v_rms_growth = max_v_rms / v_rms_0 if v_rms_0 > 0 else 1.0
    vort_growth = max_vorticity / W0 if W0 > 0 else 1.0
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"Total runtime: {elapsed_total:.1f} seconds ({elapsed_total/60:.1f} minutes)")
    print(f"Total steps: {step}")
    print(f"Final time: {solver.current_time:.4f} / {args.time}")
    
    print(f"\nKey results:")
    print(f"  Maximum v_rms growth: {v_rms_growth:.2f}x")
    print(f"  Maximum vorticity growth: {vort_growth:.2f}x")
    print(f"  Final phase: {current_phase}")
    
    if phase_transitions:
        print(f"\nPhase transitions:")
        for t, old_phase, new_phase in phase_transitions:
            print(f"  t={t:.3f}: {old_phase} -> {new_phase}")
    
    print(f"\nResults saved to: {output_dir}/")
    
    # Success criteria
    success = (v_rms_growth >= 1.5 or vort_growth >= 1.1) and solver.current_time >= 0.3 * args.time
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description='Kelvin-Helmholtz shear layer test with comprehensive data export',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Grid and physics
    parser.add_argument('--N', type=int, default=128,
                       help='Grid resolution (N x N x N)')
    parser.add_argument('--Re', type=float, default=800,
                       help='Reynolds number')
    
    # Shear layer parameters
    parser.add_argument('--thickness', type=float, default=0.03,
                       help='Shear layer thickness (fraction of domain)')
    parser.add_argument('--amplitude', type=float, default=0.05,
                       help='Perturbation amplitude')
    parser.add_argument('--velocity', type=float, default=3.0,
                       help='Velocity difference across layer')
    
    # Time integration
    parser.add_argument('--time', type=float, default=1.0,
                       help='Target simulation time')
    parser.add_argument('--max-steps', type=int, default=100000,
                       help='Maximum number of steps')
    
    # Output control
    parser.add_argument('--save-fields', action='store_true',
                       help='Save full 3D velocity and vorticity fields')
    parser.add_argument('--save-every', type=int, default=50,
                       help='Steps between field checkpoints')
    
    args = parser.parse_args()
    
    success = run_test(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()