#!/usr/bin/env python3
"""
custom_initial_condition.py
============================
Demonstrates how to use custom initial conditions with the BKM Engine.

Shows three examples:
1. Lamb-Chaplygin dipole
2. Random turbulent field
3. Loaded from file
"""

import sys
import os
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.unified_bkm_engine import CUDABKMSolver


def lamb_chaplygin_dipole(solver, radius=1.0, strength=2.0):
    """
    Initialize a Lamb-Chaplygin dipole vortex.
    
    Two counter-rotating vortices that translate together.
    """
    N = solver.nx
    L = 2 * np.pi
    xp = solver.xp
    
    # Create coordinate arrays
    x = xp.linspace(0, L, N, endpoint=False)
    y = xp.linspace(0, L, N, endpoint=False)
    z = xp.linspace(0, L, N, endpoint=False)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    # Dipole centers
    y1 = L/2 + radius/2
    y2 = L/2 - radius/2
    x_center = L/2
    z_center = L/2
    
    # Distance from vortex centers
    r1 = xp.sqrt((X - x_center)**2 + (Y - y1)**2)
    r2 = xp.sqrt((X - x_center)**2 + (Y - y2)**2)
    
    # Gaussian vortices
    sigma = radius / 3
    psi1 = strength * xp.exp(-(r1/sigma)**2)
    psi2 = -strength * xp.exp(-(r2/sigma)**2)
    psi = psi1 + psi2
    
    # Velocity from streamfunction (2D, extended to 3D)
    # u = -∂ψ/∂y, v = ∂ψ/∂x
    psi_hat = xp.fft.fftn(psi)
    kx = xp.fft.fftfreq(N, L/N) * 2*xp.pi
    ky = kx.copy()
    KX, KY, _ = xp.meshgrid(kx, ky, kx, indexing='ij')
    
    solver.u = -xp.real(xp.fft.ifftn(1j * KY * psi_hat))
    solver.v = xp.real(xp.fft.ifftn(1j * KX * psi_hat))
    solver.w = xp.zeros_like(solver.u)
    
    # Add small 3D perturbation
    solver.w += 0.01 * xp.sin(2*Z/L) * xp.exp(-(r1/sigma)**2)
    
    # Project to ensure divergence-free
    solver.u, solver.v, solver.w = solver.project_div_free(
        solver.u, solver.v, solver.w
    )
    
    print(f"Initialized Lamb-Chaplygin dipole:")
    print(f"  Radius: {radius}")
    print(f"  Strength: {strength}")
    print(f"  Initial energy: {solver.compute_energy():.6f}")


def random_turbulent_field(solver, energy_level=0.5, peak_k=4):
    """
    Initialize random turbulent field with prescribed spectrum.
    """
    N = solver.nx
    L = 2 * np.pi
    xp = solver.xp
    
    # Wavenumber space
    k = xp.fft.fftfreq(N, L/N) * 2*xp.pi
    kx, ky, kz = xp.meshgrid(k, k, k, indexing='ij')
    k_mag = xp.sqrt(kx**2 + ky**2 + kz**2)
    
    # Energy spectrum E(k) ~ k^4 * exp(-2*(k/k_peak)^2)
    E_k = xp.zeros_like(k_mag)
    mask = k_mag > 0
    E_k[mask] = (k_mag[mask]**4) * xp.exp(-2*(k_mag[mask]/peak_k)**2)
    
    # Random phases
    phase_u = xp.random.uniform(-xp.pi, xp.pi, (N, N, N))
    phase_v = xp.random.uniform(-xp.pi, xp.pi, (N, N, N))
    phase_w = xp.random.uniform(-xp.pi, xp.pi, (N, N, N))
    
    # Velocity in Fourier space
    u_hat = xp.sqrt(E_k) * xp.exp(1j * phase_u)
    v_hat = xp.sqrt(E_k) * xp.exp(1j * phase_v)
    w_hat = xp.sqrt(E_k) * xp.exp(1j * phase_w)
    
    # Project to divergence-free
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1.0
    k_dot_u = kx*u_hat + ky*v_hat + kz*w_hat
    u_hat -= kx * k_dot_u / k2
    v_hat -= ky * k_dot_u / k2
    w_hat -= kz * k_dot_u / k2
    
    # Transform to physical space
    solver.u = xp.real(xp.fft.ifftn(u_hat))
    solver.v = xp.real(xp.fft.ifftn(v_hat))
    solver.w = xp.real(xp.fft.ifftn(w_hat))
    
    # Rescale to target energy
    current_energy = solver.compute_energy()
    scale = xp.sqrt(energy_level / current_energy) if current_energy > 0 else 1
    solver.u *= scale
    solver.v *= scale
    solver.w *= scale
    
    print(f"Initialized random turbulent field:")
    print(f"  Peak wavenumber: {peak_k}")
    print(f"  Target energy: {energy_level}")
    print(f"  Actual energy: {solver.compute_energy():.6f}")


def load_from_file(solver, filename):
    """
    Load initial condition from HDF5 file.
    """
    import h5py
    
    print(f"Loading initial condition from: {filename}")
    
    try:
        with h5py.File(filename, 'r') as f:
            # Check if velocity fields exist
            if 'velocity' in f:
                u = f['velocity/u'][:]
                v = f['velocity/v'][:]
                w = f['velocity/w'][:]
            elif 'u' in f:
                u = f['u'][:]
                v = f['v'][:]
                w = f['w'][:]
            else:
                raise ValueError("No velocity fields found in file")
            
            # Transfer to GPU if needed
            xp = solver.xp
            solver.u = xp.asarray(u)
            solver.v = xp.asarray(v)
            solver.w = xp.asarray(w)
            
            # Verify grid size
            if solver.u.shape != (solver.nx, solver.ny, solver.nz):
                raise ValueError(f"Grid size mismatch: file has {u.shape}, "
                               f"solver expects ({solver.nx}, {solver.ny}, {solver.nz})")
            
            print(f"  ✓ Loaded successfully")
            print(f"  Energy: {solver.compute_energy():.6f}")
            
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Custom initial conditions demonstration'
    )
    parser.add_argument('--type', choices=['dipole', 'random', 'file'],
                       default='dipole', help='Type of initial condition')
    parser.add_argument('--file', type=str, help='HDF5 file for loading')
    parser.add_argument('--N', type=int, default=128, help='Grid resolution')
    parser.add_argument('--Re', type=float, default=1000, help='Reynolds number')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps')
    args = parser.parse_args()
    
    print("="*70)
    print("CUSTOM INITIAL CONDITIONS DEMONSTRATION")
    print("="*70)
    
    # Create solver
    print(f"\nCreating solver: {args.N}³ grid, Re={args.Re}")
    solver = CUDABKMSolver(
        grid_size=args.N,
        reynolds_number=args.Re,
        dt=0.001,
        adapt_dt=True
    )
    
    # Set initial condition
    print(f"\nInitializing: {args.type}")
    if args.type == 'dipole':
        lamb_chaplygin_dipole(solver)
    elif args.type == 'random':
        random_turbulent_field(solver)
    elif args.type == 'file':
        if not args.file:
            print("Error: --file required for file type")
            return
        load_from_file(solver, args.file)
    
    # Check divergence
    div_max, div_l2 = solver.compute_divergence_metrics()
    print(f"  Divergence: max={div_max:.2e}, L2={div_l2:.2e}")
    
    # Run simulation
    print(f"\nEvolving for {args.steps} steps...")
    E0 = solver.compute_energy()
    
    for step in range(args.steps):
        metrics = solver.evolve_one_timestep(use_rk2=True)
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}: t={solver.current_time:.3f}, "
                  f"E/E0={metrics['energy']/E0:.4f}, "
                  f"ω_max={metrics['vorticity']:.2f}")
    
    print(f"\nFinal state:")
    print(f"  Time: {solver.current_time:.4f}")
    print(f"  Energy ratio: {solver.compute_energy()/E0:.6f}")
    print(f"  Max divergence: {solver.compute_divergence_metrics()[0]:.2e}")
    
    # Save option
    save = input("\nSave final state? (y/N): ").lower() == 'y'
    if save:
        import h5py
        filename = f"custom_{args.type}_final.h5"
        xp = solver.xp
        
        with h5py.File(filename, 'w') as f:
            f.attrs['time'] = solver.current_time
            f.attrs['Re'] = args.Re
            f.attrs['grid_size'] = args.N
            
            # Transfer from GPU if needed
            u = xp.asnumpy(solver.u) if solver.use_gpu else solver.u
            v = xp.asnumpy(solver.v) if solver.use_gpu else solver.v
            w = xp.asnumpy(solver.w) if solver.use_gpu else solver.w
            
            f.create_dataset('velocity/u', data=u)
            f.create_dataset('velocity/v', data=v)
            f.create_dataset('velocity/w', data=w)
        
        print(f"  Saved to: {filename}")


if __name__ == "__main__":
    main()