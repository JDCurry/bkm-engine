#!/usr/bin/env python3
"""
test_isotropic_decay.py
=======================
Test isotropic turbulence decay with stability improvements

Key improvements from stable version:
- Conservative solver parameters to prevent guard triggering
- Smoother initial conditions option
- Adaptive spectrum fitting based on Re_lambda
- Better vorticity spike handling
- Fixed undefined variable errors
"""

import numpy as np
import time
import os
import h5py
from datetime import datetime
import json
import argparse
import sys

# Import the BKM engine
from unified_bkm_engine import CUDABKMSolver

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_divide(a, b, default=0.0):
    """Safe division with default for zero denominator"""
    return a / b if abs(b) > 1e-30 else default

def generate_smooth_initial_field(N, E_target=0.05, use_gpu=False):
    """
    Generate smoother initial conditions without sharp spectral cutoff
    This avoids the high vorticity spikes that trigger the guard system.
    """
    if use_gpu:
        try:
            import cupy as xp
        except ImportError:
            import numpy as xp
    else:
        import numpy as xp
    
    # Simple Taylor-Green-like initial condition with multiple modes
    L = 2 * np.pi
    x = xp.linspace(0, L, N, endpoint=False)
    X, Y, Z = xp.meshgrid(x, x, x, indexing='ij')
    
    # Initialize fields
    u = xp.zeros((N, N, N))
    v = xp.zeros((N, N, N))
    w = xp.zeros((N, N, N))
    
    # Random number generator
    rng = xp.random.RandomState(42)
    
    # Add a few low wavenumber modes for isotropy
    for k in [1, 2, 3]:
        # Random phases for each mode
        phase_x = rng.rand() * 2 * np.pi
        phase_y = rng.rand() * 2 * np.pi
        phase_z = rng.rand() * 2 * np.pi
        
        # Random amplitudes that decay with wavenumber
        amp = rng.rand() * np.exp(-k/3.0)
        
        # Add contribution from this mode
        u += amp * xp.sin(k*X + phase_x) * xp.cos(k*Y) * xp.cos(k*Z)
        v += amp * xp.cos(k*X) * xp.sin(k*Y + phase_y) * xp.cos(k*Z)
        w += amp * xp.cos(k*X) * xp.cos(k*Y) * xp.sin(k*Z + phase_z)
    
    # Make divergence-free by projection
    u_hat = xp.fft.fftn(u)
    v_hat = xp.fft.fftn(v)
    w_hat = xp.fft.fftn(w)
    
    k = xp.fft.fftfreq(N, L/N) * 2*np.pi
    kx, ky, kz = xp.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0  # Avoid division by zero
    
    # Project to divergence-free
    kdotu = kx * u_hat + ky * v_hat + kz * w_hat
    u_hat -= kx * kdotu / k2
    v_hat -= ky * kdotu / k2
    w_hat -= kz * kdotu / k2
    
    # Transform back to physical space
    u = xp.fft.ifftn(u_hat).real
    v = xp.fft.ifftn(v_hat).real
    w = xp.fft.ifftn(w_hat).real
    
    # Normalize to target energy
    E_current = 0.5 * xp.mean(u**2 + v**2 + w**2)
    if E_current > 0:
        scale = xp.sqrt(E_target / E_current)
        u *= scale
        v *= scale
        w *= scale
    
    return u, v, w

def generate_isotropic_field(N, k_peak=4, E_target=0.05, k_cutoff=None, use_gpu=False):
    """
    Generate initial isotropic turbulence field with specified spectrum
    Updated with lower defaults for stability
    """
    if use_gpu:
        try:
            import cupy as xp
        except ImportError:
            import numpy as xp
    else:
        import numpy as xp
    
    # Wavenumber grid
    L = 2 * np.pi
    k = xp.fft.fftfreq(N, L/N) * 2*np.pi
    kx, ky, kz = xp.meshgrid(k, k, k, indexing='ij')
    k_mag = xp.sqrt(kx**2 + ky**2 + kz**2)
    
    # Energy spectrum E(k) ~ k^4 exp(-2(k/k_peak)^2)
    E_k = xp.zeros_like(k_mag)
    mask = k_mag > 0
    E_k[mask] = (k_mag[mask]**4) * xp.exp(-2*(k_mag[mask]/k_peak)**2)
    
    # Apply smooth cutoff if specified
    if k_cutoff is not None:
        cutoff_width = 2.0
        E_k *= 0.5 * (1 - xp.tanh((k_mag - k_cutoff)/cutoff_width))
    
    # Random phases for each component
    rng = xp.random.RandomState(42)
    phases_u = rng.uniform(0, 2*np.pi, (N, N, N))
    phases_v = rng.uniform(0, 2*np.pi, (N, N, N))
    phases_w = rng.uniform(0, 2*np.pi, (N, N, N))
    
    # Create velocity field in Fourier space with random phases
    amplitude = xp.sqrt(E_k)
    u_hat = amplitude * xp.exp(1j * phases_u)
    v_hat = amplitude * xp.exp(1j * phases_v)
    w_hat = amplitude * xp.exp(1j * phases_w)
    
    # Ensure zero mean flow
    u_hat[0, 0, 0] = 0
    v_hat[0, 0, 0] = 0
    w_hat[0, 0, 0] = 0
    
    # Project to divergence-free
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0
    
    kdotu = kx * u_hat + ky * v_hat + kz * w_hat
    u_hat -= kx * kdotu / k2
    v_hat -= ky * kdotu / k2
    w_hat -= kz * kdotu / k2
    
    # Transform to physical space
    u = xp.fft.ifftn(u_hat).real
    v = xp.fft.ifftn(v_hat).real
    w = xp.fft.ifftn(w_hat).real
    
    # Normalize to target energy
    E_current = 0.5 * xp.mean(u**2 + v**2 + w**2)
    if E_current > 0:
        scale = xp.sqrt(E_target / E_current)
        u *= scale
        v *= scale
        w *= scale
    
    return u, v, w

def compute_turbulence_statistics(solver):
    """Compute key turbulence statistics"""
    xp = solver.xp
    
    # RMS velocities
    u_rms = xp.sqrt(xp.mean(solver.u**2))
    v_rms = xp.sqrt(xp.mean(solver.v**2))
    w_rms = xp.sqrt(xp.mean(solver.w**2))
    u_total_rms = xp.sqrt(xp.mean(solver.u**2 + solver.v**2 + solver.w**2))
    
    # Energy and enstrophy
    energy = solver.compute_energy()
    enstrophy = solver.compute_enstrophy()
    
    # Dissipation rate
    epsilon = 2.0 * solver.viscosity * enstrophy
    
    # Taylor microscale and Reynolds number
    if epsilon > 0 and energy > 0:
        taylor_lambda = xp.sqrt(15 * solver.viscosity * energy / epsilon)
        Re_lambda = u_total_rms * taylor_lambda / solver.viscosity
    else:
        taylor_lambda = 0.0
        Re_lambda = 0.0
    
    # Integral scale (approximate)
    L_int = energy**(3/2) / epsilon if epsilon > 0 else 0.0
    
    return {
        'u_rms': float(u_rms),
        'v_rms': float(v_rms),
        'w_rms': float(w_rms),
        'u_total_rms': float(u_total_rms),
        'energy': float(energy),
        'enstrophy': float(enstrophy),
        'epsilon': float(epsilon),
        'taylor_lambda': float(taylor_lambda),
        'Re_lambda': float(Re_lambda),
        'L_int': float(L_int)
    }

def compute_energy_spectrum(solver):
    """
    Compute energy spectrum with adaptive slope fitting
    """
    xp = solver.xp
    N = solver.nx
    
    # FFT of velocity fields
    u_hat = xp.fft.fftn(solver.u)
    v_hat = xp.fft.fftn(solver.v)
    w_hat = xp.fft.fftn(solver.w)
    
    # Energy in spectral space
    E_k = 0.5 * (xp.abs(u_hat)**2 + xp.abs(v_hat)**2 + xp.abs(w_hat)**2) / N**6
    
    # Wavenumber grid
    k = xp.fft.fftfreq(N, 2*np.pi/N) * 2*np.pi
    kx, ky, kz = xp.meshgrid(k, k, k, indexing='ij')
    k_mag = xp.sqrt(kx**2 + ky**2 + kz**2)
    
    # Radial binning with mode counting
    k_max = float(xp.max(k_mag))
    n_bins = min(N // 2, 128)
    k_bins = xp.linspace(0, k_max, n_bins)
    E_spectrum = xp.zeros(n_bins - 1)
    mode_counts = xp.zeros(n_bins - 1)
    
    for i in range(n_bins - 1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        E_spectrum[i] = xp.sum(E_k[mask])
        mode_counts[i] = xp.sum(mask)
    
    # Convert to numpy for analysis
    if solver.use_gpu:
        k_bins = k_bins.get()
        E_spectrum = E_spectrum.get()
        mode_counts = mode_counts.get()
    else:
        k_bins = np.array(k_bins)
        E_spectrum = np.array(E_spectrum)
        mode_counts = np.array(mode_counts)
    
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    # Compute compensated spectrum
    E_compensated = E_spectrum * k_centers**(5/3)
    
    # Total energy check
    E_total = np.sum(E_spectrum)
    
    # Adaptive slope fitting based on Re_lambda
    slope = np.nan
    stats = compute_turbulence_statistics(solver)
    Re_lambda = stats['Re_lambda']
    
    # Only attempt to find inertial range if Re_lambda is high enough
    if Re_lambda > 20 and E_total > 0:
        # Find wavenumber range containing 15-70% of energy
        E_cumsum = np.cumsum(E_spectrum)
        E_cumsum_norm = E_cumsum / E_total
        
        k_lo_idx = np.where(E_cumsum_norm >= 0.15)[0]
        k_hi_idx = np.where(E_cumsum_norm >= 0.70)[0]
        
        if len(k_lo_idx) > 0 and len(k_hi_idx) > 0:
            k_lo_idx = k_lo_idx[0]
            k_hi_idx = k_hi_idx[0]
            
            # Build fit mask with minimum requirements
            E_floor = 1e-14 * E_total
            mode_min = 200 if N >= 256 else 50
            
            fit_mask = np.zeros_like(k_centers, dtype=bool)
            fit_mask[k_lo_idx:k_hi_idx+1] = True
            fit_mask &= (E_spectrum > E_floor)
            fit_mask &= (mode_counts >= mode_min)
            
            # Need at least 12 points for meaningful fit
            if np.sum(fit_mask) >= 12:
                log_k = np.log10(k_centers[fit_mask])
                log_E = np.log10(E_spectrum[fit_mask])
                try:
                    slope, _ = np.polyfit(log_k, log_E, 1)
                    # Sanity check on slope
                    if slope < -3 or slope > -1:
                        slope = np.nan
                except:
                    slope = np.nan
    
    return k_centers, E_spectrum, E_compensated, E_total, slope

def create_isotropic_solver(N, Re, use_stable_params=True):
    """
    Create and configure solver for isotropic turbulence simulation
    
    Parameters:
    -----------
    N : int
        Grid resolution
    Re : float
        Reynolds number
    use_stable_params : bool
        Use conservative parameters for stability
    """
    if use_stable_params:
        # Conservative parameters to avoid guard triggering
        solver = CUDABKMSolver(
            grid_size=N,
            reynolds_number=Re,
            dt=0.0001,
            adapt_dt=True,
            CFL_target=0.2,  # Very conservative
            track_alignment=True,
            use_dealiasing=True,
            dealias_fraction=2.0/3.0,
            rho_soft=10.0,  # Very relaxed
            rho_hard=20.0,  # Very relaxed
            dt_min=1e-6,
            dt_max=0.001,  # Conservative maximum
            projection_tol=3e-9,
            extra_projection_iters=0,
            divergence_threshold=1e-6,
            startup_steps=0,
            dt_max_startup=5e-4,
            growth_cap_startup=1.05,
            precision='mixed'  # Use mixed precision
        )
    else:
        # Standard parameters
        solver = CUDABKMSolver(
            grid_size=N,
            reynolds_number=Re,
            dt=0.001,
            adapt_dt=True,
            CFL_target=0.4,
            track_alignment=True,
            use_dealiasing=True,
            precision='mixed'
        )
    
    # Verify viscosity
    print(f"  Solver created: nu={solver.viscosity:.6e} (nu=1/Re convention)")
    
    return solver

def save_field_checkpoint(solver, filename, budget_residual=0.0, P_inj=0.0):
    """Save full 3D field checkpoint"""
    xp = solver.xp
    
    # Convert to numpy if on GPU
    if solver.use_gpu:
        u_save = solver.u.get()
        v_save = solver.v.get()
        w_save = solver.w.get()
    else:
        u_save = solver.u
        v_save = solver.v
        w_save = solver.w
    
    with h5py.File(filename, 'w') as f:
        # Save fields with compression
        f.create_dataset('u', data=u_save, compression='gzip', compression_opts=1)
        f.create_dataset('v', data=v_save, compression='gzip', compression_opts=1)
        f.create_dataset('w', data=w_save, compression='gzip', compression_opts=1)
        
        # Metadata
        f.attrs['time'] = solver.current_time
        f.attrs['total_steps'] = solver.total_steps
        f.attrs['energy'] = solver.compute_energy()
        f.attrs['enstrophy'] = solver.compute_enstrophy()
        f.attrs['max_vorticity'] = solver.compute_max_vorticity()
        f.attrs['bkm_integral'] = solver.bkm_integral
        f.attrs['budget_residual'] = budget_residual
        f.attrs['P_inj'] = P_inj
        
        # Compute and save statistics
        stats = compute_turbulence_statistics(solver)
        for key, val in stats.items():
            f.attrs[key] = val
    
    print(f"  Field checkpoint saved: {filename}")

# ============================================================================
# MAIN SIMULATION FUNCTION
# ============================================================================

def run_decay_simulation(N=256, Re=300, t_final=15.0, quick_test=False,
                        save_fields=False, field_save_interval=3.0, 
                        save_spectra=True, use_smooth_ic=True,
                        use_stable_params=True):
    """
    Run isotropic turbulence decay simulation with stability improvements
    """
    
    # Adjust parameters for resolution
    if N == 128:
        if Re > 300:
            Re = 300
            print(f"Note: Adjusted Re to {Re} for N=128 stability")
        E_init = 0.02
        k_peak_init = 2
    elif N == 256:
        if Re > 500:
            Re = 500
            print(f"Note: Adjusted Re to {Re} for stability")
        E_init = 0.03
        k_peak_init = 3
    else:
        E_init = 0.03
        k_peak_init = 3
    
    # Override for quick test
    if quick_test:
        t_final = 3.0
        print("\n" + "="*70)
        print("QUICK VALIDATION TEST - DECAY VERIFICATION")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("ISOTROPIC TURBULENCE DECAY SIMULATION")
        print("="*70)
    
    print(f"Resolution: {N}^3")
    print(f"Reynolds number: {Re}")
    print(f"Target time: {t_final}")
    print(f"Initial energy: {E_init}")
    print(f"Initial conditions: {'Smooth' if use_smooth_ic else 'Spectral'}")
    print(f"Stability mode: {'Conservative' if use_stable_params else 'Standard'}")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"decay_N{N}_Re{Re}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create solver
    solver = create_isotropic_solver(N, Re, use_stable_params)
    
    # Save simulation parameters
    params = {
        'simulation_type': 'isotropic_decay',
        'N': int(N),
        'Re': float(Re),
        'nu': float(solver.viscosity),
        'L': float(2*np.pi),
        't_final': float(t_final),
        'quick_test': quick_test,
        'use_smooth_ic': use_smooth_ic,
        'use_stable_params': use_stable_params,
        'initial_energy': E_init,
        'timestamp': timestamp
    }
    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        json.dump(params, f, indent=2)
    
    # Generate initial conditions
    if use_smooth_ic:
        print("Generating smooth initial conditions...")
        u, v, w = generate_smooth_initial_field(N, E_target=E_init, use_gpu=solver.use_gpu)
    else:
        print("Generating spectral initial conditions...")
        u, v, w = generate_isotropic_field(N, k_peak=k_peak_init, E_target=E_init,
                                          k_cutoff=None, use_gpu=solver.use_gpu)
    
    solver.u = u
    solver.v = v
    solver.w = w
    
    # Initial diagnostics
    E0 = solver.compute_energy()
    vort0 = solver.compute_max_vorticity()
    enst0 = solver.compute_enstrophy()
    stats0 = compute_turbulence_statistics(solver)
    
    # Check initial resolution
    eps0 = 2.0 * solver.viscosity * enst0
    if eps0 > 0:
        eta0 = (solver.viscosity**3 / eps0)**0.25
        k_max_eta0 = (N/3.0) * eta0
    else:
        k_max_eta0 = np.inf
    
    print(f"\nInitial conditions:")
    print(f"  Energy: {E0:.4f}")
    print(f"  Max vorticity: {vort0:.2f}")
    print(f"  Enstrophy: {enst0:.3f}")
    print(f"  Re_lambda: {stats0['Re_lambda']:.1f}")
    print(f"  k_max*eta: {k_max_eta0:.2f}")
    
    if k_max_eta0 < 1.0:
        print(f"  WARNING: Under-resolved (k_max*eta < 1)")
    
    if vort0 > 50:
        print(f"  WARNING: High initial vorticity")
    
    # Initialize time series storage
    time_series = {
        't': [],
        'energy': [],
        'enstrophy': [],
        'vorticity_max': [],
        'dissipation': [],
        'k_max_eta': [],
        'bkm': [],
        'u_rms': [],
        'Re_lambda': [],
        'taylor_lambda': [],
        'L_int': [],
        'spectrum_slope': []
    }
    
    # Save initial spectrum if requested
    if save_spectra:
        k, E, E_comp, E_total, slope = compute_energy_spectrum(solver)
        spec_file = os.path.join(output_dir, f"spectrum_t{0:.1f}.h5")
        
        with h5py.File(spec_file, 'w') as f:
            f.create_dataset('k', data=k)
            f.create_dataset('E', data=E)
            f.create_dataset('E_compensated', data=E_comp)
            f.attrs['E_total'] = E_total
            f.attrs['slope'] = slope
            f.attrs['time'] = 0.0
            f.attrs['Re_lambda'] = stats0['Re_lambda']
        
        print(f"  Initial spectrum saved")
    
    print(f"\nStarting evolution...")
    print("-"*70)
    
    step = 0
    t_start = time.time()
    prev_vorticity = vort0
    vorticity_spike_count = 0
    
    # Main time evolution loop
    while solver.current_time < t_final and step < 50000:
        # Adaptive timestep with safety check
        if solver.adapt_dt:
            dt_new, cfl, rho, mode, _ = solver.choose_dt()
            
            # Safety check for vorticity spikes
            current_vort = solver.compute_max_vorticity()
            if current_vort > 2 * prev_vorticity and step > 100:
                dt_new = min(dt_new, solver.dt * 0.5)
                vorticity_spike_count += 1
                if vorticity_spike_count > 3:
                    dt_new = min(dt_new, 1e-4)
                    print(f"  [STABILITY] Vorticity spike - limiting dt to {dt_new:.2e}")
            else:
                vorticity_spike_count = max(0, vorticity_spike_count - 1)
            
            solver.dt = dt_new
            prev_vorticity = current_vort
        
        # Time evolution step
        try:
            metrics = solver.evolve_one_timestep(use_rk2=True)
        except Exception as e:
            print(f"Evolution failed at step {step}: {e}")
            break
        
        # Record metrics periodically
        if step % 10 == 0:
            stats = compute_turbulence_statistics(solver)
            
            # Resolution metrics
            current_enstrophy = metrics.get('enstrophy', solver.compute_enstrophy())
            eps_diss = 2.0 * solver.viscosity * current_enstrophy
            
            if eps_diss > 0:
                eta = (solver.viscosity**3 / eps_diss)**0.25
                k_max_eta = (N/3.0) * eta
            else:
                k_max_eta = np.inf
            
            # Spectrum slope
            _, _, _, _, slope = compute_energy_spectrum(solver)
            
            # Store time series
            time_series['t'].append(float(solver.current_time))
            time_series['energy'].append(float(metrics['energy']))
            time_series['enstrophy'].append(float(current_enstrophy))
            time_series['vorticity_max'].append(float(metrics['vorticity']))
            time_series['dissipation'].append(float(eps_diss))
            time_series['k_max_eta'].append(float(k_max_eta))
            time_series['bkm'].append(float(metrics.get('bkm_integral', 0)))
            time_series['u_rms'].append(float(stats['u_total_rms']))
            time_series['Re_lambda'].append(float(stats['Re_lambda']))
            time_series['taylor_lambda'].append(float(stats['taylor_lambda']))
            time_series['L_int'].append(float(stats['L_int']))
            time_series['spectrum_slope'].append(float(slope))
        
        # Progress output
        if step % 100 == 0:
            dt_ms = solver.dt * 1000
            print(f"Step {step:5d} | t={solver.current_time:.2f} | "
                  f"dt={dt_ms:.2f}ms | E={metrics['energy']:.4f}")
            
            if step % 500 == 0 and len(time_series['Re_lambda']) > 0:
                slope_str = (f"slope={time_series['spectrum_slope'][-1]:.2f}" 
                            if not np.isnan(time_series['spectrum_slope'][-1])
                            else "viscous")
                print(f"  Re_λ={time_series['Re_lambda'][-1]:.1f} | "
                      f"E/E0={metrics['energy']/E0:.3f} | "
                      f"{slope_str} | "
                      f"k_max*η={time_series['k_max_eta'][-1]:.2f}")
        
        # Save spectrum periodically
        if save_spectra and solver.current_time > 0 and step % 1000 == 0:
            k, E, E_comp, E_total, slope = compute_energy_spectrum(solver)
            spec_file = os.path.join(output_dir, f"spectrum_t{solver.current_time:.1f}.h5")
            
            with h5py.File(spec_file, 'w') as f:
                f.create_dataset('k', data=k)
                f.create_dataset('E', data=E)
                f.create_dataset('E_compensated', data=E_comp)
                f.attrs['E_total'] = E_total
                f.attrs['slope'] = slope
                f.attrs['time'] = solver.current_time
        
        # Save field checkpoint periodically
        if save_fields and step % int(field_save_interval / solver.dt) == 0:
            field_file = os.path.join(output_dir, f"fields_t{solver.current_time:06.2f}.h5")
            save_field_checkpoint(solver, field_file)
        
        step += 1
    
    # Save time series to HDF5
    print("\nSaving time series data...")
    ts_file = os.path.join(output_dir, 'time_series.h5')
    with h5py.File(ts_file, 'w') as f:
        for key, data in time_series.items():
            if len(data) > 0:
                f.create_dataset(key, data=np.array(data, dtype=np.float64))
        
        # Save metadata
        f.attrs['simulation_type'] = 'isotropic_decay'
        f.attrs['N'] = N
        f.attrs['Re'] = Re
        f.attrs['nu'] = solver.viscosity
        f.attrs['L'] = 2*np.pi
        f.attrs['initial_energy'] = E0
        f.attrs['final_time'] = solver.current_time
        f.attrs['total_steps'] = step
        f.attrs['use_smooth_ic'] = use_smooth_ic
        f.attrs['use_stable_params'] = use_stable_params
    
    print(f"  Time series saved: {len(time_series['t'])} points")
    
    # Final summary
    elapsed = time.time() - t_start
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"Final time: t={solver.current_time:.2f}")
    print(f"Total steps: {step}")
    print(f"Wall time: {elapsed/60:.1f} minutes")
    
    if len(time_series['energy']) > 0:
        print(f"\nFinal state:")
        print(f"  Energy: {time_series['energy'][-1]:.4f} ({time_series['energy'][-1]/E0:.1%} of initial)")
        print(f"  Re_lambda: {time_series['Re_lambda'][-1]:.1f}")
        
        # Check for monotonic decay
        energy_array = np.array(time_series['energy'])
        if np.all(np.diff(energy_array) <= 0):
            print("  Energy decay: MONOTONIC (physically correct)")
        else:
            print("  WARNING: Non-monotonic energy decay detected")
    
    print(f"\nResults saved to: {output_dir}/")
    
    return output_dir

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(description='Run isotropic turbulence decay simulation')
    
    parser.add_argument('--N', type=int, default=256,
                       help='Grid resolution (default: 256)')
    parser.add_argument('--Re', type=float, default=300,
                       help='Reynolds number (default: 300)')
    parser.add_argument('--time', type=float, default=15.0,
                       help='Final simulation time (default: 15.0)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick 3-second validation test')
    parser.add_argument('--save-fields', action='store_true',
                       help='Save full 3D field checkpoints')
    parser.add_argument('--field-interval', type=float, default=3.0,
                       help='Interval between field saves (default: 3.0)')
    parser.add_argument('--no-spectra', action='store_true',
                       help='Disable spectrum saves')
    parser.add_argument('--spectral-ic', action='store_true',
                       help='Use spectral initial conditions (default: smooth)')
    parser.add_argument('--standard-params', action='store_true',
                       help='Use standard solver parameters (default: conservative)')
    
    args = parser.parse_args()
    
    try:
        output_dir = run_decay_simulation(
            N=args.N,
            Re=args.Re,
            t_final=args.time,
            quick_test=args.quick,
            save_fields=args.save_fields,
            field_save_interval=args.field_interval,
            save_spectra=not args.no_spectra,
            use_smooth_ic=not args.spectral_ic,
            use_stable_params=not args.standard_params
        )
        
        print(f"\nSimulation completed successfully!")
        print(f"Output directory: {output_dir}")
        
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()