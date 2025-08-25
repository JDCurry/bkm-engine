#!/usr/bin/env python3
"""
test_base.py
============
Unified base module for all BKM test scripts.
Provides standardized checkpointing, diagnostics, and shared utilities.

This module ensures consistent data output across all test cases for
peer review and reproducibility.
"""

import numpy as np
import h5py
import csv
import json
import hashlib
import os
from datetime import datetime


class UnifiedCheckpointer:
    """
    Standardized HDF5 checkpointer used by all tests.
    Ensures comprehensive and consistent data export.
    """
    
    def __init__(self, output_dir, test_name, grid_shape, params):
        """
        Initialize checkpointer.
        
        Parameters:
        -----------
        output_dir : str
            Output directory path
        test_name : str
            Name of the test (e.g., 'taylor_green', 'kida_pelz')
        grid_shape : tuple
            Grid dimensions (nx, ny, nz)
        params : dict
            Test parameters (Re, nu, dt, etc.)
        """
        self.output_dir = output_dir
        self.test_name = test_name
        self.grid_shape = grid_shape
        self.params = params
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize time series file
        self.ts_file = os.path.join(output_dir, 'time_series.h5')
        self.csv_file = os.path.join(output_dir, 'diagnostics.csv')
        self._init_time_series()
        self._init_csv()
    
    def _init_time_series(self):
        """Initialize HDF5 time series with comprehensive diagnostics."""
        with h5py.File(self.ts_file, 'w') as f:
            # Metadata
            f.attrs['test_name'] = self.test_name
            f.attrs['grid_shape'] = self.grid_shape
            f.attrs['created_utc'] = datetime.utcnow().isoformat()
            
            # Parameters
            for key, value in self.params.items():
                if isinstance(value, (int, float, str, bool)):
                    f.attrs[f'param_{key}'] = value
            
            # Create expandable datasets for all diagnostics
            diagnostics = [
                'time', 'step', 'energy', 'enstrophy', 'vorticity_max',
                'dissipation', 'helicity', 'divergence_max', 'divergence_l2',
                'budget_residual', 'budget_relative', 'Re_lambda', 
                'taylor_lambda', 'L_int', 'k_max_eta', 'dt', 'cfl',
                'u_rms', 'v_rms', 'w_rms', 'u_total_rms', 'spectrum_slope',
                # Additional fields for forced simulations
                'forcing_alpha', 'forcing_P_inj', 'band_fraction'
            ]
            
            for diag in diagnostics:
                f.create_dataset(diag, (0,), maxshape=(None,), dtype='f8')
    
    def _init_csv(self):
        """Initialize CSV for human-readable output."""
        self.csv_handle = open(self.csv_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_handle)
        self.csv_writer.writerow([
            'step', 'time', 'energy', 'enstrophy', 'vorticity_max',
            'dissipation', 'Re_lambda', 'k_max_eta', 'dt', 'cfl',
            'budget_residual', 'budget_relative'
        ])
    
    def save_checkpoint(self, step, solver, save_fields=True, save_slices=True):
        """
        Save comprehensive checkpoint with all fields and diagnostics.
        
        Parameters:
        -----------
        step : int
            Current timestep
        solver : CUDABKMSolver
            Solver instance
        save_fields : bool
            Save full 3D fields (velocity and vorticity)
        save_slices : bool
            Save 2D slices for visualization
        """
        filename = os.path.join(self.output_dir, f'checkpoint_{step:07d}.h5')
        
        with h5py.File(filename, 'w') as f:
            # Metadata
            f.attrs['step'] = step
            f.attrs['time'] = solver.current_time
            f.attrs['test_name'] = self.test_name
            f.attrs['grid_shape'] = self.grid_shape
            f.attrs['reynolds_number'] = solver.Re
            f.attrs['viscosity'] = solver.viscosity
            f.attrs['created_utc'] = datetime.utcnow().isoformat()
            
            if save_fields:
                # Transfer from GPU if needed
                xp = solver.xp
                if solver.use_gpu:
                    u = xp.asnumpy(solver.u)
                    v = xp.asnumpy(solver.v)
                    w = xp.asnumpy(solver.w)
                else:
                    u, v, w = solver.u, solver.v, solver.w
                
                # Velocity fields
                grp = f.create_group('velocity')
                grp.create_dataset('u', data=u, compression='gzip', compression_opts=4)
                grp.create_dataset('v', data=v, compression='gzip', compression_opts=4)
                grp.create_dataset('w', data=w, compression='gzip', compression_opts=4)
                
                # Compute vorticity
                omega_x = solver.compute_derivatives_fft(solver.w, 1) - \
                         solver.compute_derivatives_fft(solver.v, 2)
                omega_y = solver.compute_derivatives_fft(solver.u, 2) - \
                         solver.compute_derivatives_fft(solver.w, 0)
                omega_z = solver.compute_derivatives_fft(solver.v, 0) - \
                         solver.compute_derivatives_fft(solver.u, 1)
                
                if solver.use_gpu:
                    omega_x = xp.asnumpy(omega_x)
                    omega_y = xp.asnumpy(omega_y)
                    omega_z = xp.asnumpy(omega_z)
                
                # Vorticity fields
                grp = f.create_group('vorticity')
                grp.create_dataset('omega_x', data=omega_x, compression='gzip', compression_opts=4)
                grp.create_dataset('omega_y', data=omega_y, compression='gzip', compression_opts=4)
                grp.create_dataset('omega_z', data=omega_z, compression='gzip', compression_opts=4)
                
                # Data integrity hashes
                grp = f.create_group('integrity')
                grp.attrs['u_sha256'] = hashlib.sha256(np.ascontiguousarray(u).data).hexdigest()
                grp.attrs['v_sha256'] = hashlib.sha256(np.ascontiguousarray(v).data).hexdigest()
                grp.attrs['w_sha256'] = hashlib.sha256(np.ascontiguousarray(w).data).hexdigest()
            
            if save_slices:
                self._save_slices(f, solver)
            
            # Current diagnostics
            grp = f.create_group('diagnostics')
            diag = compute_comprehensive_diagnostics(solver)
            for key, value in diag.items():
                if not np.isnan(value) and not np.isinf(value):
                    grp.attrs[key] = value
        
        return filename
    
    def _save_slices(self, f, solver):
        """Save 2D slices at midplanes for visualization."""
        grp = f.create_group('slices')
        
        xp = solver.xp
        nx, ny, nz = solver.u.shape
        
        # Midplane indices
        mx, my, mz = nx//2, ny//2, nz//2
        
        # Get numpy arrays
        if solver.use_gpu:
            u = xp.asnumpy(solver.u)
            v = xp.asnumpy(solver.v)
            w = xp.asnumpy(solver.w)
        else:
            u, v, w = solver.u, solver.v, solver.w
        
        # XY plane (z=mid)
        grp.create_dataset('u_xy', data=u[:, :, mz], compression='gzip', compression_opts=2)
        grp.create_dataset('v_xy', data=v[:, :, mz], compression='gzip', compression_opts=2)
        grp.create_dataset('w_xy', data=w[:, :, mz], compression='gzip', compression_opts=2)
        
        # XZ plane (y=mid)
        grp.create_dataset('u_xz', data=u[:, my, :], compression='gzip', compression_opts=2)
        grp.create_dataset('v_xz', data=v[:, my, :], compression='gzip', compression_opts=2)
        grp.create_dataset('w_xz', data=w[:, my, :], compression='gzip', compression_opts=2)
        
        # YZ plane (x=mid)
        grp.create_dataset('u_yz', data=u[mx, :, :], compression='gzip', compression_opts=2)
        grp.create_dataset('v_yz', data=v[mx, :, :], compression='gzip', compression_opts=2)
        grp.create_dataset('w_yz', data=w[mx, :, :], compression='gzip', compression_opts=2)
    
    def append_time_series(self, data):
        """
        Append data point to time series.
        
        Parameters:
        -----------
        data : dict
            Dictionary containing all diagnostic values
        """
        # Append to HDF5
        with h5py.File(self.ts_file, 'a') as f:
            for key, value in data.items():
                if key in f:
                    if not np.isnan(value) and not np.isinf(value):
                        dataset = f[key]
                        dataset.resize(dataset.shape[0] + 1, axis=0)
                        dataset[-1] = value
        
        # Write to CSV (subset of key metrics)
        self.csv_writer.writerow([
            data.get('step', 0),
            data.get('time', 0),
            data.get('energy', 0),
            data.get('enstrophy', 0),
            data.get('vorticity_max', 0),
            data.get('dissipation', 0),
            data.get('Re_lambda', 0),
            data.get('k_max_eta', 0),
            data.get('dt', 0),
            data.get('cfl', 0),
            data.get('budget_residual', 0),
            data.get('budget_relative', 0)
        ])
        
        # Periodic flush
        if data.get('step', 0) % 100 == 0:
            self.csv_handle.flush()
    
    def save_spectrum(self, step, solver):
        """Save energy spectrum with compensated form."""
        filename = os.path.join(self.output_dir, f'spectrum_{step:07d}.h5')
        
        if hasattr(solver, 'compute_energy_spectrum'):
            result = solver.compute_energy_spectrum()
            
            # Handle different return formats
            if len(result) == 3:
                k, E, slope = result
            else:
                k, E = result[:2]
                slope = -5/3  # Default Kolmogorov
            
            # Convert to numpy if needed
            xp = solver.xp
            if solver.use_gpu:
                k = xp.asnumpy(k) if hasattr(k, 'get') else k
                E = xp.asnumpy(E) if hasattr(E, 'get') else E
            
            with h5py.File(filename, 'w') as f:
                f.attrs['step'] = step
                f.attrs['time'] = solver.current_time
                f.attrs['slope'] = slope
                f.attrs['energy_total'] = solver.compute_energy()
                
                f.create_dataset('k', data=k)
                f.create_dataset('E', data=E)
                f.create_dataset('E_compensated', data=E * k**(5/3))
        
        return filename
    
    def close(self):
        """Close open file handles."""
        if hasattr(self, 'csv_handle'):
            self.csv_handle.close()


def compute_comprehensive_diagnostics(solver, prev_energy=None, prev_time=None):
    """
    Compute comprehensive diagnostics for any test.
    
    Parameters:
    -----------
    solver : CUDABKMSolver
        Solver instance
    prev_energy : float
        Energy from previous timestep (for budget)
    prev_time : float
        Time from previous timestep (for budget)
    
    Returns:
    --------
    dict : All diagnostic quantities
    """
    xp = solver.xp
    
    # Basic quantities
    energy = solver.compute_energy()
    enstrophy = solver.compute_enstrophy()
    vorticity_max = solver.compute_max_vorticity()
    div_max, div_l2 = solver.compute_divergence_metrics()
    
    # Dissipation rate
    dissipation = 2.0 * solver.viscosity * enstrophy
    
    # Velocity statistics
    u2 = solver.u**2
    v2 = solver.v**2
    w2 = solver.w**2
    
    u_rms = float(xp.sqrt(xp.mean(u2)).get() if solver.use_gpu else xp.sqrt(xp.mean(u2)))
    v_rms = float(xp.sqrt(xp.mean(v2)).get() if solver.use_gpu else xp.sqrt(xp.mean(v2)))
    w_rms = float(xp.sqrt(xp.mean(w2)).get() if solver.use_gpu else xp.sqrt(xp.mean(w2)))
    u_total_rms = float(xp.sqrt(xp.mean(u2 + v2 + w2)).get() if solver.use_gpu 
                       else xp.sqrt(xp.mean(u2 + v2 + w2)))
    
    # Taylor microscale and Reynolds number
    dudx = solver.compute_derivatives_fft(solver.u, 0)
    dvdy = solver.compute_derivatives_fft(solver.v, 1)
    dwdz = solver.compute_derivatives_fft(solver.w, 2)
    
    mean_u2 = xp.mean(u2 + v2 + w2)
    mean_grad2 = xp.mean(dudx**2 + dvdy**2 + dwdz**2)
    
    if solver.use_gpu:
        mean_u2 = float(mean_u2.get())
        mean_grad2 = float(mean_grad2.get())
    else:
        mean_u2 = float(mean_u2)
        mean_grad2 = float(mean_grad2)
    
    taylor_lambda = np.sqrt(mean_u2 / mean_grad2) if mean_grad2 > 0 else 0.1
    Re_lambda = u_total_rms * taylor_lambda * solver.Re
    
    # Integral scale
    L_int = np.pi / 2 * u_total_rms / np.sqrt(enstrophy) if enstrophy > 0 else 1.0
    
    # Kolmogorov scale
    if dissipation > 0:
        eta = (solver.viscosity**3 / dissipation)**0.25
        k_max_eta = (solver.nx/3.0) * eta
    else:
        eta = 0.1
        k_max_eta = np.inf
    
    # Helicity
    omega_x = solver.compute_derivatives_fft(solver.w, 1) - solver.compute_derivatives_fft(solver.v, 2)
    omega_y = solver.compute_derivatives_fft(solver.u, 2) - solver.compute_derivatives_fft(solver.w, 0)
    omega_z = solver.compute_derivatives_fft(solver.v, 0) - solver.compute_derivatives_fft(solver.u, 1)
    
    helicity = xp.mean(solver.u * omega_x + solver.v * omega_y + solver.w * omega_z)
    helicity = float(helicity.get() if solver.use_gpu else helicity)
    
    # Budget calculation
    budget_residual = 0.0
    budget_relative = 0.0
    if prev_energy is not None and prev_time is not None:
        dt_actual = solver.current_time - prev_time
        if dt_actual > 0:
            dE_dt = (energy - prev_energy) / dt_actual
            budget_residual = dE_dt + dissipation  # For decay: should be ~0
            if dissipation > 0:
                budget_relative = abs(budget_residual) / dissipation
    
    # Spectrum slope (if available)
    spectrum_slope = -5/3  # Default
    if hasattr(solver, 'compute_energy_spectrum'):
        try:
            result = solver.compute_energy_spectrum()
            if len(result) >= 3:
                spectrum_slope = result[2]
        except:
            pass
    
    return {
        'energy': energy,
        'enstrophy': enstrophy,
        'vorticity_max': vorticity_max,
        'dissipation': dissipation,
        'helicity': helicity,
        'divergence_max': div_max,
        'divergence_l2': div_l2,
        'Re_lambda': Re_lambda,
        'taylor_lambda': taylor_lambda,
        'L_int': L_int,
        'k_max_eta': k_max_eta,
        'u_rms': u_rms,
        'v_rms': v_rms,
        'w_rms': w_rms,
        'u_total_rms': u_total_rms,
        'budget_residual': budget_residual,
        'budget_relative': budget_relative,
        'spectrum_slope': spectrum_slope
    }


# ============================================================================
# ISOTROPIC TURBULENCE UTILITIES (shared by decay and forced tests)
# ============================================================================

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
    None (modifies solver.u, v, w in place)
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


def compute_budget_residual(dE_dt, epsilon, P_inj=0.0):
    """
    Compute energy budget residual.
    
    The energy equation is: dE/dt = -epsilon + P_inj
    Residual R = dE/dt + epsilon - P_inj
    
    Parameters:
    -----------
    dE_dt : float
        Time derivative of energy
    epsilon : float
        Dissipation rate
    P_inj : float
        Power injection (0 for decay, non-zero for forced)
        
    Returns:
    --------
    residual : float
        Budget residual
    relative_error : float
        |R|/epsilon
    """
    residual = dE_dt + epsilon - P_inj
    relative_error = abs(residual) / epsilon if epsilon > 0 else 0.0
    return residual, relative_error


def get_spectrum_save_schedule(t_final, test_type='decay'):
    """
    Get standardized spectrum save schedule.
    
    Parameters:
    -----------
    t_final : float
        Final simulation time
    test_type : str
        'decay' or 'forced' to select appropriate schedule
        
    Returns:
    --------
    times : list
        List of times to save spectra
    """
    if test_type == 'decay':
        if t_final <= 5.0:
            times = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        elif t_final <= 15.0:
            times = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 12.0, 15.0]
        else:
            times = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    else:  # forced
        if t_final <= 5.0:
            times = [1.0, 2.0, 3.0, 4.0, 5.0]
        elif t_final <= 20.0:
            times = [2.0, 5.0, 10.0, 15.0, 20.0]
        else:
            times = [2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    
    # Filter to times within simulation range
    times = [t for t in times if t <= t_final]
    return times


# ============================================================================
# FORCING CONTROLLER FOR ISOTROPIC FORCED TEST
# ============================================================================

class FractionController:
    """
    PI controller for maintaining spectral shape through fraction control.
    
    This controller maintains a target fraction of total energy in the forcing
    band, allowing natural energy decay while preserving large-scale content.
    """
    
    def __init__(self, f_band=0.25, c_fracP=1.5, c_fracI=0.1, alpha_max=0.15):
        """
        Initialize fraction controller.
        
        Parameters:
        -----------
        f_band : float
            Target fraction of total energy in forcing band
        c_fracP : float
            Proportional gain for fraction error
        c_fracI : float
            Integral gain for fraction error
        alpha_max : float
            Maximum forcing amplitude (safety limit)
        """
        self.f_band = f_band
        self.c_fracP = c_fracP
        self.c_fracI = c_fracI
        self.alpha_max = alpha_max
        self.epsilon_smooth = 0.01  # Initial dissipation estimate
        self.beta_eps = 0.1  # EMA factor for smoothing
        self.df_integral = 0.0  # Integral of fraction error
        
    def update(self, E_total, E_band, epsilon, dt, t_current=0.0):
        """
        Compute forcing amplitude using fraction-only PI control.
        
        Returns:
        --------
        alpha : float
            Forcing amplitude (negative = injection, positive = removal)
        f_error : float
            Fraction error
        E_band_target : float
            Target energy in band
        eps_smooth : float
            Smoothed dissipation estimate
        f_current : float
            Current band fraction
        """
        # Update smoothed dissipation estimate
        self.epsilon_smooth = (1 - self.beta_eps) * self.epsilon_smooth + self.beta_eps * epsilon
        
        # Use instantaneous epsilon for control
        eps_inst = epsilon
        
        # Two-stage target: relaxed initially, then tighten
        if t_current < 4.0:
            f_target_current = 0.35  # Relaxed during startup
        else:
            f_target_current = self.f_band  # Final target
        
        # Current band fraction
        f_current = E_band / E_total if E_total > 0 else 0.0
        
        # Error: positive when band is over target
        e_frac = f_current - f_target_current
        
        # Deadband to prevent chatter (3% tolerance)
        if abs(e_frac) < 0.03:
            P_frac = 0.0
            self.df_integral *= 0.9  # Decay integrator
        else:
            # Update integral after initial transient
            if t_current > 1.0:
                self.df_integral = np.clip(self.df_integral + e_frac * dt, -0.5, 0.5)
            
            # PI control on fraction error
            P_frac = self.c_fracP * e_frac + self.c_fracI * self.df_integral
        
        # Safe energy band for division
        E_band_safe = max(E_band, 1e-12)
        
        # Apply startup ramp
        ramp = min(1.0, t_current / 2.0)  # Ramp over 2 time units
        
        # Compute forcing amplitude
        alpha = ramp * eps_inst * P_frac / (2.0 * E_band_safe)
        
        # Safety check: don't push in wrong direction
        if (e_frac > 0.03 and alpha < 0) or (e_frac < -0.03 and alpha > 0):
            alpha = 0.0
        
        # Dynamic clamp based on physical scale
        alpha_phys = abs(eps_inst / (2.0 * E_band_safe)) if E_band_safe > 0 else 0.0
        alpha_cap = min(self.alpha_max, 1.5 * alpha_phys)
        
        # Apply clamp
        alpha = np.clip(alpha, -alpha_cap, alpha_cap)
        
        # Return values for diagnostics
        E_band_target = f_target_current * E_total
        f_error = f_target_current - f_current
        
        return alpha, f_error, E_band_target, self.epsilon_smooth, f_current


def apply_spectral_forcing(solver, controller, k_min=2, k_max=4, dt_force=1.0, 
                          apply_sponge=False, mu_sponge=0.02, ramp_factor=1.0):
    """
    Apply forcing in spectral band with fraction control.
    
    Sign convention:
    - alpha < 0: injects energy (amplifies modes)
    - alpha > 0: removes energy (damps modes)
    - Power injection: P_inj = -2 * alpha * E_band
    
    Returns:
    --------
    alpha : float
        Applied forcing amplitude
    E_band : float
        Energy in forcing band
    f_error : float
        Fraction error
    E_band_target : float
        Target band energy
    eps_smooth : float
        Smoothed dissipation
    f_current : float
        Current band fraction
    P_inj : float
        Power injection rate
    """
    xp = solver.xp
    N = solver.u.shape[0]
    L = 2*np.pi
    
    # Wavenumber grid
    k = xp.fft.fftfreq(N, L/N) * 2*np.pi
    kx, ky, kz = xp.meshgrid(k, k, k, indexing='ij')
    k_mag = xp.sqrt(kx**2 + ky**2 + kz**2)
    
    # Transform to spectral space
    u_hat = xp.fft.fftn(solver.u)
    v_hat = xp.fft.fftn(solver.v)
    w_hat = xp.fft.fftn(solver.w)
    
    # Compute energy in forcing band
    band_mask = (k_mag >= k_min) & (k_mag <= k_max)
    E_band = 0.5 * xp.sum(
        (xp.abs(u_hat[band_mask])**2 + 
         xp.abs(v_hat[band_mask])**2 + 
         xp.abs(w_hat[band_mask])**2) / N**6
    )
    
    if solver.use_gpu:
        E_band = float(E_band.get())
    else:
        E_band = float(E_band)
    
    # Get total energy and dissipation
    E_total = solver.compute_energy()
    current_enstrophy = solver.compute_enstrophy()
    epsilon = 2.0 * solver.viscosity * current_enstrophy
    
    # Get control signal
    alpha, f_error, E_band_target, eps_smooth, f_current = controller.update(
        E_total, E_band, epsilon, dt_force, t_current=solver.current_time
    )
    
    # Apply ramp factor
    alpha *= ramp_factor
    
    # Apply forcing to band modes
    if abs(alpha) > 1e-10:
        forcing_factor = 1.0 - alpha * dt_force
        u_hat[band_mask] *= forcing_factor
        v_hat[band_mask] *= forcing_factor
        w_hat[band_mask] *= forcing_factor
    
    # Optional high-k sponge layer
    if apply_sponge:
        sponge_k_min = N//4  # Start sponge at k = N/4
        sponge_mask = k_mag >= sponge_k_min
        sponge_factor = 1.0 - mu_sponge * dt_force * (k_mag[sponge_mask] / (N//2))**2
        sponge_factor = xp.maximum(sponge_factor, 0.0)
        u_hat[sponge_mask] *= sponge_factor
        v_hat[sponge_mask] *= sponge_factor
        w_hat[sponge_mask] *= sponge_factor
    
    # Transform back to physical space
    solver.u = xp.fft.ifftn(u_hat).real
    solver.v = xp.fft.ifftn(v_hat).real
    solver.w = xp.fft.ifftn(w_hat).real
    
    # Power injection: P_inj = -2 * alpha * E_band
    P_inj = -2.0 * alpha * E_band
    
    return alpha, E_band, f_error, E_band_target, eps_smooth, f_current, P_inj