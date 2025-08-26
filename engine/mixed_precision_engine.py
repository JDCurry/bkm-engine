#!/usr/bin/env python3
"""
mixed_precision_bkm_engine.py - Unified Version with Mixed Precision
============================================================================
CUDA-accelerated BKM integral solver with mixed precision support.

This merges all features from:
- unified_bkm_engine.py: All v2/v3 fixes, alignment tracking, safety checks
- corrected_mixed_precision_engine.py: Mixed precision memory optimization

Mixed Precision Implementation:
- Velocity fields: float32 (50% memory savings in mixed mode)
- Critical computations (derivatives, wavenumbers): float64 (accuracy)
- Diagnostics (energy, vorticity): float64 (accuracy)
- FFT operations: complex128 for projections (numerical stability)

Key Features:
- Proper 3D spectral operations with shape validation
- Energy-conserving skew-symmetric convection
- Adaptive timestepping with guard conditions
- BKM integral tracking
- Vorticity-stretching alignment metrics
- Safe 2D slice extraction for visualization
- Checkpoint/resume capability
- Mixed precision memory optimization
"""

# Engine provenance stamp for traceability
ENGINE_ID = "unified_mixed_precision_bkm@2025-01-25"
ENGINE_VERSION = "3.0.0-mixed"

import argparse

def choose_backend(force=None):
    if force == "cpu":
        import numpy as xp
        return xp, "cpu"
    try:
        import cupy as cp
        from cupy.cuda import runtime as rt
        if rt.getDeviceCount() > 0:
            return cp, "gpu"
    except Exception as e:
        print(f"[GPU disabled] {e!r}")
    import numpy as xp
    return xp, "cpu"

import matplotlib.pyplot as plt
import matplotlib
import h5py
import os
import time
from datetime import datetime
import psutil
from scipy.ndimage import gaussian_filter
from scipy import stats
import warnings
import ctypes
import gc
import json
import hashlib
import sys
import csv
import numpy as np  # Always import numpy for some operations

matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=RuntimeWarning)

parser = argparse.ArgumentParser(description="Unified Mixed Precision BKM Engine")
parser.add_argument("--device", choices=["cpu", "gpu"], help="Force backend")
args, unknown = parser.parse_known_args()

# Choose backend
xp, backend = choose_backend(args.device if hasattr(args, "device") else None)
print(f"Using {backend.upper()} for grid")

if hasattr(args, "device") and args.device == "gpu" and backend != "gpu":
    raise RuntimeError("Requested --device gpu but no CUDA device is available.")


class AlignmentLogger:
    """
    Lightweight CSV logger for vorticity-stretching alignment metrics
    """
    def __init__(self, output_dir=".", filename="alignment_history.csv"):
        """
        Initialize alignment logger
        
        Args:
            output_dir: Directory for output file
            filename: CSV filename
        """
        self.filepath = os.path.join(output_dir, filename)
        self.file = None
        self.writer = None
        self._initialize_file()
    
    def _initialize_file(self):
        """Create CSV with headers"""
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['t', 'mean_abs_cos_theta', 'vol_frac_aligned', 'aligned_engaged'])
        self.file.flush()
    
    def write(self, t, mean_abs_cos_theta, vol_frac_aligned, aligned_engaged):
        """
        Write alignment metrics to CSV
        
        Args:
            t: Current time
            mean_abs_cos_theta: Volume mean of |cos(theta)|
            vol_frac_aligned: Fraction of points with |cos(theta)| > c_star
            aligned_engaged: 1 if vol_frac_aligned > f_star, else 0
        """
        self.writer.writerow([f"{t:.6e}", f"{mean_abs_cos_theta:.6f}", 
                             f"{vol_frac_aligned:.6f}", int(aligned_engaged)])
        self.file.flush()
    
    def close(self):
        """Close the CSV file"""
        if self.file:
            self.file.close()


class CUDABKMSolver:
    """
    Production-ready CUDA-accelerated Navier-Stokes solver with mixed precision support
    """

    def __init__(self, grid_size=64, dt=0.001, reynolds_number=1000.0,
                 use_dealiasing=True, dealias_fraction=2.0/3.0,
                 force_cpu=False, checkpoint_interval=1000,
                 CFL_target=0.8, adapt_dt=True, guard_reduce_factor=0.5,
                 rho_soft=0.985, rho_hard=0.995, dt_min=1e-6, dt_max=0.1,
                 C_adv=1.0, p_value=xp.nan, initial_amplitude=1.0,
                 viscosity_ramp_time=0.0, viscosity_ramp_factor=1.0,
                 rng_seed=None, projection_tol=3e-9, extra_projection_iters=0,
                 divergence_threshold=1e-6,
                 startup_steps=2000, dt_max_startup=5e-4, growth_cap_startup=1.2,
                 align_every=10, c_star=0.95, f_star=0.01, 
                 alignment_subsample=1, track_alignment=True,
                 precision='mixed', dtype=None):
        """
        Initialize solver with mixed precision support
        
        Args:
            precision: 'float32', 'float64', or 'mixed'
                - 'float32': All operations in float32
                - 'float64': All operations in float64  
                - 'mixed': Fields in float32, critical ops in float64
            dtype: Legacy parameter for backward compatibility
        """
        
        # Set up precision modes FIRST
        self.precision_mode = precision
        if precision == 'float32':
            self.dtype_field = xp.float32
            self.dtype_diag = xp.float32
            self.dtype_compute = xp.float32
            self.complex_dtype = xp.complex64
        elif precision == 'float64':
            self.dtype_field = xp.float64
            self.dtype_diag = xp.float64
            self.dtype_compute = xp.float64
            self.complex_dtype = xp.complex128
        elif precision == 'mixed':
            self.dtype_field = xp.float32      # Fields stored in float32
            self.dtype_diag = xp.float64       # Diagnostics in float64
            self.dtype_compute = xp.float64    # Critical computations in float64
            self.complex_dtype = xp.complex128 # FFTs in complex128 for projection
        else:
            raise ValueError(f"Unknown precision mode: {precision}")
        
        # Handle legacy dtype parameter
        if dtype is not None:
            self.dtype = dtype  # Keep for backward compatibility
            if dtype == xp.float32:
                self.dtype_field = xp.float32
                self.dtype_diag = xp.float32
                self.dtype_compute = xp.float32
                self.complex_dtype = xp.complex64
            else:
                self.dtype_field = xp.float64
                self.dtype_diag = xp.float64
                self.dtype_compute = xp.float64
                self.complex_dtype = xp.complex128
        else:
            self.dtype = self.dtype_field  # For backward compatibility

        print(f"Using {backend.upper()} for {grid_size}^3 grid")
        print(f"Precision mode: {precision}")
        if precision == 'mixed':
            memory_gb = grid_size**3 * 3 * 4 / 1e9  # 3 fields × 4 bytes (float32)
            print(f"  Fields: float32 ({memory_gb:.1f} GB)")
            print(f"  Diagnostics: float64")
            print(f"  Memory savings: ~50% vs full float64")

        # Basic parameters
        self.nx = self.ny = self.nz = grid_size
        self.Lx = self.Ly = self.Lz = 2 * xp.pi
        self.dx = self.dy = self.dz = self.Lx / grid_size
        self.dt = dt
        self.dt_prev = dt
        self.dt_used = dt  # Track actual dt used for time advancement
        self.Re = reynolds_number
        self.viscosity_target = 1.0 / reynolds_number
        self.viscosity = self.viscosity_target
        self.use_dealiasing = use_dealiasing
        self.dealias_fraction = dealias_fraction
        self.checkpoint_interval = checkpoint_interval
        self.CFL_target = CFL_target
        self.adapt_dt = adapt_dt
        self.guard_reduce_factor = guard_reduce_factor
        self.rho_soft = rho_soft
        self.rho_hard = rho_hard
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.C_adv = C_adv
        self.p_value = p_value
        self.initial_amplitude = initial_amplitude
        self.viscosity_ramp_time = viscosity_ramp_time
        self.viscosity_ramp_factor = viscosity_ramp_factor
        self.projection_tol = projection_tol
        self.extra_projection_iters = extra_projection_iters
        self.divergence_threshold = divergence_threshold
        
        # Startup phase parameters
        self.startup_steps = startup_steps
        self.dt_max_startup = dt_max_startup
        self.growth_cap_startup = growth_cap_startup
        self.startup_complete = False
        
        # Alignment tracking parameters
        self.align_every = align_every
        self.c_star = c_star
        self.f_star = f_star
        self.alignment_subsample = alignment_subsample
        self.track_alignment = track_alignment
        
        # Initialize alignment tracking
        if self.track_alignment:
            self.alignment_logger = AlignmentLogger(output_dir=".")
            self.alignment_engaged_steps = 0
            self.total_alignment_checks = 0
            self.peak_vol_frac_aligned = 0.0
            self.mean_abs_cos_theta_history = []
            self.vol_frac_aligned_history = []
            
        # Leaky guard parameters
        self.dt_min_consecutive = 0
        self.guard_relaxation = 0.0
        self.dt_lockdown_steps = 0
        self.guard_eval_counter = 0
        
        # Safety monitoring
        self.budget_error_count = 0
        self.high_div_count = 0
        self.should_stop = False
        self.nan_detected = False
        self.energy_violation_count = 0
        self.initial_energy = None
        
        # Enhanced budget tracking
        self.eps_filter_history = []
        self.dt_used_history = []
        self.cumulative_dissipation = 0.0
        self.budget_check_interval = 100
        
        # Set up reproducible RNG
        if rng_seed is None:
            rng_seed = int(time.time() * 1000) % 2**32
        self.rng_seed = rng_seed
        self.rng = np.random.Generator(np.random.PCG64(rng_seed))
        print(f"RNG initialized with seed: {rng_seed}")
        np.random.seed(rng_seed)

        # Set backend
        self.use_gpu = backend == "gpu"
        self.xp = xp
        self.fft = xp.fft

        # Pre-compute FFT frequencies with appropriate precision
        kx = self.xp.fft.fftfreq(grid_size, 2 * xp.pi / grid_size) * 2 * xp.pi
        ky = self.xp.fft.fftfreq(grid_size, 2 * xp.pi / grid_size) * 2 * xp.pi
        kz = self.xp.fft.fftfreq(grid_size, 2 * xp.pi / grid_size) * 2 * xp.pi

        # Wavenumbers in compute precision for accurate derivatives
        self.kx = kx.reshape(-1, 1, 1).astype(self.dtype_compute)
        self.ky = ky.reshape(1, -1, 1).astype(self.dtype_compute)
        self.kz = kz.reshape(1, 1, -1).astype(self.dtype_compute)
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2

        # Store full grids for alignment computation
        self.kx_grid, self.ky_grid, self.kz_grid = self.xp.meshgrid(kx, ky, kz, indexing='ij')

        # Pre-compute i*k with appropriate complex precision
        self.ikx = (1j * self.kx).astype(self.complex_dtype)
        self.iky = (1j * self.ky).astype(self.complex_dtype)
        self.ikz = (1j * self.kz).astype(self.complex_dtype)

        # Maximum eigenvalue for guard condition
        kxmax = float(self.xp.max(self.xp.abs(self.kx)))
        kymax = float(self.xp.max(self.xp.abs(self.ky)))
        kzmax = float(self.xp.max(self.xp.abs(self.kz)))
        self.kappa_h = (kxmax**2 + kymax**2 + kzmax**2)

        # De-aliasing mask
        if use_dealiasing:
            kx_cut = dealias_fraction * kxmax
            ky_cut = dealias_fraction * kymax
            kz_cut = dealias_fraction * kzmax
            self.dealias_mask = (self.xp.abs(self.kx) <= kx_cut) & \
                                (self.xp.abs(self.ky) <= ky_cut) & \
                                (self.xp.abs(self.kz) <= kz_cut)
            modes_kept = float(self.xp.sum(self.dealias_mask))
            total_modes = grid_size**3
            self.modes_kept_percent = (modes_kept / total_modes) * 100
            print(f"De-aliasing: keeping {self.modes_kept_percent:.1f}% of modes")
        else:
            self.dealias_mask = self.xp.ones_like(self.k2, dtype=bool)
            self.modes_kept_percent = 100.0

        # Initialize velocity fields with field precision (float32 in mixed mode)
        self.u = self.xp.zeros((grid_size, grid_size, grid_size), dtype=self.dtype_field)
        self.v = self.xp.zeros((grid_size, grid_size, grid_size), dtype=self.dtype_field)
        self.w = self.xp.zeros((grid_size, grid_size, grid_size), dtype=self.dtype_field)

        # Critical quantities
        self.bkm_integral = 0.0
        self.current_time = 0.0
        self.total_steps = 0
        self.time_accumulated = 0.0

        # Enhanced diagnostics
        self.time_history = []
        self.vorticity_max_history = []
        self.bkm_history = []
        self.energy_history = []
        self.enstrophy_history = []
        self.cfl_history = []
        self.rho_history = []
        self.dt_history = []
        self.dt_adv_history = []
        self.dt_diff_history = []
        self.div_max_history = []
        self.div_l2_history = []
        self.eps_in_history = []
        self.eps_diss_history = []
        self.guard_iters_history = []
        self.limiter_history = []
        self.divergence_history = []
        self.lipschitz_history = []
        self.spectra_history = []
        self.spectrum_slopes = []
        self.backoff_events = 0
        self.dt_min_hits = 0
        self.dt_max_hits = 0
        self.initial_vort_ratio = None

        print(f"Solver initialized (mixed precision): Re={reynolds_number}, CFL={CFL_target}")
        print(f"  Engine: {ENGINE_ID} (v{ENGINE_VERSION})")
        print(f"  Guard: rho_soft={rho_soft}, rho_hard={rho_hard}")
        print(f"  Projection: tol={projection_tol}, extra_iters={extra_projection_iters}")
        print(f"  Divergence threshold: {divergence_threshold}")
        print(f"  Startup: {startup_steps} steps, dt_max={dt_max_startup:.1e}, growth={growth_cap_startup}")
        if track_alignment:
            print(f"  Alignment tracking: every {align_every} steps, c_star={c_star}, f_star={f_star}")

    def _require_3d(self, *fields, field_names=None):
        """
        Validate that all fields are 3D arrays with correct shape.
        Loud and clear error messages for debugging.
        
        Args:
            *fields: Variable number of fields to check
            field_names: Optional list of field names for better error messages
        """
        for i, f in enumerate(fields):
            field_name = field_names[i] if field_names and i < len(field_names) else f"field_{i}"
            
            # Check dimensionality
            if f.ndim != 3:
                raise ValueError(
                    f"[3D SAFETY CHECK FAILED] {field_name}: "
                    f"Expected 3D field, got {f.ndim}D with shape {f.shape}. "
                    f"Engine requires 3D fields for spectral operations."
                )
            
            # Check shape matches grid
            if f.shape != (self.nx, self.ny, self.nz):
                raise ValueError(
                    f"[3D SAFETY CHECK FAILED] {field_name}: "
                    f"Shape {f.shape} doesn't match grid ({self.nx}, {self.ny}, {self.nz})"
                )

    def emergency_brake(self, trigger_type, value, threshold, action="lockdown"):
        """
        Emergency brake system with detailed logging.
        
        Args:
            trigger_type: What triggered the brake (e.g., "rho", "divergence", "energy")
            value: Current value that triggered
            threshold: Threshold that was exceeded
            action: What action to take ("lockdown", "stop", "checkpoint")
        """
        print(f"\n{'='*60}")
        print(f"[EMERGENCY BRAKE ACTIVATED]")
        print(f"  Trigger: {trigger_type}")
        print(f"  Value: {value:.6f}")
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Action: {action}")
        print(f"  Time: t={self.current_time:.4f}")
        print(f"  Step: {self.total_steps}")
        print(f"{'='*60}\n")
        
        if action == "lockdown":
            self.dt_lockdown_steps = 200
            self.dt *= 0.25
        elif action == "stop":
            self.should_stop = True
            self.save_checkpoint(f'emergency_{trigger_type}_step{self.total_steps}.h5')
        elif action == "checkpoint":
            self.save_checkpoint(f'emergency_{trigger_type}_step{self.total_steps}.h5')

    def compute_derivatives_fft(self, field, axis):
        """
        Spectral derivatives with mixed precision handling
        
        CRITICAL: This function ONLY accepts 3D fields to ensure shape compatibility
        with the de-aliasing mask. For 2D slices, compute derivatives on the full
        3D field first, then extract the slice.
        """
        # Defensive check - MUST be 3D
        if field.ndim != 3:
            raise ValueError(
                f"compute_derivatives_fft requires a 3D field, got {field.ndim}D with shape {field.shape}. "
                f"For 2D slices, compute derivatives on the full 3D field first, then extract the slice."
            )
        
        # Ensure field shape matches expected grid
        if field.shape != (self.nx, self.ny, self.nz):
            raise ValueError(
                f"Field shape {field.shape} doesn't match grid ({self.nx}, {self.ny}, {self.nz})"
            )
        
        # Convert to compute precision for accurate FFT operations if needed
        if self.precision_mode == 'mixed' and field.dtype != self.dtype_compute:
            field_compute = field.astype(self.dtype_compute)
        else:
            field_compute = field
        
        # Forward FFT
        field_hat = self.fft.fftn(field_compute)
        
        # Apply de-aliasing mask if enabled
        if self.use_dealiasing:
            # Verify mask shape matches FFT output
            if self.dealias_mask.shape != field_hat.shape:
                raise ValueError(
                    f"De-aliasing mask shape {self.dealias_mask.shape} doesn't match "
                    f"FFT output shape {field_hat.shape}"
                )
            field_hat *= self.dealias_mask
        
        # Use pre-computed i*k and in-place multiply to avoid allocation
        if axis == 0:
            ik = self.ikx
        elif axis == 1:
            ik = self.iky
        elif axis == 2:
            ik = self.ikz
        else:
            raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
        
        # In-place multiply to save memory
        self.xp.multiply(ik, field_hat, out=field_hat)
        
        # Inverse FFT
        deriv = self.fft.ifftn(field_hat).real
        
        # Convert back to field precision if needed
        if self.precision_mode == 'mixed':
            deriv = deriv.astype(self.dtype_field)
        
        return deriv

    def compute_energy(self):
        """Total kinetic energy - computed in diagnostic precision for accuracy"""
        if self.precision_mode == 'mixed':
            # Cast to float64 for accurate energy computation
            u_diag = self.u.astype(self.dtype_diag)
            v_diag = self.v.astype(self.dtype_diag)
            w_diag = self.w.astype(self.dtype_diag)
            energy = 0.5 * self.xp.mean(u_diag**2 + v_diag**2 + w_diag**2)
        else:
            energy = 0.5 * self.xp.mean(self.u**2 + self.v**2 + self.w**2)
        return float(energy)

    def compute_enstrophy(self):
        """Enstrophy - computed in diagnostic precision for accuracy"""
        # Use field precision for derivatives, diagnostic precision for final calculation
        omega_x = self.compute_derivatives_fft(self.w, 1) - self.compute_derivatives_fft(self.v, 2)
        omega_y = self.compute_derivatives_fft(self.u, 2) - self.compute_derivatives_fft(self.w, 0)
        omega_z = self.compute_derivatives_fft(self.v, 0) - self.compute_derivatives_fft(self.u, 1)
        
        if self.precision_mode == 'mixed':
            # Cast to diagnostic precision for accurate enstrophy
            omega_x = omega_x.astype(self.dtype_diag)
            omega_y = omega_y.astype(self.dtype_diag)
            omega_z = omega_z.astype(self.dtype_diag)
        
        enstrophy = 0.5 * self.xp.mean(omega_x**2 + omega_y**2 + omega_z**2)
        return float(enstrophy)

    def compute_max_vorticity(self):
        """Compute ||omega||_inf in diagnostic precision for accuracy"""
        omega_x = self.compute_derivatives_fft(self.w, 1) - self.compute_derivatives_fft(self.v, 2)
        omega_y = self.compute_derivatives_fft(self.u, 2) - self.compute_derivatives_fft(self.w, 0)
        omega_z = self.compute_derivatives_fft(self.v, 0) - self.compute_derivatives_fft(self.u, 1)

        if self.precision_mode == 'mixed':
            # Use diagnostic precision for accurate max computation
            omega_x = omega_x.astype(self.dtype_diag)
            omega_y = omega_y.astype(self.dtype_diag)  
            omega_z = omega_z.astype(self.dtype_diag)
        
        vorticity_mag = self.xp.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        max_vort = float(self.xp.max(vorticity_mag))
        return max_vort if not self.xp.isnan(max_vort) else 0.0

    def umax(self):
        """Measure current max |u| - diagnostic precision for accuracy"""
        u_clean = self.dealias_field(self.u)
        v_clean = self.dealias_field(self.v)
        w_clean = self.dealias_field(self.w)
        
        if self.precision_mode == 'mixed':
            u_clean = u_clean.astype(self.dtype_diag)
            v_clean = v_clean.astype(self.dtype_diag)
            w_clean = w_clean.astype(self.dtype_diag)
        
        speed = self.xp.sqrt(u_clean**2 + v_clean**2 + w_clean**2)
        return float(self.xp.max(speed))

    def dealias_field(self, field):
        """
        Properly de-alias a field via spectral truncation
        Handles mixed precision appropriately
        """
        # Convert to compute precision for accurate FFT if needed
        if self.precision_mode == 'mixed' and field.dtype != self.dtype_compute:
            field_compute = field.astype(self.dtype_compute)
        else:
            field_compute = field
            
        field_hat = self.fft.fftn(field_compute)
        if self.use_dealiasing:
            field_hat *= self.dealias_mask
        result = self.fft.ifftn(field_hat).real
        
        # Convert back to field precision if needed
        if self.precision_mode == 'mixed':
            result = result.astype(self.dtype_field)
            
        return result

    def compute_laplacian_fft(self, field):
        """Spectral Laplacian with mixed precision handling"""
        # Convert to compute precision for accurate FFT if needed
        if self.precision_mode == 'mixed' and field.dtype != self.dtype_compute:
            field_compute = field.astype(self.dtype_compute)
        else:
            field_compute = field
            
        field_hat = self.fft.fftn(field_compute)
        if self.use_dealiasing:
            field_hat *= self.dealias_mask
        laplacian_hat = -self.k2 * field_hat
        result = self.fft.ifftn(laplacian_hat).real
        
        # Convert back to field precision if needed
        if self.precision_mode == 'mixed':
            result = result.astype(self.dtype_field)
            
        return result

    def compute_divergence_field(self, u, v, w):
        """Compute divergence field"""
        dudx = self.compute_derivatives_fft(u, 0)
        dvdy = self.compute_derivatives_fft(v, 1)
        dwdz = self.compute_derivatives_fft(w, 2)
        return dudx + dvdy + dwdz

    def compute_divergence_metrics_fields(self, u, v, w):
        """Compute divergence metrics - diagnostic precision for accuracy"""
        div = self.compute_divergence_field(u, v, w)
        
        if self.precision_mode == 'mixed':
            div = div.astype(self.dtype_diag)
        
        div_max = float(self.xp.max(self.xp.abs(div)))
        div_l2 = float(self.xp.sqrt(self.xp.mean(div**2)))
        
        if self.use_gpu:
            del div
        return div_max, div_l2

    def compute_divergence_metrics(self):
        """Compute divergence metrics for current fields"""
        return self.compute_divergence_metrics_fields(self.u, self.v, self.w)

    def compute_max_divergence(self):
        """Legacy interface for max divergence"""
        div_max, _ = self.compute_divergence_metrics()
        return div_max

    def compute_slices(self, axis='z', index='mid', include_diagnostics=True):
        """
        Extract 2D slices with stable semantics and optional diagnostics.
        Enhanced version with speed and vorticity magnitude.
        
        Args:
            axis: 'x', 'y', or 'z' - axis perpendicular to slice
            index: 'mid' or integer - position along axis
            include_diagnostics: If True, compute derived quantities
            
        Returns:
            Dictionary with slice data including velocity and optional diagnostics
        """
        xp = self.xp
        
        # Validate inputs are 3D
        self._require_3d(self.u, self.v, self.w, field_names=['u', 'v', 'w'])
        
        # Determine slice index
        if index == 'mid':
            if axis == 'z':
                idx = self.nz // 2
            elif axis == 'y':
                idx = self.ny // 2
            elif axis == 'x':
                idx = self.nx // 2
            else:
                raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")
        else:
            idx = int(index)
        
        # Extract 2D slices based on axis
        if axis == 'z':
            slicer = (slice(None), slice(None), idx)
        elif axis == 'y':
            slicer = (slice(None), idx, slice(None))
        elif axis == 'x':
            slicer = (idx, slice(None), slice(None))
        else:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")
        
        # Extract velocity slices and convert to diagnostic precision
        u_slice = self.u[slicer].astype(self.dtype_diag)
        v_slice = self.v[slicer].astype(self.dtype_diag)
        w_slice = self.w[slicer].astype(self.dtype_diag)
        
        # Compute speed magnitude
        speed_slice = xp.sqrt(u_slice**2 + v_slice**2 + w_slice**2)
        
        result = {
            'axis': axis,
            'index': idx,
            'u': u_slice,
            'v': v_slice,
            'w': w_slice,
            'speed': speed_slice
        }
        
        # Add diagnostics if requested
        if include_diagnostics:
            # Compute vorticity on full 3D fields first (for proper spectral ops)
            omega_x = self.compute_derivatives_fft(self.w, 1) - self.compute_derivatives_fft(self.v, 2)
            omega_y = self.compute_derivatives_fft(self.u, 2) - self.compute_derivatives_fft(self.w, 0)
            omega_z = self.compute_derivatives_fft(self.v, 0) - self.compute_derivatives_fft(self.u, 1)
            
            # Extract vorticity slices and convert to diagnostic precision
            omega_x_slice = omega_x[slicer].astype(self.dtype_diag)
            omega_y_slice = omega_y[slicer].astype(self.dtype_diag)
            omega_z_slice = omega_z[slicer].astype(self.dtype_diag)
            
            # Compute vorticity magnitude
            vorticity_mag_slice = xp.sqrt(omega_x_slice**2 + omega_y_slice**2 + omega_z_slice**2)
            
            result.update({
                'omega_x': omega_x_slice,
                'omega_y': omega_y_slice,
                'omega_z': omega_z_slice,
                'vorticity_mag': vorticity_mag_slice
            })
            
            # Clean up 3D arrays if using GPU
            if self.use_gpu:
                del omega_x, omega_y, omega_z
        
        return result
    
    def compute_slices_simple(self):
        """
        Simple slice helper with stable semantics (as suggested).
        Returns mid-plane slices of speed in all three orientations.
        
        Returns:
            Dictionary with xy, xz, yz slices of speed magnitude
        """
        xp = self.xp
        nx, ny, nz = self.nx, self.ny, self.nz
        xmid, ymid, zmid = nx//2, ny//2, nz//2
        
        # Compute speed magnitude in diagnostic precision
        if self.precision_mode == 'mixed':
            u_diag = self.u.astype(self.dtype_diag)
            v_diag = self.v.astype(self.dtype_diag)
            w_diag = self.w.astype(self.dtype_diag)
            speed = xp.sqrt(u_diag**2 + v_diag**2 + w_diag**2)
        else:
            speed = xp.sqrt(self.u**2 + self.v**2 + self.w**2)
        
        return {
            "xy": speed[:, :, zmid].astype(self.dtype_diag),
            "xz": speed[:, ymid, :].astype(self.dtype_diag),
            "yz": speed[xmid, :, :].astype(self.dtype_diag),
        }

    def compute_alignment_metrics(self, u, v, w):
        """
        Compute vorticity-stretching alignment metrics
        
        This measures how aligned vorticity omega is with its stretching direction S*omega,
        where S is the strain-rate tensor. This is the key quantity for vortex stretching.
        
        Returns:
            mean_abs_cos_theta: Volume mean of |cos(theta)|
            vol_frac_aligned: Fraction of points with |cos(theta)| > c_star
            aligned_engaged: Boolean, True if vol_frac_aligned > f_star
        """
        xp = self.xp
        subsample = self.alignment_subsample
        
        # Subsample if requested (for computational efficiency)
        if subsample > 1:
            u_sub = u[::subsample, ::subsample, ::subsample]
            v_sub = v[::subsample, ::subsample, ::subsample]
            w_sub = w[::subsample, ::subsample, ::subsample]
        else:
            u_sub, v_sub, w_sub = u, v, w
        
        # Convert to compute precision for accurate FFT
        if self.precision_mode == 'mixed':
            u_sub = u_sub.astype(self.dtype_compute)
            v_sub = v_sub.astype(self.dtype_compute)
            w_sub = w_sub.astype(self.dtype_compute)
        
        # Transform to Fourier space for derivatives
        uk_hat = xp.fft.fftn(u_sub)
        vk_hat = xp.fft.fftn(v_sub)
        wk_hat = xp.fft.fftn(w_sub)
        
        # Get wavenumbers (adjust for subsampling)
        if subsample > 1:
            nx_sub = u_sub.shape[0]
            kx = xp.fft.fftfreq(nx_sub, d=self.Lx/nx_sub) * 2 * xp.pi
            ky = xp.fft.fftfreq(nx_sub, d=self.Ly/nx_sub) * 2 * xp.pi
            kz = xp.fft.fftfreq(nx_sub, d=self.Lz/nx_sub) * 2 * xp.pi
            kx_grid, ky_grid, kz_grid = xp.meshgrid(kx, ky, kz, indexing='ij')
        else:
            kx_grid, ky_grid, kz_grid = self.kx_grid, self.ky_grid, self.kz_grid
        
        # Compute all velocity gradients (streaming to minimize memory)
        dudx = xp.fft.ifftn(1j * kx_grid * uk_hat).real
        dudy = xp.fft.ifftn(1j * ky_grid * uk_hat).real
        dudz = xp.fft.ifftn(1j * kz_grid * uk_hat).real
        
        dvdx = xp.fft.ifftn(1j * kx_grid * vk_hat).real
        dvdy = xp.fft.ifftn(1j * ky_grid * vk_hat).real
        dvdz = xp.fft.ifftn(1j * kz_grid * vk_hat).real
        
        dwdx = xp.fft.ifftn(1j * kx_grid * wk_hat).real
        dwdy = xp.fft.ifftn(1j * ky_grid * wk_hat).real
        dwdz = xp.fft.ifftn(1j * kz_grid * wk_hat).real
        
        # Compute vorticity components
        omega_x = dwdy - dvdz
        omega_y = dudz - dwdx
        omega_z = dvdx - dudy
        
        # Build strain-rate tensor S = 0.5 * (grad_u + grad_u^T)
        # Diagonal components
        S_xx = dudx
        S_yy = dvdy
        S_zz = dwdz
        
        # Off-diagonal components (symmetric)
        S_xy = 0.5 * (dudy + dvdx)
        S_xz = 0.5 * (dudz + dwdx)
        S_yz = 0.5 * (dvdz + dwdy)
        
        # Compute S*omega (matrix-vector product)
        S_omega_x = S_xx * omega_x + S_xy * omega_y + S_xz * omega_z
        S_omega_y = S_xy * omega_x + S_yy * omega_y + S_yz * omega_z
        S_omega_z = S_xz * omega_x + S_yz * omega_y + S_zz * omega_z
        
        # Compute alignment metric in diagnostic precision
        if self.precision_mode == 'mixed':
            omega_x = omega_x.astype(self.dtype_diag)
            omega_y = omega_y.astype(self.dtype_diag)
            omega_z = omega_z.astype(self.dtype_diag)
            S_omega_x = S_omega_x.astype(self.dtype_diag)
            S_omega_y = S_omega_y.astype(self.dtype_diag)
            S_omega_z = S_omega_z.astype(self.dtype_diag)
        
        omega_dot_Somega = omega_x * S_omega_x + omega_y * S_omega_y + omega_z * S_omega_z
        omega_mag = xp.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        Somega_mag = xp.sqrt(S_omega_x**2 + S_omega_y**2 + S_omega_z**2)
        
        # Small epsilon to avoid division by zero
        eps = 1e-10
        cos_theta = xp.abs(omega_dot_Somega) / (omega_mag * Somega_mag + eps)
        
        # Compute metrics
        mean_abs_cos_theta = xp.mean(cos_theta)
        vol_frac_aligned = xp.mean(cos_theta > self.c_star)
        aligned_engaged = vol_frac_aligned > self.f_star
        mean_abs_cos_theta = float(mean_abs_cos_theta)
        vol_frac_aligned = float(vol_frac_aligned)
        return mean_abs_cos_theta, vol_frac_aligned, aligned_engaged

    def compute_cumulative_budget(self):
        """
        Compute cumulative energy budget accounting for all dissipation sources
        Fixes budget accounting errors from dt changes
        """
        if self.initial_energy is None or len(self.eps_diss_history) == 0:
            return 0.0, 0.0
        
        # Current energy
        e_current = self.compute_energy()
        
        # Total energy change
        de_total = e_current - self.initial_energy
        
        # Cumulative dissipation using actual dt history
        total_dissipated = 0.0
        
        # Use dt_used_history if available for accurate accounting
        if hasattr(self, 'dt_used_history') and len(self.dt_used_history) > 0:
            # Use actual timesteps from history
            n_steps = min(len(self.eps_diss_history), len(self.dt_used_history))
            for i in range(n_steps):
                total_dissipated += self.eps_diss_history[i] * self.dt_used_history[i]
        elif hasattr(self, 'dt_history') and len(self.dt_history) > 0:
            # Fallback to dt_history
            n_steps = min(len(self.eps_diss_history), len(self.dt_history))
            for i in range(n_steps):
                total_dissipated += self.eps_diss_history[i] * self.dt_history[i]
        else:
            # Last fallback: use average dt
            if self.total_steps > 0:
                avg_dt = self.current_time / self.total_steps
                total_dissipated = sum(self.eps_diss_history) * avg_dt
        
        # Add filtering dissipation if tracked
        if hasattr(self, 'eps_filter_history') and len(self.eps_filter_history) > 0:
            if len(self.dt_used_history) > 0:
                n_steps = min(len(self.eps_filter_history), len(self.dt_used_history))
                for i in range(n_steps):
                    total_dissipated += self.eps_filter_history[i] * self.dt_used_history[i]
        
        # Expected energy (for unforced flow)
        e_expected = self.initial_energy - total_dissipated
        
        # Budget error
        budget_error = abs(e_current - e_expected)
        budget_relative = budget_error / abs(self.initial_energy) if self.initial_energy != 0 else 0
        
        return budget_error, budget_relative

    def dt_advective(self, umax):
        """Advective CFL-based dt limit"""
        if umax < 1e-12:
            return 1e9
        dx = self.dx
        return self.CFL_target * dx / umax

    def dt_diffusive(self):
        """
        Diffusive dt limit for explicit viscosity
        Proper stability bound without unsafe multiplier
        """
        d = 3.0
        safety_factor = 0.8
        return safety_factor * (self.dx**2) / (2.0 * d * self.viscosity)

    def update_viscosity(self):
        """Update viscosity based on ramp schedule"""
        if self.viscosity_ramp_time > 0 and self.current_time < self.viscosity_ramp_time:
            t_frac = self.current_time / self.viscosity_ramp_time
            # Use numpy for scalar operations (works with both backends)
            ramp = 0.5 * (1 + np.cos(np.pi * t_frac))
            factor = 1.0 + ramp * (self.viscosity_ramp_factor - 1.0)
            self.viscosity = self.viscosity_target * factor
        else:
            self.viscosity = self.viscosity_target

    def compute_max_vorticity_filtered(self):
        """
        Compute max vorticity from de-aliased fields
        """
        # De-alias velocity fields first
        u_clean = self.dealias_field(self.u)
        v_clean = self.dealias_field(self.v)
        w_clean = self.dealias_field(self.w)
        
        # Compute vorticity from clean fields
        omega_x = self.compute_derivatives_fft(w_clean, 1) - self.compute_derivatives_fft(v_clean, 2)
        omega_y = self.compute_derivatives_fft(u_clean, 2) - self.compute_derivatives_fft(w_clean, 0)
        omega_z = self.compute_derivatives_fft(v_clean, 0) - self.compute_derivatives_fft(u_clean, 1)
        
        # Use diagnostic precision for max computation
        if self.precision_mode == 'mixed':
            omega_x = omega_x.astype(self.dtype_diag)
            omega_y = omega_y.astype(self.dtype_diag)
            omega_z = omega_z.astype(self.dtype_diag)
        
        vorticity_mag = self.xp.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        max_vort = float(self.xp.max(vorticity_mag))
        
        return max_vort if not self.xp.isnan(max_vort) else 0.0

    def guard_rho(self, dt, umax, c_eff=None):
        """
        Compute guard rho with filtered Lipschitz estimate
        """
        dx = self.dx
        
        # Use filtered max vorticity for cleaner Lipschitz
        vort_max_filtered = self.compute_max_vorticity_filtered()
        Lip = self.C_adv * 2.0 * vort_max_filtered  # 2*||omega||_inf is more robust
        
        # Store for diagnostics
        if not hasattr(self, 'lipschitz_history'):
            self.lipschitz_history = []
        self.lipschitz_history.append(Lip)
        
        # Log every 1000 steps for debugging (reduce frequency)
        if self.total_steps % 1000 == 0 and self.total_steps > 0:
            dt_adv = self.CFL_target * dx / max(umax, 1e-12)
            dt_diff = self.dt_diffusive()
            print(f"    [GUARD DEBUG] dt={dt:.3e}, dt_adv={dt_adv:.3e}, dt_diff={dt_diff:.3e}")
            print(f"                  Lip={Lip:.3e}, nu*kappa_h={self.viscosity*self.kappa_h:.3e}")
        
        if dt < 1e-8:
            # Taylor expansion for small dt
            rho = 1.0 + dt * (Lip - self.viscosity * self.kappa_h)
        else:
            denom = 1.0 + self.viscosity * dt * self.kappa_h
            numer = 1.0 + dt * Lip
            rho = numer / denom
        
        return rho

    def choose_dt(self):
        """
        Enhanced dt selection with startup clamping and emergency brakes
        CRITICAL FIX: Emergency brake respects configured rho_hard threshold
        """
        umax_now = self.umax()
        
        # Check startup completion criteria
        if not self.startup_complete:
            if self.total_steps >= self.startup_steps:
                # Additional criteria: stable CFL and rho
                if len(self.cfl_history) >= 200 and len(self.rho_history) >= 200:
                    # Convert to numpy for min/max operations if using CuPy
                    if self.use_gpu:
                        recent_cfl = self.xp.array(self.cfl_history[-200:])
                        recent_rho = self.xp.array(self.rho_history[-200:])
                        min_cfl = float(self.xp.min(recent_cfl))
                        max_rho = float(self.xp.max(recent_rho))
                    else:
                        recent_cfl = np.array(self.cfl_history[-200:])
                        recent_rho = np.array(self.rho_history[-200:])
                        min_cfl = np.min(recent_cfl)
                        max_rho = np.max(recent_rho)
                    
                    if min_cfl > 5e-3 and max_rho < 0.995:
                        self.startup_complete = True
                        print(f"  [STARTUP] Complete at step {self.total_steps}")
        
        # Detect startup phase and apply clamping
        startup_phase = not self.startup_complete
        if startup_phase:
            effective_dt_max = min(self.dt_max_startup, self.dt_max)
            effective_growth_cap = self.growth_cap_startup
        else:
            effective_dt_max = self.dt_max
            effective_growth_cap = 1.3  # Normal growth after startup
        
        # Check if in lockdown
        if self.dt_lockdown_steps > 0:
            self.dt_lockdown_steps -= 1
            dt_final = self.dt_prev * 0.5
            # Use Python's min/max for scalar clipping
            dt_final = max(self.dt_min, min(dt_final, effective_dt_max))
            actual_cfl = umax_now * dt_final / max(self.dx, 1e-12)
            rho_final = self.guard_rho(dt_final, umax_now)
            self.dt_used = dt_final
            return dt_final, actual_cfl, rho_final, "lockdown", 0
        
        # Initialize PI controller state if needed
        if not hasattr(self, 'pi_integral'):
            self.pi_integral = 0.0
            self.pi_last_error = 0.0
            self.pi_active = False
        
        # Base constraints
        dt_cfl = self.dt_advective(umax_now)
        dt_nu = self.dt_diffusive()
        
        # Store these for history tracking
        dt_adv = dt_cfl
        dt_diff = dt_nu
        
        # Apply leaky guard relaxation
        effective_rho_soft = self.rho_soft + self.guard_relaxation
        effective_rho_hard = self.rho_hard + self.guard_relaxation
        
        # Target rho for PI control
        rho_target = effective_rho_soft + 0.5 * (effective_rho_hard - effective_rho_soft)
        
        # Adaptive PI gains based on grid size
        if self.nx <= 64:
            Kp = 0.05
            Ki = 0.005
            shrink_cap = 0.8
            deadband = 0.003
        elif self.nx <= 128:
            Kp = 0.08
            Ki = 0.008
            shrink_cap = 0.7
            deadband = 0.002
        else:
            Kp = 0.1
            Ki = 0.01
            shrink_cap = 0.5
            deadband = 0.001
        
        # Initial proposal
        if self.dt_prev > 0:
            # Use PI control if stable and past startup
            if len(self.rho_history) > 10 and self.pi_active and not startup_phase:
                rho_current = self.rho_history[-1] if self.rho_history else 1.0
                error = rho_target - rho_current
                
                if abs(error) < deadband:
                    pi_factor = 1.0
                else:
                    # Use Python's min/max for scalar clipping
                    self.pi_integral = max(-0.2, min(self.pi_integral + Ki * error, 0.2))
                    pi_factor = 1.0 + Kp * error + self.pi_integral
                    pi_factor = max(shrink_cap, min(pi_factor, effective_growth_cap))
                
                dt_proposal = self.dt_prev * pi_factor
            else:
                # Standard growth limiting with startup cap
                dt_proposal = self.dt_prev * effective_growth_cap
            
            dt_proposal = min(dt_proposal, dt_cfl, dt_nu)
        else:
            dt_proposal = min(dt_cfl, dt_nu)
        
        # Quick check
        rho_initial = self.guard_rho(dt_proposal, umax_now)
        if rho_initial >= effective_rho_hard * 0.99:
            dt_proposal = self.dt_min * 10
            self.pi_active = False
        else:
            self.pi_active = not startup_phase
        
        # Use Python's min/max for scalar clipping (works with both NumPy and CuPy)
        dt_proposal = max(self.dt_min, min(dt_proposal, effective_dt_max))
        
        if dt_cfl <= dt_nu:
            limiter = "advective"
        else:
            limiter = "diffusive"
        
        # Bisection search with relaxed thresholds
        dt = dt_proposal
        dt_lo = None
        dt_hi = None
        
        iters = 0
        max_iters = 30
        
        while iters < max_iters:
            rho = self.guard_rho(dt, umax_now)
            
            if rho < effective_rho_soft:
                dt_hi = dt
                if dt_lo is None:
                    dt_new = min(dt * 1.1, effective_dt_max)
                    if dt_new > dt * 1.05:
                        dt = dt_new
                    else:
                        break
                else:
                    dt_mid = 0.5 * (dt_lo + dt_hi)
                    if abs(dt_mid - dt) / dt < 0.01:
                        break
                    dt = dt_mid
                    
            elif rho < effective_rho_hard:
                dt_hi = dt
                limiter = "guard"
                break
                
            else:
                dt_lo = dt
                limiter = "guard"
                self.backoff_events += 1
                
                if dt_hi is None:
                    dt = max(dt * 0.5, self.dt_min)
                else:
                    dt_mid = 0.5 * (dt_lo + dt_hi)
                    dt = dt_mid
            
            iters += 1
        
        # Final clipping
        dt_final = max(self.dt_min, min(dt, effective_dt_max))
        
        # CRITICAL FIX: Emergency brake uses configured rho_hard, not hardcoded 1.0
        rho_final = self.guard_rho(dt_final, umax_now)
        if rho_final >= self.rho_hard:  # THIS IS THE FIX
            self.emergency_brake("rho", rho_final, threshold=self.rho_hard, action="lockdown")
            dt_final = min(dt_final * 0.25, self.dt_min * 2)
            self.dt_lockdown_steps = 200
            limiter = "emergency"
        
        # Track dt_min jail
        if dt_final == self.dt_min:
            self.dt_min_consecutive += 1
            self.dt_min_hits += 1
        else:
            self.dt_min_consecutive = 0
        
        if dt_final == effective_dt_max:
            self.dt_max_hits += 1
        
        # Compute final metrics
        actual_cfl = umax_now * dt_final / max(self.dx, 1e-12)
        
        # Store history
        self.cfl_history.append(actual_cfl)
        self.rho_history.append(rho_final)
        self.dt_history.append(dt_final)
        self.dt_adv_history.append(dt_adv)
        self.dt_diff_history.append(dt_diff)
        self.guard_iters_history.append(iters)
        self.limiter_history.append(limiter)
        
        self.dt_prev = dt_final
        self.dt_used = dt_final
        
        return dt_final, actual_cfl, rho_final, limiter, iters

    def compute_skew_symmetric_convection(self, u, v, w):
        """
        Compute energy-conserving skew-symmetric convection term
        C = 0.5 * (u·∇u + ∇·(uu))
        Memory-optimized: stream derivatives to avoid holding all 9 at once
        """
        # Initialize accumulation arrays
        conv_u_std = self.xp.zeros_like(u)
        conv_v_std = self.xp.zeros_like(v)
        conv_w_std = self.xp.zeros_like(w)
        
        # Standard convection u·∇u (streamed to save memory)
        tmp = self.compute_derivatives_fft(u, 0)
        conv_u_std += u * tmp
        del tmp
        tmp = self.compute_derivatives_fft(u, 1)
        conv_u_std += v * tmp
        del tmp
        tmp = self.compute_derivatives_fft(u, 2)
        conv_u_std += w * tmp
        del tmp
        
        tmp = self.compute_derivatives_fft(v, 0)
        conv_v_std += u * tmp
        del tmp
        tmp = self.compute_derivatives_fft(v, 1)
        conv_v_std += v * tmp
        del tmp
        tmp = self.compute_derivatives_fft(v, 2)
        conv_v_std += w * tmp
        del tmp
        
        tmp = self.compute_derivatives_fft(w, 0)
        conv_w_std += u * tmp
        del tmp
        tmp = self.compute_derivatives_fft(w, 1)
        conv_w_std += v * tmp
        del tmp
        tmp = self.compute_derivatives_fft(w, 2)
        conv_w_std += w * tmp
        del tmp
        
        # Divergence form ∇·(uu) (also streamed)
        conv_u_div = self.compute_derivatives_fft(u * u, 0)
        tmp = self.compute_derivatives_fft(u * v, 1)
        conv_u_div += tmp
        del tmp
        tmp = self.compute_derivatives_fft(u * w, 2)
        conv_u_div += tmp
        del tmp
        
        conv_v_div = self.compute_derivatives_fft(v * u, 0)
        tmp = self.compute_derivatives_fft(v * v, 1)
        conv_v_div += tmp
        del tmp
        tmp = self.compute_derivatives_fft(v * w, 2)
        conv_v_div += tmp
        del tmp
        
        conv_w_div = self.compute_derivatives_fft(w * u, 0)
        tmp = self.compute_derivatives_fft(w * v, 1)
        conv_w_div += tmp
        del tmp
        tmp = self.compute_derivatives_fft(w * w, 2)
        conv_w_div += tmp
        del tmp
        
        # Skew-symmetric form (energy-conserving)
        conv_u = 0.5 * (conv_u_std + conv_u_div)
        conv_v = 0.5 * (conv_v_std + conv_v_div)
        conv_w = 0.5 * (conv_w_std + conv_w_div)
        
        # Clean up intermediate arrays
        del conv_u_std, conv_v_std, conv_w_std
        del conv_u_div, conv_v_div, conv_w_div
        
        # De-alias the result
        conv_u = self.dealias_field(conv_u)
        conv_v = self.dealias_field(conv_v)
        conv_w = self.dealias_field(conv_w)
        
        return conv_u, conv_v, conv_w

    def project_div_free(self, u, v, w, extra_iters=None):
        """Enhanced projection with tighter tolerance and optional extra iterations"""
        if extra_iters is None:
            extra_iters = self.extra_projection_iters
        
        # Convert to compute precision for accurate projection
        if self.precision_mode == 'mixed':
            u = u.astype(self.dtype_compute)
            v = v.astype(self.dtype_compute)
            w = w.astype(self.dtype_compute)
            
        uhat = self.fft.fftn(u)
        vhat = self.fft.fftn(v)
        what = self.fft.fftn(w)

        k2_safe = self.k2.copy()
        k2_safe[0, 0, 0] = 1.0

        # Initial projection
        kdotu = self.kx * uhat + self.ky * vhat + self.kz * what
        uhat -= self.kx * kdotu / k2_safe
        vhat -= self.ky * kdotu / k2_safe
        what -= self.kz * kdotu / k2_safe

        if self.use_dealiasing:
            uhat *= self.dealias_mask
            vhat *= self.dealias_mask
            what *= self.dealias_mask

        # Convert back to physical space
        u_proj = self.fft.ifftn(uhat).real
        v_proj = self.fft.ifftn(vhat).real
        w_proj = self.fft.ifftn(what).real
        
        # Extra iterations for high-vorticity cases
        for _ in range(extra_iters):
            div_max, _ = self.compute_divergence_metrics_fields(u_proj, v_proj, w_proj)
            if div_max < self.projection_tol:
                break
                
            # Additional correction
            div = self.compute_divergence_field(u_proj, v_proj, w_proj)
            div_hat = self.fft.fftn(div)
            phi_hat = div_hat / k2_safe
            
            u_corr = self.fft.ifftn(-self.kx * phi_hat).real
            v_corr = self.fft.ifftn(-self.ky * phi_hat).real
            w_corr = self.fft.ifftn(-self.kz * phi_hat).real
            
            u_proj -= u_corr
            v_proj -= v_corr
            w_proj -= w_corr
        
        # Convert back to field precision if needed
        if self.precision_mode == 'mixed':
            u_proj = u_proj.astype(self.dtype_field)
            v_proj = v_proj.astype(self.dtype_field)
            w_proj = w_proj.astype(self.dtype_field)

        return u_proj, v_proj, w_proj

    def compute_injection_rate(self, f_u, f_v, f_w):
        """Compute energy injection rate from forcing"""
        # Use diagnostic precision for accurate energy computation
        if self.precision_mode == 'mixed':
            u_diag = self.u.astype(self.dtype_diag)
            v_diag = self.v.astype(self.dtype_diag)
            w_diag = self.w.astype(self.dtype_diag)
            f_u_diag = f_u.astype(self.dtype_diag)
            f_v_diag = f_v.astype(self.dtype_diag)
            f_w_diag = f_w.astype(self.dtype_diag)
            eps_in = self.xp.mean(u_diag * f_u_diag + v_diag * f_v_diag + w_diag * f_w_diag)
        else:
            eps_in = self.xp.mean(self.u * f_u + self.v * f_v + self.w * f_w)
        return float(eps_in)

    def compute_energy_spectrum(self):
        """Compute energy spectrum E(k) with slope analysis"""
        # Convert to compute precision for accurate FFT
        if self.precision_mode == 'mixed':
            u_compute = self.u.astype(self.dtype_compute)
            v_compute = self.v.astype(self.dtype_compute)
            w_compute = self.w.astype(self.dtype_compute)
        else:
            u_compute = self.u
            v_compute = self.v
            w_compute = self.w
        
        uhat = self.fft.fftn(u_compute)
        vhat = self.fft.fftn(v_compute)
        what = self.fft.fftn(w_compute)
        
        # Energy in spectral space
        E_k = 0.5 * (self.xp.abs(uhat)**2 + self.xp.abs(vhat)**2 + self.xp.abs(what)**2)
        
        # Radial binning
        k_mag = self.xp.sqrt(self.k2)
        k_max = float(self.xp.max(k_mag))
        n_bins = min(self.nx // 2, 128)
        k_bins = self.xp.linspace(0, k_max, n_bins)
        E_spectrum = self.xp.zeros(n_bins - 1)
        
        for i in range(n_bins - 1):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            E_spectrum[i] = self.xp.sum(E_k[mask])
        
        # Convert to numpy arrays for polyfit
        if self.use_gpu:
            k_bins = k_bins.get()
            E_spectrum = E_spectrum.get()
        else:
            k_bins = np.array(k_bins)
            E_spectrum = np.array(E_spectrum)
        
        # Compute spectral slope in inertial range
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        k_min_fit = 0.05 * k_max
        k_max_fit = 0.3 * k_max
        fit_mask = (k_centers > k_min_fit) & (k_centers < k_max_fit) & (E_spectrum > 0)
        
        slope = np.nan
        n_fit_points = np.sum(fit_mask)
        
        if n_fit_points >= 8:
            log_k = np.log10(k_centers[fit_mask])
            log_E = np.log10(E_spectrum[fit_mask])
            try:
                slope, intercept = np.polyfit(log_k, log_E, 1)
            except:
                slope = np.nan
        
        return k_centers, E_spectrum, slope

    def check_spectrum_health(self, is_forced=False):
        """
        Enhanced spectrum health check with slope analysis
        """
        k_centers, E_spectrum, slope = self.compute_energy_spectrum()
        
        # Check for high-k pile-up
        n_check = max(1, len(E_spectrum) // 10)
        high_k_energy = np.sum(E_spectrum[-n_check:])
        total_energy = np.sum(E_spectrum)
        
        health_ok = True
        
        if total_energy > 0:
            high_k_fraction = high_k_energy / total_energy
            if high_k_fraction > 0.1:
                print(f"  [SPECTRUM] High-k energy pile-up: {high_k_fraction:.1%} in top 10% modes")
                health_ok = False
        
        # Only check slope for forced turbulence
        if not np.isnan(slope):
            self.spectrum_slopes.append(slope)
            
            if is_forced:
                expected_slope = -5.0/3.0
                slope_error = abs(slope - expected_slope) / abs(expected_slope)
                
                k_max = np.max(k_centers) * (k_centers[-1] / k_centers[-2])
                k_min_fit = 0.05 * k_max
                k_max_fit = 0.3 * k_max
                fit_mask = (k_centers > k_min_fit) & (k_centers < k_max_fit) & (E_spectrum > 0)
                n_fit_points = np.sum(fit_mask)
                
                if slope_error > 0.3 and n_fit_points >= 8:
                    print(f"  [SPECTRUM] Slope = {slope:.2f} (expected {expected_slope:.2f} for forced, {n_fit_points} points)")
                    if slope > -1.0:
                        print(f"    WARNING: Very shallow spectrum, possible numerical issues")
                        health_ok = False
            else:
                if self.total_steps % 5000 == 0:
                    print(f"  [SPECTRUM INFO] Slope = {slope:.2f} (unforced flow)")
        
        return health_ok

    def evolve_one_timestep(self, forcing_func=None, forcing_params=None, use_rk2=True):
        """
        Enhanced evolution with proper de-aliasing and energy conservation
        
        Critical features:
        - De-alias fields before computing derivatives
        - Use skew-symmetric convection form
        - De-alias nonlinear products
        - Emergency energy checks for unforced flows
        - Proper time accounting
        - Alignment metrics computation
        - Mixed precision handling throughout
        """
        # Store initial state for diagnostics
        vort_before = self.compute_max_vorticity()
        energy_before = self.compute_energy()
        
        # Store initial energy if not set
        if self.initial_energy is None:
            self.initial_energy = energy_before
        
        # Update viscosity if ramping
        self.update_viscosity()
        
        # De-alias velocity fields BEFORE any computation
        u_clean = self.dealias_field(self.u)
        v_clean = self.dealias_field(self.v)
        w_clean = self.dealias_field(self.w)
        
        if use_rk2:
            # ==== RK2 TIME INTEGRATION WITH PROPER DE-ALIASING ====
            conv_u, conv_v, conv_w = self.compute_skew_symmetric_convection(u_clean, v_clean, w_clean)
            visc_u = self.viscosity * self.compute_laplacian_fft(u_clean)
            visc_v = self.viscosity * self.compute_laplacian_fft(v_clean)
            visc_w = self.viscosity * self.compute_laplacian_fft(w_clean)
            force_u = force_v = force_w = 0
            eps_in = 0.0
            if forcing_func is not None:
                if forcing_params is None:
                    forcing_params = {}
                force = forcing_func(self, **forcing_params)
                if isinstance(force, (tuple, list)) and len(force) == 3:
                    force_u, force_v, force_w = force
                    eps_in = self.compute_injection_rate(force_u, force_v, force_w)
                else:
                    raise ValueError("forcing_func must return (force_u, force_v, force_w)")
            k1_u = -conv_u + visc_u + force_u
            k1_v = -conv_v + visc_v + force_v
            k1_w = -conv_w + visc_w + force_w
            u_star = self.u + self.dt * k1_u
            v_star = self.v + self.dt * k1_v
            w_star = self.w + self.dt * k1_w
            u_star, v_star, w_star = self.project_div_free(u_star, v_star, w_star)
            u_star_clean = self.dealias_field(u_star)
            v_star_clean = self.dealias_field(v_star)
            w_star_clean = self.dealias_field(w_star)
            conv_u, conv_v, conv_w = self.compute_skew_symmetric_convection(u_star_clean, v_star_clean, w_star_clean)
            visc_u = self.viscosity * self.compute_laplacian_fft(u_star_clean)
            visc_v = self.viscosity * self.compute_laplacian_fft(v_star_clean)
            visc_w = self.viscosity * self.compute_laplacian_fft(w_star_clean)
            k2_u = -conv_u + visc_u + force_u
            k2_v = -conv_v + visc_v + force_v
            k2_w = -conv_w + visc_w + force_w
            u_new = self.u + 0.5 * self.dt * (k1_u + k2_u)
            v_new = self.v + 0.5 * self.dt * (k1_v + k2_v)
            w_new = self.w + 0.5 * self.dt * (k1_w + k2_w)
        else:
            # ==== FORWARD EULER (with de-aliasing and skew-symmetric) ====
            conv_u, conv_v, conv_w = self.compute_skew_symmetric_convection(u_clean, v_clean, w_clean)
            visc_u = self.viscosity * self.compute_laplacian_fft(u_clean)
            visc_v = self.viscosity * self.compute_laplacian_fft(v_clean)
            visc_w = self.viscosity * self.compute_laplacian_fft(w_clean)
            force_u = force_v = force_w = 0
            eps_in = 0.0
            if forcing_func is not None:
                if forcing_params is None:
                    forcing_params = {}
                force = forcing_func(self, **forcing_params)
                if isinstance(force, (tuple, list)) and len(force) == 3:
                    force_u, force_v, force_w = force
                    eps_in = self.compute_injection_rate(force_u, force_v, force_w)
                else:
                    raise ValueError("forcing_func must return (force_u, force_v, force_w)")
            u_new = self.u + self.dt * (-conv_u + visc_u + force_u)
            v_new = self.v + self.dt * (-conv_v + visc_v + force_v)
            w_new = self.w + self.dt * (-conv_w + visc_w + force_w)
        
        # Project to divergence-free (final projection)
        extra_iters = self.extra_projection_iters
        if forcing_func is not None or vort_before > 100:
            extra_iters = max(1, extra_iters)
        u_new, v_new, w_new = self.project_div_free(u_new, v_new, w_new, extra_iters)
        
        # Update fields
        self.u = u_new
        self.v = v_new
        self.w = w_new
        
        # Compute alignment metrics periodically
        if self.track_alignment and self.total_steps % self.align_every == 0:
            mean_abs, vol_frac, engaged = self.compute_alignment_metrics(self.u, self.v, self.w)
            self.alignment_logger.write(self.current_time, mean_abs, vol_frac, int(engaged))
            self.total_alignment_checks += 1
            if engaged:
                self.alignment_engaged_steps += 1
            self.peak_vol_frac_aligned = max(self.peak_vol_frac_aligned, vol_frac)
            self.mean_abs_cos_theta_history.append(mean_abs)
            self.vol_frac_aligned_history.append(vol_frac)
            if engaged:
                print(f"  [ALIGNMENT] Engaged at t={self.current_time:.4f}: vol_frac={vol_frac:.3%}")
        
        # Post-evolution diagnostics
        vort_after = self.compute_max_vorticity()
        energy_after = self.compute_energy()
        enstrophy = self.compute_enstrophy()
        div_max, div_l2 = self.compute_divergence_metrics()
        
        # Update BKM integral (trapezoidal rule)
        self.bkm_integral += 0.5 * (vort_before + vort_after) * self.dt_used
        
        # Energy budget check
        eps_diss = 2 * self.viscosity * enstrophy
        dE_dt = (energy_after - energy_before) / self.dt_used
        eps_filter = 0.0
        if self.use_dealiasing:
            expected_visc_loss = eps_diss * self.dt_used
            actual_loss = energy_before - energy_after
            if actual_loss > expected_visc_loss * 1.1:
                eps_filter = (actual_loss - expected_visc_loss) / self.dt_used
                self.eps_filter_history.append(eps_filter)
        
        if forcing_func is not None:
            budget_error = abs(dE_dt + eps_diss + eps_filter - eps_in)
            relative_error = budget_error / max(abs(eps_in), abs(eps_diss), 1e-10)
        else:
            budget_error = abs(dE_dt + eps_diss + eps_filter)
            relative_error = budget_error / max(abs(eps_diss), 1e-10)
            if self.total_steps % self.budget_check_interval == 0:
                budget_error_cum, relative_error_cum = self.compute_cumulative_budget()
                if relative_error_cum > 0.01:
                    if relative_error_cum > 0.02:
                        print(f"  [BUDGET] Cumulative energy error: {relative_error_cum:.1%}")
                    relative_error = relative_error_cum
        
        # Critical energy check for unforced flows
        if forcing_func is None and energy_after > energy_before * 1.001:
            self.energy_violation_count += 1
            if self.energy_violation_count > 5:
                print(f"  [CRITICAL] Energy violation in unforced flow!")
                print(f"    E_new/E_old = {energy_after/energy_before:.6f}")
                print(f"    Stopping simulation for safety")
                self.should_stop = True
                self.save_checkpoint('emergency_energy_violation.h5')
        else:
            self.energy_violation_count = max(0, self.energy_violation_count - 1)
        
        # Check for runaway vorticity
        if vort_after > vort_before * 2.0 and self.total_steps > 100:
            print(f"  [CRITICAL] Vorticity doubling: {vort_before:.1f} -> {vort_after:.1f}")
            self.should_stop = True
            self.save_checkpoint('emergency_vorticity_explosion.h5')
        
        # Regular budget error check
        if relative_error > 0.02:
            self.budget_error_count += 1
            if self.budget_error_count > 100:
                print(f"  [SAFETY] Energy budget error > 2% for 100 steps")
                self.should_stop = True
        else:
            self.budget_error_count = max(0, self.budget_error_count - 1)
        
        # NaN/Inf check
        if self.xp.isnan(vort_after) or self.xp.isinf(vort_after) or \
           self.xp.isnan(energy_after) or self.xp.isinf(energy_after):
            print(f"  [SAFETY] NaN/Inf detected at step {self.total_steps}")
            self.save_checkpoint('emergency_checkpoint.h5')
            self.nan_detected = True
            self.should_stop = True
        
        # Divergence check
        if div_max > self.divergence_threshold:
            self.high_div_count += 1
            if self.high_div_count > 100:
                print(f"  [DIVERGENCE] Persistent high divergence: {div_max:.3e}")
                self.dt *= 0.5
                self.dt_lockdown_steps = 200
                self.high_div_count = 0
        else:
            self.high_div_count = max(0, self.high_div_count - 1)
        
        # Update time and verify consistency
        self.current_time += self.dt_used
        self.time_accumulated += self.dt_used
        self.dt_used_history.append(self.dt_used)
        if abs(self.current_time - self.time_accumulated) > 1e-10:
            print(f"  [WARNING] Time accounting mismatch: {self.current_time:.6f} vs {self.time_accumulated:.6f}")
        
        self.total_steps += 1
        
        # Store diagnostics
        self.time_history.append(self.current_time)
        self.vorticity_max_history.append(vort_after)
        self.bkm_history.append(self.bkm_integral)
        self.energy_history.append(energy_after)
        self.enstrophy_history.append(enstrophy)
        self.div_max_history.append(div_max)
        self.div_l2_history.append(div_l2)
        self.eps_in_history.append(eps_in)
        self.eps_diss_history.append(eps_diss)
        
        # Periodic spectrum check
        if self.total_steps % 1000 == 0:
            is_forced = (forcing_func is not None)
            spectrum_healthy = self.check_spectrum_health(is_forced=is_forced)
            if not spectrum_healthy and is_forced:
                k_centers, E_spectrum, slope = self.compute_energy_spectrum()
                self.spectra_history.append((self.current_time, k_centers, E_spectrum, slope))
        
        # Return diagnostics
        return {
            'vorticity': vort_after,
            'vort_max': vort_after,
            'energy': energy_after,
            'enstrophy': enstrophy,
            'bkm_integral': self.bkm_integral,
            'bkm_rate': vort_after,
            'div_max': div_max,
            'div_l2': div_l2,
            'eps_in': eps_in,
            'eps_diss': eps_diss,
            'budget_error': budget_error,
            'budget_relative': relative_error
        }

    def save_checkpoint(self, filename):
        """Save checkpoint with current state and diagnostics"""
        print(f"  [CHECKPOINT] Saving checkpoint: {filename}")
        
        try:
            import h5py
            
            # Convert CuPy arrays to NumPy for saving if necessary
            if self.use_gpu:
                u_save = self.u.get()
                v_save = self.v.get()
                w_save = self.w.get()
            else:
                u_save = self.u
                v_save = self.v
                w_save = self.w
            
            with h5py.File(filename, 'w') as f:
                # Save velocity fields
                f.create_dataset('u', data=u_save)
                f.create_dataset('v', data=v_save)
                f.create_dataset('w', data=w_save)
                
                # Save scalar quantities
                f.attrs['time'] = self.current_time
                f.attrs['total_steps'] = self.total_steps
                f.attrs['bkm_integral'] = self.bkm_integral
                f.attrs['dt'] = self.dt
                f.attrs['dt_prev'] = self.dt_prev
                f.attrs['reynolds'] = self.Re
                f.attrs['grid_size'] = self.nx
                f.attrs['precision_mode'] = self.precision_mode
                
                # Save histories (convert to numpy if needed)
                if len(self.time_history) > 0:
                    f.create_dataset('time_history', data=np.array(self.time_history))
                if len(self.vorticity_max_history) > 0:
                    f.create_dataset('vorticity_history', data=np.array(self.vorticity_max_history))
                if len(self.bkm_history) > 0:
                    f.create_dataset('bkm_history', data=np.array(self.bkm_history))
                if len(self.energy_history) > 0:
                    f.create_dataset('energy_history', data=np.array(self.energy_history))
                
            print(f"    Checkpoint saved successfully")
            
        except ImportError:
            print(f"    [WARNING] h5py not available, checkpoint not saved")
        except Exception as e:
            print(f"    [ERROR] Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, filename):
        """Load checkpoint to resume simulation"""
        print(f"  [CHECKPOINT] Loading checkpoint: {filename}")
        
        try:
            import h5py
            
            with h5py.File(filename, 'r') as f:
                # Load velocity fields
                u_load = np.array(f['u'])
                v_load = np.array(f['v'])
                w_load = np.array(f['w'])
                
                # Convert to CuPy if using GPU and ensure correct precision
                if self.use_gpu:
                    self.u = self.xp.asarray(u_load, dtype=self.dtype_field)
                    self.v = self.xp.asarray(v_load, dtype=self.dtype_field)
                    self.w = self.xp.asarray(w_load, dtype=self.dtype_field)
                else:
                    self.u = u_load.astype(self.dtype_field)
                    self.v = v_load.astype(self.dtype_field)
                    self.w = w_load.astype(self.dtype_field)
                
                # Load scalar quantities
                self.current_time = f.attrs.get('time', 0.0)
                self.total_steps = f.attrs.get('total_steps', 0)
                self.bkm_integral = f.attrs.get('bkm_integral', 0.0)
                self.dt = f.attrs.get('dt', self.dt)
                self.dt_prev = f.attrs.get('dt_prev', self.dt)
                
                # Load histories
                if 'time_history' in f:
                    self.time_history = list(np.array(f['time_history']))
                if 'vorticity_history' in f:
                    self.vorticity_max_history = list(np.array(f['vorticity_history']))
                if 'bkm_history' in f:
                    self.bkm_history = list(np.array(f['bkm_history']))
                if 'energy_history' in f:
                    self.energy_history = list(np.array(f['energy_history']))
                
            print(f"    Checkpoint loaded: t={self.current_time:.4f}, step={self.total_steps}")
            
        except ImportError:
            print(f"    [ERROR] h5py not available, cannot load checkpoint")
            raise
        except Exception as e:
            print(f"    [ERROR] Failed to load checkpoint: {e}")
            raise


# Simple test if run directly
if __name__ == "__main__":
    print("Testing Unified Mixed Precision BKM Engine...")
    print(f"Backend detected: {backend}")
    
    # Try to create a small solver instance with mixed precision
    solver = CUDABKMSolver(grid_size=32, reynolds_number=100, precision='mixed')
    print(f"Solver created successfully using {solver.xp.__name__}")
    print(f"Field precision: {solver.dtype_field}")
    print(f"Diagnostic precision: {solver.dtype_diag}")
    print(f"Compute precision: {solver.dtype_compute}")
    
    # Run a few steps
    for i in range(5):
        solver.choose_dt()
        result = solver.evolve_one_timestep()
        print(f"Step {i+1}: Energy={result['energy']:.6f}, Vorticity={result['vorticity']:.6f}")
    
    print("Mixed precision test completed!")