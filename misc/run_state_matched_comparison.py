#!/usr/bin/env python3
"""
run_state_matched_comparison.py
===============================
Definitive test to resolve the N=128 vs N=256 discrepancy.

Key fixes based on GPT feedback:
1. Match comparisons by DYNAMICAL STATE (ω level), not wall-clock time
2. Grid-invariant k_eff: use k_rms in physical units, plus k_eff/k_nyquist
3. Run N=256 until ω matches N=128 levels (or T=10)
4. Output actual spectra for visual comparison

The question: Is N=128's low τ_norm real physics or numerical artifact?

Author: Josh Curry & Claude
Date: December 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_bkm_engine import CUDABKMSolver, backend
from heat_kernel_ns_integration_v5_1 import (
    HeatKernelConfigV51,
    HeatKernelAlignmentTrackerV51,
)

print(f"Backend: {backend}")


# ============================================================================
# Grid-Invariant Metrics
# ============================================================================

def compute_grid_invariant_k(solver) -> dict:
    """
    Compute multiple k metrics that are grid-invariant.
    
    Returns:
        k_eff_abs: Enstrophy-weighted k in physical units
        k_rms: RMS wavenumber from energy spectrum
        k_eff_norm: k_eff / k_nyquist (dimensionless, 0-1)
        k_peak: Peak of enstrophy spectrum
    """
    xp = solver.xp
    N = solver.nx
    
    # Compute vorticity components in Fourier space
    u_hat = xp.fft.fftn(solver.u)
    v_hat = xp.fft.fftn(solver.v)
    w_hat = xp.fft.fftn(solver.w)
    
    # Wavenumber grid
    k_1d = xp.fft.fftfreq(N, d=1.0/N) * 2 * np.pi / solver.Lx
    kx, ky, kz = xp.meshgrid(k_1d, k_1d, k_1d, indexing='ij')
    k_mag = xp.sqrt(kx**2 + ky**2 + kz**2)
    
    # Vorticity in Fourier space: ω = ∇ × u
    # ω_x = ∂w/∂y - ∂v/∂z = i(ky*w - kz*v)
    # ω_y = ∂u/∂z - ∂w/∂x = i(kz*u - kx*w)
    # ω_z = ∂v/∂x - ∂u/∂y = i(kx*v - ky*u)
    omega_x_hat = 1j * (ky * w_hat - kz * v_hat)
    omega_y_hat = 1j * (kz * u_hat - kx * w_hat)
    omega_z_hat = 1j * (kx * v_hat - ky * u_hat)
    
    # Enstrophy spectrum |ω̂|²
    enstrophy_spec = (xp.abs(omega_x_hat)**2 + 
                      xp.abs(omega_y_hat)**2 + 
                      xp.abs(omega_z_hat)**2)
    
    # Physical parameters
    k_nyquist = np.pi * N / solver.Lx  # Physical Nyquist
    k_dealias = k_nyquist * 2/3  # After 2/3 dealiasing
    
    # Enstrophy-weighted k (physical units)
    total_enstrophy = xp.sum(enstrophy_spec)
    if float(total_enstrophy) > 1e-30:
        k_eff_abs = float(xp.sum(k_mag * enstrophy_spec) / total_enstrophy)
    else:
        k_eff_abs = 1.0
    
    # RMS wavenumber from energy (alternative metric)
    # k_rms = sqrt(∑ k² E(k) / ∑ E(k))
    energy_spec = 0.5 * (xp.abs(u_hat)**2 + xp.abs(v_hat)**2 + xp.abs(w_hat)**2)
    total_energy = xp.sum(energy_spec)
    if float(total_energy) > 1e-30:
        k_rms = float(xp.sqrt(xp.sum(k_mag**2 * energy_spec) / total_energy))
    else:
        k_rms = 1.0
    
    # Shell-averaged enstrophy spectrum for peak finding
    k_bins = xp.arange(0, int(k_dealias) + 1, 1.0)
    shell_enstrophy = xp.zeros(len(k_bins) - 1)
    for i in range(len(k_bins) - 1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        shell_enstrophy[i] = xp.sum(enstrophy_spec[mask])
    
    # Convert to numpy for peak finding
    if hasattr(shell_enstrophy, 'get'):
        shell_enstrophy_np = shell_enstrophy.get()
    else:
        shell_enstrophy_np = np.array(shell_enstrophy)
    
    k_peak_idx = np.argmax(shell_enstrophy_np)
    if hasattr(k_bins, 'get'):
        k_bins_np = k_bins.get()
    else:
        k_bins_np = np.array(k_bins)
    k_peak = float(k_bins_np[k_peak_idx])
    
    return {
        'k_eff_abs': k_eff_abs,
        'k_rms': k_rms,
        'k_eff_norm': k_eff_abs / k_nyquist,  # Dimensionless, 0-1
        'k_rms_norm': k_rms / k_nyquist,
        'k_peak': k_peak,
        'k_nyquist': float(k_nyquist),
        'k_dealias': float(k_dealias),
    }


def compute_shell_spectrum(solver, n_shells: int = 64) -> dict:
    """Compute shell-averaged energy and enstrophy spectra."""
    xp = solver.xp
    N = solver.nx
    
    # Wavenumber grid
    k_1d = xp.fft.fftfreq(N, d=1.0/N) * 2 * np.pi / solver.Lx
    kx, ky, kz = xp.meshgrid(k_1d, k_1d, k_1d, indexing='ij')
    k_mag = xp.sqrt(kx**2 + ky**2 + kz**2)
    
    k_nyquist = np.pi * N / solver.Lx
    k_max = k_nyquist * 2/3
    
    # FFTs
    u_hat = xp.fft.fftn(solver.u)
    v_hat = xp.fft.fftn(solver.v)
    w_hat = xp.fft.fftn(solver.w)
    
    # Compute vorticity in Fourier space
    ox_hat = 1j * (ky * w_hat - kz * v_hat)
    oy_hat = 1j * (kz * u_hat - kx * w_hat)
    oz_hat = 1j * (kx * v_hat - ky * u_hat)
    
    energy_spec = 0.5 * (xp.abs(u_hat)**2 + xp.abs(v_hat)**2 + xp.abs(w_hat)**2)
    enstrophy_spec = xp.abs(ox_hat)**2 + xp.abs(oy_hat)**2 + xp.abs(oz_hat)**2
    
    # Shell binning
    k_bins = np.linspace(0, k_max, n_shells + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    E_k = np.zeros(n_shells)
    Z_k = np.zeros(n_shells)
    
    for i in range(n_shells):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        E_k[i] = float(xp.sum(energy_spec[mask]))
        Z_k[i] = float(xp.sum(enstrophy_spec[mask]))
    
    return {
        'k': k_centers,
        'E_k': E_k,
        'Z_k': Z_k,
        'k_nyquist': float(k_nyquist),
    }


# ============================================================================
# Initial Condition
# ============================================================================

def initialize_taylor_green(solver, amplitude=1.0):
    xp = solver.xp
    x = xp.linspace(0, 2*xp.pi, solver.nx, endpoint=False)
    y = xp.linspace(0, 2*xp.pi, solver.ny, endpoint=False)
    z = xp.linspace(0, 2*xp.pi, solver.nz, endpoint=False)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    solver.u = amplitude * xp.sin(X) * xp.cos(Y) * xp.cos(Z)
    solver.v = -amplitude * xp.cos(X) * xp.sin(Y) * xp.cos(Z)
    solver.w = xp.zeros_like(solver.u)
    
    solver.u = solver.u.astype(solver.dtype)
    solver.v = solver.v.astype(solver.dtype)
    solver.w = solver.w.astype(solver.dtype)
    solver.u, solver.v, solver.w = solver.project_div_free(solver.u, solver.v, solver.w)
    solver.initial_energy = solver.compute_energy()


# ============================================================================
# State-Matched Recording
# ============================================================================

def run_to_omega_target(grid_size: int, reynolds_number: float, 
                        omega_targets: list, T_max: float = 15.0,
                        output_dir: str = "./results") -> dict:
    """
    Run simulation and record state at specific ω levels.
    
    This enables state-matched comparison across resolutions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dx = 2 * np.pi / grid_size
    dt_safe = 0.2 * dx / 1.0
    
    label = f"TG_N{grid_size}_Re{int(reynolds_number)}"
    
    print(f"\n{'='*70}")
    print(f"{label} - Running to ω targets: {omega_targets}")
    print(f"  N={grid_size}, Re={reynolds_number}, T_max={T_max}")
    print(f"{'='*70}")
    
    solver = CUDABKMSolver(
        grid_size=grid_size, reynolds_number=reynolds_number, dt=dt_safe,
        CFL_target=0.2, adapt_dt=False, track_alignment=False,
        rho_soft=0.99, rho_hard=0.999, startup_steps=0,
    )
    
    initialize_taylor_green(solver, amplitude=1.0)
    
    config = HeatKernelConfigV51(
        s_values=[1.0, 1.25],
        t_viscosity_mult=10.0,
        track_every=10,
        omega_percentile=99.5,
    )
    
    tracker = HeatKernelAlignmentTrackerV51(solver, config)
    
    step = 0
    last_print = 0.0
    print_interval = 1.0
    
    # State snapshots at target ω levels
    omega_targets_remaining = sorted(omega_targets, reverse=False)
    state_snapshots = {}
    
    # Full time series
    time_series = {
        'time': [], 'omega_max': [], 'omega_percentile': [],
        'tau_norm': [], 'k_eff_abs': [], 'k_eff_norm': [], 'k_rms': [],
    }
    
    # Spectra at key times
    spectra_snapshots = {}
    
    while solver.current_time < T_max and not solver.should_stop:
        result = solver.evolve_one_timestep()
        
        if step % config.track_every == 0:
            metrics = tracker.record(step)
            omega_pct = metrics.get('omega_percentile', 0)
            
            # Grid-invariant k metrics
            k_metrics = compute_grid_invariant_k(solver)
            
            # Compute tau_norm with grid-invariant k
            tau_norm_invariant = np.sqrt(k_metrics['k_eff_abs']) / (omega_pct + 1e-10)
            
            time_series['time'].append(solver.current_time)
            time_series['omega_max'].append(result['vorticity'])
            time_series['omega_percentile'].append(omega_pct)
            time_series['tau_norm'].append(tau_norm_invariant)
            time_series['k_eff_abs'].append(k_metrics['k_eff_abs'])
            time_series['k_eff_norm'].append(k_metrics['k_eff_norm'])
            time_series['k_rms'].append(k_metrics['k_rms'])
            
            # Check if we've hit a target ω level
            while omega_targets_remaining and omega_pct >= omega_targets_remaining[0]:
                target = omega_targets_remaining.pop(0)
                print(f"  *** Hit ω target {target} at t={solver.current_time:.3f}")
                
                state_snapshots[target] = {
                    'time': solver.current_time,
                    'omega_percentile': omega_pct,
                    'omega_max': result['vorticity'],
                    'tau_norm': tau_norm_invariant,
                    'k_eff_abs': k_metrics['k_eff_abs'],
                    'k_eff_norm': k_metrics['k_eff_norm'],
                    'k_rms': k_metrics['k_rms'],
                    'k_rms_norm': k_metrics['k_rms_norm'],
                    'k_peak': k_metrics['k_peak'],
                }
                
                # Save spectrum at this state
                spectra_snapshots[target] = compute_shell_spectrum(solver)
            
            if solver.current_time - last_print >= print_interval:
                print(f"  t={solver.current_time:.2f} | ω={omega_pct:.2f} | "
                      f"τ={tau_norm_invariant:.4f} | k_eff={k_metrics['k_eff_abs']:.2f} | "
                      f"k_norm={k_metrics['k_eff_norm']:.3f}")
                last_print = solver.current_time
        
        step += 1
        if step > 2000000:
            break
    
    # Final state
    final_k = compute_grid_invariant_k(solver)
    final_omega = solver.compute_max_vorticity()
    
    # Compute final tau_norm using grid-invariant k
    # Get omega percentile from the last recorded value
    if time_series['omega_percentile']:
        omega_pct_final = time_series['omega_percentile'][-1]
    else:
        omega_pct_final = final_omega * 0.7  # Rough estimate
    
    tau_norm_final = np.sqrt(final_k['k_eff_abs']) / (omega_pct_final + 1e-10)
    
    results = {
        'label': label,
        'grid_size': grid_size,
        'reynolds_number': reynolds_number,
        'T_final': solver.current_time,
        'k_nyquist': final_k['k_nyquist'],
        
        'final_omega_max': final_omega,
        'final_omega_percentile': omega_pct_final,
        'final_tau_norm': tau_norm_final,
        'final_k_eff_abs': final_k['k_eff_abs'],
        'final_k_eff_norm': final_k['k_eff_norm'],
        'final_k_rms': final_k['k_rms'],
        
        'min_tau_norm': min(time_series['tau_norm']),
        'max_k_eff_abs': max(time_series['k_eff_abs']),
        'max_omega_percentile': max(time_series['omega_percentile']),
        
        'state_snapshots': state_snapshots,
        'spectra_snapshots': spectra_snapshots,
        'time_series': time_series,
    }
    
    print(f"\n  Complete at t={solver.current_time:.2f}")
    print(f"  Final ω_99.5%: {omega_pct_final:.2f}")
    print(f"  Min τ_norm: {results['min_tau_norm']:.4f}")
    print(f"  Max k_eff (physical): {results['max_k_eff_abs']:.2f}")
    print(f"  States captured: {list(state_snapshots.keys())}")
    
    return results


def create_state_matched_plots(results_128: dict, results_256: dict, output_dir: str):
    """Create comparison plots at matched dynamical states."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('State-Matched Comparison: N=128 vs N=256 (Grid-Invariant Metrics)', fontsize=14)
    
    # Colors
    c128 = 'blue'
    c256 = 'red'
    
    # 1. τ_norm vs ω (state-matched)
    ax = axes[0, 0]
    
    ts128 = results_128['time_series']
    ts256 = results_256['time_series']
    
    ax.plot(ts128['omega_percentile'], ts128['tau_norm'], 
           c=c128, linewidth=2, label=f"N=128")
    ax.plot(ts256['omega_percentile'], ts256['tau_norm'], 
           c=c256, linewidth=2, label=f"N=256")
    
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Cascade threshold')
    ax.set_xlabel('ω (99.5 percentile)')
    ax.set_ylabel('τ_norm (grid-invariant)')
    ax.set_title('τ_norm vs ω (State-Matched)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. k_eff (physical) vs ω
    ax = axes[0, 1]
    ax.plot(ts128['omega_percentile'], ts128['k_eff_abs'], 
           c=c128, linewidth=2, label=f"N=128")
    ax.plot(ts256['omega_percentile'], ts256['k_eff_abs'], 
           c=c256, linewidth=2, label=f"N=256")
    ax.set_xlabel('ω (99.5 percentile)')
    ax.set_ylabel('k_eff (physical units)')
    ax.set_title('k_eff vs ω (Physical Units)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. k_eff_norm vs ω (should be comparable across N)
    ax = axes[0, 2]
    ax.plot(ts128['omega_percentile'], ts128['k_eff_norm'], 
           c=c128, linewidth=2, label=f"N=128")
    ax.plot(ts256['omega_percentile'], ts256['k_eff_norm'], 
           c=c256, linewidth=2, label=f"N=256")
    ax.set_xlabel('ω (99.5 percentile)')
    ax.set_ylabel('k_eff / k_nyquist')
    ax.set_title('Normalized k_eff vs ω')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Time evolution comparison
    ax = axes[1, 0]
    ax.plot(ts128['time'], ts128['tau_norm'], c=c128, linewidth=2, label='N=128')
    ax.plot(ts256['time'], ts256['tau_norm'], c=c256, linewidth=2, label='N=256')
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('τ_norm')
    ax.set_title('τ_norm vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Spectra comparison at matched ω
    ax = axes[1, 1]
    
    # Find common ω targets
    common_targets = set(results_128['state_snapshots'].keys()) & set(results_256['state_snapshots'].keys())
    
    if common_targets:
        target = max(common_targets)  # Highest common ω
        
        spec128 = results_128['spectra_snapshots'].get(target, {})
        spec256 = results_256['spectra_snapshots'].get(target, {})
        
        if spec128 and spec256:
            # Normalize spectra for comparison
            k128 = spec128['k']
            Z128 = spec128['Z_k'] / (np.sum(spec128['Z_k']) + 1e-30)
            
            k256 = spec256['k']
            Z256 = spec256['Z_k'] / (np.sum(spec256['Z_k']) + 1e-30)
            
            ax.loglog(k128, Z128, c=c128, linewidth=2, label=f'N=128')
            ax.loglog(k256, Z256, c=c256, linewidth=2, label=f'N=256')
            
            # Mark Nyquist
            ax.axvline(x=spec128['k_nyquist']*2/3, color=c128, linestyle=':', alpha=0.5)
            ax.axvline(x=spec256['k_nyquist']*2/3, color=c256, linestyle=':', alpha=0.5)
            
            ax.set_xlabel('k')
            ax.set_ylabel('Z(k) / ∫Z(k)dk (normalized)')
            ax.set_title(f'Enstrophy Spectrum at ω≈{target}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No common ω targets reached', 
               transform=ax.transAxes, ha='center', va='center')
    
    # 6. State-matched comparison table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Build comparison table
    table_data = [['Metric', 'N=128', 'N=256', 'Δ%']]
    
    for target in sorted(common_targets)[:3]:  # Top 3 common targets
        s128 = results_128['state_snapshots'].get(target, {})
        s256 = results_256['state_snapshots'].get(target, {})
        
        if s128 and s256:
            table_data.append([f'--- ω={target} ---', '', '', ''])
            
            tau128 = s128.get('tau_norm', 0)
            tau256 = s256.get('tau_norm', 0)
            delta_tau = (tau256 - tau128) / (tau128 + 1e-10) * 100
            table_data.append(['τ_norm', f'{tau128:.4f}', f'{tau256:.4f}', f'{delta_tau:+.1f}%'])
            
            k128 = s128.get('k_eff_abs', 0)
            k256 = s256.get('k_eff_abs', 0)
            delta_k = (k256 - k128) / (k128 + 1e-10) * 100
            table_data.append(['k_eff (phys)', f'{k128:.2f}', f'{k256:.2f}', f'{delta_k:+.1f}%'])
            
            kn128 = s128.get('k_eff_norm', 0)
            kn256 = s256.get('k_eff_norm', 0)
            delta_kn = (kn256 - kn128) / (kn128 + 1e-10) * 100
            table_data.append(['k_eff/k_nyq', f'{kn128:.3f}', f'{kn256:.3f}', f'{delta_kn:+.1f}%'])
    
    # Add final state comparison
    table_data.append(['--- Final State ---', '', '', ''])
    table_data.append(['min τ_norm', 
                      f"{results_128['min_tau_norm']:.4f}",
                      f"{results_256['min_tau_norm']:.4f}",
                      f"{(results_256['min_tau_norm']-results_128['min_tau_norm'])/results_128['min_tau_norm']*100:+.1f}%"])
    table_data.append(['max ω', 
                      f"{results_128['max_omega_percentile']:.2f}",
                      f"{results_256['max_omega_percentile']:.2f}",
                      ''])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                    colWidths=[0.3, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title('State-Matched Comparison')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'state_matched_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: state_matched_comparison.png")


# ============================================================================
# Main
# ============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = f"./state_matched_{timestamp}"
    os.makedirs(base_output, exist_ok=True)
    
    print("="*70)
    print("STATE-MATCHED COMPARISON")
    print("Question: Is N=128's low τ_norm real physics or numerical artifact?")
    print("="*70)
    
    Re = 3200
    omega_targets = [8, 10, 12, 15, 18, 20]  # Record state at these ω levels
    
    # Run N=128 to T=10
    print("\n" + "="*70)
    print("N=128: Running to T=10 (reference)")
    print("="*70)
    
    results_128 = run_to_omega_target(
        grid_size=128, reynolds_number=Re,
        omega_targets=omega_targets, T_max=10.0,
        output_dir=os.path.join(base_output, "N128")
    )
    
    # Run N=256 to T=10 (or until it matches N=128's max ω)
    print("\n" + "="*70)
    print("N=256: Running to T=10 (the decisive test)")
    print("="*70)
    
    results_256 = run_to_omega_target(
        grid_size=256, reynolds_number=Re,
        omega_targets=omega_targets, T_max=10.0,
        output_dir=os.path.join(base_output, "N256")
    )
    
    # Create comparison plots
    print("\nCreating state-matched comparison plots...")
    create_state_matched_plots(results_128, results_256, base_output)
    
    # Summary
    print("\n" + "="*70)
    print("STATE-MATCHED COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'N=128':>12} {'N=256':>12} {'Verdict':>15}")
    print("-"*70)
    
    # Compare at common ω levels
    common = set(results_128['state_snapshots'].keys()) & set(results_256['state_snapshots'].keys())
    
    for target in sorted(common):
        s128 = results_128['state_snapshots'][target]
        s256 = results_256['state_snapshots'][target]
        
        print(f"\n  At ω = {target}:")
        print(f"    τ_norm:      {s128['tau_norm']:.4f}      {s256['tau_norm']:.4f}")
        print(f"    k_eff (abs): {s128['k_eff_abs']:.2f}        {s256['k_eff_abs']:.2f}")
        print(f"    k_eff/k_nyq: {s128['k_eff_norm']:.3f}       {s256['k_eff_norm']:.3f}")
    
    # Final verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    # Compare min τ_norm at matched ω levels
    if common:
        max_common = max(common)
        tau_128_at_match = results_128['state_snapshots'][max_common]['tau_norm']
        tau_256_at_match = results_256['state_snapshots'][max_common]['tau_norm']
        
        delta_pct = (tau_256_at_match - tau_128_at_match) / tau_128_at_match * 100
        
        print(f"\nAt matched ω = {max_common}:")
        print(f"  N=128 τ_norm: {tau_128_at_match:.4f}")
        print(f"  N=256 τ_norm: {tau_256_at_match:.4f}")
        print(f"  Difference: {delta_pct:+.1f}%")
        
        if abs(delta_pct) < 15:
            print("\n  → CONVERGED: τ_norm is grid-independent at matched states")
            print("  → N=128 results are VALID (not numerical artifacts)")
        elif delta_pct > 15:
            print("\n  → N=256 shows HIGHER τ_norm at matched ω")
            print("  → N=128 may be artificially cascade-y (under-resolution)")
        else:
            print("\n  → N=256 shows LOWER τ_norm at matched ω")
            print("  → N=128 may be under-estimating cascade activity")
    
    # Save summary
    summary = {
        'N128': {k: v for k, v in results_128.items() if k not in ['time_series', 'spectra_snapshots']},
        'N256': {k: v for k, v in results_256.items() if k not in ['time_series', 'spectra_snapshots']},
    }
    
    with open(os.path.join(base_output, 'state_matched_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {base_output}")
    
    return results_128, results_256


if __name__ == "__main__":
    results_128, results_256 = main()
