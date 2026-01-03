#!/usr/bin/env python3
"""
run_scaling_study.py
====================
Systematic scaling study for heat kernel diagnostics.

Based on GPT recommendations:
1. Matrix of Re × N × IC (not just "max everything")
2. Event-conditioned lead/lag analysis (around ω surges)
3. Convergence check: does τ_norm stabilize with N?
4. Detrended cross-correlation for robust lag detection

Tests:
- Re ∈ {400, 800, 1600} (3200 if time permits)
- N ∈ {64, 96, 128}
- IC ∈ {taylor_green, kida_pelz} (focus on coherent flows)

Key outputs per run:
- min τ_norm, time_of_min
- Dwell fractions {diffusion, transitional, cascade}
- Crossing counts
- Event-conditioned lead/lag
- Peak ω_percentile
- Convergence metrics (ω vs N, k_eff vs N)

Author: Josh Curry & Claude
Date: December 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from scipy import signal as scipy_signal
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_bkm_engine import CUDABKMSolver, backend
from heat_kernel_ns_integration_v5_1 import (
    HeatKernelConfigV51,
    HeatKernelAlignmentTrackerV51,
)

print(f"Backend: {backend}")


# ============================================================================
# Initial Conditions
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
    return {'type': 'taylor_green', 'E0': solver.initial_energy, 'W0': solver.compute_max_vorticity()}


def initialize_kida_pelz(solver, amplitude=1.3):
    xp = solver.xp
    x = xp.linspace(0, solver.Lx, solver.nx, endpoint=False)
    y = xp.linspace(0, solver.Ly, solver.ny, endpoint=False)
    z = xp.linspace(0, solver.Lz, solver.nz, endpoint=False)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    r1_sq = (X - np.pi)**2 + (Y - np.pi)**2
    r2_sq = (Y - np.pi)**2 + (Z - np.pi)**2
    sigma = 0.8
    
    vort1 = amplitude * xp.exp(-r1_sq / (2*sigma**2))
    vort2 = amplitude * xp.exp(-r2_sq / (2*sigma**2))
    
    eps = 0.1
    solver.u = -vort1 * (Y - np.pi) / xp.sqrt(r1_sq + eps**2)
    solver.v = vort1 * (X - np.pi) / xp.sqrt(r1_sq + eps**2) + vort2 * (Z - np.pi) / xp.sqrt(r2_sq + eps**2)
    solver.w = -vort2 * (Y - np.pi) / xp.sqrt(r2_sq + eps**2)
    
    solver.u = solver.u.astype(xp.float64)
    solver.v = solver.v.astype(xp.float64)
    solver.w = solver.w.astype(xp.float64)
    solver.u, solver.v, solver.w = solver.project_div_free(solver.u, solver.v, solver.w)
    solver.initial_energy = solver.compute_energy()
    return {'type': 'kida_pelz', 'E0': solver.initial_energy, 'W0': solver.compute_max_vorticity()}


# ============================================================================
# Improved Lead/Lag Analysis
# ============================================================================

def detrended_cross_correlation(series_a: np.ndarray, series_b: np.ndarray, 
                                 max_lag: int = 30) -> Dict:
    """
    Compute cross-correlation after detrending both series.
    
    Detrending removes drift that can cause spurious lag detection.
    """
    if len(series_a) < 20 or len(series_b) < 20:
        return {'peak_lag': 0, 'peak_correlation': 0, 'significant': False}
    
    # Ensure same length
    min_len = min(len(series_a), len(series_b))
    a = series_a[:min_len].copy()
    b = series_b[:min_len].copy()
    
    # Detrend: subtract rolling mean (window = 1/4 of series)
    window = max(5, min_len // 4)
    
    def rolling_mean(x, w):
        return np.convolve(x, np.ones(w)/w, mode='same')
    
    a_detrend = a - rolling_mean(a, window)
    b_detrend = b - rolling_mean(b, window)
    
    # Normalize
    a_norm = (a_detrend - np.mean(a_detrend)) / (np.std(a_detrend) + 1e-10)
    b_norm = (b_detrend - np.mean(b_detrend)) / (np.std(b_detrend) + 1e-10)
    
    # Cross-correlation
    max_lag = min(max_lag, min_len // 3)
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(a_norm[:lag], b_norm[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(a_norm[lag:], b_norm[:-lag])[0, 1]
        else:
            corr = np.corrcoef(a_norm, b_norm)[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0)
    
    correlations = np.array(correlations)
    peak_idx = np.argmax(np.abs(correlations))
    peak_lag = list(lags)[peak_idx]
    peak_corr = correlations[peak_idx]
    
    return {
        'peak_lag': int(peak_lag),
        'peak_correlation': float(peak_corr),
        'significant': abs(peak_corr) > 0.5,
        'detrended': True,
    }


def event_conditioned_lag(tau_norm: np.ndarray, omega: np.ndarray, 
                          times: np.ndarray, window_half: int = 10) -> Dict:
    """
    Compute lead/lag around ω surge events only.
    
    This is more robust than correlating entire series because it focuses
    on the physically relevant episodes (vorticity peaks).
    
    Returns average lag: positive = τ_norm leads (predictive)
    """
    if len(omega) < 30:
        return {'event_lag': 0, 'n_events': 0, 'significant': False}
    
    # Find local maxima of omega (surge events)
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(omega, prominence=0.1 * np.max(omega), distance=10)
    
    if len(peaks) == 0:
        return {'event_lag': 0, 'n_events': 0, 'significant': False}
    
    # For each peak, look for preceding τ_norm minimum
    lags = []
    for peak_idx in peaks:
        # Window around peak
        start = max(0, peak_idx - window_half * 2)
        end = min(len(tau_norm), peak_idx + window_half)
        
        if end - start < 5:
            continue
        
        # Find minimum τ_norm in window before peak
        window_tau = tau_norm[start:peak_idx]
        if len(window_tau) == 0:
            continue
        
        min_idx_in_window = np.argmin(window_tau)
        min_idx_global = start + min_idx_in_window
        
        # Lag = peak_idx - min_idx (positive = τ_norm dip preceded peak)
        lag = peak_idx - min_idx_global
        lags.append(lag)
    
    if len(lags) == 0:
        return {'event_lag': 0, 'n_events': 0, 'significant': False}
    
    mean_lag = np.mean(lags)
    std_lag = np.std(lags)
    
    return {
        'event_lag': float(mean_lag),
        'event_lag_std': float(std_lag),
        'n_events': len(lags),
        'significant': len(lags) >= 2 and mean_lag > 0,
        'predictive': mean_lag > 2,  # τ_norm dip at least 2 steps before ω peak
    }


# ============================================================================
# Simulation Runner
# ============================================================================

def run_simulation(grid_size: int, reynolds_number: float, ic_type: str,
                   T_final: float = 5.0, output_dir: str = "./results") -> Dict:
    """Run single simulation with comprehensive diagnostics."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    dx = 2 * np.pi / grid_size
    dt_safe = 0.2 * dx / 1.0
    
    label = f"{ic_type[:2].upper()}_N{grid_size}_Re{reynolds_number}"
    
    print(f"\n  {label}: ", end="", flush=True)
    
    solver = CUDABKMSolver(
        grid_size=grid_size, reynolds_number=reynolds_number, dt=dt_safe,
        CFL_target=0.2, adapt_dt=False, track_alignment=False,
        rho_soft=0.99, rho_hard=0.999, startup_steps=0,
    )
    
    # Initialize
    if ic_type == 'taylor_green':
        ic_info = initialize_taylor_green(solver, amplitude=1.0)
    elif ic_type == 'kida_pelz':
        ic_info = initialize_kida_pelz(solver, amplitude=1.3)
    else:
        raise ValueError(f"Unknown IC: {ic_type}")
    
    config = HeatKernelConfigV51(
        s_values=[0.7, 1.0, 1.25, 1.5],
        t_viscosity_mult=10.0,
        track_every=5,
        omega_percentile=99.5,
    )
    
    tracker = HeatKernelAlignmentTrackerV51(solver, config)
    tracker.start_logging(output_dir, filename=f"hk_{label}.csv")
    
    step = 0
    time_series = {
        'time': [], 'omega_max': [], 'omega_percentile': [], 
        'tau_norm': [], 'k_eff': [], 'ew_score': []
    }
    
    while solver.current_time < T_final and not solver.should_stop:
        result = solver.evolve_one_timestep()
        
        if step % config.track_every == 0:
            metrics = tracker.record(step)
            
            time_series['time'].append(solver.current_time)
            time_series['omega_max'].append(result['vorticity'])
            time_series['omega_percentile'].append(metrics.get('omega_percentile', 0))
            time_series['tau_norm'].append(metrics.get('tau_norm', 1.0))
            time_series['k_eff'].append(metrics.get('k_eff', 1.0))
            time_series['ew_score'].append(metrics.get('ew_score', 0.0))
        
        step += 1
        if step > 500000:
            break
    
    tracker.close()
    summary = tracker.get_summary()
    
    # Convert to arrays for analysis
    tau_arr = np.array(time_series['tau_norm'])
    omega_arr = np.array(time_series['omega_percentile'])
    time_arr = np.array(time_series['time'])
    
    # Detrended lag analysis
    detrend_lag = detrended_cross_correlation(tau_arr, omega_arr)
    
    # Event-conditioned lag analysis
    event_lag = event_conditioned_lag(tau_arr, omega_arr, time_arr)
    
    # Compile results
    rc = summary.get('regime_crossings', {})
    
    results = {
        'label': label,
        'ic_type': ic_type,
        'grid_size': grid_size,
        'reynolds_number': reynolds_number,
        'T_final': T_final,
        
        # Peak values (for convergence check)
        'peak_omega_max': float(np.max(time_series['omega_max'])),
        'peak_omega_percentile': float(np.max(omega_arr)),
        'mean_k_eff': float(np.mean(time_series['k_eff'])),
        'max_k_eff': float(np.max(time_series['k_eff'])),
        
        # τ_norm statistics
        'min_tau_norm': rc.get('min_tau_norm', 0),
        'min_tau_norm_time': rc.get('min_tau_norm_time', 0),
        'mean_tau_norm': summary.get('mean_tau_norm', 0),
        
        # Regime dwell fractions
        'dwell_cascade': rc.get('dwell_frac_cascade', 0),
        'dwell_transitional': rc.get('dwell_frac_transitional', 0),
        'dwell_diffusion': rc.get('dwell_frac_diffusion', 0),
        
        # Crossings
        'total_crossings': rc.get('total_crossings', 0),
        'first_cascade_entry': rc.get('first_cascade_entry'),
        
        # Lead/lag analysis
        'detrend_lag': detrend_lag['peak_lag'],
        'detrend_corr': detrend_lag['peak_correlation'],
        'event_lag': event_lag['event_lag'],
        'event_lag_n': event_lag['n_events'],
        'predictive': event_lag.get('predictive', False),
        
        # Early warning
        'max_ew_score': summary.get('max_ew_score', 0),
        
        # Time series (for plotting)
        'time_series': time_series,
    }
    
    status = "✓" if results['min_tau_norm'] < 0.3 else "·"
    print(f"{status} ω={results['peak_omega_max']:.1f}, τ={results['min_tau_norm']:.3f}, "
          f"lag={results['detrend_lag']:+d}/{results['event_lag']:.1f}")
    
    return results


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_convergence(results_by_N: Dict[int, Dict]) -> Dict:
    """
    Check if diagnostics converge with resolution.
    
    Compares ω_percentile, k_eff, and τ_norm across N values.
    """
    Ns = sorted(results_by_N.keys())
    if len(Ns) < 2:
        return {'converged': None, 'message': 'Need at least 2 resolutions'}
    
    # Extract metrics at each N
    omega_peaks = [results_by_N[N]['peak_omega_percentile'] for N in Ns]
    k_effs = [results_by_N[N]['mean_k_eff'] for N in Ns]
    tau_mins = [results_by_N[N]['min_tau_norm'] for N in Ns]
    
    # Check relative change between highest two resolutions
    if len(Ns) >= 2:
        N_high = Ns[-1]
        N_low = Ns[-2]
        
        omega_change = abs(omega_peaks[-1] - omega_peaks[-2]) / (omega_peaks[-2] + 1e-10)
        k_eff_change = abs(k_effs[-1] - k_effs[-2]) / (k_effs[-2] + 1e-10)
        tau_change = abs(tau_mins[-1] - tau_mins[-2]) / (tau_mins[-2] + 1e-10)
        
        converged = omega_change < 0.1 and tau_change < 0.15
        
        return {
            'converged': converged,
            'omega_change': omega_change,
            'k_eff_change': k_eff_change,
            'tau_norm_change': tau_change,
            'N_values': Ns,
            'omega_peaks': omega_peaks,
            'k_effs': k_effs,
            'tau_mins': tau_mins,
        }
    
    return {'converged': None}


def create_scaling_plots(all_results: Dict, output_dir: str):
    """Create comprehensive scaling study plots."""
    
    # Organize by IC type
    by_ic = {}
    for label, res in all_results.items():
        ic = res['ic_type']
        if ic not in by_ic:
            by_ic[ic] = {}
        by_ic[ic][label] = res
    
    for ic_type, ic_results in by_ic.items():
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'{ic_type.replace("_", " ").title()} - Scaling Study', fontsize=14)
        
        # Organize by Re and N
        by_Re = {}
        by_N = {}
        for label, res in ic_results.items():
            Re = res['reynolds_number']
            N = res['grid_size']
            if Re not in by_Re:
                by_Re[Re] = []
            by_Re[Re].append(res)
            if N not in by_N:
                by_N[N] = []
            by_N[N].append(res)
        
        colors_Re = {400: 'blue', 800: 'orange', 1600: 'red', 3200: 'purple'}
        markers_N = {64: 'o', 96: 's', 128: '^', 256: 'D'}
        
        # Plot 1: min τ_norm vs Re (by N)
        ax = axes[0, 0]
        for N, results in sorted(by_N.items()):
            Res = [r['reynolds_number'] for r in results]
            taus = [r['min_tau_norm'] for r in results]
            ax.plot(Res, taus, marker=markers_N.get(N, 'o'), label=f'N={N}', linewidth=2, markersize=8)
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Cascade threshold')
        ax.set_xlabel('Reynolds Number')
        ax.set_ylabel('min τ_norm')
        ax.set_title('τ_norm vs Re (lower = more cascade)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Plot 2: Peak ω vs Re (by N) - convergence check
        ax = axes[0, 1]
        for N, results in sorted(by_N.items()):
            Res = [r['reynolds_number'] for r in results]
            omegas = [r['peak_omega_percentile'] for r in results]
            ax.plot(Res, omegas, marker=markers_N.get(N, 'o'), label=f'N={N}', linewidth=2, markersize=8)
        ax.set_xlabel('Reynolds Number')
        ax.set_ylabel('Peak ω (99.5%)')
        ax.set_title('Vorticity vs Re (convergence check)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Plot 3: k_eff vs Re (by N)
        ax = axes[0, 2]
        for N, results in sorted(by_N.items()):
            Res = [r['reynolds_number'] for r in results]
            k_effs = [r['max_k_eff'] for r in results]
            ax.plot(Res, k_effs, marker=markers_N.get(N, 'o'), label=f'N={N}', linewidth=2, markersize=8)
        ax.set_xlabel('Reynolds Number')
        ax.set_ylabel('max k_eff')
        ax.set_title('Effective Wavenumber vs Re')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Plot 4: Dwell fractions vs Re
        ax = axes[1, 0]
        Res = sorted(by_Re.keys())
        # Use highest N for each Re
        cascade_fracs = []
        trans_fracs = []
        diff_fracs = []
        for Re in Res:
            results_at_Re = by_Re[Re]
            # Pick highest N
            best = max(results_at_Re, key=lambda r: r['grid_size'])
            cascade_fracs.append(best['dwell_cascade'])
            trans_fracs.append(best['dwell_transitional'])
            diff_fracs.append(best['dwell_diffusion'])
        
        x = np.arange(len(Res))
        ax.bar(x, cascade_fracs, 0.6, label='Cascade', color='red', alpha=0.8)
        ax.bar(x, trans_fracs, 0.6, bottom=cascade_fracs, label='Transitional', color='orange', alpha=0.8)
        ax.bar(x, diff_fracs, 0.6, bottom=np.array(cascade_fracs)+np.array(trans_fracs), 
               label='Diffusion', color='green', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Re={Re}' for Re in Res])
        ax.set_ylabel('Dwell Fraction')
        ax.set_title('Regime Dwell vs Re (highest N)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Lead/lag vs Re
        ax = axes[1, 1]
        for N, results in sorted(by_N.items()):
            Res = [r['reynolds_number'] for r in results]
            detrend_lags = [r['detrend_lag'] for r in results]
            event_lags = [r['event_lag'] for r in results]
            ax.plot(Res, detrend_lags, marker=markers_N.get(N, 'o'), label=f'N={N} (detrend)', 
                   linewidth=2, markersize=8, linestyle='-')
            ax.plot(Res, event_lags, marker=markers_N.get(N, 'o'), 
                   linewidth=1, markersize=6, linestyle='--', alpha=0.6)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Reynolds Number')
        ax.set_ylabel('Lag (steps)')
        ax.set_title('Lead/Lag vs Re\n(positive = τ_norm leads = predictive)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Plot 6: Convergence with N (at highest Re)
        ax = axes[1, 2]
        if len(by_Re) > 0:
            highest_Re = max(by_Re.keys())
            results_at_Re = by_Re[highest_Re]
            Ns = [r['grid_size'] for r in sorted(results_at_Re, key=lambda r: r['grid_size'])]
            taus = [r['min_tau_norm'] for r in sorted(results_at_Re, key=lambda r: r['grid_size'])]
            omegas = [r['peak_omega_percentile'] for r in sorted(results_at_Re, key=lambda r: r['grid_size'])]
            
            ax.plot(Ns, taus, 'ro-', label='min τ_norm', linewidth=2, markersize=10)
            ax2 = ax.twinx()
            ax2.plot(Ns, omegas, 'bs--', label='peak ω', linewidth=2, markersize=10)
            
            ax.set_xlabel('Grid Size N')
            ax.set_ylabel('min τ_norm', color='red')
            ax2.set_ylabel('peak ω (99.5%)', color='blue')
            ax.set_title(f'Convergence with N (Re={highest_Re})')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'scaling_{ic_type}.png'), dpi=150)
        plt.close()
        print(f"  Saved: scaling_{ic_type}.png")


# ============================================================================
# Main
# ============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = f"./scaling_study_{timestamp}"
    os.makedirs(base_output, exist_ok=True)
    
    print("="*70)
    print("Systematic Scaling Study")
    print("="*70)
    print(f"Output: {base_output}")
    
    # Define test matrix
    Re_values = [400, 800, 1600]
    N_values = [64, 96, 128]
    IC_types = ['taylor_green', 'kida_pelz']
    T_final = 5.0
    
    print(f"\nTest matrix:")
    print(f"  Re: {Re_values}")
    print(f"  N: {N_values}")
    print(f"  IC: {IC_types}")
    print(f"  Total runs: {len(Re_values) * len(N_values) * len(IC_types)}")
    
    all_results = {}
    
    for ic_type in IC_types:
        print(f"\n{'='*60}")
        print(f"IC: {ic_type}")
        print(f"{'='*60}")
        
        for Re in Re_values:
            for N in N_values:
                label = f"{ic_type[:2].upper()}_N{N}_Re{Re}"
                output_dir = os.path.join(base_output, ic_type, f"N{N}_Re{Re}")
                
                results = run_simulation(
                    grid_size=N,
                    reynolds_number=Re,
                    ic_type=ic_type,
                    T_final=T_final,
                    output_dir=output_dir,
                )
                all_results[label] = results
    
    # Create plots
    print("\nCreating scaling plots...")
    create_scaling_plots(all_results, base_output)
    
    # Convergence analysis
    print("\nConvergence Analysis:")
    for ic_type in IC_types:
        for Re in Re_values:
            results_by_N = {}
            for N in N_values:
                label = f"{ic_type[:2].upper()}_N{N}_Re{Re}"
                if label in all_results:
                    results_by_N[N] = all_results[label]
            
            conv = analyze_convergence(results_by_N)
            if conv.get('converged') is not None:
                status = "✓ CONVERGED" if conv['converged'] else "✗ NOT CONVERGED"
                print(f"  {ic_type} Re={Re}: {status}")
                print(f"    Δω={conv['omega_change']*100:.1f}%, Δτ={conv['tau_norm_change']*100:.1f}%")
    
    # Save summary
    summary_data = {}
    for label, res in all_results.items():
        summary_data[label] = {
            'ic_type': res['ic_type'],
            'grid_size': int(res['grid_size']),
            'reynolds_number': float(res['reynolds_number']),
            'peak_omega': float(res['peak_omega_percentile']),
            'min_tau_norm': float(res['min_tau_norm']),
            'dwell_cascade': float(res['dwell_cascade']),
            'dwell_transitional': float(res['dwell_transitional']),
            'detrend_lag': int(res['detrend_lag']),
            'event_lag': float(res['event_lag']),
            'predictive': bool(res['predictive']),
        }
    
    with open(os.path.join(base_output, 'scaling_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("SCALING STUDY SUMMARY")
    print("="*80)
    print(f"{'Label':<18} {'N':>4} {'Re':>5} {'ω_peak':>7} {'min τ':>7} {'Casc%':>6} {'Lag_d':>6} {'Lag_e':>6} {'Pred':>5}")
    print("-"*80)
    
    for label in sorted(all_results.keys()):
        d = summary_data[label]
        pred = "Yes" if d['predictive'] else "No"
        print(f"{label:<18} {d['grid_size']:>4} {d['reynolds_number']:>5} "
              f"{d['peak_omega']:>7.2f} {d['min_tau_norm']:>7.3f} "
              f"{d['dwell_cascade']*100:>5.0f}% {d['detrend_lag']:>+6} "
              f"{d['event_lag']:>6.1f} {pred:>5}")
    
    print(f"\nAll results saved to: {base_output}")
    
    return all_results


if __name__ == "__main__":
    all_results = main()
