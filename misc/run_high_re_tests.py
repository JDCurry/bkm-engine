#!/usr/bin/env python3
"""
run_high_re_tests.py
====================
Focused high-Re tests based on scaling study findings.

Tests:
1. TG at Re=3200, N=128 - Push toward cascade threshold
2. TG at Re=1600, N=128, T=10 - Extended time to capture decay phase
3. KP at Re=1600 with tighter dt - Convergence fix

Also fixes the event-conditioned lag detection:
- For monotonic growth (like TG), detect "acceleration events" instead of peaks
- Compute lag between τ_norm minima and ω acceleration phases

Author: Josh Curry & Claude
Date: December 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_bkm_engine import CUDABKMSolver, backend
from heat_kernel_ns_integration_v5_1 import (
    HeatKernelConfigV51,
    HeatKernelAlignmentTrackerV51,
)

print(f"Backend: {backend}")


# ============================================================================
# Improved Event Detection
# ============================================================================

def detect_acceleration_events(omega: np.ndarray, times: np.ndarray, 
                                smooth_sigma: float = 3.0) -> list:
    """
    Detect ω acceleration events (for monotonically growing flows like TG).
    
    Instead of looking for peaks (which TG doesn't have until late),
    we look for periods where d²ω/dt² > threshold (acceleration phases).
    
    Returns list of (start_idx, peak_accel_idx, end_idx) tuples.
    """
    if len(omega) < 20:
        return []
    
    # Smooth to reduce noise
    omega_smooth = gaussian_filter1d(omega, sigma=smooth_sigma)
    
    # Compute derivatives
    dt = np.gradient(times)
    d_omega = np.gradient(omega_smooth) / (dt + 1e-10)
    d2_omega = np.gradient(d_omega) / (dt + 1e-10)
    
    # Find acceleration peaks (where d²ω/dt² is maximal)
    accel_threshold = 0.1 * np.max(np.abs(d2_omega))
    
    # Find regions where acceleration is high
    high_accel = d2_omega > accel_threshold
    
    events = []
    in_event = False
    start_idx = 0
    
    for i, is_high in enumerate(high_accel):
        if is_high and not in_event:
            in_event = True
            start_idx = i
        elif not is_high and in_event:
            in_event = False
            end_idx = i
            # Find peak acceleration in this window
            window_accel = d2_omega[start_idx:end_idx]
            if len(window_accel) > 0:
                peak_idx = start_idx + np.argmax(window_accel)
                events.append((start_idx, peak_idx, end_idx))
    
    return events


def compute_acceleration_lag(tau_norm: np.ndarray, omega: np.ndarray, 
                              times: np.ndarray) -> dict:
    """
    Compute lag between τ_norm changes and ω acceleration events.
    
    For each acceleration event, find the preceding τ_norm minimum.
    Positive lag = τ_norm dip preceded acceleration = predictive.
    """
    events = detect_acceleration_events(omega, times)
    
    if len(events) == 0:
        # Fallback: just look at overall correlation of derivatives
        if len(tau_norm) < 20:
            return {'accel_lag': 0, 'n_events': 0, 'method': 'none'}
        
        dt = np.gradient(times)
        d_tau = np.gradient(tau_norm) / (dt + 1e-10)
        d_omega = np.gradient(omega) / (dt + 1e-10)
        
        # Cross-correlation of derivatives
        d_tau_norm = (d_tau - np.mean(d_tau)) / (np.std(d_tau) + 1e-10)
        d_omega_norm = (d_omega - np.mean(d_omega)) / (np.std(d_omega) + 1e-10)
        
        max_lag = min(30, len(d_tau) // 3)
        best_lag = 0
        best_corr = 0
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(d_tau_norm[:lag], d_omega_norm[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(d_tau_norm[lag:], d_omega_norm[:-lag])[0, 1]
            else:
                corr = np.corrcoef(d_tau_norm, d_omega_norm)[0, 1]
            
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        
        return {
            'accel_lag': best_lag,
            'accel_corr': float(best_corr),
            'n_events': 0,
            'method': 'derivative_correlation',
            'predictive': best_lag > 0 and best_corr < -0.3,  # Negative corr = τ decreasing while ω accelerating
        }
    
    # Event-based analysis
    lags = []
    for start_idx, peak_idx, end_idx in events:
        # Look for τ_norm minimum in window before this event
        lookback = min(20, start_idx)
        if lookback < 3:
            continue
        
        tau_window = tau_norm[start_idx - lookback:start_idx]
        if len(tau_window) == 0:
            continue
        
        min_idx_in_window = np.argmin(tau_window)
        min_idx_global = start_idx - lookback + min_idx_in_window
        
        # Lag = event_start - tau_min (positive = τ dip preceded event)
        lag = start_idx - min_idx_global
        lags.append(lag)
    
    if len(lags) == 0:
        return {'accel_lag': 0, 'n_events': 0, 'method': 'events_no_lags'}
    
    mean_lag = np.mean(lags)
    
    return {
        'accel_lag': float(mean_lag),
        'accel_lag_std': float(np.std(lags)),
        'n_events': len(events),
        'lags': lags,
        'method': 'acceleration_events',
        'predictive': mean_lag > 2,
    }


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
# Simulation Runner
# ============================================================================

def run_simulation(grid_size: int, reynolds_number: float, ic_type: str,
                   T_final: float = 5.0, dt_mult: float = 1.0,
                   output_dir: str = "./results", label: str = None) -> dict:
    """Run simulation with comprehensive diagnostics."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    dx = 2 * np.pi / grid_size
    dt_base = 0.2 * dx / 1.0
    dt_safe = dt_base * dt_mult  # Allow tighter dt via dt_mult < 1
    
    if label is None:
        label = f"{ic_type[:2].upper()}_N{grid_size}_Re{reynolds_number}"
    
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"  N={grid_size}, Re={reynolds_number}, T={T_final}, dt_mult={dt_mult}")
    print(f"{'='*60}")
    
    solver = CUDABKMSolver(
        grid_size=grid_size, reynolds_number=reynolds_number, dt=dt_safe,
        CFL_target=0.2 * dt_mult, adapt_dt=False, track_alignment=False,
        rho_soft=0.99, rho_hard=0.999, startup_steps=0,
    )
    
    # Initialize
    if ic_type == 'taylor_green':
        ic_info = initialize_taylor_green(solver, amplitude=1.0)
    elif ic_type == 'kida_pelz':
        ic_info = initialize_kida_pelz(solver, amplitude=1.3)
    else:
        raise ValueError(f"Unknown IC: {ic_type}")
    
    print(f"  E0={ic_info['E0']:.4f}, W0={ic_info['W0']:.2f}")
    
    config = HeatKernelConfigV51(
        s_values=[0.7, 1.0, 1.25, 1.5],
        t_viscosity_mult=10.0,
        track_every=5,
        omega_percentile=99.5,
    )
    
    tracker = HeatKernelAlignmentTrackerV51(solver, config)
    tracker.start_logging(output_dir, filename=f"hk_{label}.csv")
    
    step = 0
    last_print = 0.0
    print_interval = max(0.5, T_final / 20)
    
    time_series = {
        'time': [], 'omega_max': [], 'omega_percentile': [], 
        'tau_norm': [], 'k_eff': [], 'ew_score': [], 'gamma_exp': []
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
            time_series['gamma_exp'].append(metrics.get('gamma_exp', 0.0))
            
            if solver.current_time - last_print >= print_interval:
                tau_norm = metrics.get('tau_norm', 0)
                regime = metrics.get('regime', 'unknown')[:5]
                k_eff = metrics.get('k_eff', 0)
                gamma = metrics.get('gamma_exp', 0)
                
                print(f"  t={solver.current_time:.2f} | ω={result['vorticity']:.2f} | "
                      f"τ={tau_norm:.3f} | k={k_eff:.1f} | γ={gamma:.2f} | {regime}")
                last_print = solver.current_time
        
        step += 1
        if step > 1000000:
            break
    
    tracker.close()
    summary = tracker.get_summary()
    
    # Convert to arrays
    tau_arr = np.array(time_series['tau_norm'])
    omega_arr = np.array(time_series['omega_percentile'])
    time_arr = np.array(time_series['time'])
    
    # Improved acceleration-based lag analysis
    accel_lag_info = compute_acceleration_lag(tau_arr, omega_arr, time_arr)
    
    # Compile results
    rc = summary.get('regime_crossings', {})
    
    results = {
        'label': label,
        'ic_type': ic_type,
        'grid_size': grid_size,
        'reynolds_number': reynolds_number,
        'T_final': T_final,
        'dt_mult': dt_mult,
        
        'peak_omega_max': float(np.max(time_series['omega_max'])),
        'peak_omega_percentile': float(np.max(omega_arr)),
        'mean_k_eff': float(np.mean(time_series['k_eff'])),
        'max_k_eff': float(np.max(time_series['k_eff'])),
        
        'min_tau_norm': float(rc.get('min_tau_norm', np.min(tau_arr))),
        'min_tau_norm_time': float(rc.get('min_tau_norm_time', 0)),
        'mean_tau_norm': float(np.mean(tau_arr)),
        
        'dwell_cascade': float(rc.get('dwell_frac_cascade', 0)),
        'dwell_transitional': float(rc.get('dwell_frac_transitional', 0)),
        'dwell_diffusion': float(rc.get('dwell_frac_diffusion', 0)),
        
        'total_crossings': int(rc.get('total_crossings', 0)),
        'first_cascade_entry': rc.get('first_cascade_entry'),
        
        # Improved lag analysis
        'accel_lag': float(accel_lag_info.get('accel_lag', 0)),
        'accel_lag_method': accel_lag_info.get('method', 'none'),
        'accel_n_events': int(accel_lag_info.get('n_events', 0)),
        'predictive': bool(accel_lag_info.get('predictive', False)),
        
        'max_ew_score': float(summary.get('max_ew_score', 0)),
        'max_gamma_exp': float(np.max(time_series['gamma_exp'])),
        
        'time_series': time_series,
    }
    
    print(f"\n  Complete: {step} steps")
    print(f"  Peak ω: {results['peak_omega_max']:.2f} (percentile: {results['peak_omega_percentile']:.2f})")
    print(f"  Min τ_norm: {results['min_tau_norm']:.4f} at t={results['min_tau_norm_time']:.2f}")
    print(f"  Regime: cascade={results['dwell_cascade']*100:.0f}%, trans={results['dwell_transitional']*100:.0f}%")
    print(f"  Accel lag: {results['accel_lag']:.1f} ({results['accel_lag_method']}, {results['accel_n_events']} events)")
    print(f"  Predictive: {results['predictive']}")
    
    return results


def create_detailed_plot(results: dict, output_path: str):
    """Create detailed 6-panel plot for a single run."""
    
    ts = results['time_series']
    t = np.array(ts['time'])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{results['label']} - Detailed Analysis", fontsize=14)
    
    # 1. Vorticity evolution
    ax = axes[0, 0]
    ax.plot(t, ts['omega_max'], 'b-', label='ω_max', linewidth=2)
    ax.plot(t, ts['omega_percentile'], 'r--', label='ω_99.5%', linewidth=1.5)
    ax.axhline(y=results['peak_omega_percentile'], color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Vorticity')
    ax.set_title('Vorticity Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. τ_norm evolution
    ax = axes[0, 1]
    ax.plot(t, ts['tau_norm'], 'g-', linewidth=2)
    ax.axhline(y=0.1, color='red', linestyle='--', label='Cascade threshold')
    ax.axhline(y=0.5, color='orange', linestyle='--', label='Diffusion threshold')
    ax.axhline(y=results['min_tau_norm'], color='purple', linestyle=':', alpha=0.7)
    ax.fill_between(t, 0, 0.1, alpha=0.15, color='red')
    ax.fill_between(t, 0.1, 0.5, alpha=0.15, color='orange')
    ax.set_xlabel('Time')
    ax.set_ylabel('τ_norm')
    ax.set_title(f'τ_norm (min={results["min_tau_norm"]:.4f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. k_eff evolution
    ax = axes[0, 2]
    ax.plot(t, ts['k_eff'], 'm-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('k_eff')
    ax.set_title(f'Effective Wavenumber (max={results["max_k_eff"]:.1f})')
    ax.grid(True, alpha=0.3)
    
    # 4. γ_exp (growth rate)
    ax = axes[1, 0]
    ax.plot(t, ts['gamma_exp'], 'orange', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Amplifier threshold')
    ax.set_xlabel('Time')
    ax.set_ylabel('γ_exp')
    ax.set_title(f'Exponential Growth Rate (max={results["max_gamma_exp"]:.2f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. τ_norm vs ω phase plot
    ax = axes[1, 1]
    omega = np.array(ts['omega_percentile'])
    tau = np.array(ts['tau_norm'])
    
    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
    for i in range(len(t)-1):
        ax.plot([omega[i], omega[i+1]], [tau[i], tau[i+1]], 
               color=colors[i], linewidth=1.5)
    
    ax.scatter([omega[0]], [tau[0]], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter([omega[-1]], [tau[-1]], c='red', s=100, marker='s', label='End', zorder=5)
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('ω_percentile')
    ax.set_ylabel('τ_norm')
    ax.set_title('Phase Space (τ vs ω)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 6. Early warning score
    ax = axes[1, 2]
    ax.plot(t, ts['ew_score'], 'purple', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Alert threshold')
    ax.set_xlabel('Time')
    ax.set_ylabel('EW Score')
    ax.set_title(f'Early Warning (max={results["max_ew_score"]:.2f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = f"./high_re_tests_{timestamp}"
    os.makedirs(base_output, exist_ok=True)
    
    print("="*70)
    print("High-Re Tests with Improved Event Detection")
    print("="*70)
    
    all_results = {}
    
    # =========================================================================
    # Test 1: TG at Re=3200, N=128 - Push toward cascade
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: Taylor-Green Re=3200, N=128")
    print("Goal: Push τ_norm toward 0.1 threshold")
    print("="*70)
    
    results = run_simulation(
        grid_size=128, reynolds_number=3200, ic_type='taylor_green',
        T_final=5.0, dt_mult=1.0,
        output_dir=os.path.join(base_output, "TG_Re3200"),
        label="TG_N128_Re3200"
    )
    all_results["TG_N128_Re3200"] = results
    create_detailed_plot(results, os.path.join(base_output, "TG_Re3200_detailed.png"))
    
    # =========================================================================
    # Test 2: TG at Re=1600, N=128, T=10 - Extended time
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: Taylor-Green Re=1600, N=128, T=10")
    print("Goal: Capture full growth → decay cycle")
    print("="*70)
    
    results = run_simulation(
        grid_size=128, reynolds_number=1600, ic_type='taylor_green',
        T_final=10.0, dt_mult=1.0,
        output_dir=os.path.join(base_output, "TG_Re1600_T10"),
        label="TG_N128_Re1600_T10"
    )
    all_results["TG_N128_Re1600_T10"] = results
    create_detailed_plot(results, os.path.join(base_output, "TG_Re1600_T10_detailed.png"))
    
    # =========================================================================
    # Test 3: KP at Re=1600, N=128, tighter dt
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 3: Kida-Pelz Re=1600, N=128, dt_mult=0.5")
    print("Goal: Improve convergence with tighter timestep")
    print("="*70)
    
    results = run_simulation(
        grid_size=128, reynolds_number=1600, ic_type='kida_pelz',
        T_final=5.0, dt_mult=0.5,  # Tighter timestep
        output_dir=os.path.join(base_output, "KP_Re1600_tight"),
        label="KP_N128_Re1600_tight"
    )
    all_results["KP_N128_Re1600_tight"] = results
    create_detailed_plot(results, os.path.join(base_output, "KP_Re1600_tight_detailed.png"))
    
    # =========================================================================
    # Test 4: TG at Re=6400 (if Re=3200 doesn't hit cascade)
    # =========================================================================
    if all_results["TG_N128_Re3200"]['min_tau_norm'] > 0.12:
        print("\n" + "="*70)
        print("TEST 4: Taylor-Green Re=6400, N=128")
        print("Goal: Finally reach cascade regime")
        print("="*70)
        
        results = run_simulation(
            grid_size=128, reynolds_number=6400, ic_type='taylor_green',
            T_final=5.0, dt_mult=0.75,  # Slightly tighter for stability
            output_dir=os.path.join(base_output, "TG_Re6400"),
            label="TG_N128_Re6400"
        )
        all_results["TG_N128_Re6400"] = results
        create_detailed_plot(results, os.path.join(base_output, "TG_Re6400_detailed.png"))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("HIGH-Re TEST SUMMARY")
    print("="*70)
    
    print(f"\n{'Label':<25} {'Re':>6} {'Peak ω':>8} {'min τ':>8} {'Casc%':>6} {'Lag':>6} {'Pred':>5}")
    print("-"*70)
    
    for label, res in all_results.items():
        pred = "Yes" if res['predictive'] else "No"
        print(f"{label:<25} {res['reynolds_number']:>6.0f} {res['peak_omega_percentile']:>8.2f} "
              f"{res['min_tau_norm']:>8.4f} {res['dwell_cascade']*100:>5.0f}% "
              f"{res['accel_lag']:>+6.1f} {pred:>5}")
    
    # Save summary
    summary_data = {label: {k: v for k, v in res.items() if k != 'time_series'} 
                    for label, res in all_results.items()}
    
    with open(os.path.join(base_output, 'high_re_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {base_output}")
    
    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    tg_3200 = all_results.get("TG_N128_Re3200", {})
    tg_1600_t10 = all_results.get("TG_N128_Re1600_T10", {})
    
    print(f"\n1. TG Re=3200: min τ_norm = {tg_3200.get('min_tau_norm', 'N/A')}")
    print(f"   → {'Entered cascade!' if tg_3200.get('dwell_cascade', 0) > 0 else 'Still transitional'}")
    
    print(f"\n2. TG Re=1600 T=10: Captured decay phase")
    print(f"   → Predictive: {tg_1600_t10.get('predictive', 'N/A')}")
    
    return all_results


if __name__ == "__main__":
    all_results = main()
