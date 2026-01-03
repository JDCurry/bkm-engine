#!/usr/bin/env python3
"""
run_resolution_disambiguation.py
================================
Definitive test to distinguish physics floor vs resolution ceiling.

Key question: Is TG's τ_norm plateau at ~0.17 due to:
  A) Physics: Coherent structure self-regulates (intrinsic floor)
  B) Resolution: N=128 bottlenecks smallest scales (resolution ceiling)

Tests:
1. TG Re=3200, N=256, T=4 - The disambiguation run
2. TG Re=3200, N=128, T=10 - Extended time at standard resolution
3. TG Re=1600, N=256, T=4 - Lower Re at high resolution (baseline)

Prediction fork:
- If physics floor: min τ_norm stays ~0.17 even as k_eff_max increases
- If resolution ceiling: k_eff_max rises AND min τ_norm drops below 0.17

Also adds lag_time (physical time units) alongside lag_steps.

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
# Event Detection (from high_re_tests.py)
# ============================================================================

def detect_acceleration_events(omega: np.ndarray, times: np.ndarray, 
                                smooth_sigma: float = 3.0) -> list:
    """Detect ω acceleration events."""
    if len(omega) < 20:
        return []
    
    omega_smooth = gaussian_filter1d(omega, sigma=smooth_sigma)
    dt = np.gradient(times)
    d_omega = np.gradient(omega_smooth) / (dt + 1e-10)
    d2_omega = np.gradient(d_omega) / (dt + 1e-10)
    
    accel_threshold = 0.1 * np.max(np.abs(d2_omega))
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
            window_accel = d2_omega[start_idx:end_idx]
            if len(window_accel) > 0:
                peak_idx = start_idx + np.argmax(window_accel)
                events.append((start_idx, peak_idx, end_idx))
    
    return events


def compute_acceleration_lag(tau_norm: np.ndarray, omega: np.ndarray, 
                              times: np.ndarray) -> dict:
    """Compute lag with both steps and physical time."""
    events = detect_acceleration_events(omega, times)
    
    if len(events) == 0:
        if len(tau_norm) < 20:
            return {'accel_lag_steps': 0, 'accel_lag_time': 0, 'n_events': 0, 
                    'method': 'none', 'predictive': False}
        
        dt = np.gradient(times)
        median_dt = np.median(dt)
        d_tau = np.gradient(tau_norm) / (dt + 1e-10)
        d_omega = np.gradient(omega) / (dt + 1e-10)
        
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
            'accel_lag_steps': best_lag,
            'accel_lag_time': float(best_lag * median_dt),
            'median_dt': float(median_dt),
            'accel_corr': float(best_corr),
            'n_events': 0,
            'method': 'derivative_correlation',
            'predictive': best_lag > 0 and best_corr < -0.3,
        }
    
    # Event-based analysis
    dt = np.gradient(times)
    median_dt = np.median(dt)
    
    lags_steps = []
    lags_time = []
    
    for start_idx, peak_idx, end_idx in events:
        lookback = min(20, start_idx)
        if lookback < 3:
            continue
        
        tau_window = tau_norm[start_idx - lookback:start_idx]
        if len(tau_window) == 0:
            continue
        
        min_idx_in_window = np.argmin(tau_window)
        min_idx_global = start_idx - lookback + min_idx_in_window
        
        lag_steps = start_idx - min_idx_global
        lag_time = times[start_idx] - times[min_idx_global]
        
        lags_steps.append(lag_steps)
        lags_time.append(lag_time)
    
    if len(lags_steps) == 0:
        return {'accel_lag_steps': 0, 'accel_lag_time': 0, 'n_events': 0, 
                'method': 'events_no_lags', 'predictive': False}
    
    mean_lag_steps = np.mean(lags_steps)
    mean_lag_time = np.mean(lags_time)
    
    return {
        'accel_lag_steps': float(mean_lag_steps),
        'accel_lag_time': float(mean_lag_time),
        'accel_lag_steps_std': float(np.std(lags_steps)),
        'accel_lag_time_std': float(np.std(lags_time)),
        'median_dt': float(median_dt),
        'n_events': len(events),
        'method': 'acceleration_events',
        'predictive': mean_lag_steps > 2,
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
    return {'type': 'taylor_green', 'E0': solver.initial_energy, 'W0': solver.compute_max_vorticity()}


# ============================================================================
# Simulation Runner
# ============================================================================

def run_simulation(grid_size: int, reynolds_number: float, T_final: float,
                   dt_mult: float = 1.0, output_dir: str = "./results", 
                   label: str = None) -> dict:
    """Run TG simulation with comprehensive diagnostics."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    dx = 2 * np.pi / grid_size
    dt_base = 0.2 * dx / 1.0
    dt_safe = dt_base * dt_mult
    
    if label is None:
        label = f"TG_N{grid_size}_Re{int(reynolds_number)}"
    
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"  N={grid_size}, Re={reynolds_number}, T={T_final}, dt_mult={dt_mult}")
    print(f"  dx={dx:.6f}, dt={dt_safe:.6f}")
    print(f"{'='*70}")
    
    solver = CUDABKMSolver(
        grid_size=grid_size, reynolds_number=reynolds_number, dt=dt_safe,
        CFL_target=0.2 * dt_mult, adapt_dt=False, track_alignment=False,
        rho_soft=0.99, rho_hard=0.999, startup_steps=0,
    )
    
    ic_info = initialize_taylor_green(solver, amplitude=1.0)
    print(f"  E0={ic_info['E0']:.4f}, W0={ic_info['W0']:.2f}")
    
    # Lighter config for 256³ runs
    track_every = 10 if grid_size >= 256 else 5
    
    config = HeatKernelConfigV51(
        s_values=[1.0, 1.25],  # Reduced for speed
        t_viscosity_mult=10.0,
        track_every=track_every,
        omega_percentile=99.5,
    )
    
    tracker = HeatKernelAlignmentTrackerV51(solver, config)
    tracker.start_logging(output_dir, filename=f"hk_{label}.csv")
    
    step = 0
    last_print = 0.0
    print_interval = max(0.25, T_final / 20)
    
    time_series = {
        'time': [], 'omega_max': [], 'omega_percentile': [], 
        'tau_norm': [], 'k_eff': [], 'ew_score': [], 'gamma_exp': [],
        'dt': []
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
            time_series['dt'].append(dt_safe)
            
            if solver.current_time - last_print >= print_interval:
                tau_norm = metrics.get('tau_norm', 0)
                regime = metrics.get('regime', 'unknown')[:5]
                k_eff = metrics.get('k_eff', 0)
                
                print(f"  t={solver.current_time:.2f} | ω={result['vorticity']:.2f} | "
                      f"τ={tau_norm:.4f} | k={k_eff:.2f} | {regime}")
                last_print = solver.current_time
        
        step += 1
        if step > 2000000:
            break
    
    tracker.close()
    summary = tracker.get_summary()
    
    # Convert to arrays
    tau_arr = np.array(time_series['tau_norm'])
    omega_arr = np.array(time_series['omega_percentile'])
    time_arr = np.array(time_series['time'])
    
    # Lag analysis with physical time
    lag_info = compute_acceleration_lag(tau_arr, omega_arr, time_arr)
    
    rc = summary.get('regime_crossings', {})
    
    results = {
        'label': label,
        'grid_size': int(grid_size),
        'reynolds_number': float(reynolds_number),
        'T_final': float(T_final),
        'dt': float(dt_safe),
        'dx': float(dx),
        
        'peak_omega_max': float(np.max(time_series['omega_max'])),
        'peak_omega_percentile': float(np.max(omega_arr)),
        'mean_k_eff': float(np.mean(time_series['k_eff'])),
        'max_k_eff': float(np.max(time_series['k_eff'])),
        
        'min_tau_norm': float(np.min(tau_arr)),
        'min_tau_norm_time': float(time_arr[np.argmin(tau_arr)]),
        'mean_tau_norm': float(np.mean(tau_arr)),
        
        'dwell_cascade': float(rc.get('dwell_frac_cascade', 0)),
        'dwell_transitional': float(rc.get('dwell_frac_transitional', 0)),
        'dwell_diffusion': float(rc.get('dwell_frac_diffusion', 0)),
        
        'total_crossings': int(rc.get('total_crossings', 0)),
        'entered_cascade': rc.get('first_cascade_entry') is not None,
        
        # Lag in both units
        'accel_lag_steps': float(lag_info.get('accel_lag_steps', 0)),
        'accel_lag_time': float(lag_info.get('accel_lag_time', 0)),
        'accel_n_events': int(lag_info.get('n_events', 0)),
        'predictive': bool(lag_info.get('predictive', False)),
        
        'max_gamma_exp': float(np.max(time_series['gamma_exp'])),
        
        'time_series': time_series,
    }
    
    print(f"\n  Complete: {step} steps")
    print(f"  Peak ω_max: {results['peak_omega_max']:.2f}")
    print(f"  Peak ω_99.5%: {results['peak_omega_percentile']:.2f}")
    print(f"  Max k_eff: {results['max_k_eff']:.2f}")
    print(f"  Min τ_norm: {results['min_tau_norm']:.4f} at t={results['min_tau_norm_time']:.2f}")
    print(f"  Entered cascade: {results['entered_cascade']}")
    print(f"  Lag: {results['accel_lag_steps']:.1f} steps = {results['accel_lag_time']:.3f} time units")
    print(f"  Predictive: {results['predictive']}")
    
    return results


def create_comparison_plot(all_results: dict, output_path: str):
    """Create comparison plot for disambiguation."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Resolution Disambiguation: Physics Floor vs Resolution Ceiling', fontsize=14)
    
    colors = {'N128': 'blue', 'N256': 'red'}
    markers = {'N128': 'o', 'N256': 's'}
    
    # Organize by N
    by_N = {}
    for label, res in all_results.items():
        N = res['grid_size']
        key = f'N{N}'
        if key not in by_N:
            by_N[key] = []
        by_N[key].append(res)
    
    # 1. min τ_norm vs Re
    ax = axes[0, 0]
    for N_key, results in sorted(by_N.items()):
        Res = [r['reynolds_number'] for r in results]
        taus = [r['min_tau_norm'] for r in results]
        ax.scatter(Res, taus, c=colors.get(N_key, 'gray'), marker=markers.get(N_key, 'o'),
                  s=150, label=N_key, zorder=5)
        if len(Res) > 1:
            ax.plot(sorted(Res), [taus[i] for i in np.argsort(Res)], 
                   c=colors.get(N_key, 'gray'), linestyle='--', alpha=0.5)
    
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='Cascade threshold')
    ax.axhline(y=0.17, color='orange', linestyle=':', linewidth=2, label='N=128 plateau')
    ax.set_xlabel('Reynolds Number')
    ax.set_ylabel('min τ_norm')
    ax.set_title('τ_norm Floor Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 2. max k_eff vs Re
    ax = axes[0, 1]
    for N_key, results in sorted(by_N.items()):
        Res = [r['reynolds_number'] for r in results]
        k_effs = [r['max_k_eff'] for r in results]
        ax.scatter(Res, k_effs, c=colors.get(N_key, 'gray'), marker=markers.get(N_key, 'o'),
                  s=150, label=N_key, zorder=5)
        if len(Res) > 1:
            ax.plot(sorted(Res), [k_effs[i] for i in np.argsort(Res)], 
                   c=colors.get(N_key, 'gray'), linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Reynolds Number')
    ax.set_ylabel('max k_eff')
    ax.set_title('Scale Resolution Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 3. Peak ω vs Re
    ax = axes[0, 2]
    for N_key, results in sorted(by_N.items()):
        Res = [r['reynolds_number'] for r in results]
        omegas = [r['peak_omega_percentile'] for r in results]
        ax.scatter(Res, omegas, c=colors.get(N_key, 'gray'), marker=markers.get(N_key, 'o'),
                  s=150, label=N_key, zorder=5)
    
    ax.set_xlabel('Reynolds Number')
    ax.set_ylabel('Peak ω (99.5%)')
    ax.set_title('Vorticity Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 4. τ_norm time evolution
    ax = axes[1, 0]
    for label, res in all_results.items():
        t = res['time_series']['time']
        tau = res['time_series']['tau_norm']
        N = res['grid_size']
        N_key = f'N{N}'
        ax.plot(t, tau, color=colors.get(N_key, 'gray'), linewidth=2, 
               label=f"{label}", alpha=0.8)
    
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7)
    ax.fill_between([0, 15], [0, 0], [0.1, 0.1], alpha=0.15, color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('τ_norm')
    ax.set_title('τ_norm Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. k_eff evolution
    ax = axes[1, 1]
    for label, res in all_results.items():
        t = res['time_series']['time']
        k = res['time_series']['k_eff']
        N = res['grid_size']
        N_key = f'N{N}'
        ax.plot(t, k, color=colors.get(N_key, 'gray'), linewidth=2, 
               label=f"{label}", alpha=0.8)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('k_eff')
    ax.set_title('k_eff Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 6. Summary bar chart
    ax = axes[1, 2]
    labels = list(all_results.keys())
    x = np.arange(len(labels))
    
    min_taus = [all_results[l]['min_tau_norm'] for l in labels]
    max_k_effs = [all_results[l]['max_k_eff'] for l in labels]
    
    bar_colors = [colors.get(f"N{all_results[l]['grid_size']}", 'gray') for l in labels]
    
    bars1 = ax.bar(x - 0.2, min_taus, 0.35, label='min τ_norm', color=bar_colors, alpha=0.7)
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0.17, color='orange', linestyle=':', alpha=0.5)
    
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, max_k_effs, 0.35, label='max k_eff', color=bar_colors, alpha=0.4, hatch='//')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('min τ_norm', color='blue')
    ax2.set_ylabel('max k_eff', color='gray')
    ax.set_title('Summary: τ_norm vs k_eff')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = f"./resolution_disambiguation_{timestamp}"
    os.makedirs(base_output, exist_ok=True)
    
    print("="*70)
    print("RESOLUTION DISAMBIGUATION TEST")
    print("Question: Is TG's τ_norm plateau physics or resolution?")
    print("="*70)
    
    all_results = {}
    
    # =========================================================================
    # Baseline: N=128 reference points
    # =========================================================================
    print("\n" + "="*70)
    print("BASELINE: N=128 Reference")
    print("="*70)
    
    # Re=1600 baseline
    results = run_simulation(
        grid_size=128, reynolds_number=1600, T_final=5.0,
        output_dir=os.path.join(base_output, "N128_Re1600"),
        label="N128_Re1600"
    )
    all_results["N128_Re1600"] = results
    
    # Re=3200 baseline
    results = run_simulation(
        grid_size=128, reynolds_number=3200, T_final=5.0,
        output_dir=os.path.join(base_output, "N128_Re3200"),
        label="N128_Re3200"
    )
    all_results["N128_Re3200"] = results
    
    # =========================================================================
    # High Resolution: N=256
    # =========================================================================
    print("\n" + "="*70)
    print("HIGH RESOLUTION: N=256 (The Disambiguation Runs)")
    print("="*70)
    
    # Re=1600 at N=256
    results = run_simulation(
        grid_size=256, reynolds_number=1600, T_final=4.0,  # Shorter for speed
        dt_mult=1.0,
        output_dir=os.path.join(base_output, "N256_Re1600"),
        label="N256_Re1600"
    )
    all_results["N256_Re1600"] = results
    
    # Re=3200 at N=256 - THE KEY TEST
    results = run_simulation(
        grid_size=256, reynolds_number=3200, T_final=4.0,
        dt_mult=0.8,  # Slightly conservative
        output_dir=os.path.join(base_output, "N256_Re3200"),
        label="N256_Re3200"
    )
    all_results["N256_Re3200"] = results
    
    # =========================================================================
    # Extended time at N=128
    # =========================================================================
    print("\n" + "="*70)
    print("EXTENDED TIME: N=128, Re=3200, T=10")
    print("="*70)
    
    results = run_simulation(
        grid_size=128, reynolds_number=3200, T_final=10.0,
        output_dir=os.path.join(base_output, "N128_Re3200_T10"),
        label="N128_Re3200_T10"
    )
    all_results["N128_Re3200_T10"] = results
    
    # =========================================================================
    # Create comparison plot
    # =========================================================================
    create_comparison_plot(all_results, os.path.join(base_output, "disambiguation_comparison.png"))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("DISAMBIGUATION RESULTS")
    print("="*70)
    
    print(f"\n{'Label':<20} {'N':>5} {'Re':>5} {'T':>4} {'min τ':>8} {'max k':>7} {'Pred':>5}")
    print("-"*70)
    
    for label, res in sorted(all_results.items(), key=lambda x: (x[1]['grid_size'], x[1]['reynolds_number'])):
        pred = "Yes" if res['predictive'] else "No"
        print(f"{label:<20} {res['grid_size']:>5} {res['reynolds_number']:>5.0f} "
              f"{res['T_final']:>4.0f} {res['min_tau_norm']:>8.4f} "
              f"{res['max_k_eff']:>7.2f} {pred:>5}")
    
    # Key comparison
    print("\n" + "="*70)
    print("KEY COMPARISON: N=128 vs N=256 at Re=3200")
    print("="*70)
    
    n128 = all_results.get("N128_Re3200", {})
    n256 = all_results.get("N256_Re3200", {})
    
    if n128 and n256:
        tau_change = (n256['min_tau_norm'] - n128['min_tau_norm']) / n128['min_tau_norm'] * 100
        k_change = (n256['max_k_eff'] - n128['max_k_eff']) / n128['max_k_eff'] * 100
        
        print(f"\n  N=128: min τ_norm = {n128['min_tau_norm']:.4f}, max k_eff = {n128['max_k_eff']:.2f}")
        print(f"  N=256: min τ_norm = {n256['min_tau_norm']:.4f}, max k_eff = {n256['max_k_eff']:.2f}")
        print(f"\n  Δτ_norm = {tau_change:+.1f}%")
        print(f"  Δk_eff = {k_change:+.1f}%")
        
        print("\n  VERDICT:")
        if tau_change < -10 and k_change > 10:
            print("  → RESOLUTION CEILING: Higher N reveals smaller scales AND lower τ_norm")
            print("  → The N=128 plateau was a resolution artifact")
        elif abs(tau_change) < 10 and k_change > 10:
            print("  → MIXED: k_eff increases but τ_norm plateau persists")
            print("  → Partial resolution effect, but physics floor may exist")
        elif abs(tau_change) < 10 and abs(k_change) < 10:
            print("  → PHYSICS FLOOR: Both τ_norm and k_eff converged")
            print("  → The plateau is intrinsic to TG dynamics")
        else:
            print("  → INCONCLUSIVE: Need more data")
    
    # Save summary
    summary_data = {label: {k: v for k, v in res.items() if k != 'time_series'} 
                    for label, res in all_results.items()}
    
    with open(os.path.join(base_output, 'disambiguation_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nResults saved to: {base_output}")
    
    return all_results


if __name__ == "__main__":
    all_results = main()
