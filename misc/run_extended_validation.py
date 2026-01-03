#!/usr/bin/env python3
"""
run_extended_validation.py
==========================
Extended validation at higher Reynolds numbers and resolutions.

Tests:
1. Re sweep: 400, 800, 1600 at 64³
2. Resolution sweep: 64³, 96³, 128³ at Re=800
3. Long-time evolution: Re=800, 128³, T=10

Looking for:
- Does higher Re push flows into cascade regime?
- Does resolution affect regime classification?
- Do crossings become more frequent at higher Re?

Author: Josh Curry & Claude  
Date: December 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_bkm_engine import CUDABKMSolver, backend
from heat_kernel_ns_integration_v5_1 import (
    HeatKernelConfigV51,
    HeatKernelAlignmentTrackerV51,
)

print(f"Using backend: {backend}")


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


def run_simulation(grid_size, reynolds_number, T_final, output_dir, label):
    """Run single simulation with v5.1 tracking."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    dx = 2 * np.pi / grid_size
    dt_safe = 0.2 * dx / 1.0
    
    print(f"\n{'='*60}")
    print(f"{label}: N={grid_size}, Re={reynolds_number}, T={T_final}")
    print(f"{'='*60}")
    
    solver = CUDABKMSolver(
        grid_size=grid_size, reynolds_number=reynolds_number, dt=dt_safe,
        CFL_target=0.2, adapt_dt=False, track_alignment=False,
        rho_soft=0.99, rho_hard=0.999, startup_steps=0,
    )
    
    ic_info = initialize_taylor_green(solver, amplitude=1.0)
    print(f"  E0={ic_info['E0']:.4f}, W0={ic_info['W0']:.2f}")
    
    config = HeatKernelConfigV51(
        s_values=[0.7, 1.0, 1.25, 1.5],
        t_viscosity_mult=10.0,
        track_every=5,
        omega_percentile=99.5,
        tau_norm_cascade_threshold=0.1,
        tau_norm_diffusion_threshold=0.5,
    )
    
    tracker = HeatKernelAlignmentTrackerV51(solver, config)
    tracker.start_logging(output_dir, filename=f"hk_{label}.csv")
    
    step = 0
    last_print = 0.0
    print_interval = max(0.5, T_final / 15)
    
    time_series = {
        'time': [], 'omega_max': [], 'tau_norm': [], 'ew_score': [],
        'k_eff': [], 'regime': [], 'gamma_exp': []
    }
    
    while solver.current_time < T_final and not solver.should_stop:
        result = solver.evolve_one_timestep()
        
        if step % config.track_every == 0:
            metrics = tracker.record(step)
            
            time_series['time'].append(solver.current_time)
            time_series['omega_max'].append(result['vorticity'])
            time_series['tau_norm'].append(metrics.get('tau_norm', 1.0))
            time_series['ew_score'].append(metrics.get('ew_score', 0.0))
            time_series['k_eff'].append(metrics.get('k_eff', 1.0))
            time_series['regime'].append(metrics.get('regime', 'unknown'))
            time_series['gamma_exp'].append(metrics.get('gamma_exp', 0))
            
            if solver.current_time - last_print >= print_interval:
                tau_norm = metrics.get('tau_norm', 0)
                regime = metrics.get('regime', 'unknown')[:5]
                k_eff = metrics.get('k_eff', 0)
                
                print(f"  t={solver.current_time:.2f} | ω={result['vorticity']:.2f} | "
                      f"τ_norm={tau_norm:.3f} | k_eff={k_eff:.1f} | {regime}")
                last_print = solver.current_time
        
        step += 1
        if step > 1000000:
            break
    
    tracker.close()
    summary = tracker.get_summary()
    
    results = {
        'label': label,
        'grid_size': grid_size,
        'reynolds_number': reynolds_number,
        'T_final': T_final,
        'summary': summary,
        'time_series': time_series,
    }
    
    # Print summary
    rc = summary.get('regime_crossings', {})
    print(f"\n  Complete: {step} steps, Peak ω: {max(time_series['omega_max']):.2f}")
    print(f"  Min τ_norm: {rc.get('min_tau_norm', 0):.3f} at t={rc.get('min_tau_norm_time', 0):.2f}")
    print(f"  Crossings: {rc.get('total_crossings', 0)}")
    print(f"  Dwell: cascade={rc.get('dwell_frac_cascade', 0)*100:.0f}%, "
          f"trans={rc.get('dwell_frac_transitional', 0)*100:.0f}%, "
          f"diff={rc.get('dwell_frac_diffusion', 0)*100:.0f}%")
    
    return results


def create_comparison_plots(all_results, output_dir, title_prefix=""):
    """Create comparison plots for a set of runs."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(all_results)))
    
    # 1. Vorticity evolution
    ax = axes[0, 0]
    for idx, (label, res) in enumerate(all_results.items()):
        t = res['time_series']['time']
        omega = res['time_series']['omega_max']
        if len(t) > 0:
            ax.plot(t, omega, label=label, color=colors[idx], linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Max Vorticity')
    ax.set_title(f'{title_prefix}Vorticity Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. τ_norm evolution
    ax = axes[0, 1]
    for idx, (label, res) in enumerate(all_results.items()):
        t = res['time_series']['time']
        tau_norm = res['time_series']['tau_norm']
        if len(t) > 0:
            ax.plot(t, tau_norm, label=label, color=colors[idx], linewidth=2)
    
    ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.fill_between([0, 20], [0, 0], [0.1, 0.1], alpha=0.15, color='red')
    ax.fill_between([0, 20], [0.1, 0.1], [0.5, 0.5], alpha=0.15, color='orange')
    ax.fill_between([0, 20], [0.5, 0.5], [1.0, 1.0], alpha=0.15, color='green')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('τ_norm')
    ax.set_title(f'{title_prefix}τ_norm Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. k_eff evolution
    ax = axes[1, 0]
    for idx, (label, res) in enumerate(all_results.items()):
        t = res['time_series']['time']
        k_eff = res['time_series']['k_eff']
        if len(t) > 0:
            ax.plot(t, k_eff, label=label, color=colors[idx], linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('k_eff')
    ax.set_title(f'{title_prefix}Effective Wavenumber')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Summary bar chart
    ax = axes[1, 1]
    labels = list(all_results.keys())
    x = np.arange(len(labels))
    
    min_tau = [all_results[l]['summary'].get('regime_crossings', {}).get('min_tau_norm', 0) for l in labels]
    crossings = [all_results[l]['summary'].get('regime_crossings', {}).get('total_crossings', 0) for l in labels]
    
    ax.bar(x - 0.2, min_tau, 0.35, label='min τ_norm', color='blue', alpha=0.7)
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, crossings, 0.35, label='Crossings', color='red', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('min τ_norm', color='blue')
    ax2.set_ylabel('Regime Crossings', color='red')
    ax.set_title(f'{title_prefix}Summary Metrics')
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = f"./extended_validation_{timestamp}"
    os.makedirs(base_output, exist_ok=True)
    
    print("="*70)
    print("Extended Validation: Higher Re and Resolution")
    print("="*70)
    
    all_results = {}
    
    # =========================================================================
    # Test 1: Reynolds number sweep at 64³
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: Reynolds Number Sweep (64³)")
    print("="*70)
    
    re_results = {}
    for Re in [400, 800, 1600]:
        label = f"Re{Re}"
        results = run_simulation(
            grid_size=64, reynolds_number=Re, T_final=5.0,
            output_dir=os.path.join(base_output, f"Re_sweep/{label}"),
            label=label
        )
        re_results[label] = results
        all_results[label] = results
    
    fig = create_comparison_plots(re_results, base_output, "Re Sweep: ")
    fig.savefig(os.path.join(base_output, 're_sweep_comparison.png'), dpi=150)
    plt.close()
    print(f"\nSaved: {base_output}/re_sweep_comparison.png")
    
    # =========================================================================
    # Test 2: Resolution sweep at Re=800
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: Resolution Sweep (Re=800)")
    print("="*70)
    
    res_results = {}
    for N in [64, 96, 128]:
        label = f"N{N}"
        results = run_simulation(
            grid_size=N, reynolds_number=800, T_final=5.0,
            output_dir=os.path.join(base_output, f"res_sweep/{label}"),
            label=label
        )
        res_results[label] = results
        all_results[f"N{N}_Re800"] = results
    
    fig = create_comparison_plots(res_results, base_output, "Resolution Sweep (Re=800): ")
    fig.savefig(os.path.join(base_output, 'resolution_sweep_comparison.png'), dpi=150)
    plt.close()
    print(f"\nSaved: {base_output}/resolution_sweep_comparison.png")
    
    # =========================================================================
    # Test 3: Long-time evolution at high Re
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 3: Long-Time Evolution (Re=1600, 64³, T=10)")
    print("="*70)
    
    long_results = run_simulation(
        grid_size=64, reynolds_number=1600, T_final=10.0,
        output_dir=os.path.join(base_output, "long_time"),
        label="Re1600_T10"
    )
    all_results["Re1600_T10"] = long_results
    
    # Create long-time specific plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t = long_results['time_series']['time']
    
    ax = axes[0, 0]
    ax.plot(t, long_results['time_series']['omega_max'], 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Max Vorticity')
    ax.set_title('Long-Time Vorticity (Re=1600, T=10)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    tau_norm = long_results['time_series']['tau_norm']
    ax.plot(t, tau_norm, 'r-', linewidth=2)
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7)
    ax.fill_between(t, 0, 0.1, alpha=0.2, color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('τ_norm')
    ax.set_title('Long-Time τ_norm Evolution')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(t, long_results['time_series']['k_eff'], 'g-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('k_eff')
    ax.set_title('Effective Wavenumber Evolution')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(t, long_results['time_series']['ew_score'], 'm-', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', label='EW threshold')
    ax.set_xlabel('Time')
    ax.set_ylabel('EW Score')
    ax.set_title('Early Warning Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(base_output, 'long_time_evolution.png'), dpi=150)
    plt.close()
    print(f"\nSaved: {base_output}/long_time_evolution.png")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("EXTENDED VALIDATION SUMMARY")
    print("="*70)
    
    summary_data = {}
    for label, res in all_results.items():
        rc = res['summary'].get('regime_crossings', {})
        summary_data[label] = {
            'grid_size': res.get('grid_size', 64),
            'reynolds_number': res.get('reynolds_number', 400),
            'peak_omega': max(res['time_series']['omega_max']),
            'min_tau_norm': rc.get('min_tau_norm', 0),
            'crossings': rc.get('total_crossings', 0),
            'dwell_cascade': rc.get('dwell_frac_cascade', 0),
            'dwell_transitional': rc.get('dwell_frac_transitional', 0),
            'max_ew': res['summary'].get('max_ew_score', 0),
        }
    
    print(f"\n{'Label':<15} {'N':>4} {'Re':>5} {'Peak ω':>8} {'min τ':>7} {'Cross':>6} {'Casc%':>6}")
    print("-"*60)
    for label, data in summary_data.items():
        print(f"{label:<15} {data['grid_size']:>4} {data['reynolds_number']:>5} "
              f"{data['peak_omega']:>8.2f} {data['min_tau_norm']:>7.3f} "
              f"{data['crossings']:>6} {data['dwell_cascade']*100:>5.0f}%")
    
    with open(os.path.join(base_output, 'extended_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nAll results saved to: {base_output}")
    
    return all_results


if __name__ == "__main__":
    all_results = main()
