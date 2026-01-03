#!/usr/bin/env python3
"""
run_comprehensive_tests_v4.py
=============================
Comprehensive test suite using v4 Heat Kernel diagnostics.

v4 Fixes:
1. τ_ratio uses k_eff (enstrophy-weighted) not k_max
2. Uses ω_percentile (99.5%) not ω_max to avoid outliers
3. Three-way regime classification (cascade/transitional/diffusion)
4. Fixed vorticity plot data wiring

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

# Import engine
try:
    from mixed_precision_engine import CUDABKMSolver, backend
    ENGINE_NAME = "mixed_precision_engine"
except ImportError:
    from unified_bkm_engine import CUDABKMSolver, backend
    ENGINE_NAME = "unified_bkm_engine"

# Use v4 tracker
from heat_kernel_ns_integration_v4 import (
    HeatKernelConfig,
    HeatKernelAlignmentTrackerV4,
    analyze_cascade_dynamics_v4,
    analyze_growth_signatures_v4,
    analyze_hyperdissipation_threshold_v4,
)

print(f"Using {ENGINE_NAME} (backend: {backend})")
print("Using v4 Heat Kernel diagnostics (fixed τ_ratio)")


def initialize_taylor_green(solver, amplitude=1.0):
    """Initialize Taylor-Green vortex."""
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


def initialize_random_turbulence(solver, energy_level=0.1, k_peak=4):
    """Initialize with random turbulence."""
    xp = solver.xp
    np_random = np.random.RandomState(42)
    
    shape = (solver.nx, solver.ny, solver.nz)
    
    u_hat = (np_random.randn(*shape) + 1j * np_random.randn(*shape)).astype(np.complex128)
    v_hat = (np_random.randn(*shape) + 1j * np_random.randn(*shape)).astype(np.complex128)
    w_hat = (np_random.randn(*shape) + 1j * np_random.randn(*shape)).astype(np.complex128)
    
    kx = np.fft.fftfreq(solver.nx, solver.dx) * 2 * np.pi
    ky = np.fft.fftfreq(solver.ny, solver.dy) * 2 * np.pi
    kz = np.fft.fftfreq(solver.nz, solver.dz) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    K[0, 0, 0] = 1.0
    
    spectrum = (K / k_peak)**4 * np.exp(-2 * (K / k_peak)**2)
    spectrum[0, 0, 0] = 0
    
    u_hat *= spectrum
    v_hat *= spectrum
    w_hat *= spectrum
    
    K2 = K**2
    K2[0, 0, 0] = 1.0
    kdotu = KX * u_hat + KY * v_hat + KZ * w_hat
    u_hat -= KX * kdotu / K2
    v_hat -= KY * kdotu / K2
    w_hat -= KZ * kdotu / K2
    
    u = np.fft.ifftn(u_hat).real
    v = np.fft.ifftn(v_hat).real
    w = np.fft.ifftn(w_hat).real
    
    current_energy = 0.5 * np.mean(u**2 + v**2 + w**2)
    scale = np.sqrt(energy_level / (current_energy + 1e-10))
    u *= scale
    v *= scale
    w *= scale
    
    if solver.use_gpu:
        solver.u = xp.asarray(u.astype(np.float64))
        solver.v = xp.asarray(v.astype(np.float64))
        solver.w = xp.asarray(w.astype(np.float64))
    else:
        solver.u = u.astype(solver.dtype)
        solver.v = v.astype(solver.dtype)
        solver.w = w.astype(solver.dtype)
    
    solver.u, solver.v, solver.w = solver.project_div_free(solver.u, solver.v, solver.w)
    solver.initial_energy = solver.compute_energy()


def run_simulation(
    grid_size: int,
    reynolds_number: float,
    T_final: float,
    init_type: str = 'taylor_green',
    track_every: int = 5,
    output_dir: str = "./results"
):
    """Run simulation with v4 tracking."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    dx = 2 * np.pi / grid_size
    dt_safe = 0.2 * dx / 1.0
    
    print(f"\n{'='*60}")
    print(f"Running: {init_type}, Re={reynolds_number}, T={T_final}")
    print(f"{'='*60}")
    
    solver = CUDABKMSolver(
        grid_size=grid_size,
        reynolds_number=reynolds_number,
        dt=dt_safe,
        CFL_target=0.2,
        adapt_dt=False,
        track_alignment=False,
        rho_soft=0.99,
        rho_hard=0.999,
        startup_steps=0,
    )
    
    if init_type == 'taylor_green':
        initialize_taylor_green(solver, amplitude=1.0)
        print(f"  Taylor-Green IC: E={solver.initial_energy:.4f}")
    elif init_type == 'random':
        initialize_random_turbulence(solver, energy_level=0.1, k_peak=4)
        print(f"  Random turbulence IC: E={solver.initial_energy:.4f}")
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
    
    # v4 config
    config = HeatKernelConfig(
        s_values=[0.7, 1.0, 1.2, 1.25, 1.3, 1.5],
        t_viscosity_mult=10.0,
        track_every=track_every,
        growth_window=20,
        n_shells=16,
        omega_percentile=99.5,  # v4: Use percentile
    )
    
    tracker = HeatKernelAlignmentTrackerV4(solver, config)
    tracker.start_logging(output_dir, filename=f"hk_v4_{init_type}_Re{int(reynolds_number)}.csv")
    
    step = 0
    last_print = 0.0
    print_interval = max(0.5, T_final / 20)
    
    # v4: Store time series properly for plotting
    time_series = []
    omega_series = []
    ratio_series = []
    k_eff_series = []
    regime_series = []
    
    while solver.current_time < T_final and not solver.should_stop:
        result = solver.evolve_one_timestep()
        
        if step % track_every == 0:
            metrics = tracker.record(step)
            
            # v4: Store for plotting
            time_series.append(solver.current_time)
            omega_series.append(result['vorticity'])
            ratio_series.append(metrics.get('timescale_ratio', 1.0))
            k_eff_series.append(metrics.get('k_eff', 1.0))
            regime_series.append(metrics.get('regime', 'unknown'))
            
            if solver.current_time - last_print >= print_interval:
                ratio = metrics.get('timescale_ratio', float('inf'))
                regime = metrics.get('regime', 'unknown')
                k_eff = metrics.get('k_eff', 0)
                gamma = metrics.get('gamma_exp', 0)
                
                print(f"  t={solver.current_time:.2f} | ω={result['vorticity']:.2f} | "
                      f"k_eff={k_eff:.1f} | τ_ratio={ratio:.3f} ({regime[:4]}) | "
                      f"γ_exp={gamma:.2f}")
                last_print = solver.current_time
        
        step += 1
        if step > 1000000:
            print("  Max steps reached")
            break
    
    tracker.close()
    
    summary = tracker.get_summary()
    history = tracker.history
    
    results = {
        'params': {
            'grid_size': grid_size,
            'reynolds_number': reynolds_number,
            'T_final': T_final,
            'init_type': init_type,
        },
        'summary': summary,
        'analysis': {
            'cascade': analyze_cascade_dynamics_v4(history),
            'growth': analyze_growth_signatures_v4(history),
            'threshold': analyze_hyperdissipation_threshold_v4(history, config.s_values),
        },
        'time_series': {
            'time': time_series,
            'omega_max': omega_series,
            'timescale_ratio': ratio_series,
            'k_eff': k_eff_series,
            'regime': regime_series,
        }
    }
    
    print(f"  Complete: {step} steps, T_final={solver.current_time:.3f}")
    print(f"  Peak ω: {max(omega_series):.2f}")
    print(f"  BKM integral: {summary.get('bkm_integral_total', 0):.3f}")
    
    # v4: Print regime breakdown
    cascade = results['analysis']['cascade']
    if cascade:
        print(f"  Regime breakdown:")
        print(f"    Cascade dominated (<0.1): {cascade.get('cascade_dominated_fraction', 0)*100:.1f}%")
        print(f"    Transitional (0.1-1.0): {cascade.get('transitional_fraction', 0)*100:.1f}%")
        print(f"    Diffusion dominated (>1.0): {cascade.get('diffusion_dominated_fraction', 0)*100:.1f}%")
    
    return results, tracker, solver


def create_comparison_plots_v4(all_results, output_dir):
    """Create comparison plots with v4 fixes."""
    
    print("\nCreating comparison plots (v4)...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Vorticity evolution (FIXED)
    ax = axes[0, 0]
    for label, res in all_results.items():
        t = res['time_series']['time']
        omega = res['time_series']['omega_max']
        if len(t) > 0 and len(omega) > 0:
            ax.plot(t, omega, label=label, linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Max Vorticity')
    ax.set_title('Vorticity Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Timescale ratio evolution (NEW)
    ax = axes[0, 1]
    for label, res in all_results.items():
        t = res['time_series']['time']
        ratio = res['time_series']['timescale_ratio']
        if len(t) > 0 and len(ratio) > 0:
            ax.semilogy(t, ratio, label=label, linewidth=2, alpha=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Critical (ratio=1)')
    ax.axhline(y=0.1, color='orange', linestyle=':', linewidth=1, label='Cascade threshold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Timescale Ratio (τ_cascade/τ_diffusion)')
    ax.set_title('v4: Timescale Ratio Evolution')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: k_eff evolution (NEW)
    ax = axes[0, 2]
    for label, res in all_results.items():
        t = res['time_series']['time']
        k_eff = res['time_series']['k_eff']
        if len(t) > 0 and len(k_eff) > 0:
            ax.plot(t, k_eff, label=label, linewidth=2, alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('k_eff (enstrophy-weighted)')
    ax.set_title('v4: Effective Wavenumber Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Regime breakdown bar chart (NEW)
    ax = axes[1, 0]
    labels = []
    cascade_fracs = []
    trans_fracs = []
    diff_fracs = []
    
    for label, res in all_results.items():
        cascade = res['analysis']['cascade']
        if cascade:
            labels.append(label)
            cascade_fracs.append(cascade.get('cascade_dominated_fraction', 0))
            trans_fracs.append(cascade.get('transitional_fraction', 0))
            diff_fracs.append(cascade.get('diffusion_dominated_fraction', 0))
    
    if labels:
        x = np.arange(len(labels))
        width = 0.6
        ax.bar(x, cascade_fracs, width, label='Cascade (<0.1)', color='red', alpha=0.8)
        ax.bar(x, trans_fracs, width, bottom=cascade_fracs, label='Transitional (0.1-1)', color='orange', alpha=0.8)
        ax.bar(x, diff_fracs, width, bottom=np.array(cascade_fracs)+np.array(trans_fracs), 
               label='Diffusion (>1)', color='green', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Fraction of Time')
        ax.set_title('v4: Regime Classification')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Hyperdissipation threshold
    ax = axes[1, 1]
    labels = []
    subcrit_vals = []
    supercrit_vals = []
    for label, res in all_results.items():
        thresh = res['analysis']['threshold']
        if thresh and 'subcritical_mean_peak_S_eff' in thresh:
            labels.append(label)
            subcrit_vals.append(thresh['subcritical_mean_peak_S_eff'])
            supercrit_vals.append(thresh['supercritical_mean_peak_S_eff'])
    
    if labels:
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, subcrit_vals, width, label='s < 1.25', alpha=0.8)
        ax.bar(x + width/2, supercrit_vals, width, label='s ≥ 1.25', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Peak S_eff')
        ax.set_title('Hyperdissipation Threshold (s = 5/4)')
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Growth signatures
    ax = axes[1, 2]
    for label, res in all_results.items():
        growth = res['analysis']['growth']
        if growth:
            exp_frac = growth.get('exponential_fraction', 0)
            quad_frac = growth.get('quadratic_fraction', 0)
            stable_frac = growth.get('stable_fraction', 0)
            
            ax.bar(label, exp_frac, color='red', alpha=0.7, label='Exponential' if label == list(all_results.keys())[0] else '')
            ax.bar(label, quad_frac, bottom=exp_frac, color='orange', alpha=0.7, 
                   label='Quadratic' if label == list(all_results.keys())[0] else '')
            ax.bar(label, stable_frac, bottom=exp_frac+quad_frac, color='green', alpha=0.7,
                   label='Stable' if label == list(all_results.keys())[0] else '')
    
    ax.set_ylabel('Fraction')
    ax.set_title('Growth Signature Distribution')
    ax.legend(fontsize=8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_summary_v4.png'), dpi=150)
    print(f"  Saved: {output_dir}/comparison_summary_v4.png")
    plt.close()


def main():
    """Run comprehensive test suite with v4."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = f"./comprehensive_results_v4_{timestamp}"
    os.makedirs(base_output, exist_ok=True)
    
    print("="*70)
    print("Comprehensive Heat Kernel + BKM Test Suite (v4)")
    print("Key fixes: k_eff instead of k_max, ω_percentile instead of ω_max")
    print(f"Output directory: {base_output}")
    print("="*70)
    
    all_results = {}
    
    # Test 1: Reynolds number sweep
    print("\n" + "="*70)
    print("TEST 1: Reynolds Number Sweep (Taylor-Green)")
    print("="*70)
    
    for Re in [400, 800, 1600]:
        label = f"TG_Re{Re}"
        output_dir = os.path.join(base_output, label)
        
        results, tracker, solver = run_simulation(
            grid_size=64,
            reynolds_number=Re,
            T_final=5.0,
            init_type='taylor_green',
            track_every=5,
            output_dir=output_dir,
        )
        all_results[label] = results
    
    # Test 2: Extended run
    print("\n" + "="*70)
    print("TEST 2: Extended Time Run (T=10)")
    print("="*70)
    
    label = "TG_Re400_T10"
    output_dir = os.path.join(base_output, label)
    
    results, tracker, solver = run_simulation(
        grid_size=64,
        reynolds_number=400,
        T_final=10.0,
        init_type='taylor_green',
        track_every=5,
        output_dir=output_dir,
    )
    all_results[label] = results
    
    # Test 3: Random turbulence
    print("\n" + "="*70)
    print("TEST 3: Random Turbulence IC")
    print("="*70)
    
    for Re in [400, 800]:
        label = f"Random_Re{Re}"
        output_dir = os.path.join(base_output, label)
        
        results, tracker, solver = run_simulation(
            grid_size=64,
            reynolds_number=Re,
            T_final=5.0,
            init_type='random',
            track_every=5,
            output_dir=output_dir,
        )
        all_results[label] = results
    
    # Create plots
    create_comparison_plots_v4(all_results, base_output)
    
    # Save summary
    summary_data = {}
    for label, res in all_results.items():
        cascade = res['analysis']['cascade']
        summary_data[label] = {
            'params': res['params'],
            'peak_omega': max(res['time_series']['omega_max']) if res['time_series']['omega_max'] else 0,
            'bkm_integral': res['summary'].get('bkm_integral_total', 0),
            'mean_timescale_ratio': cascade.get('mean_timescale_ratio', 0) if cascade else 0,
            'cascade_dominated_frac': cascade.get('cascade_dominated_fraction', 0) if cascade else 0,
            'transitional_frac': cascade.get('transitional_fraction', 0) if cascade else 0,
            'diffusion_dominated_frac': cascade.get('diffusion_dominated_fraction', 0) if cascade else 0,
            'threshold_effect': res['analysis']['threshold'].get('threshold_effect', 'N/A'),
        }
    
    with open(os.path.join(base_output, 'summary_v4.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE (v4)")
    print("="*70)
    print(f"{'Run':<15} {'Peak ω':>8} {'BKM':>8} {'τ_ratio':>10} {'Cascade%':>9} {'Trans%':>8} {'Diff%':>7}")
    print("-"*70)
    for label, data in summary_data.items():
        print(f"{label:<15} {data['peak_omega']:>8.2f} {data['bkm_integral']:>8.3f} "
              f"{data['mean_timescale_ratio']:>10.4f} {data['cascade_dominated_frac']*100:>8.1f}% "
              f"{data['transitional_frac']*100:>7.1f}% {data['diffusion_dominated_frac']*100:>6.1f}%")
    
    print("\n" + "="*70)
    print("All tests complete!")
    print(f"Results saved to: {base_output}")
    print("="*70)
    
    return all_results, base_output


if __name__ == "__main__":
    all_results, output_dir = main()
