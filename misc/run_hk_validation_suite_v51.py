#!/usr/bin/env python3
"""
run_hk_validation_suite_v51.py
==============================
Heat Kernel Diagnostic Validation Suite (v5.1)

New features:
- Regime crossing detection and analysis
- Early Warning Score tracking
- Lead/lag analysis between diagnostics
- Prediction validation

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
    from unified_bkm_engine import CUDABKMSolver, backend
    ENGINE_NAME = "unified_bkm_engine"
except ImportError:
    print("ERROR: Could not import unified_bkm_engine.py")
    sys.exit(1)

# Import v5.1 tracker
from heat_kernel_ns_integration_v5_1 import (
    HeatKernelConfigV51,
    HeatKernelAlignmentTrackerV51,
)

print(f"Using {ENGINE_NAME} (backend: {backend})")
print("Using v5.1 Heat Kernel (regime crossings + early warning)")


# ============================================================================
# Initial Condition Generators
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


def initialize_kelvin_helmholtz(solver, thickness=0.03, amplitude=0.05, U0=3.0):
    xp = solver.xp
    x = xp.linspace(0, solver.Lx, solver.nx, endpoint=False)
    y = xp.linspace(0, solver.Ly, solver.ny, endpoint=False)
    X, Y = xp.meshgrid(x, y, indexing='ij')
    
    delta = thickness * solver.Ly
    y0 = solver.Ly / 2.0
    U_base = U0 * xp.tanh((Y - y0) / delta)
    envelope = xp.exp(-((Y - y0)**2) / (2.0 * delta**2))
    
    kx = 2.0 * xp.pi * 2 / solver.Lx
    psi = amplitude * U0 * envelope * xp.sin(kx * X)
    
    u_p = xp.gradient(psi, solver.Ly/solver.ny, axis=1)
    v_p = -xp.gradient(psi, solver.Lx/solver.nx, axis=0)
    
    solver.u = xp.tile((U_base + u_p)[:, :, xp.newaxis], (1, 1, solver.nz)).astype(xp.float64)
    solver.v = xp.tile(v_p[:, :, xp.newaxis], (1, 1, solver.nz)).astype(xp.float64)
    solver.w = xp.zeros_like(solver.u)
    
    z = xp.linspace(0, solver.Lz, solver.nz, endpoint=False)
    solver.w += 0.1 * amplitude * U0 * envelope[:, :, xp.newaxis] * xp.sin(2*xp.pi*z[xp.newaxis, xp.newaxis, :]/solver.Lz)
    
    solver.u, solver.v, solver.w = solver.project_div_free(solver.u, solver.v, solver.w)
    solver.initial_energy = solver.compute_energy()
    return {'type': 'kelvin_helmholtz', 'E0': solver.initial_energy, 'W0': solver.compute_max_vorticity()}


def initialize_random_isotropic(solver, energy_level=0.1, k_peak=4):
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
    
    u_hat *= spectrum; v_hat *= spectrum; w_hat *= spectrum
    
    K2 = K**2; K2[0, 0, 0] = 1.0
    kdotu = KX * u_hat + KY * v_hat + KZ * w_hat
    u_hat -= KX * kdotu / K2
    v_hat -= KY * kdotu / K2
    w_hat -= KZ * kdotu / K2
    
    u = np.fft.ifftn(u_hat).real
    v = np.fft.ifftn(v_hat).real
    w = np.fft.ifftn(w_hat).real
    
    scale = np.sqrt(energy_level / (0.5 * np.mean(u**2 + v**2 + w**2) + 1e-10))
    u *= scale; v *= scale; w *= scale
    
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
    return {'type': 'random_isotropic', 'E0': solver.initial_energy, 'W0': solver.compute_max_vorticity()}


# ============================================================================
# Simulation Runner
# ============================================================================

def run_hk_simulation_v51(ic_type, grid_size=64, reynolds_number=400, T_final=5.0,
                          track_every=5, output_dir="./results", **ic_kwargs):
    """Run simulation with v5.1 heat kernel tracking."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    dx = 2 * np.pi / grid_size
    dt_safe = 0.2 * dx / 1.0
    if ic_type == 'kelvin_helmholtz':
        dt_safe = 0.15 * dx / ic_kwargs.get('U0', 3.0)
    
    print(f"\n{'='*60}")
    print(f"Running: {ic_type}, Re={reynolds_number}, T={T_final}")
    print(f"{'='*60}")
    
    solver = CUDABKMSolver(
        grid_size=grid_size, reynolds_number=reynolds_number, dt=dt_safe,
        CFL_target=0.2, adapt_dt=False, track_alignment=False,
        rho_soft=0.99, rho_hard=0.999, startup_steps=0,
    )
    
    # Initialize
    init_funcs = {
        'taylor_green': initialize_taylor_green,
        'kida_pelz': initialize_kida_pelz,
        'kelvin_helmholtz': initialize_kelvin_helmholtz,
        'random_isotropic': initialize_random_isotropic,
    }
    ic_info = init_funcs[ic_type](solver, **ic_kwargs)
    print(f"  IC: {ic_type}, E0={ic_info['E0']:.4f}, W0={ic_info['W0']:.2f}")
    
    # v5.1 config
    config = HeatKernelConfigV51(
        s_values=[0.7, 1.0, 1.2, 1.25, 1.3, 1.5],
        t_viscosity_mult=10.0,
        track_every=track_every,
        omega_percentile=99.5,
        tau_norm_cascade_threshold=0.1,
        tau_norm_diffusion_threshold=0.5,
        trend_window=10,
        prediction_horizon=20,
    )
    
    tracker = HeatKernelAlignmentTrackerV51(solver, config)
    tracker.start_logging(output_dir, filename=f"hk_v51_{ic_type}.csv")
    
    # Run
    step = 0
    last_print = 0.0
    print_interval = max(0.2, T_final / 20)
    
    time_series = {
        'time': [], 'omega_max': [], 'tau_norm': [], 'ew_score': [],
        'k_eff': [], 'regime': [], 'gamma_exp': [], 'regime_crossing': []
    }
    
    while solver.current_time < T_final and not solver.should_stop:
        result = solver.evolve_one_timestep()
        
        if step % track_every == 0:
            metrics = tracker.record(step)
            
            time_series['time'].append(solver.current_time)
            time_series['omega_max'].append(result['vorticity'])
            time_series['tau_norm'].append(metrics.get('tau_norm', 1.0))
            time_series['ew_score'].append(metrics.get('ew_score', 0.0))
            time_series['k_eff'].append(metrics.get('k_eff', 1.0))
            time_series['regime'].append(metrics.get('regime', 'unknown'))
            time_series['gamma_exp'].append(metrics.get('gamma_exp', 0))
            time_series['regime_crossing'].append(metrics.get('regime_crossing', False))
            
            if solver.current_time - last_print >= print_interval:
                tau_norm = metrics.get('tau_norm', 0)
                ew = metrics.get('ew_score', 0)
                regime = metrics.get('regime', 'unknown')[:5]
                crossing = "CROSS!" if metrics.get('regime_crossing') else ""
                
                print(f"  t={solver.current_time:.2f} | ω={result['vorticity']:.2f} | "
                      f"τ_norm={tau_norm:.3f} | EW={ew:.2f} | {regime} {crossing}")
                last_print = solver.current_time
        
        step += 1
        if step > 500000:
            break
    
    tracker.close()
    summary = tracker.get_summary()
    
    results = {
        'ic_type': ic_type,
        'ic_info': ic_info,
        'params': {'grid_size': grid_size, 'reynolds_number': reynolds_number, 'T_final': T_final},
        'summary': summary,
        'time_series': time_series,
    }
    
    # Print v5.1 specific results
    print(f"\n  Complete: {step} steps")
    print(f"  Peak ω: {max(time_series['omega_max']):.2f}")
    
    if 'regime_crossings' in summary:
        rc = summary['regime_crossings']
        print(f"  Regime crossings: {rc.get('total_crossings', 0)}")
        if rc.get('first_cascade_entry'):
            print(f"    First cascade entry: t={rc['first_cascade_entry']:.2f}")
        print(f"    Min τ_norm: {rc.get('min_tau_norm', 0):.3f} at t={rc.get('min_tau_norm_time', 0):.2f}")
        print(f"    Dwell: cascade={rc.get('dwell_frac_cascade', 0)*100:.0f}%, "
              f"trans={rc.get('dwell_frac_transitional', 0)*100:.0f}%, "
              f"diff={rc.get('dwell_frac_diffusion', 0)*100:.0f}%")
    
    if 'ew_prediction_stats' in summary:
        ew_stats = summary['ew_prediction_stats']
        print(f"  EW predictions: {ew_stats.get('total_predictions', 0)} total, "
              f"accuracy={ew_stats.get('accuracy', 'N/A')}")
    
    if 'lead_lag_analysis' in summary:
        ll = summary['lead_lag_analysis']
        if 'tau_norm_vs_omega_percentile' in ll:
            lag_info = ll['tau_norm_vs_omega_percentile']
            print(f"  Lead/lag (τ_norm vs ω): lag={lag_info.get('peak_lag', 0)}, "
                  f"corr={lag_info.get('peak_correlation', 0):.2f}")
    
    return results


# ============================================================================
# Visualization
# ============================================================================

def create_validation_plots_v51(all_results, output_dir):
    """Create v5.1 validation plots."""
    
    print("\nCreating v5.1 validation plots...")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    colors = {
        'taylor_green': 'blue', 'kida_pelz': 'red',
        'kelvin_helmholtz': 'green', 'random_isotropic': 'purple',
    }
    
    # 1. Vorticity with regime crossing markers
    ax = axes[0, 0]
    for label, res in all_results.items():
        t = res['time_series']['time']
        omega = res['time_series']['omega_max']
        crossings = res['time_series']['regime_crossing']
        ic = res['ic_type']
        
        if len(t) > 0:
            ax.plot(t, omega, label=label, color=colors.get(ic, 'gray'), linewidth=2)
            
            # Mark crossings
            cross_times = [t[i] for i, c in enumerate(crossings) if c]
            cross_omegas = [omega[i] for i, c in enumerate(crossings) if c]
            if cross_times:
                ax.scatter(cross_times, cross_omegas, color=colors.get(ic, 'gray'),
                          marker='X', s=100, zorder=5, edgecolors='black')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Max Vorticity')
    ax.set_title('Vorticity Evolution (X = regime crossing)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. τ_norm with regime bands
    ax = axes[0, 1]
    for label, res in all_results.items():
        t = res['time_series']['time']
        tau_norm = res['time_series']['tau_norm']
        ic = res['ic_type']
        if len(t) > 0:
            ax.plot(t, tau_norm, label=label, color=colors.get(ic, 'gray'), linewidth=2)
    
    ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2)
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=2)
    ax.fill_between([0, 10], [0, 0], [0.1, 0.1], alpha=0.2, color='red')
    ax.fill_between([0, 10], [0.1, 0.1], [0.5, 0.5], alpha=0.2, color='orange')
    ax.fill_between([0, 10], [0.5, 0.5], [1.0, 1.0], alpha=0.2, color='green')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('τ_norm')
    ax.set_title('Re-Normalized Timescale (v5)')
    ax.set_ylim(0, 0.7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Early Warning Score
    ax = axes[1, 0]
    for label, res in all_results.items():
        t = res['time_series']['time']
        ew = res['time_series']['ew_score']
        ic = res['ic_type']
        if len(t) > 0:
            ax.plot(t, ew, label=label, color=colors.get(ic, 'gray'), linewidth=2)
    
    ax.axhline(y=0.5, color='red', linestyle='--', label='EW threshold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Early Warning Score')
    ax.set_title('v5.1: Early Warning Score')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Dwell time breakdown
    ax = axes[1, 1]
    labels = []
    cascade_dwell = []
    trans_dwell = []
    diff_dwell = []
    
    for label, res in all_results.items():
        if 'regime_crossings' in res['summary']:
            rc = res['summary']['regime_crossings']
            labels.append(label)
            cascade_dwell.append(rc.get('dwell_frac_cascade', 0))
            trans_dwell.append(rc.get('dwell_frac_transitional', 0))
            diff_dwell.append(rc.get('dwell_frac_diffusion', 0))
    
    if labels:
        x = np.arange(len(labels))
        ax.bar(x, cascade_dwell, 0.6, label='Cascade', color='red', alpha=0.8)
        ax.bar(x, trans_dwell, 0.6, bottom=cascade_dwell, label='Transitional', color='orange', alpha=0.8)
        ax.bar(x, diff_dwell, 0.6, bottom=np.array(cascade_dwell)+np.array(trans_dwell),
               label='Diffusion', color='green', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Dwell Fraction')
        ax.set_title('v5.1: Regime Dwell Times')
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. Lead/lag analysis
    ax = axes[2, 0]
    pairs = ['tau_norm_vs_omega_percentile', 'ew_score_vs_omega_percentile', 'gamma_exp_vs_omega_percentile']
    pair_labels = ['τ_norm', 'EW score', 'γ_exp']
    
    for idx, label in enumerate(all_results.keys()):
        res = all_results[label]
        if 'lead_lag_analysis' in res['summary']:
            ll = res['summary']['lead_lag_analysis']
            lags = []
            corrs = []
            for pair in pairs:
                if pair in ll:
                    lags.append(ll[pair].get('peak_lag', 0))
                    corrs.append(ll[pair].get('peak_correlation', 0))
                else:
                    lags.append(0)
                    corrs.append(0)
            
            x = np.arange(len(pairs)) + idx * 0.2
            ax.bar(x, lags, 0.18, label=label, alpha=0.8)
    
    ax.set_xticks(np.arange(len(pairs)) + 0.3)
    ax.set_xticklabels(pair_labels)
    ax.set_ylabel('Peak Lag (steps)')
    ax.set_title('v5.1: Lead/Lag vs ω_percentile\n(positive = leads)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # 6. Crossing events timeline
    ax = axes[2, 1]
    for idx, (label, res) in enumerate(all_results.items()):
        if 'regime_crossings' in res['summary']:
            rc = res['summary']['regime_crossings']
            crossings = rc.get('crossings', [])
            
            for t, from_r, to_r in crossings:
                marker = 'v' if 'cascade' in to_r else '^'
                color = 'red' if 'cascade' in to_r else 'green'
                ax.scatter([t], [idx], marker=marker, s=150, c=color, edgecolors='black', zorder=5)
            
            # Mark min τ_norm time
            min_t = rc.get('min_tau_norm_time', 0)
            ax.scatter([min_t], [idx], marker='*', s=200, c='gold', edgecolors='black', zorder=6)
    
    ax.set_yticks(range(len(all_results)))
    ax.set_yticklabels(list(all_results.keys()))
    ax.set_xlabel('Time')
    ax.set_title('v5.1: Regime Crossings Timeline\n(▼=enter cascade, ▲=exit cascade, ★=min τ_norm)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max([res['time_series']['time'][-1] for res in all_results.values() if res['time_series']['time']]))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hk_validation_v51.png'), dpi=150)
    print(f"  Saved: {output_dir}/hk_validation_v51.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = f"./hk_validation_v51_{timestamp}"
    os.makedirs(base_output, exist_ok=True)
    
    print("="*70)
    print("Heat Kernel Validation Suite v5.1")
    print("New: Regime crossings, Early Warning Score, Lead/Lag analysis")
    print("="*70)
    
    all_results = {}
    
    grid_size = 64
    Re = 400
    T_final = 5.0
    
    for ic_type, kwargs in [
        ('taylor_green', {'amplitude': 1.0}),
        ('kida_pelz', {'amplitude': 1.3}),
        ('kelvin_helmholtz', {'thickness': 0.03, 'amplitude': 0.05, 'U0': 3.0}),
        ('random_isotropic', {'energy_level': 0.1, 'k_peak': 4}),
    ]:
        label = f"{ic_type[:2].upper()}_Re{Re}"
        results = run_hk_simulation_v51(
            ic_type, grid_size=grid_size, reynolds_number=Re, T_final=T_final,
            output_dir=os.path.join(base_output, ic_type), **kwargs
        )
        all_results[label] = results
    
    create_validation_plots_v51(all_results, base_output)
    
    # Save summary
    summary_data = {}
    for label, res in all_results.items():
        rc = res['summary'].get('regime_crossings', {})
        ll = res['summary'].get('lead_lag_analysis', {})
        
        summary_data[label] = {
            'ic_type': res['ic_type'],
            'peak_omega': max(res['time_series']['omega_max']),
            'mean_tau_norm': res['summary'].get('mean_tau_norm', 0),
            'min_tau_norm': rc.get('min_tau_norm', 0),
            'min_tau_norm_time': rc.get('min_tau_norm_time', 0),
            'total_crossings': rc.get('total_crossings', 0),
            'first_cascade_entry': rc.get('first_cascade_entry'),
            'dwell_cascade': rc.get('dwell_frac_cascade', 0),
            'dwell_transitional': rc.get('dwell_frac_transitional', 0),
            'dwell_diffusion': rc.get('dwell_frac_diffusion', 0),
            'max_ew_score': res['summary'].get('max_ew_score', 0),
            'tau_norm_leads_omega': ll.get('tau_norm_vs_omega_percentile', {}).get('peak_lag', 0),
        }
    
    with open(os.path.join(base_output, 'validation_summary_v51.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("v5.1 VALIDATION SUMMARY")
    print("="*70)
    print(f"{'IC Type':<18} {'Peak ω':>7} {'min τ':>7} {'Cross':>6} {'Casc%':>6} {'MaxEW':>6} {'Lag':>5}")
    print("-"*70)
    for label, data in summary_data.items():
        print(f"{data['ic_type']:<18} {data['peak_omega']:>7.2f} {data['min_tau_norm']:>7.3f} "
              f"{data['total_crossings']:>6} {data['dwell_cascade']*100:>5.0f}% "
              f"{data['max_ew_score']:>6.2f} {data['tau_norm_leads_omega']:>5}")
    
    print(f"\nResults saved to: {base_output}")
    
    return all_results


if __name__ == "__main__":
    all_results = main()
