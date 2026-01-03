#!/usr/bin/env python3
"""
measure_cascade_scaling.py
==========================
Empirical test of the "self-limiting cascade" conjecture.

Key measurements (from GPT's analysis):
1. Scaling exponent: k_eff ~ ω^a  → implies D ~ ω^(2+2a)
2. Cascade occupancy: time spent with τ_norm ≤ θ for various thresholds
3. Dissipation vs stretching: does D ramp faster than growth rate?

This is the "Clay-adjacent" empirical work - measuring the exponent
that determines whether dissipation can outrun stretching.

Author: Josh Curry & Claude
Date: December 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from scipy import stats
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_bkm_engine import CUDABKMSolver, backend
from heat_kernel_ns_integration_v5_1 import (
    HeatKernelConfigV51,
    HeatKernelAlignmentTrackerV51,
)

print(f"Backend: {backend}")


def compute_vorticity_and_gradient_norms(solver):
    """
    Compute vorticity norms and gradient norms for dissipation estimation.
    
    Returns:
        omega_L2: ||ω||_2
        grad_omega_L2: ||∇ω||_2 (proxy for dissipation)
        omega_max: max|ω|
        omega_percentile: 99.5th percentile of |ω|
    """
    xp = solver.xp
    N = solver.nx
    
    # Wavenumber grid
    k_1d = xp.fft.fftfreq(N, d=1.0/N) * 2 * np.pi / solver.Lx
    kx, ky, kz = xp.meshgrid(k_1d, k_1d, k_1d, indexing='ij')
    k_mag = xp.sqrt(kx**2 + ky**2 + kz**2)
    k_sq = kx**2 + ky**2 + kz**2
    
    # Velocity FFT
    u_hat = xp.fft.fftn(solver.u)
    v_hat = xp.fft.fftn(solver.v)
    w_hat = xp.fft.fftn(solver.w)
    
    # Vorticity in Fourier space
    omega_x_hat = 1j * (ky * w_hat - kz * v_hat)
    omega_y_hat = 1j * (kz * u_hat - kx * w_hat)
    omega_z_hat = 1j * (kx * v_hat - ky * u_hat)
    
    # ||ω||_2^2 = ∫|ω|² dx = ∑|ω̂|² (Parseval)
    enstrophy = float(xp.sum(xp.abs(omega_x_hat)**2 + 
                              xp.abs(omega_y_hat)**2 + 
                              xp.abs(omega_z_hat)**2))
    omega_L2 = np.sqrt(enstrophy)
    
    # ||∇ω||_2^2 = ∑ k² |ω̂|² (dissipation of enstrophy scales with this)
    grad_enstrophy = float(xp.sum(k_sq * (xp.abs(omega_x_hat)**2 + 
                                           xp.abs(omega_y_hat)**2 + 
                                           xp.abs(omega_z_hat)**2)))
    grad_omega_L2 = np.sqrt(grad_enstrophy)
    
    # k_eff from enstrophy weighting
    if enstrophy > 1e-30:
        k_eff = float(xp.sum(k_mag * (xp.abs(omega_x_hat)**2 + 
                                       xp.abs(omega_y_hat)**2 + 
                                       xp.abs(omega_z_hat)**2)) / enstrophy)
    else:
        k_eff = 1.0
    
    # Physical space vorticity magnitude for percentiles
    omega_x = xp.fft.ifftn(omega_x_hat).real
    omega_y = xp.fft.ifftn(omega_y_hat).real
    omega_z = xp.fft.ifftn(omega_z_hat).real
    omega_mag = xp.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    
    if hasattr(omega_mag, 'get'):
        omega_np = omega_mag.get()
    else:
        omega_np = np.array(omega_mag)
    
    omega_max = float(np.max(omega_np))
    omega_percentile = float(np.percentile(omega_np, 99.5))
    
    return {
        'omega_L2': omega_L2,
        'grad_omega_L2': grad_omega_L2,
        'enstrophy': enstrophy,
        'k_eff': k_eff,
        'omega_max': omega_max,
        'omega_percentile': omega_percentile,
    }


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


def run_scaling_measurement(grid_size: int, reynolds_number: float, T_max: float = 10.0):
    """
    Run simulation and collect data for scaling analysis.
    """
    dx = 2 * np.pi / grid_size
    dt_safe = 0.2 * dx / 1.0
    nu = 1.0 / reynolds_number
    
    print(f"\n{'='*70}")
    print(f"SCALING MEASUREMENT: N={grid_size}, Re={reynolds_number}, T={T_max}")
    print(f"{'='*70}")
    
    solver = CUDABKMSolver(
        grid_size=grid_size, reynolds_number=reynolds_number, dt=dt_safe,
        CFL_target=0.2, adapt_dt=False, track_alignment=False,
        rho_soft=0.99, rho_hard=0.999, startup_steps=0,
    )
    
    initialize_taylor_green(solver, amplitude=1.0)
    
    step = 0
    track_every = 10
    
    # Data collection
    data = {
        'time': [],
        'omega_percentile': [],
        'omega_L2': [],
        'k_eff': [],
        'grad_omega_L2': [],
        'dissipation_proxy': [],  # D = ν ||∇ω||²
        'tau_norm': [],
        'growth_rate': [],  # d/dt log(ω)
    }
    
    prev_omega = None
    prev_time = None
    
    while solver.current_time < T_max and not solver.should_stop:
        result = solver.evolve_one_timestep()
        
        if step % track_every == 0:
            metrics = compute_vorticity_and_gradient_norms(solver)
            
            omega_pct = metrics['omega_percentile']
            k_eff = metrics['k_eff']
            grad_omega = metrics['grad_omega_L2']
            
            # Dissipation proxy: D = ν ||∇ω||²
            D = nu * grad_omega**2
            
            # τ_norm (grid-invariant version)
            tau_norm = np.sqrt(k_eff) / (omega_pct + 1e-10)
            
            # Growth rate: d/dt log(ω)
            if prev_omega is not None and prev_time is not None:
                dt = solver.current_time - prev_time
                if dt > 1e-10 and prev_omega > 1e-10:
                    growth_rate = (np.log(omega_pct) - np.log(prev_omega)) / dt
                else:
                    growth_rate = 0.0
            else:
                growth_rate = 0.0
            
            data['time'].append(solver.current_time)
            data['omega_percentile'].append(omega_pct)
            data['omega_L2'].append(metrics['omega_L2'])
            data['k_eff'].append(k_eff)
            data['grad_omega_L2'].append(grad_omega)
            data['dissipation_proxy'].append(D)
            data['tau_norm'].append(tau_norm)
            data['growth_rate'].append(growth_rate)
            
            prev_omega = omega_pct
            prev_time = solver.current_time
        
        step += 1
        if step > 2000000:
            break
    
    # Convert to arrays
    for key in data:
        data[key] = np.array(data[key])
    
    data['reynolds_number'] = reynolds_number
    data['grid_size'] = grid_size
    data['nu'] = nu
    
    return data


def compute_scaling_exponent(data, omega_min=5.0, omega_max=None):
    """
    Compute the scaling exponent a in k_eff ~ ω^a.
    
    Uses data points where omega is in the "high-ω" window.
    """
    omega = data['omega_percentile']
    k_eff = data['k_eff']
    
    if omega_max is None:
        omega_max = np.max(omega) * 0.95
    
    # Select high-ω window
    mask = (omega >= omega_min) & (omega <= omega_max)
    
    if np.sum(mask) < 5:
        return None, None, None
    
    log_omega = np.log(omega[mask])
    log_k = np.log(k_eff[mask])
    
    # Linear regression: log(k_eff) = a * log(ω) + b
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_omega, log_k)
    
    return slope, r_value**2, std_err


def compute_cascade_occupancy(data, thresholds=[0.2, 0.15, 0.12, 0.10]):
    """
    Compute time spent in cascade-dominant regime for various thresholds.
    
    Returns:
        occupancy: dict of {threshold: fraction of time with τ_norm ≤ threshold}
        min_tau: minimum τ_norm achieved
        dip_episodes: list of (start, end, duration) for each dip below lowest threshold
    """
    tau = data['tau_norm']
    time = data['time']
    
    T_total = time[-1] - time[0]
    dt = np.diff(time)
    
    occupancy = {}
    for theta in thresholds:
        mask = tau[:-1] <= theta
        time_in = np.sum(dt[mask])
        occupancy[theta] = time_in / T_total
    
    min_tau = np.min(tau)
    min_tau_time = time[np.argmin(tau)]
    
    # Find contiguous episodes below lowest threshold
    theta_low = min(thresholds)
    below = tau <= theta_low
    
    episodes = []
    in_episode = False
    start_idx = 0
    
    for i, b in enumerate(below):
        if b and not in_episode:
            in_episode = True
            start_idx = i
        elif not b and in_episode:
            in_episode = False
            episodes.append({
                'start_time': time[start_idx],
                'end_time': time[i-1],
                'duration': time[i-1] - time[start_idx],
                'min_tau': np.min(tau[start_idx:i]),
            })
    
    return {
        'occupancy': occupancy,
        'min_tau': min_tau,
        'min_tau_time': min_tau_time,
        'episodes': episodes,
        'n_episodes': len(episodes),
    }


def compute_dissipation_vs_stretching(data):
    """
    Compare dissipation scaling with stretching (growth rate).
    
    Key question: Does D ramp faster than G = d/dt log(ω)?
    """
    omega = data['omega_percentile']
    D = data['dissipation_proxy']
    G = data['growth_rate']
    
    # Smooth growth rate (noisy from finite differences)
    G_smooth = gaussian_filter1d(G, sigma=3)
    
    # Fit D ~ ω^β in high-ω region
    mask = omega > 5.0
    if np.sum(mask) < 5:
        return None
    
    log_omega = np.log(omega[mask])
    log_D = np.log(D[mask] + 1e-30)
    
    slope_D, intercept_D, r_D, _, _ = stats.linregress(log_omega, log_D)
    
    # For comparison: if G ~ ω^γ (stretching rate)
    # Only use positive growth regions
    pos_growth = (G_smooth > 0.01) & mask
    if np.sum(pos_growth) > 5:
        log_G = np.log(G_smooth[pos_growth])
        log_omega_g = np.log(omega[pos_growth])
        slope_G, _, r_G, _, _ = stats.linregress(log_omega_g, log_G)
    else:
        slope_G = None
        r_G = None
    
    return {
        'D_exponent': slope_D,  # β in D ~ ω^β
        'D_r_squared': r_D**2,
        'G_exponent': slope_G,  # γ in G ~ ω^γ (if measurable)
        'G_r_squared': r_G**2 if r_G else None,
    }


def create_scaling_plots(all_data: dict, output_dir: str):
    """Create comprehensive scaling analysis plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Self-Limiting Cascade Analysis: Scaling Exponents & Occupancy', fontsize=14)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_data)))
    
    # 1. k_eff vs ω (log-log) with scaling fits
    ax = axes[0, 0]
    for (label, data), color in zip(all_data.items(), colors):
        omega = data['omega_percentile']
        k_eff = data['k_eff']
        ax.loglog(omega, k_eff, 'o-', color=color, alpha=0.7, markersize=3, label=label)
        
        # Add fit line
        a, r2, _ = compute_scaling_exponent(data)
        if a is not None:
            omega_fit = np.linspace(5, np.max(omega)*0.95, 50)
            k_fit = np.exp(np.log(omega_fit) * a + np.log(k_eff[omega > 5][0]) - np.log(omega[omega > 5][0]) * a)
            ax.loglog(omega_fit, k_fit, '--', color=color, alpha=0.5)
    
    ax.set_xlabel('ω (99.5%)')
    ax.set_ylabel('k_eff')
    ax.set_title('k_eff ~ ω^a Scaling')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Scaling exponent vs Re
    ax = axes[0, 1]
    Re_list = []
    a_list = []
    a_err_list = []
    
    for label, data in all_data.items():
        a, r2, err = compute_scaling_exponent(data)
        if a is not None:
            Re_list.append(data['reynolds_number'])
            a_list.append(a)
            a_err_list.append(err if err else 0)
    
    ax.errorbar(Re_list, a_list, yerr=a_err_list, fmt='o-', capsize=5, markersize=10)
    ax.axhline(y=1.0, color='red', linestyle='--', label='a=1 (D~ω⁴)')
    ax.axhline(y=0.5, color='orange', linestyle='--', label='a=0.5 (D~ω³)')
    ax.set_xlabel('Reynolds Number')
    ax.set_ylabel('Scaling exponent a')
    ax.set_title('k_eff ~ ω^a: Exponent vs Re')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text showing implied dissipation exponent
    for Re, a in zip(Re_list, a_list):
        ax.annotate(f'D~ω^{2+2*a:.1f}', (Re, a), textcoords="offset points", 
                   xytext=(5, 5), fontsize=8)
    
    # 3. Dissipation vs ω
    ax = axes[0, 2]
    for (label, data), color in zip(all_data.items(), colors):
        omega = data['omega_percentile']
        D = data['dissipation_proxy']
        ax.loglog(omega, D, 'o-', color=color, alpha=0.7, markersize=3, label=label)
    
    ax.set_xlabel('ω (99.5%)')
    ax.set_ylabel('D = ν||∇ω||²')
    ax.set_title('Dissipation vs Vorticity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. τ_norm evolution with cascade threshold
    ax = axes[1, 0]
    for (label, data), color in zip(all_data.items(), colors):
        ax.plot(data['time'], data['tau_norm'], '-', color=color, alpha=0.7, label=label)
    
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='Cascade threshold')
    ax.set_xlabel('Time')
    ax.set_ylabel('τ_norm')
    ax.set_title('τ_norm Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # 5. Cascade occupancy vs Re
    ax = axes[1, 1]
    thresholds = [0.2, 0.15, 0.12, 0.10]
    
    for theta in thresholds:
        Re_occ = []
        occ_frac = []
        for label, data in all_data.items():
            occ = compute_cascade_occupancy(data, thresholds)
            Re_occ.append(data['reynolds_number'])
            occ_frac.append(occ['occupancy'].get(theta, 0) * 100)
        
        ax.plot(Re_occ, occ_frac, 'o-', label=f'τ ≤ {theta}')
    
    ax.set_xlabel('Reynolds Number')
    ax.set_ylabel('Time in regime (%)')
    ax.set_title('Cascade Occupancy vs Re')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    table_data = [['Re', 'a', 'D~ω^β', 'min τ', 'T(τ<0.2)']]
    
    for label, data in all_data.items():
        Re = data['reynolds_number']
        a, r2, _ = compute_scaling_exponent(data)
        occ = compute_cascade_occupancy(data)
        diss = compute_dissipation_vs_stretching(data)
        
        beta = 2 + 2*a if a else 'N/A'
        beta_str = f'{beta:.2f}' if isinstance(beta, float) else beta
        
        table_data.append([
            f'{int(Re)}',
            f'{a:.3f}' if a else 'N/A',
            beta_str,
            f'{occ["min_tau"]:.3f}',
            f'{occ["occupancy"].get(0.2, 0)*100:.1f}%'
        ])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax.set_title('Scaling Summary')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cascade_scaling_analysis.png'), dpi=150)
    plt.close()
    print(f"Saved: cascade_scaling_analysis.png")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./cascade_scaling_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("SELF-LIMITING CASCADE: SCALING EXPONENT MEASUREMENT")
    print("="*70)
    print("\nKey question: What is 'a' in k_eff ~ ω^a?")
    print("This determines dissipation scaling: D ~ ω^(2+2a)")
    print("  a > 1  → D ~ ω^4+ (super-quartic, strong self-limiting)")
    print("  a ≈ 0.5 → D ~ ω³ (cubic)")
    print("  a ≈ 0  → D ~ ω² (quadratic, weak self-limiting)")
    
    # Run at multiple Reynolds numbers
    reynolds_numbers = [800, 1600, 3200]
    grid_size = 128
    T_max = 10.0
    
    all_data = {}
    
    for Re in reynolds_numbers:
        data = run_scaling_measurement(grid_size, Re, T_max)
        label = f"Re={int(Re)}"
        all_data[label] = data
        
        # Compute and report scaling
        a, r2, err = compute_scaling_exponent(data)
        occ = compute_cascade_occupancy(data)
        
        print(f"\n{label}:")
        print(f"  Scaling: k_eff ~ ω^{a:.3f} (R²={r2:.3f})")
        print(f"  Implied: D ~ ω^{2+2*a:.2f}")
        print(f"  min τ_norm: {occ['min_tau']:.4f}")
        print(f"  Time with τ<0.2: {occ['occupancy'].get(0.2, 0)*100:.1f}%")
        print(f"  Time with τ<0.1: {occ['occupancy'].get(0.1, 0)*100:.1f}%")
    
    # Create plots
    create_scaling_plots(all_data, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("SELF-LIMITING CASCADE: VERDICT")
    print("="*70)
    
    a_values = []
    for label, data in all_data.items():
        a, _, _ = compute_scaling_exponent(data)
        if a is not None:
            a_values.append(a)
    
    if a_values:
        mean_a = np.mean(a_values)
        implied_beta = 2 + 2*mean_a
        
        print(f"\nMean scaling exponent: a = {mean_a:.3f}")
        print(f"Implied dissipation exponent: β = {implied_beta:.2f}")
        print(f"  (D ~ ω^{implied_beta:.1f})")
        
        if implied_beta > 3:
            print("\n✓ STRONG self-limiting: Dissipation grows faster than cubic")
            print("  This supports the 'cascade can't sustain blow-up' narrative")
        elif implied_beta > 2:
            print("\n~ MODERATE self-limiting: Dissipation grows super-quadratically")
        else:
            print("\n⚠ WEAK self-limiting: Dissipation only grows quadratically")
    
    # Save results
    results = {
        'timestamp': timestamp,
        'grid_size': grid_size,
        'T_max': T_max,
        'scaling_exponents': {},
    }
    
    for label, data in all_data.items():
        a, r2, err = compute_scaling_exponent(data)
        occ = compute_cascade_occupancy(data)
        diss = compute_dissipation_vs_stretching(data)
        
        results['scaling_exponents'][label] = {
            'reynolds_number': data['reynolds_number'],
            'a': float(a) if a else None,
            'r_squared': float(r2) if r2 else None,
            'implied_beta': float(2 + 2*a) if a else None,
            'min_tau_norm': float(occ['min_tau']),
            'cascade_occupancy': {str(k): float(v) for k, v in occ['occupancy'].items()},
            'D_exponent': float(diss['D_exponent']) if diss and diss['D_exponent'] else None,
        }
    
    with open(os.path.join(output_dir, 'scaling_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    return all_data, results


if __name__ == "__main__":
    all_data, results = main()
