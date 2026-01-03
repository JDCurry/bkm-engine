#!/usr/bin/env python3
"""
heat_kernel_ns_integration_v5_1.py
==================================
Heat Kernel Integration for Navier-Stokes / BKM Monitoring

v5.1 Updates (Dec 2025) - Regime crossing events & early warning:
  1. Track regime boundary crossings (entering/exiting cascade)
  2. Early Warning Score combining multiple danger signals
  3. Lead/lag analysis between τ_norm and ω_percentile
  4. Dwell time tracking in each regime
  5. Predictive validation: does EW predict ω surges?

Building on v5's Re-normalized τ_norm = √k_eff / ω

Author: Josh Curry & Claude
Date: December 2025 (v5.1)
"""

import numpy as np
import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from scipy import signal as scipy_signal

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@dataclass
class HeatKernelConfigV51:
    """Configuration for v5.1 heat kernel diagnostics."""
    
    # Smoothing parameters
    s: float = 1.0
    t_diffusion: float = 0.1
    use_viscous_timescale: bool = True
    t_viscosity_mult: float = 10.0
    track_every: int = 10
    epsilon: float = 1e-6
    
    # Fractional orders
    s_values: List[float] = field(default_factory=lambda: [0.7, 1.0, 1.2, 1.25, 1.3, 1.5])
    s_reference: float = 1.0
    
    # Growth signature detection
    growth_window: int = 20
    
    # Shell-based energy tracking
    n_shells: int = 16
    
    # v5: Percentile for vorticity
    omega_percentile: float = 99.5
    
    # v5: Re-normalized thresholds
    tau_norm_cascade_threshold: float = 0.1
    tau_norm_diffusion_threshold: float = 0.5
    
    # v5.1: Early warning parameters
    trend_window: int = 10
    ew_tau_trend_weight: float = 0.3      # Weight for τ_norm trend component
    ew_gamma_weight: float = 0.3          # Weight for exponential growth component
    ew_cascade_rate_weight: float = 0.2   # Weight for cascade rate component
    ew_high_k_weight: float = 0.2         # Weight for high-k fraction component
    
    # v5.1: Thresholds for EW components
    gamma_exp_threshold: float = 0.1      # γ_exp above this contributes to EW
    cascade_rate_threshold: float = 0.5   # Cascade rate above this contributes
    high_k_increase_threshold: float = 0.01  # High-k fraction increase rate threshold
    
    # v5.1: Prediction horizon for validation
    prediction_horizon: int = 20          # Steps to look ahead for ω surge


class RegimeCrossingTracker:
    """
    v5.1: Track regime boundary crossings and dwell times.
    
    Events tracked:
    - First entry into cascade regime
    - All boundary crossings (with direction)
    - Dwell time in each regime
    - Minimum τ_norm and its timing
    """
    
    def __init__(self, config: HeatKernelConfigV51):
        self.config = config
        
        # Crossing events: list of (time, from_regime, to_regime)
        self.crossings: List[Tuple[float, str, str]] = []
        
        # First entry times
        self.first_cascade_entry: Optional[float] = None
        self.first_transitional_entry: Optional[float] = None
        self.first_diffusion_entry: Optional[float] = None
        
        # Dwell times (accumulated)
        self.dwell_times = {
            'cascade_dominated': 0.0,
            'transitional': 0.0,
            'diffusion_dominated': 0.0,
        }
        
        # Minimum τ_norm tracking
        self.min_tau_norm = float('inf')
        self.min_tau_norm_time = 0.0
        
        # State
        self.current_regime: Optional[str] = None
        self.last_time = 0.0
        self.last_regime_entry_time = 0.0
    
    def classify_regime(self, tau_norm: float) -> str:
        """Classify regime based on τ_norm thresholds."""
        if tau_norm < self.config.tau_norm_cascade_threshold:
            return 'cascade_dominated'
        elif tau_norm < self.config.tau_norm_diffusion_threshold:
            return 'transitional'
        else:
            return 'diffusion_dominated'
    
    def update(self, time: float, tau_norm: float) -> Optional[Dict]:
        """
        Update tracker with new τ_norm value.
        
        Returns crossing event dict if a boundary was crossed, else None.
        """
        new_regime = self.classify_regime(tau_norm)
        dt = time - self.last_time
        
        # Track minimum
        if tau_norm < self.min_tau_norm:
            self.min_tau_norm = tau_norm
            self.min_tau_norm_time = time
        
        # First observation
        if self.current_regime is None:
            self.current_regime = new_regime
            self.last_regime_entry_time = time
            self._record_first_entry(new_regime, time)
            self.last_time = time
            return None
        
        # Accumulate dwell time
        if dt > 0:
            self.dwell_times[self.current_regime] += dt
        
        # Check for crossing
        event = None
        if new_regime != self.current_regime:
            event = {
                'time': time,
                'from_regime': self.current_regime,
                'to_regime': new_regime,
                'tau_norm': tau_norm,
                'dwell_in_previous': time - self.last_regime_entry_time,
            }
            self.crossings.append((time, self.current_regime, new_regime))
            self._record_first_entry(new_regime, time)
            self.current_regime = new_regime
            self.last_regime_entry_time = time
        
        self.last_time = time
        return event
    
    def _record_first_entry(self, regime: str, time: float):
        """Record first entry into a regime."""
        if regime == 'cascade_dominated' and self.first_cascade_entry is None:
            self.first_cascade_entry = time
        elif regime == 'transitional' and self.first_transitional_entry is None:
            self.first_transitional_entry = time
        elif regime == 'diffusion_dominated' and self.first_diffusion_entry is None:
            self.first_diffusion_entry = time
    
    def get_summary(self) -> Dict:
        """Get summary of crossing events and dwell times."""
        total_time = sum(self.dwell_times.values())
        
        return {
            'total_crossings': len(self.crossings),
            'first_cascade_entry': self.first_cascade_entry,
            'first_transitional_entry': self.first_transitional_entry,
            'first_diffusion_entry': self.first_diffusion_entry,
            'min_tau_norm': self.min_tau_norm,
            'min_tau_norm_time': self.min_tau_norm_time,
            'dwell_cascade': self.dwell_times['cascade_dominated'],
            'dwell_transitional': self.dwell_times['transitional'],
            'dwell_diffusion': self.dwell_times['diffusion_dominated'],
            'dwell_frac_cascade': self.dwell_times['cascade_dominated'] / (total_time + 1e-10),
            'dwell_frac_transitional': self.dwell_times['transitional'] / (total_time + 1e-10),
            'dwell_frac_diffusion': self.dwell_times['diffusion_dominated'] / (total_time + 1e-10),
            'crossings': self.crossings,
        }


class EarlyWarningSystem:
    """
    v5.1: Early Warning Score combining multiple danger signals.
    
    EW(t) = a·I(τ_norm trending down) + b·(γ_exp>threshold) + 
            c·(cascade_rate>threshold) + d·(high_k_fraction increase)
    
    Also validates predictions against actual ω surges.
    """
    
    def __init__(self, config: HeatKernelConfigV51):
        self.config = config
        
        # History for trend detection
        self.tau_norm_history = deque(maxlen=config.trend_window)
        self.high_k_history = deque(maxlen=config.trend_window)
        self.time_history = deque(maxlen=config.trend_window)
        
        # EW score history
        self.ew_scores: List[Tuple[float, float]] = []  # (time, score)
        
        # For prediction validation
        self.omega_history: List[Tuple[float, float]] = []  # (time, omega)
        self.predictions: List[Dict] = []  # Predictions to validate
    
    def compute_ew_score(
        self,
        time: float,
        tau_norm: float,
        gamma_exp: float,
        cascade_rate: float,
        high_k_fraction: float,
        omega_percentile: float
    ) -> Dict[str, float]:
        """
        Compute Early Warning score from multiple components.
        
        Returns dict with total score and component breakdown.
        """
        # Update histories
        self.tau_norm_history.append(tau_norm)
        self.high_k_history.append(high_k_fraction)
        self.time_history.append(time)
        self.omega_history.append((time, omega_percentile))
        
        # Component 1: τ_norm trend (is it decreasing?)
        tau_trend_component = 0.0
        if len(self.tau_norm_history) >= 3:
            tau_arr = np.array(self.tau_norm_history)
            x = np.arange(len(tau_arr))
            slope = np.polyfit(x, tau_arr, 1)[0]
            # Negative slope = danger (τ_norm decreasing)
            if slope < -0.01:
                tau_trend_component = min(1.0, abs(slope) / 0.05)
        
        # Component 2: Exponential growth rate
        gamma_component = 0.0
        if gamma_exp > self.config.gamma_exp_threshold:
            gamma_component = min(1.0, gamma_exp / 0.5)
        
        # Component 3: Cascade rate
        cascade_component = 0.0
        if cascade_rate > self.config.cascade_rate_threshold:
            cascade_component = min(1.0, cascade_rate / 2.0)
        
        # Component 4: High-k fraction increase
        high_k_component = 0.0
        if len(self.high_k_history) >= 3:
            hk_arr = np.array(self.high_k_history)
            hk_change = (hk_arr[-1] - hk_arr[0]) / (len(hk_arr) - 1)
            if hk_change > self.config.high_k_increase_threshold:
                high_k_component = min(1.0, hk_change / 0.05)
        
        # Weighted sum
        ew_score = (
            self.config.ew_tau_trend_weight * tau_trend_component +
            self.config.ew_gamma_weight * gamma_component +
            self.config.ew_cascade_rate_weight * cascade_component +
            self.config.ew_high_k_weight * high_k_component
        )
        
        self.ew_scores.append((time, ew_score))
        
        # Make prediction if EW is high
        if ew_score > 0.5:
            self.predictions.append({
                'prediction_time': time,
                'ew_score': ew_score,
                'current_omega': omega_percentile,
                'horizon': self.config.prediction_horizon,
                'validated': False,
                'actual_max_omega': None,
                'surge_detected': None,
            })
        
        return {
            'ew_score': ew_score,
            'tau_trend_component': tau_trend_component,
            'gamma_component': gamma_component,
            'cascade_component': cascade_component,
            'high_k_component': high_k_component,
        }
    
    def validate_predictions(self) -> List[Dict]:
        """
        Validate past predictions against actual ω evolution.
        
        A prediction is "correct" if ω_percentile increased significantly
        within the prediction horizon.
        """
        validated = []
        
        for pred in self.predictions:
            if pred['validated']:
                continue
            
            pred_time = pred['prediction_time']
            horizon_end = pred_time + pred['horizon'] * 0.1  # Approximate dt
            
            # Find omega values in prediction window
            future_omegas = [
                omega for t, omega in self.omega_history
                if pred_time < t <= horizon_end
            ]
            
            if len(future_omegas) >= 5:
                pred['validated'] = True
                pred['actual_max_omega'] = max(future_omegas)
                
                # Surge = 20% increase from prediction time
                surge_threshold = pred['current_omega'] * 1.2
                pred['surge_detected'] = pred['actual_max_omega'] > surge_threshold
                validated.append(pred)
        
        return validated
    
    def get_prediction_accuracy(self) -> Dict:
        """Compute prediction accuracy statistics."""
        validated_preds = [p for p in self.predictions if p['validated']]
        
        if not validated_preds:
            return {
                'total_predictions': len(self.predictions),
                'validated_predictions': 0,
                'true_positives': 0,
                'false_positives': 0,
                'accuracy': None,
            }
        
        true_positives = sum(1 for p in validated_preds if p['surge_detected'])
        false_positives = sum(1 for p in validated_preds if not p['surge_detected'])
        
        return {
            'total_predictions': len(self.predictions),
            'validated_predictions': len(validated_preds),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'accuracy': true_positives / len(validated_preds) if validated_preds else None,
        }


class LeadLagAnalyzer:
    """
    v5.1: Analyze lead/lag relationships between diagnostics.
    
    Key questions:
    - Does τ_norm crossing predict ω_percentile surges?
    - Does γ_exp spike before ω_percentile peaks?
    - Does cascade_rate lead high_k_fraction?
    """
    
    def __init__(self, max_lag: int = 30):
        self.max_lag = max_lag
        self.histories = {}
    
    def add_series(self, name: str, time: float, value: float):
        """Add a data point to a named series."""
        if name not in self.histories:
            self.histories[name] = []
        self.histories[name].append((time, value))
    
    def compute_cross_correlation(
        self, 
        series_a: str, 
        series_b: str
    ) -> Dict[str, float]:
        """
        Compute cross-correlation between two series.
        
        Positive lag means series_a leads series_b.
        """
        if series_a not in self.histories or series_b not in self.histories:
            return {'peak_lag': 0, 'peak_correlation': 0, 'significant': False}
        
        a_data = np.array([v for t, v in self.histories[series_a]])
        b_data = np.array([v for t, v in self.histories[series_b]])
        
        # Ensure same length
        min_len = min(len(a_data), len(b_data))
        if min_len < 10:
            return {'peak_lag': 0, 'peak_correlation': 0, 'significant': False}
        
        a_data = a_data[:min_len]
        b_data = b_data[:min_len]
        
        # Normalize
        a_norm = (a_data - np.mean(a_data)) / (np.std(a_data) + 1e-10)
        b_norm = (b_data - np.mean(b_data)) / (np.std(b_data) + 1e-10)
        
        # Cross-correlation
        max_lag = min(self.max_lag, min_len // 3)
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
            'peak_lag': peak_lag,
            'peak_correlation': float(peak_corr),
            'significant': abs(peak_corr) > 0.5,
            'a_leads_b': peak_lag > 0 and peak_corr > 0.5,
            'b_leads_a': peak_lag < 0 and peak_corr > 0.5,
        }
    
    def analyze_all_pairs(self) -> Dict[str, Dict]:
        """Analyze lead/lag for key diagnostic pairs."""
        pairs = [
            ('tau_norm', 'omega_percentile'),
            ('gamma_exp', 'omega_percentile'),
            ('cascade_rate', 'high_k_fraction'),
            ('ew_score', 'omega_percentile'),
        ]
        
        results = {}
        for a, b in pairs:
            if a in self.histories and b in self.histories:
                results[f'{a}_vs_{b}'] = self.compute_cross_correlation(a, b)
        
        return results


# Import base classes from v5
from heat_kernel_ns_integration_v5 import (
    SpectralHeatKernelV5,
    TaoInspiredDiagnosticsV5,
)


class HeatKernelAlignmentTrackerV51:
    """v5.1: Enhanced tracker with regime crossing and early warning."""
    
    def __init__(self, solver, config: HeatKernelConfigV51 = None):
        self.solver = solver
        self.config = config or HeatKernelConfigV51()
        
        self.kernel = SpectralHeatKernelV5(
            grid_size=solver.nx,
            domain_size=solver.Lx,
            use_gpu=solver.use_gpu
        )
        
        self.tao_diag = TaoInspiredDiagnosticsV5(self.config)
        
        # v5.1: New trackers
        self.regime_tracker = RegimeCrossingTracker(self.config)
        self.ew_system = EarlyWarningSystem(self.config)
        self.lead_lag = LeadLagAnalyzer(max_lag=30)
        
        # History tracking
        self.history = {
            'time': [], 'step': [], 'dt': [],
            'omega_max': [], 'omega_percentile': [],
            'alignment_max': [], 'bkm_integral_hk': [],
            'tau_ratio': [], 'tau_norm': [], 'k_ratio': [], 'k_eff': [], 'k_eta': [],
            'regime': [],
            'gamma_exp': [], 'gamma_quad': [], 'growth_type': [],
            'centroid_shell': [], 'cascade_rate': [], 'high_k_fraction': [],
            # v5.1 additions
            'ew_score': [], 'tau_trend_component': [], 'gamma_component': [],
            'cascade_component': [], 'high_k_component': [],
            'regime_crossing': [],  # True if crossing occurred this step
        }
        
        for s in self.config.s_values:
            self.history[f'S_eff_max_s{s:.2f}'] = []
            self.history[f'control_ratio_s{s:.2f}'] = []
        
        self.cumulative_radius = {s: 0.0 for s in self.config.s_values}
        self.bkm_integral = 0.0
        
        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None
    
    def start_logging(self, output_dir: str = ".", filename: str = "heat_kernel_v51.csv"):
        """Initialize CSV logging."""
        self.csv_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        header = [
            'time', 'step', 'omega_max', 'omega_percentile',
            'tau_ratio', 'tau_norm', 'k_eff', 'regime',
            'gamma_exp', 'cascade_rate', 'high_k_fraction',
            'ew_score', 'regime_crossing'
        ]
        
        self.csv_writer.writerow(header)
        self.csv_file.flush()
    
    def compute_alignment_field(self) -> Tuple[Any, Any]:
        """Compute alignment and vorticity fields."""
        xp = self.solver.xp
        
        omega_x = self.solver.compute_derivatives_fft(self.solver.w, 1) - \
                  self.solver.compute_derivatives_fft(self.solver.v, 2)
        omega_y = self.solver.compute_derivatives_fft(self.solver.u, 2) - \
                  self.solver.compute_derivatives_fft(self.solver.w, 0)
        omega_z = self.solver.compute_derivatives_fft(self.solver.v, 0) - \
                  self.solver.compute_derivatives_fft(self.solver.u, 1)
        
        dudx = self.solver.compute_derivatives_fft(self.solver.u, 0)
        dudy = self.solver.compute_derivatives_fft(self.solver.u, 1)
        dudz = self.solver.compute_derivatives_fft(self.solver.u, 2)
        dvdx = self.solver.compute_derivatives_fft(self.solver.v, 0)
        dvdy = self.solver.compute_derivatives_fft(self.solver.v, 1)
        dvdz = self.solver.compute_derivatives_fft(self.solver.v, 2)
        dwdx = self.solver.compute_derivatives_fft(self.solver.w, 0)
        dwdy = self.solver.compute_derivatives_fft(self.solver.w, 1)
        dwdz = self.solver.compute_derivatives_fft(self.solver.w, 2)
        
        S_xx, S_yy, S_zz = dudx, dvdy, dwdz
        S_xy = 0.5 * (dudy + dvdx)
        S_xz = 0.5 * (dudz + dwdx)
        S_yz = 0.5 * (dvdz + dwdy)
        
        S_omega_x = S_xx * omega_x + S_xy * omega_y + S_xz * omega_z
        S_omega_y = S_xy * omega_x + S_yy * omega_y + S_yz * omega_z
        S_omega_z = S_xz * omega_x + S_yz * omega_y + S_zz * omega_z
        
        omega_mag = xp.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        S_omega_mag = xp.sqrt(S_omega_x**2 + S_omega_y**2 + S_omega_z**2)
        
        omega_dot_S_omega = omega_x * S_omega_x + omega_y * S_omega_y + omega_z * S_omega_z
        cos_theta = xp.abs(omega_dot_S_omega) / (omega_mag * S_omega_mag + 1e-10)
        
        return cos_theta, omega_mag
    
    def compute_omega_percentile(self, omega_mag: Any) -> float:
        """Compute percentile of vorticity magnitude."""
        if self.solver.use_gpu:
            omega_flat = omega_mag.flatten().get()
        else:
            omega_flat = omega_mag.flatten()
        return float(np.percentile(omega_flat, self.config.omega_percentile))
    
    def compute_metrics(self, step: int = None) -> Dict[str, Any]:
        """Compute all metrics including v5.1 additions."""
        xp = self.solver.xp
        time = self.solver.current_time
        nu = self.solver.viscosity
        dt = self.solver.dt_used if hasattr(self.solver, 'dt_used') else self.solver.dt
        
        t_diff = self.config.t_viscosity_mult * nu if self.config.use_viscous_timescale else self.config.t_diffusion
        
        cos_theta, omega_mag = self.compute_alignment_field()
        
        omega_max = float(xp.max(omega_mag))
        omega_percentile = self.compute_omega_percentile(omega_mag)
        alignment_max = float(xp.max(cos_theta))
        
        enstrophy = float(xp.mean(omega_mag**2))
        k_eff = self.kernel.compute_enstrophy_weighted_k(omega_mag)
        
        # BKM integral
        self.bkm_integral += omega_max * dt
        
        metrics = {
            'time': time, 'step': step or 0, 'dt': dt,
            'omega_max': omega_max, 'omega_percentile': omega_percentile,
            'alignment_max': alignment_max, 'bkm_integral_hk': self.bkm_integral,
        }
        
        # Timescale metrics
        timescale_info = self.tao_diag.compute_timescales_v5(omega_percentile, k_eff, nu, enstrophy)
        metrics.update({
            'tau_ratio': timescale_info['tau_ratio'],
            'tau_norm': timescale_info['tau_norm'],
            'k_ratio': timescale_info['k_ratio'],
            'k_eff': timescale_info['k_eff'],
            'k_eta': timescale_info['k_eta'],
            'regime': timescale_info['regime'],
        })
        
        # Growth signature
        S_eff_field = omega_mag * cos_theta
        S_eff_max = float(xp.max(S_eff_field))
        growth_info = self.tao_diag.compute_growth_signature(S_eff_max, time)
        metrics.update({
            'gamma_exp': growth_info['gamma_exp'],
            'gamma_quad': growth_info['gamma_quad'],
            'growth_type': growth_info['growth_type'],
        })
        
        # Shell cascade
        shell_energies = self.kernel.compute_shell_energies(omega_mag, self.config.n_shells)
        cascade_info = self.tao_diag.track_shell_cascade(shell_energies, time)
        metrics.update({
            'centroid_shell': cascade_info['centroid_shell'],
            'cascade_rate': cascade_info['cascade_rate'],
            'high_k_fraction': cascade_info['high_k_fraction'],
        })
        
        # Smoothed metrics
        for s in self.config.s_values:
            S_eff_smoothed = self.kernel.apply(S_eff_field, t_diff, s)
            S_eff_smooth_max = float(xp.max(S_eff_smoothed))
            metrics[f'S_eff_max_s{s:.2f}'] = S_eff_smooth_max
            metrics[f'control_ratio_s{s:.2f}'] = S_eff_smooth_max / (S_eff_max + 1e-10)
        
        # v5.1: Regime crossing
        crossing_event = self.regime_tracker.update(time, timescale_info['tau_norm'])
        metrics['regime_crossing'] = crossing_event is not None
        if crossing_event:
            metrics['crossing_from'] = crossing_event['from_regime']
            metrics['crossing_to'] = crossing_event['to_regime']
        
        # v5.1: Early warning score
        ew_info = self.ew_system.compute_ew_score(
            time, timescale_info['tau_norm'],
            growth_info['gamma_exp'], cascade_info['cascade_rate'],
            cascade_info['high_k_fraction'], omega_percentile
        )
        metrics.update({
            'ew_score': ew_info['ew_score'],
            'tau_trend_component': ew_info['tau_trend_component'],
            'gamma_component': ew_info['gamma_component'],
            'cascade_component': ew_info['cascade_component'],
            'high_k_component': ew_info['high_k_component'],
        })
        
        # v5.1: Update lead/lag analyzer
        self.lead_lag.add_series('tau_norm', time, timescale_info['tau_norm'])
        self.lead_lag.add_series('omega_percentile', time, omega_percentile)
        self.lead_lag.add_series('gamma_exp', time, growth_info['gamma_exp'])
        self.lead_lag.add_series('cascade_rate', time, cascade_info['cascade_rate'])
        self.lead_lag.add_series('high_k_fraction', time, cascade_info['high_k_fraction'])
        self.lead_lag.add_series('ew_score', time, ew_info['ew_score'])
        
        return metrics
    
    def record(self, step: int = None) -> Dict[str, Any]:
        """Compute and record metrics."""
        metrics = self.compute_metrics(step)
        
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        if self.csv_writer is not None:
            row = [
                metrics.get('time', 0), metrics.get('step', 0),
                metrics.get('omega_max', 0), metrics.get('omega_percentile', 0),
                metrics.get('tau_ratio', 0), metrics.get('tau_norm', 0),
                metrics.get('k_eff', 0), metrics.get('regime', ''),
                metrics.get('gamma_exp', 0), metrics.get('cascade_rate', 0),
                metrics.get('high_k_fraction', 0), metrics.get('ew_score', 0),
                metrics.get('regime_crossing', False),
            ]
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        
        return metrics
    
    def close(self):
        if self.csv_file is not None:
            self.csv_file.close()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary including v5.1 analysis."""
        if len(self.history['time']) == 0:
            return {}
        
        summary = {
            'total_time': self.history['time'][-1],
            'total_steps': len(self.history['time']),
            'bkm_integral_total': self.bkm_integral,
        }
        
        # Regime statistics
        if self.history['tau_norm']:
            tau_norms = np.array(self.history['tau_norm'])
            summary['mean_tau_norm'] = float(np.mean(tau_norms))
            summary['min_tau_norm'] = float(np.min(tau_norms))
            
            cascade_thresh = self.config.tau_norm_cascade_threshold
            diffusion_thresh = self.config.tau_norm_diffusion_threshold
            summary['cascade_dominated_fraction'] = float(np.mean(tau_norms < cascade_thresh))
            summary['transitional_fraction'] = float(np.mean(
                (tau_norms >= cascade_thresh) & (tau_norms < diffusion_thresh)))
            summary['diffusion_dominated_fraction'] = float(np.mean(tau_norms >= diffusion_thresh))
        
        # v5.1: Regime crossing summary
        crossing_summary = self.regime_tracker.get_summary()
        summary['regime_crossings'] = crossing_summary
        
        # v5.1: Early warning analysis
        self.ew_system.validate_predictions()
        summary['ew_prediction_stats'] = self.ew_system.get_prediction_accuracy()
        summary['mean_ew_score'] = float(np.mean(self.history['ew_score'])) if self.history['ew_score'] else 0
        summary['max_ew_score'] = float(np.max(self.history['ew_score'])) if self.history['ew_score'] else 0
        
        # v5.1: Lead/lag analysis
        summary['lead_lag_analysis'] = self.lead_lag.analyze_all_pairs()
        
        # Growth statistics
        if self.history['growth_type']:
            types = self.history['growth_type']
            total = len(types)
            summary['exponential_fraction'] = types.count('exponential_amplifier') / total
            summary['quadratic_fraction'] = types.count('quadratic_pump') / total
            summary['stable_fraction'] = types.count('stable_or_decaying') / total
        
        # Peak values
        for key in ['omega_max', 'omega_percentile', 'alignment_max']:
            if self.history.get(key):
                summary[f'peak_{key}'] = max(self.history[key])
        
        # s=5/4 threshold effect
        subcrit = [s for s in self.config.s_values if s < 1.25]
        supercrit = [s for s in self.config.s_values if s >= 1.25]
        
        for s_list, name in [(subcrit, 'subcritical'), (supercrit, 'supercritical')]:
            if s_list:
                peaks = [max(self.history.get(f'S_eff_max_s{s:.2f}', [0])) for s in s_list]
                summary[f'{name}_mean_peak_S_eff'] = float(np.mean(peaks))
        
        if 'subcritical_mean_peak_S_eff' in summary and 'supercritical_mean_peak_S_eff' in summary:
            ratio = summary['supercritical_mean_peak_S_eff'] / (summary['subcritical_mean_peak_S_eff'] + 1e-10)
            summary['threshold_effect_ratio'] = ratio
            summary['threshold_effect'] = 'strong' if ratio < 0.5 else 'moderate' if ratio < 0.8 else 'weak'
        
        return summary


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Heat Kernel NS Integration v5.1")
    print("Regime Crossing Detection + Early Warning System")
    print("=" * 70)
    
    config = HeatKernelConfigV51()
    
    print("\nv5.1 New Features:")
    print("  1. Regime crossing tracker (enter/exit cascade)")
    print("  2. Early Warning Score (EW) combining:")
    print(f"     - τ_norm trend (weight={config.ew_tau_trend_weight})")
    print(f"     - γ_exp growth (weight={config.ew_gamma_weight})")
    print(f"     - Cascade rate (weight={config.ew_cascade_rate_weight})")
    print(f"     - High-k fraction (weight={config.ew_high_k_weight})")
    print("  3. Lead/lag analysis between diagnostics")
    print("  4. Prediction validation (does EW predict ω surges?)")
    
    # Test regime tracker
    print("\n" + "-"*50)
    print("Testing Regime Crossing Tracker:")
    print("-"*50)
    
    tracker = RegimeCrossingTracker(config)
    
    # Simulate τ_norm trajectory
    test_trajectory = [
        (0.0, 0.6),   # Start in diffusion
        (0.5, 0.4),   # Transitional
        (1.0, 0.3),   # Transitional
        (1.5, 0.08),  # Enter cascade!
        (2.0, 0.05),  # Deep cascade
        (2.5, 0.12),  # Exit cascade to transitional
        (3.0, 0.25),  # Transitional
    ]
    
    for time, tau_norm in test_trajectory:
        event = tracker.update(time, tau_norm)
        if event:
            print(f"  t={time:.1f}: CROSSING {event['from_regime'][:4]} → {event['to_regime'][:4]} (τ_norm={tau_norm:.2f})")
    
    summary = tracker.get_summary()
    print(f"\n  Total crossings: {summary['total_crossings']}")
    print(f"  First cascade entry: t={summary['first_cascade_entry']}")
    print(f"  Min τ_norm: {summary['min_tau_norm']:.3f} at t={summary['min_tau_norm_time']}")
    print(f"  Dwell fractions: cascade={summary['dwell_frac_cascade']:.1%}, "
          f"trans={summary['dwell_frac_transitional']:.1%}, diff={summary['dwell_frac_diffusion']:.1%}")
    
    print("\n" + "="*70)
    print("v5.1 ready for validation suite!")
    print("="*70)
