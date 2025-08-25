#!/usr/bin/env python3
"""
gpu_performance_demo.py
=======================
Demonstrates GPU acceleration benefits of the BKM Engine.

Compares performance across different grid sizes and shows speedup.
"""

import sys
import os
import numpy as np
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.unified_bkm_engine import CUDABKMSolver


def benchmark_grid_size(N, Re=1000, steps=100):
    """Run benchmark for specific grid size."""
    
    print(f"\n{N}³ Grid:")
    print("-"*30)
    
    try:
        # Create solver - it auto-detects GPU
        solver = CUDABKMSolver(
            grid_size=N,
            reynolds_number=Re,
            dt=0.001,
            adapt_dt=True
        )
        
        # Check if GPU is being used
        device = 'GPU' if solver.use_gpu else 'CPU'
        
        # Initialize with random field
        xp = solver.xp  # This is cupy if GPU, numpy if CPU
        solver.u = xp.random.randn(N, N, N) * 0.1
        solver.v = xp.random.randn(N, N, N) * 0.1
        solver.w = xp.random.randn(N, N, N) * 0.1
        
        # Project to div-free
        solver.u, solver.v, solver.w = solver.project_div_free(
            solver.u, solver.v, solver.w
        )
        
        # Warmup
        for _ in range(5):
            solver.evolve_one_timestep()
        
        # Benchmark
        t_start = time.time()
        for _ in range(steps):
            metrics = solver.evolve_one_timestep(use_rk2=True)
        t_elapsed = time.time() - t_start
        
        # Results
        steps_per_sec = steps / t_elapsed
        ms_per_step = (t_elapsed / steps) * 1000
        
        print(f"  Device: {device}")
        print(f"  Performance: {steps_per_sec:.1f} steps/sec")
        print(f"  Time per step: {ms_per_step:.1f} ms")
        print(f"  Memory used: ~{(N**3 * 8 * 6) / 1e9:.1f} GB")
        
        return steps_per_sec, ms_per_step, device
        
    except Exception as e:
        print(f"  Failed: {e}")
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description='GPU Performance Demonstration')
    parser.add_argument('--steps', type=int, default=100, help='Steps per benchmark')
    parser.add_argument('--max-grid', type=int, default=256, 
                       help='Maximum grid size to test')
    args = parser.parse_args()
    
    print("="*70)
    print("BKM ENGINE - PERFORMANCE DEMONSTRATION")
    print("="*70)
    
    # Check for GPU availability
    try:
        import cupy
        gpu_available = True
        print("\n✓ CuPy detected - GPU acceleration available")
    except ImportError:
        gpu_available = False
        print("\n⚠ CuPy not found - running on CPU")
        print("  Install CuPy for GPU acceleration: pip install cupy-cuda11x")
    
    # Grid sizes to test
    if gpu_available:
        if args.max_grid >= 512:
            grid_sizes = [64, 128, 256, 512]
        else:
            grid_sizes = [64, 128, 256]
    else:
        grid_sizes = [32, 64, 128]  # CPU can't handle large grids
        print("\n  CPU mode: limiting to smaller grids")
    
    print(f"\nTesting grid sizes: {grid_sizes}")
    
    results = {}
    
    # Run benchmarks
    for N in grid_sizes:
        perf, ms, device = benchmark_grid_size(N, steps=args.steps)
        if perf is not None:
            results[N] = (perf, ms, device)
    
    # Summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    if not results:
        print("\nNo successful benchmarks. Check your installation.")
        return
    
    print("\n{:<10} {:<10} {:<15} {:<15} {:<15}".format(
        "Grid", "Device", "Steps/sec", "ms/step", "Speedup"
    ))
    print("-"*65)
    
    baseline = results.get(64, (1, 1000, 'N/A'))[0] if 64 in results else 1
    
    for N in sorted(results.keys()):
        perf, ms, device = results[N]
        speedup = perf / baseline if baseline > 0 else 1
        print("{:<10} {:<10} {:<15.1f} {:<15.1f} {:<15.1f}x".format(
            f"{N}³", device, perf, ms, speedup
        ))
    
    # Scaling analysis
    if len(results) >= 2:
        sizes = sorted(results.keys())
        print(f"\nScaling Analysis:")
        for i in range(len(sizes)-1):
            N1, N2 = sizes[i], sizes[i+1]
            perf1, perf2 = results[N1][0], results[N2][0]
            size_ratio = (N2/N1)**3
            perf_ratio = perf1/perf2 if perf2 > 0 else 0
            efficiency = perf_ratio / size_ratio * 100
            print(f"  {N1}³ → {N2}³: {efficiency:.1f}% efficiency")
    
    print("\nNotes:")
    if gpu_available:
        print("  - GPU provides 10-100x speedup over CPU")
        print("  - 512³ simulations only feasible on GPU")
        print("  - Performance depends on GPU model and memory")
        print("  - Use 'nvidia-smi' to monitor GPU utilization")
    else:
        print("  - Install CuPy to enable GPU acceleration")
        print("  - CPU performance is limited for large grids")


if __name__ == "__main__":
    main()