#!/usr/bin/env python3
"""
Navier-Stokes Simulation HDF5 to CSV Converter
Specialized for 3D fluid dynamics and turbulence simulation data
"""

import h5py
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def explore_navier_stokes_h5(file_path):
    """
    Explore and summarize Navier-Stokes simulation HDF5 file
    """
    print(f"\nExploring Navier-Stokes simulation file: {file_path}")
    print("-" * 60)
    
    with h5py.File(file_path, 'r') as f:
        # Print structure
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: Dataset {obj.shape} {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}{name}: Group")
        
        f.visititems(print_structure)
        
        # Check for common Navier-Stokes datasets
        print("\n" + "-" * 60)
        print("Detected simulation components:")
        
        # Velocity fields
        velocity_fields = ['u', 'v', 'w', 'velocity_x', 'velocity_y', 'velocity_z']
        found_velocities = [field for field in velocity_fields if field in f]
        if found_velocities:
            print(f"  Velocity fields: {', '.join(found_velocities)}")
            for field in found_velocities:
                shape = f[field].shape
                print(f"    {field}: {shape} grid points")
        
        # Time series data
        time_series = ['time', 'vorticity_max', 'energy', 'alignment', 'alpha', 
                      'enstrophy', 'dissipation', 'reynolds_stress']
        found_series = [field for field in time_series if field in f]
        if found_series:
            print(f"  Time series data: {', '.join(found_series)}")
        
        # Check attributes
        if f.attrs:
            print("\nSimulation parameters:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
        
        print("-" * 60)
        return list(f.keys())

def extract_velocity_slice(file_path, axis='z', index=None, output_path=None):
    """
    Extract a 2D slice from 3D velocity field for visualization
    
    Args:
        axis: 'x', 'y', or 'z' - axis to slice along
        index: slice index (defaults to middle)
    """
    if output_path is None:
        base_name = Path(file_path).stem
        output_path = f"{base_name}_velocity_slice_{axis}.csv"
    
    with h5py.File(file_path, 'r') as f:
        # Find velocity components
        u_field = None
        v_field = None
        w_field = None
        
        for u_name in ['u', 'velocity_x']:
            if u_name in f:
                u_field = f[u_name][:]
                break
        
        for v_name in ['v', 'velocity_y']:
            if v_name in f:
                v_field = f[v_name][:]
                break
        
        for w_name in ['w', 'velocity_z']:
            if w_name in f:
                w_field = f[w_name][:]
                break
        
        if u_field is None:
            print("No velocity field found in file")
            return None
        
        # Default to middle slice
        if index is None:
            if axis == 'x':
                index = u_field.shape[0] // 2
            elif axis == 'y':
                index = u_field.shape[1] // 2
            else:  # z
                index = u_field.shape[2] // 2
        
        # Extract slice
        if axis == 'x':
            u_slice = u_field[index, :, :] if u_field is not None else None
            v_slice = v_field[index, :, :] if v_field is not None else None
            w_slice = w_field[index, :, :] if w_field is not None else None
        elif axis == 'y':
            u_slice = u_field[:, index, :] if u_field is not None else None
            v_slice = v_field[:, index, :] if v_field is not None else None
            w_slice = w_field[:, index, :] if w_field is not None else None
        else:  # z
            u_slice = u_field[:, :, index] if u_field is not None else None
            v_slice = v_field[:, :, index] if v_field is not None else None
            w_slice = w_field[:, :, index] if w_field is not None else None
        
        # Create DataFrame with flattened data and coordinates
        ny, nx = u_slice.shape if u_slice is not None else (0, 0)
        
        data = {}
        if axis == 'x':
            data['y'] = np.repeat(np.arange(ny), nx)
            data['z'] = np.tile(np.arange(nx), ny)
        elif axis == 'y':
            data['x'] = np.repeat(np.arange(ny), nx)
            data['z'] = np.tile(np.arange(nx), ny)
        else:  # z
            data['x'] = np.repeat(np.arange(ny), nx)
            data['y'] = np.tile(np.arange(nx), ny)
        
        if u_slice is not None:
            data['u'] = u_slice.flatten()
        if v_slice is not None:
            data['v'] = v_slice.flatten()
        if w_slice is not None:
            data['w'] = w_slice.flatten()
        
        # Add velocity magnitude
        if u_slice is not None and v_slice is not None and w_slice is not None:
            data['velocity_magnitude'] = np.sqrt(u_slice**2 + v_slice**2 + w_slice**2).flatten()
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Saved velocity slice ({axis}={index}) to {output_path}")
        return output_path

def extract_time_series(file_path, output_path=None):
    """
    Extract time series data (vorticity, energy, etc.) to CSV
    """
    if output_path is None:
        base_name = Path(file_path).stem
        output_path = f"{base_name}_time_series.csv"
    
    with h5py.File(file_path, 'r') as f:
        data = {}
        
        # Common time series in Navier-Stokes simulations
        time_series_fields = [
            'time', 't', 'timestep',
            'vorticity_max', 'vorticity', 'enstrophy',
            'energy', 'kinetic_energy', 'total_energy',
            'dissipation', 'dissipation_rate',
            'alignment', 'alpha',
            'reynolds_number', 'Re',
            'bkm_integral', 'BKM'
        ]
        
        for field in time_series_fields:
            if field in f:
                data[field] = f[field][:]
                print(f"  Found time series: {field} ({len(data[field])} points)")
        
        if not data:
            print("No time series data found")
            return None
        
        df = pd.DataFrame(data)
        
        # Add derived quantities if possible
        if 'vorticity_max' in df.columns and 'time' in df.columns:
            # Compute running BKM integral if not present
            if 'bkm_integral' not in df.columns:
                dt = np.diff(df['time'].values)
                if len(dt) > 0:
                    dt_avg = np.mean(dt)
                    df['bkm_integral_computed'] = np.cumsum(df['vorticity_max'].values) * dt_avg
        
        df.to_csv(output_path, index=False)
        print(f"Saved time series data to {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {', '.join(df.columns)}")
        return output_path

def compute_vorticity_from_velocity(file_path, output_path=None):
    """
    Compute vorticity field from velocity components and save statistics
    """
    if output_path is None:
        base_name = Path(file_path).stem
        output_path = f"{base_name}_vorticity_stats.csv"
    
    with h5py.File(file_path, 'r') as f:
        # Load velocity fields
        u = f['u'][:] if 'u' in f else None
        v = f['v'][:] if 'v' in f else None
        w = f['w'][:] if 'w' in f else None
        
        if u is None or v is None or w is None:
            print("Complete velocity field (u,v,w) not found")
            return None
        
        print(f"Computing vorticity from {u.shape} velocity field...")
        
        # Compute vorticity components using finite differences
        # Assuming periodic boundary conditions
        dx = dy = dz = 1.0  # Can be adjusted based on actual grid spacing
        
        # ω_x = ∂w/∂y - ∂v/∂z
        dwdy = np.gradient(w, axis=1) / dy
        dvdz = np.gradient(v, axis=2) / dz
        omega_x = dwdy - dvdz
        
        # ω_y = ∂u/∂z - ∂w/∂x
        dudz = np.gradient(u, axis=2) / dz
        dwdx = np.gradient(w, axis=0) / dx
        omega_y = dudz - dwdx
        
        # ω_z = ∂v/∂x - ∂u/∂y
        dvdx = np.gradient(v, axis=0) / dx
        dudy = np.gradient(u, axis=1) / dy
        omega_z = dvdx - dudy
        
        # Compute vorticity magnitude
        vorticity_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        
        # Compute statistics
        stats = {
            'vorticity_max': np.max(vorticity_mag),
            'vorticity_mean': np.mean(vorticity_mag),
            'vorticity_std': np.std(vorticity_mag),
            'enstrophy': 0.5 * np.mean(vorticity_mag**2),
            'omega_x_max': np.max(np.abs(omega_x)),
            'omega_y_max': np.max(np.abs(omega_y)),
            'omega_z_max': np.max(np.abs(omega_z)),
        }
        
        # Save statistics
        df = pd.DataFrame([stats])
        df.to_csv(output_path, index=False)
        print(f"Saved vorticity statistics to {output_path}")
        
        for key, value in stats.items():
            print(f"  {key}: {value:.6f}")
        
        return output_path

def extract_checkpoint_summary(file_path, output_dir=None):
    """
    Extract summary from checkpoint files (multiple timesteps)
    """
    if output_dir is None:
        output_dir = Path(file_path).stem + "_summary"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if it's a checkpoint file
    with h5py.File(file_path, 'r') as f:
        if 'step' in f.attrs or 'time' in f.attrs:
            # Single checkpoint
            step = f.attrs.get('step', 0)
            time = f.attrs.get('time', 0.0)
            
            summary = {
                'step': step,
                'time': time,
            }
            
            # Add other attributes
            for key in f.attrs.keys():
                if key not in summary:
                    summary[key] = f.attrs[key]
            
            # Check field sizes
            if 'u' in f:
                summary['grid_points'] = np.prod(f['u'].shape)
                summary['grid_shape'] = str(f['u'].shape)
            
            df = pd.DataFrame([summary])
            output_path = os.path.join(output_dir, f"checkpoint_{step:06d}_summary.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved checkpoint summary to {output_path}")
            return output_path

def main():
    """
    Main function with Navier-Stokes specific options
    """
    if len(sys.argv) < 2:
        print("Navier-Stokes HDF5 to CSV Converter")
        print("=" * 60)
        print("\nUsage: python ns_h5_to_csv.py <h5_file> [options]")
        print("\nOptions:")
        print("  --explore                  : Explore file structure and parameters")
        print("  --time-series             : Extract time series data (vorticity, energy, etc.)")
        print("  --slice <axis> [index]    : Extract 2D velocity slice (axis: x/y/z)")
        print("  --vorticity               : Compute vorticity statistics from velocity")
        print("  --checkpoint              : Extract checkpoint summary")
        print("  --all                     : Extract all available data")
        print("  --output <path>           : Specify output file/directory")
        print("\nExamples:")
        print("  python ns_h5_to_csv.py simulation.h5 --explore")
        print("  python ns_h5_to_csv.py simulation.h5 --time-series")
        print("  python ns_h5_to_csv.py simulation.h5 --slice z 256")
        print("  python ns_h5_to_csv.py checkpoint.h5 --vorticity")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    
    # Default action is explore
    if len(sys.argv) == 2 or '--explore' in sys.argv:
        explore_navier_stokes_h5(file_path)
        if len(sys.argv) == 2:
            print("\nUse options above to extract specific data.")
    
    # Time series extraction
    if '--time-series' in sys.argv:
        output_path = None
        if '--output' in sys.argv:
            idx = sys.argv.index('--output')
            if idx + 1 < len(sys.argv):
                output_path = sys.argv[idx + 1]
        extract_time_series(file_path, output_path)
    
    # Velocity slice extraction
    if '--slice' in sys.argv:
        idx = sys.argv.index('--slice')
        if idx + 1 < len(sys.argv):
            axis = sys.argv[idx + 1].lower()
            index = None
            if idx + 2 < len(sys.argv) and sys.argv[idx + 2].isdigit():
                index = int(sys.argv[idx + 2])
            
            output_path = None
            if '--output' in sys.argv:
                out_idx = sys.argv.index('--output')
                if out_idx + 1 < len(sys.argv):
                    output_path = sys.argv[out_idx + 1]
            
            extract_velocity_slice(file_path, axis, index, output_path)
        else:
            print("Error: --slice requires axis argument (x/y/z)")
    
    # Vorticity computation
    if '--vorticity' in sys.argv:
        output_path = None
        if '--output' in sys.argv:
            idx = sys.argv.index('--output')
            if idx + 1 < len(sys.argv):
                output_path = sys.argv[idx + 1]
        compute_vorticity_from_velocity(file_path, output_path)
    
    # Checkpoint summary
    if '--checkpoint' in sys.argv:
        output_dir = None
        if '--output' in sys.argv:
            idx = sys.argv.index('--output')
            if idx + 1 < len(sys.argv):
                output_dir = sys.argv[idx + 1]
        extract_checkpoint_summary(file_path, output_dir)
    
    # Extract all
    if '--all' in sys.argv:
        base_name = Path(file_path).stem
        output_dir = f"{base_name}_extracted"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nExtracting all data to {output_dir}/")
        
        # Time series
        extract_time_series(file_path, os.path.join(output_dir, "time_series.csv"))
        
        # Velocity slices at multiple positions
        with h5py.File(file_path, 'r') as f:
            if 'u' in f:
                shape = f['u'].shape
                for axis in ['x', 'y', 'z']:
                    for frac in [0.25, 0.5, 0.75]:
                        if axis == 'x':
                            index = int(shape[0] * frac)
                        elif axis == 'y':
                            index = int(shape[1] * frac)
                        else:
                            index = int(shape[2] * frac)
                        
                        output_path = os.path.join(output_dir, f"slice_{axis}_{int(frac*100)}.csv")
                        extract_velocity_slice(file_path, axis, index, output_path)
        
        # Vorticity statistics
        compute_vorticity_from_velocity(file_path, os.path.join(output_dir, "vorticity_stats.csv"))
        
        print(f"\nAll data extracted to {output_dir}/")

if __name__ == "__main__":
    main()