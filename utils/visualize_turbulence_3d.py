#!/usr/bin/env python3
"""
visualize_turbulence_3d.py
===========================
Generate publication-quality 3D visualizations from turbulence simulation HDF5 files.

This script creates various visualization types suitable for academic papers:
- Volume rendering of vorticity magnitude
- Isosurfaces of Q-criterion (vortex identification)
- Slice planes with velocity/vorticity fields
- Streamlines and pathlines
- Combined multi-panel figures

Requires:
- matplotlib
- numpy
- h5py
- mayavi or pyvista (for 3D rendering)
- scipy (for interpolation)

Usage:
    python visualize_turbulence_3d.py fields_t10.00.h5
    python visualize_turbulence_3d.py fields_t10.00.h5 --type vorticity --colormap plasma
    python visualize_turbulence_3d.py fields_t10.00.h5 --type q-criterion --threshold 0.5

"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import os
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D

# Try to import 3D visualization libraries
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista not available. Install with: pip install pyvista")

try:
    from mayavi import mlab
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False
    print("Mayavi not available. Some 3D features will be limited.")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_field_data(filepath):
    """
    Load velocity and vorticity fields from HDF5 file.
    
    Parameters:
    -----------
    filepath : str
        Path to HDF5 field checkpoint file
        
    Returns:
    --------
    data : dict
        Dictionary containing field arrays and metadata
    """
    data = {}
    
    with h5py.File(filepath, 'r') as f:
        # Load velocity fields
        data['u'] = f['u'][:]
        data['v'] = f['v'][:]
        data['w'] = f['w'][:]
        
        # Load vorticity fields
        data['omega_x'] = f['omega_x'][:]
        data['omega_y'] = f['omega_y'][:]
        data['omega_z'] = f['omega_z'][:]
        
        # Load metadata
        data['time'] = f.attrs['time']
        data['Re'] = f.attrs['reynolds_number']
        data['energy'] = f.attrs['energy']
        data['enstrophy'] = f.attrs['enstrophy']
        data['Re_lambda'] = f.attrs.get('Re_lambda', 0)
        
        # Grid info
        data['nx'] = f.attrs['nx']
        data['ny'] = f.attrs['ny']
        data['nz'] = f.attrs['nz']
        data['Lx'] = f.attrs.get('Lx', 2*np.pi)
        data['Ly'] = f.attrs.get('Ly', 2*np.pi)
        data['Lz'] = f.attrs.get('Lz', 2*np.pi)
        
        # Load spectrum if available
        if 'k_spectrum' in f:
            data['k_spectrum'] = f['k_spectrum'][:]
            data['E_spectrum'] = f['E_spectrum'][:]
    
    print(f"Loaded field data:")
    print(f"  Time: {data['time']:.2f}")
    print(f"  Grid: {data['nx']}x{data['ny']}x{data['nz']}")
    print(f"  Re_lambda: {data['Re_lambda']:.1f}")
    print(f"  Energy: {data['energy']:.4f}")
    
    return data


def compute_vorticity_magnitude(data):
    """Compute vorticity magnitude field."""
    omega_mag = np.sqrt(data['omega_x']**2 + 
                       data['omega_y']**2 + 
                       data['omega_z']**2)
    return omega_mag


def compute_q_criterion(data):
    """
    Compute Q-criterion for vortex identification.
    Q = 0.5 * (||Ω||² - ||S||²)
    where Ω is vorticity tensor and S is strain rate tensor.
    """
    nx, ny, nz = data['nx'], data['ny'], data['nz']
    dx = data['Lx'] / nx
    dy = data['Ly'] / ny
    dz = data['Lz'] / nz
    
    # Compute velocity gradients (simplified - use FFT for better accuracy)
    dudx = np.gradient(data['u'], dx, axis=0)
    dudy = np.gradient(data['u'], dy, axis=1)
    dudz = np.gradient(data['u'], dz, axis=2)
    
    dvdx = np.gradient(data['v'], dx, axis=0)
    dvdy = np.gradient(data['v'], dy, axis=1)
    dvdz = np.gradient(data['v'], dz, axis=2)
    
    dwdx = np.gradient(data['w'], dx, axis=0)
    dwdy = np.gradient(data['w'], dy, axis=1)
    dwdz = np.gradient(data['w'], dz, axis=2)
    
    # Strain rate tensor components
    S11 = dudx
    S22 = dvdy
    S33 = dwdz
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S23 = 0.5 * (dvdz + dwdy)
    
    # Vorticity tensor components
    W12 = 0.5 * (dudy - dvdx)
    W13 = 0.5 * (dudz - dwdx)
    W23 = 0.5 * (dvdz - dwdy)
    
    # Q-criterion
    S_squared = 2 * (S11**2 + S22**2 + S33**2) + 4 * (S12**2 + S13**2 + S23**2)
    W_squared = 4 * (W12**2 + W13**2 + W23**2)
    Q = 0.5 * (W_squared - S_squared)
    
    return Q


def compute_lambda2_criterion(data):
    """
    Compute λ₂ criterion for vortex identification.
    Based on the second eigenvalue of S² + Ω².
    """
    # This is a simplified version - full implementation would compute
    # eigenvalues of the S² + Ω² tensor at each point
    # For now, return Q-criterion as approximation
    return compute_q_criterion(data)


# ============================================================================
# 3D VISUALIZATION WITH PYVISTA
# ============================================================================

def create_pyvista_volume_rendering(data, field='vorticity', cmap='plasma', 
                                   opacity='sigmoid', save_path=None):
    """
    Create volume rendering using PyVista.
    
    Parameters:
    -----------
    data : dict
        Field data dictionary
    field : str
        Field to visualize ('vorticity', 'q-criterion', 'velocity')
    cmap : str
        Colormap name
    opacity : str or array
        Opacity mapping ('linear', 'sigmoid', 'geom', or custom array)
    save_path : str
        Path to save image
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available. Skipping volume rendering.")
        return
    
    # Select field to visualize
    if field == 'vorticity':
        scalar_field = compute_vorticity_magnitude(data)
        field_name = "Vorticity Magnitude"
    elif field == 'q-criterion':
        scalar_field = compute_q_criterion(data)
        field_name = "Q-Criterion"
    elif field == 'velocity':
        scalar_field = np.sqrt(data['u']**2 + data['v']**2 + data['w']**2)
        field_name = "Velocity Magnitude"
    else:
        raise ValueError(f"Unknown field: {field}")
    
    # Create structured grid
    nx, ny, nz = data['nx'], data['ny'], data['nz']
    grid = pv.ImageData(dimensions=(nx, ny, nz))
    grid.spacing = (data['Lx']/nx, data['Ly']/ny, data['Lz']/nz)
    grid.origin = (0, 0, 0)
    
    # Add scalar field
    grid[field_name] = scalar_field.flatten(order='F')
    
    # Create plotter
    plotter = pv.Plotter(window_size=[1920, 1080], off_screen=save_path is not None)
    
    # Add volume with opacity mapping
    if opacity == 'sigmoid':
        opacity_values = [0, 0, 0.1, 0.3, 0.6, 0.9, 1.0]
    elif opacity == 'linear':
        opacity_values = 'linear'
    elif opacity == 'geom':
        opacity_values = 'geom_r'
    else:
        opacity_values = opacity
    
    plotter.add_volume(
        grid,
        scalars=field_name,
        cmap=cmap,
        opacity=opacity_values,
        shade=True,
        ambient=0.3,
        diffuse=0.6,
        specular=0.2
    )
    
    # Add axes and labels
    plotter.show_axes()
    plotter.add_text(
        f"{field_name}\nt = {data['time']:.2f}, Re_λ = {data['Re_lambda']:.1f}",
        position='upper_left',
        font_size=12
    )
    
    # Set camera position for good view
    plotter.camera_position = [(2*data['Lx'], 2*data['Ly'], 2*data['Lz']),
                               (data['Lx']/2, data['Ly']/2, data['Lz']/2),
                               (0, 0, 1)]
    
    if save_path:
        plotter.screenshot(save_path)
        print(f"Saved volume rendering to: {save_path}")
    else:
        plotter.show()
    
    plotter.close()


def create_pyvista_isosurfaces(data, field='q-criterion', threshold=None,
                              cmap='coolwarm', save_path=None):
    """
    Create isosurface visualization using PyVista.
    
    Parameters:
    -----------
    data : dict
        Field data dictionary
    field : str
        Field for isosurface ('q-criterion', 'lambda2', 'vorticity')
    threshold : float or list
        Isosurface threshold value(s)
    cmap : str
        Colormap for surface coloring
    save_path : str
        Path to save image
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available. Skipping isosurface rendering.")
        return
    
    # Select field
    if field == 'q-criterion':
        scalar_field = compute_q_criterion(data)
        field_name = "Q-Criterion"
        if threshold is None:
            threshold = np.percentile(scalar_field[scalar_field > 0], 90)
    elif field == 'lambda2':
        scalar_field = compute_lambda2_criterion(data)
        field_name = "λ₂ Criterion"
        if threshold is None:
            threshold = np.percentile(scalar_field[scalar_field < 0], 10)
    elif field == 'vorticity':
        scalar_field = compute_vorticity_magnitude(data)
        field_name = "Vorticity Magnitude"
        if threshold is None:
            threshold = np.percentile(scalar_field, 95)
    
    # Create structured grid
    nx, ny, nz = data['nx'], data['ny'], data['nz']
    grid = pv.ImageData(dimensions=(nx, ny, nz))
    grid.spacing = (data['Lx']/nx, data['Ly']/ny, data['Lz']/nz)
    grid.origin = (0, 0, 0)
    
    # Add scalar field
    grid[field_name] = scalar_field.flatten(order='F')
    
    # Also add velocity magnitude for coloring
    vel_mag = np.sqrt(data['u']**2 + data['v']**2 + data['w']**2)
    grid["Velocity"] = vel_mag.flatten(order='F')
    
    # Create plotter
    plotter = pv.Plotter(window_size=[1920, 1080], off_screen=save_path is not None)
    
    # Extract and add isosurfaces
    if isinstance(threshold, list):
        for thresh in threshold:
            iso = grid.contour([thresh], scalars=field_name)
            if iso.n_points > 0:
                plotter.add_mesh(iso, scalars="Velocity", cmap=cmap,
                               opacity=0.8, smooth_shading=True)
    else:
        iso = grid.contour([threshold], scalars=field_name)
        if iso.n_points > 0:
            plotter.add_mesh(iso, scalars="Velocity", cmap=cmap,
                           opacity=0.9, smooth_shading=True)
    
    # Add bounding box
    plotter.add_mesh(grid.outline(), color='black', line_width=2)
    
    # Add axes and labels
    plotter.show_axes()
    plotter.add_text(
        f"{field_name} Isosurfaces\nt = {data['time']:.2f}, Re_λ = {data['Re_lambda']:.1f}",
        position='upper_left',
        font_size=12
    )
    
    # Set camera and lighting
    plotter.camera_position = [(1.5*data['Lx'], 1.5*data['Ly'], 1.5*data['Lz']),
                               (data['Lx']/2, data['Ly']/2, data['Lz']/2),
                               (0, 0, 1)]
    plotter.add_light(pv.Light(position=(data['Lx'], data['Ly'], 2*data['Lz']), 
                               light_type='headlight'))
    
    if save_path:
        plotter.screenshot(save_path)
        print(f"Saved isosurface rendering to: {save_path}")
    else:
        plotter.show()
    
    plotter.close()


def create_slice_visualization(data, normal='z', origin=None, 
                              field='vorticity', cmap='RdBu_r', save_path=None):
    """
    Create slice plane visualization with contours.
    
    Parameters:
    -----------
    data : dict
        Field data dictionary
    normal : str or tuple
        Slice plane normal ('x', 'y', 'z' or vector)
    origin : tuple
        Slice plane origin (default: domain center)
    field : str
        Field to visualize on slice
    cmap : str
        Colormap
    save_path : str
        Path to save figure
    """
    if not PYVISTA_AVAILABLE:
        print("Using matplotlib for 2D slice (install PyVista for 3D context)")
        create_matplotlib_slice(data, normal, field, cmap, save_path)
        return
    
    # Create structured grid
    nx, ny, nz = data['nx'], data['ny'], data['nz']
    grid = pv.ImageData(dimensions=(nx, ny, nz))
    grid.spacing = (data['Lx']/nx, data['Ly']/ny, data['Lz']/nz)
    grid.origin = (0, 0, 0)
    
    # Add fields
    if field == 'vorticity_z':
        grid["Field"] = data['omega_z'].flatten(order='F')
        field_name = "Vorticity (z-component)"
    elif field == 'vorticity':
        omega_mag = compute_vorticity_magnitude(data)
        grid["Field"] = omega_mag.flatten(order='F')
        field_name = "Vorticity Magnitude"
    elif field == 'velocity':
        vel_mag = np.sqrt(data['u']**2 + data['v']**2 + data['w']**2)
        grid["Field"] = vel_mag.flatten(order='F')
        field_name = "Velocity Magnitude"
    
    # Create plotter
    plotter = pv.Plotter(window_size=[1920, 1080], off_screen=save_path is not None)
    
    # Add slice
    if origin is None:
        origin = (data['Lx']/2, data['Ly']/2, data['Lz']/2)
    
    if normal == 'x':
        normal_vec = (1, 0, 0)
    elif normal == 'y':
        normal_vec = (0, 1, 0)
    elif normal == 'z':
        normal_vec = (0, 0, 1)
    else:
        normal_vec = normal
    
    slice_mesh = grid.slice(normal=normal_vec, origin=origin)
    
    plotter.add_mesh(slice_mesh, scalars="Field", cmap=cmap,
                    show_scalar_bar=True, scalar_bar_args={'title': field_name})
    
    # Add grid outline
    plotter.add_mesh(grid.outline(), color='black', line_width=2)
    
    # Add title
    plotter.add_text(
        f"{field_name} - {normal.upper()}-normal slice\nt = {data['time']:.2f}",
        position='upper_left',
        font_size=12
    )
    
    # Set view
    if normal == 'z':
        plotter.view_xy()
    elif normal == 'y':
        plotter.view_xz()
    elif normal == 'x':
        plotter.view_yz()
    
    if save_path:
        plotter.screenshot(save_path)
        print(f"Saved slice visualization to: {save_path}")
    else:
        plotter.show()
    
    plotter.close()


# ============================================================================
# MATPLOTLIB FALLBACK VISUALIZATIONS
# ============================================================================

def create_matplotlib_slice(data, axis='z', field='vorticity', cmap='RdBu_r', 
                           save_path=None):
    """
    Create 2D slice visualization using matplotlib.
    
    Parameters:
    -----------
    data : dict
        Field data dictionary
    axis : str
        Slice axis ('x', 'y', or 'z')
    field : str
        Field to visualize
    cmap : str
        Colormap
    save_path : str
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Determine slice indices
    if axis == 'z':
        slice_idx = data['nz'] // 2
        x = np.linspace(0, data['Lx'], data['nx'])
        y = np.linspace(0, data['Ly'], data['ny'])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Velocity magnitude
        vel_slice = np.sqrt(data['u'][:, :, slice_idx]**2 + 
                          data['v'][:, :, slice_idx]**2 + 
                          data['w'][:, :, slice_idx]**2)
        
        # Vorticity z-component
        vort_slice = data['omega_z'][:, :, slice_idx]
        
        xlabel, ylabel = 'x', 'y'
        
    elif axis == 'y':
        slice_idx = data['ny'] // 2
        x = np.linspace(0, data['Lx'], data['nx'])
        z = np.linspace(0, data['Lz'], data['nz'])
        X, Y = np.meshgrid(x, z, indexing='ij')
        
        vel_slice = np.sqrt(data['u'][:, slice_idx, :]**2 + 
                          data['v'][:, slice_idx, :]**2 + 
                          data['w'][:, slice_idx, :]**2)
        vort_slice = data['omega_y'][:, slice_idx, :]
        
        xlabel, ylabel = 'x', 'z'
        
    else:  # x
        slice_idx = data['nx'] // 2
        y = np.linspace(0, data['Ly'], data['ny'])
        z = np.linspace(0, data['Lz'], data['nz'])
        X, Y = np.meshgrid(y, z, indexing='ij')
        
        vel_slice = np.sqrt(data['u'][slice_idx, :, :]**2 + 
                          data['v'][slice_idx, :, :]**2 + 
                          data['w'][slice_idx, :, :]**2)
        vort_slice = data['omega_x'][slice_idx, :, :]
        
        xlabel, ylabel = 'y', 'z'
    
    # Plot velocity magnitude
    im1 = axes[0, 0].pcolormesh(X, Y, vel_slice, cmap='viridis', shading='gouraud')
    axes[0, 0].set_title('Velocity Magnitude')
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].set_ylabel(ylabel)
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot vorticity component
    vmax = np.abs(vort_slice).max()
    im2 = axes[0, 1].pcolormesh(X, Y, vort_slice, cmap='RdBu_r', 
                                vmin=-vmax, vmax=vmax, shading='gouraud')
    axes[0, 1].set_title(f'Vorticity ({axis}-component)')
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel(ylabel)
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot Q-criterion slice
    Q = compute_q_criterion(data)
    if axis == 'z':
        Q_slice = Q[:, :, slice_idx]
    elif axis == 'y':
        Q_slice = Q[:, slice_idx, :]
    else:
        Q_slice = Q[slice_idx, :, :]
    
    im3 = axes[1, 0].pcolormesh(X, Y, Q_slice, cmap='coolwarm', shading='gouraud')
    axes[1, 0].set_title('Q-Criterion')
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 0].set_ylabel(ylabel)
    axes[1, 0].set_aspect('equal')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot energy spectrum if available
    if 'k_spectrum' in data and 'E_spectrum' in data:
        k = data['k_spectrum']
        E = data['E_spectrum']
        
        axes[1, 1].loglog(k, E, 'b-', linewidth=2, label='E(k)')
        
        # Add Kolmogorov -5/3 reference
        k_ref = k[k > 0]
        E_ref = E[k > 0][0] * (k_ref / k_ref[0])**(-5/3)
        axes[1, 1].loglog(k_ref, E_ref, 'k--', alpha=0.5, label='k^{-5/3}')
        
        axes[1, 1].set_xlabel('k')
        axes[1, 1].set_ylabel('E(k)')
        axes[1, 1].set_title('Energy Spectrum')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        # If no spectrum, show statistics
        axes[1, 1].text(0.1, 0.9, f"Time: {data['time']:.2f}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.8, f"Re: {data['Re']:.0f}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f"Re_λ: {data['Re_lambda']:.1f}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Energy: {data['energy']:.4f}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f"Enstrophy: {data['enstrophy']:.2f}", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Simulation Parameters')
        axes[1, 1].axis('off')
    
    plt.suptitle(f'Isotropic Turbulence - {axis.upper()}-normal Slice at t={data["time"]:.2f}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved matplotlib figure to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_multiplane_figure(data, save_path=None):
    """
    Create publication-quality figure with multiple slice planes.
    
    Parameters:
    -----------
    data : dict
        Field data dictionary
    save_path : str
        Path to save figure
    """
    fig = plt.figure(figsize=(15, 12))
    
    # Compute vorticity magnitude
    omega_mag = compute_vorticity_magnitude(data)
    
    # Create slices at different positions
    slices_z = [data['nz']//4, data['nz']//2, 3*data['nz']//4]
    
    for i, z_idx in enumerate(slices_z):
        ax = fig.add_subplot(3, 3, i+1)
        
        slice_data = omega_mag[:, :, z_idx]
        im = ax.imshow(slice_data.T, cmap='hot', origin='lower',
                      extent=[0, data['Lx'], 0, data['Ly']])
        ax.set_title(f'z = {z_idx * data["Lz"]/data["nz"]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Y-slices
    slices_y = [data['ny']//4, data['ny']//2, 3*data['ny']//4]
    
    for i, y_idx in enumerate(slices_y):
        ax = fig.add_subplot(3, 3, i+4)
        
        slice_data = omega_mag[:, y_idx, :]
        im = ax.imshow(slice_data.T, cmap='hot', origin='lower',
                      extent=[0, data['Lx'], 0, data['Lz']])
        ax.set_title(f'y = {y_idx * data["Ly"]/data["ny"]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # X-slices
    slices_x = [data['nx']//4, data['nx']//2, 3*data['nx']//4]
    
    for i, x_idx in enumerate(slices_x):
        ax = fig.add_subplot(3, 3, i+7)
        
        slice_data = omega_mag[x_idx, :, :]
        im = ax.imshow(slice_data.T, cmap='hot', origin='lower',
                      extent=[0, data['Ly'], 0, data['Lz']])
        ax.set_title(f'x = {x_idx * data["Lx"]/data["nx"]:.2f}')
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'Vorticity Magnitude - Multiple Slices\n'
                f't = {data["time"]:.2f}, Re_λ = {data["Re_lambda"]:.1f}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-plane figure to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for visualization script."""
    
    parser = argparse.ArgumentParser(
        description='Generate 3D visualizations from turbulence HDF5 files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('input_file', help='Path to HDF5 field checkpoint file')
    parser.add_argument('--type', choices=['volume', 'isosurface', 'slice', 'multiplane'],
                       default='slice', help='Visualization type')
    parser.add_argument('--field', default='vorticity',
                       help='Field to visualize (vorticity, velocity, q-criterion)')
    parser.add_argument('--colormap', default='plasma',
                       help='Colormap name')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Isosurface threshold value')
    parser.add_argument('--axis', choices=['x', 'y', 'z'], default='z',
                       help='Slice plane normal axis')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save output image')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved images')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input_file}")
    data = load_field_data(args.input_file)
    
    # Generate visualization based on type
    if args.type == 'volume':
        if PYVISTA_AVAILABLE:
            create_pyvista_volume_rendering(data, field=args.field, 
                                          cmap=args.colormap, save_path=args.save)
        else:
            print("Volume rendering requires PyVista. Install with: pip install pyvista")
            
    elif args.type == 'isosurface':
        if PYVISTA_AVAILABLE:
            create_pyvista_isosurfaces(data, field=args.field, 
                                     threshold=args.threshold,
                                     cmap=args.colormap, save_path=args.save)
        else:
            print("Isosurface rendering requires PyVista. Install with: pip install pyvista")
            
    elif args.type == 'slice':
        if PYVISTA_AVAILABLE:
            create_slice_visualization(data, normal=args.axis, field=args.field,
                                     cmap=args.colormap, save_path=args.save)
        else:
            create_matplotlib_slice(data, axis=args.axis, field=args.field,
                                  cmap=args.colormap, save_path=args.save)
            
    elif args.type == 'multiplane':
        create_multiplane_figure(data, save_path=args.save)
    
    print("Visualization complete")


if __name__ == "__main__":
    main()
