"""Parameter study for RFF GP Field.

Explores effect of different:
- Lengthscales: spatial correlation range
- Sigma: amplitude/variance
- Nu: smoothness parameter
- Number of features: approximation quality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from src.env.field import RFFGPField
from src.env.utils.types import GridConfig


def study_lengthscale_effect():
    """Visualize effect of lengthscale on field correlation."""
    print("=" * 70)
    print("LENGTHSCALE STUDY")
    print("=" * 70)
    
    config = GridConfig.create(n_x=30, n_y=30)
    lengthscales = [1.0, 3.0, 6.0, 12.0]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    seed = 42
    for col, ell in enumerate(lengthscales):
        field = RFFGPField(config, d_max=5, sigma=1.0, lengthscale=ell, 
                          nu=2.5, num_features=500)
        field.reset(jax.random.PRNGKey(seed))
        
        u_field = np.asarray(field._precomputed_u)
        
        # Field visualization
        im0 = axes[0, col].imshow(u_field.T, origin='lower', cmap='RdBu_r',
                                   vmin=-3, vmax=3)
        axes[0, col].set_title(f'lengthscale = {ell}', fontsize=12)
        axes[0, col].set_xlabel('x')
        if col == 0:
            axes[0, col].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0, col], shrink=0.8)
        
        # Histogram of values
        axes[1, col].hist(u_field.flatten(), bins=30, density=True, 
                         edgecolor='black', alpha=0.7)
        axes[1, col].axvline(0, color='red', linestyle='--', lw=2)
        axes[1, col].set_xlabel('Displacement u', fontsize=10)
        if col == 0:
            axes[1, col].set_ylabel('Density', fontsize=10)
        axes[1, col].set_title(f'std={u_field.std():.2f}', fontsize=10)
        axes[1, col].grid(alpha=0.3)
    
    fig.suptitle('Effect of Lengthscale on Field Correlation\n(sigma=1.0, nu=2.5, L=500)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "param_study_lengthscale.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    
    plt.show()
    
    print("\nObservations:")
    print("  - Small lengthscale: rapid spatial variation, fine structure")
    print("  - Large lengthscale: smooth, slowly varying field")
    print("  - Lengthscale ~ grid units where values are correlated")


def study_sigma_effect():
    """Visualize effect of sigma (amplitude) on field magnitude."""
    print("\n" + "=" * 70)
    print("SIGMA (AMPLITUDE) STUDY")
    print("=" * 70)
    
    config = GridConfig.create(n_x=25, n_y=25)
    sigmas = [0.5, 1.0, 2.0, 4.0]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    seed = 42
    for col, sigma in enumerate(sigmas):
        field = RFFGPField(config, d_max=10, sigma=sigma, lengthscale=4.0,
                          nu=2.5, num_features=500)
        field.reset(jax.random.PRNGKey(seed))
        
        u_field = np.asarray(field._precomputed_u)
        
        # Field visualization (scaled to show relative differences)
        vmax = 3 * sigma
        im0 = axes[0, col].imshow(u_field.T, origin='lower', cmap='RdBu_r',
                                   vmin=-vmax, vmax=vmax)
        axes[0, col].set_title(f'sigma = {sigma}', fontsize=12)
        axes[0, col].set_xlabel('x')
        if col == 0:
            axes[0, col].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0, col], shrink=0.8)
        
        # Histogram
        axes[1, col].hist(u_field.flatten(), bins=30, density=True,
                         edgecolor='black', alpha=0.7)
        axes[1, col].axvline(0, color='red', linestyle='--', lw=2)
        axes[1, col].set_xlabel('Displacement u', fontsize=10)
        if col == 0:
            axes[1, col].set_ylabel('Density', fontsize=10)
        axes[1, col].set_title(f'std={u_field.std():.2f} (target={sigma:.1f})', fontsize=10)
        axes[1, col].grid(alpha=0.3)
        axes[1, col].set_xlim(-vmax, vmax)
    
    fig.suptitle('Effect of Sigma on Field Amplitude\n(lengthscale=4.0, nu=2.5, L=500)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "param_study_sigma.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    
    plt.show()
    
    print("\nObservations:")
    print("  - Sigma controls marginal standard deviation (amplitude)")
    print("  - Larger sigma = larger displacement magnitudes")
    print("  - Spatial structure (correlation) remains same")


def study_nu_effect():
    """Visualize effect of nu (smoothness) on field regularity."""
    print("\n" + "=" * 70)
    print("NU (SMOOTHNESS) STUDY")
    print("=" * 70)
    
    config = GridConfig.create(n_x=30, n_y=30)
    nus = [0.5, 1.5, 2.5, 10.0]  # 10.0 approximates RBF
    nu_names = ['nu=0.5 (rough)', 'nu=1.5', 'nu=2.5', 'nu=10 (~RBF)']
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    seed = 42
    for col, (nu, name) in enumerate(zip(nus, nu_names)):
        field = RFFGPField(config, d_max=5, sigma=1.0, lengthscale=4.0,
                          nu=nu, num_features=500)
        field.reset(jax.random.PRNGKey(seed))
        
        u_field = np.asarray(field._precomputed_u)
        
        # Field visualization
        im0 = axes[0, col].imshow(u_field.T, origin='lower', cmap='RdBu_r',
                                   vmin=-3, vmax=3)
        axes[0, col].set_title(name, fontsize=12)
        axes[0, col].set_xlabel('x')
        if col == 0:
            axes[0, col].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0, col], shrink=0.8)
        
        # Cross-section to show smoothness
        mid_idx = config.n_y // 2
        axes[1, col].plot(u_field[:, mid_idx], 'b-', lw=1.5)
        axes[1, col].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1, col].set_xlabel('x', fontsize=10)
        if col == 0:
            axes[1, col].set_ylabel('u(x, y=mid)', fontsize=10)
        axes[1, col].set_title('Cross-section at y=mid', fontsize=10)
        axes[1, col].grid(alpha=0.3)
        axes[1, col].set_ylim(-3.5, 3.5)
    
    fig.suptitle('Effect of Nu on Field Smoothness\n(sigma=1.0, lengthscale=4.0, L=500)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "param_study_nu.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    
    plt.show()
    
    print("\nObservations:")
    print("  - nu=0.5: Exponential kernel, rough/jagged paths")
    print("  - nu=1.5: Once differentiable, some regularity")
    print("  - nu=2.5: Twice differentiable, smooth")
    print("  - nu->inf: RBF kernel, infinitely differentiable, very smooth")


def study_num_features_effect():
    """Visualize effect of number of RFF features on approximation quality."""
    print("\n" + "=" * 70)
    print("NUMBER OF FEATURES STUDY")
    print("=" * 70)
    
    config = GridConfig.create(n_x=25, n_y=25)
    num_features_list = [20, 100, 500, 2000]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    # Use same base seed for fair comparison
    base_seed = 42
    
    for col, L in enumerate(num_features_list):
        field = RFFGPField(config, d_max=5, sigma=1.0, lengthscale=4.0,
                          nu=2.5, num_features=L)
        field.reset(jax.random.PRNGKey(base_seed))
        
        u_field = np.asarray(field._precomputed_u)
        
        # Field visualization
        im0 = axes[0, col].imshow(u_field.T, origin='lower', cmap='RdBu_r',
                                   vmin=-3, vmax=3)
        axes[0, col].set_title(f'L = {L} features', fontsize=12)
        axes[0, col].set_xlabel('x')
        if col == 0:
            axes[0, col].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0, col], shrink=0.8)
        
        # Histogram
        axes[1, col].hist(u_field.flatten(), bins=30, density=True,
                         edgecolor='black', alpha=0.7)
        axes[1, col].axvline(0, color='red', linestyle='--', lw=2)
        axes[1, col].set_xlabel('Displacement u', fontsize=10)
        if col == 0:
            axes[1, col].set_ylabel('Density', fontsize=10)
        axes[1, col].set_title(f'std={u_field.std():.2f}', fontsize=10)
        axes[1, col].grid(alpha=0.3)
        axes[1, col].set_xlim(-3.5, 3.5)
    
    fig.suptitle('Effect of Number of RFF Features\n(sigma=1.0, lengthscale=4.0, nu=2.5)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "param_study_num_features.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    
    plt.show()
    
    print("\nObservations:")
    print("  - L=20: Poor approximation, artifacts visible")
    print("  - L=100: Reasonable approximation")
    print("  - L=500: Good approximation (recommended)")
    print("  - L=2000: Excellent approximation, diminishing returns")


def study_3d_field():
    """Visualize 3D divergence-free field at different z-levels."""
    print("\n" + "=" * 70)
    print("3D FIELD VISUALIZATION")
    print("=" * 70)
    
    config = GridConfig.create(n_x=20, n_y=20, n_z=8)
    field = RFFGPField(config, d_max=3, sigma=1.0, lengthscale=4.0,
                      nu=2.5, num_features=500)
    field.reset(jax.random.PRNGKey(42))
    
    mean_field = field.get_mean_displacement_field()
    
    # Select z-levels to visualize
    z_levels = [0, 2, 4, 7]  # 0-indexed
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    for col, z_idx in enumerate(z_levels):
        u_slice = mean_field[:, :, z_idx, 0]
        v_slice = mean_field[:, :, z_idx, 1]
        speed = np.sqrt(u_slice**2 + v_slice**2)
        
        # Speed colormap with quiver
        im0 = axes[0, col].imshow(speed.T, origin='lower', cmap='viridis',
                                   extent=[1, config.n_x, 1, config.n_y])
        skip = 2
        x_coords = np.arange(1, config.n_x + 1)
        y_coords = np.arange(1, config.n_y + 1)
        X, Y = np.meshgrid(x_coords, y_coords)
        axes[0, col].quiver(X[::skip, ::skip], Y[::skip, ::skip],
                           u_slice.T[::skip, ::skip], v_slice.T[::skip, ::skip],
                           color='white', alpha=0.8, scale=15)
        axes[0, col].set_title(f'z = {z_idx + 1}', fontsize=12)
        axes[0, col].set_xlabel('x')
        if col == 0:
            axes[0, col].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0, col], label='Speed', shrink=0.8)
        
        # Streamplot - needs 1D coordinate arrays
        axes[1, col].streamplot(x_coords, y_coords, u_slice.T, v_slice.T, density=1.5,
                               color=speed.T, cmap='viridis', linewidth=1)
        axes[1, col].set_title(f'Streamlines at z={z_idx + 1}', fontsize=11)
        axes[1, col].set_xlabel('x')
        if col == 0:
            axes[1, col].set_ylabel('y')
        axes[1, col].set_xlim(1, config.n_x)
        axes[1, col].set_ylim(1, config.n_y)
        axes[1, col].set_aspect('equal')
    
    fig.suptitle('3D Divergence-Free Velocity Field at Different Z-Levels\n'
                 '(streamfunction method: u=-dpsi/dy, v=dpsi/dx)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "param_study_3d_field.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    
    plt.show()
    
    print("\nObservations:")
    print("  - Field varies smoothly with z (3D streamfunction)")
    print("  - Streamlines show divergence-free flow (no sources/sinks)")
    print("  - Each z-level has coherent but different flow pattern")


if __name__ == "__main__":
    study_lengthscale_effect()
    study_sigma_effect()
    study_nu_effect()
    study_num_features_effect()
    study_3d_field()
