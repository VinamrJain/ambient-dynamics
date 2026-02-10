"""Divergence verification for RFF GP Field.

Verifies that 3D streamfunction-based fields are divergence-free using JAX autodiff.
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


def compute_divergence_at_point(field: RFFGPField, x: float, y: float, z: float) -> float:
    """Compute divergence div(u,v) = du/dx + dv/dy at a point using autodiff."""
    def u_func(x, y, z):
        u, _ = field.velocity_at_point(x, y, z)
        return u
    
    def v_func(x, y, z):
        _, v = field.velocity_at_point(x, y, z)
        return v
    
    du_dx = jax.grad(u_func, argnums=0)(x, y, z)
    dv_dy = jax.grad(v_func, argnums=1)(x, y, z)
    return float(du_dx + dv_dy)


def compute_divergence_grid(field: RFFGPField, n_samples: int = 20) -> tuple:
    """Compute divergence on a grid of points."""
    # Sample points within the grid bounds
    x_vals = np.linspace(1.5, field.config.n_x - 0.5, n_samples)
    y_vals = np.linspace(1.5, field.config.n_y - 0.5, n_samples)
    z_val = field.config.n_z / 2  # Middle z-level
    
    X, Y = np.meshgrid(x_vals, y_vals)
    div_grid = np.zeros_like(X)
    
    for i in range(n_samples):
        for j in range(n_samples):
            div_grid[i, j] = compute_divergence_at_point(field, X[i,j], Y[i,j], z_val)
    
    return X, Y, div_grid


def visualize_divergence_single_field():
    """Visualize divergence of a single field realization."""
    print("=" * 70)
    print("DIVERGENCE VERIFICATION - Single Field")
    print("=" * 70)
    
    # Create 3D field
    config = GridConfig.create(n_x=15, n_y=15, n_z=8)
    field = RFFGPField(
        config, d_max=2,
        sigma=1.0, lengthscale=3.0, nu=2.5, num_features=500
    )
    field.reset(jax.random.PRNGKey(42))
    
    print(f"\nGrid: {config.n_x} x {config.n_y} x {config.n_z}")
    print(f"Parameters: sigma=1.0, lengthscale=3.0, nu=2.5, L=500")
    
    # Compute divergence grid
    print("\nComputing divergence field (using JAX autodiff)...")
    X, Y, div_grid = compute_divergence_grid(field, n_samples=25)
    
    # Get velocity field for visualization
    mean_field = field.get_mean_displacement_field()
    z_idx = config.n_z // 2
    u_field = mean_field[:, :, z_idx, 0]
    v_field = mean_field[:, :, z_idx, 1]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Velocity magnitude
    speed = np.sqrt(u_field**2 + v_field**2)
    im0 = axes[0].imshow(speed.T, origin='lower', cmap='viridis', 
                         extent=[1, config.n_x, 1, config.n_y])
    skip = 2
    x_coords = np.arange(1, config.n_x + 1)
    y_coords = np.arange(1, config.n_y + 1)
    X_vel, Y_vel = np.meshgrid(x_coords, y_coords)
    axes[0].quiver(X_vel[::skip, ::skip], Y_vel[::skip, ::skip],
                   u_field.T[::skip, ::skip], v_field.T[::skip, ::skip],
                   color='white', alpha=0.7, scale=20)
    axes[0].set_title(f'Velocity Field at z={z_idx+1}', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], label='Speed')
    
    # Divergence field
    vmax = max(abs(div_grid.min()), abs(div_grid.max()))
    im1 = axes[1].contourf(X, Y, div_grid, levels=20, cmap='RdBu_r', 
                           vmin=-vmax, vmax=vmax)
    axes[1].set_title(f'Divergence Field\n(max|div|={np.abs(div_grid).max():.2e})', fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1], label='Divergence')
    
    # Divergence histogram
    axes[2].hist(div_grid.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[2].axvline(0, color='red', linestyle='--', linewidth=2, label='Expected (0)')
    axes[2].set_xlabel('Divergence value')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Divergence Distribution', fontsize=12)
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "divergence_single.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved to: {output_path}")
    
    plt.show()
    
    # Statistics
    print(f"\nDivergence Statistics:")
    print(f"  Max absolute: {np.abs(div_grid).max():.2e}")
    print(f"  Mean absolute: {np.abs(div_grid).mean():.2e}")
    print(f"  Std: {div_grid.std():.2e}")
    print(f"\n  (Values should be ~1e-6 to 1e-8 for numerical precision)")


def visualize_divergence_multiple_realizations():
    """Check divergence across multiple field realizations."""
    print("\n" + "=" * 70)
    print("DIVERGENCE VERIFICATION - Multiple Realizations")
    print("=" * 70)
    
    config = GridConfig.create(n_x=12, n_y=12, n_z=6)
    field = RFFGPField(
        config, d_max=2,
        sigma=1.0, lengthscale=3.0, nu=2.5, num_features=500
    )
    
    n_realizations = 20
    n_test_points = 50
    
    print(f"\nTesting {n_realizations} field realizations")
    print(f"Sampling {n_test_points} random points per realization")
    
    all_divergences = []
    
    for seed in range(n_realizations):
        field.reset(jax.random.PRNGKey(seed))
        
        # Random test points
        rng = np.random.default_rng(seed)
        x_pts = rng.uniform(1.5, config.n_x - 0.5, n_test_points)
        y_pts = rng.uniform(1.5, config.n_y - 0.5, n_test_points)
        z_pts = rng.uniform(1.5, config.n_z - 0.5, n_test_points)
        
        for x, y, z in zip(x_pts, y_pts, z_pts):
            div = compute_divergence_at_point(field, x, y, z)
            all_divergences.append(div)
        
        if (seed + 1) % 5 == 0:
            print(f"  Completed {seed + 1}/{n_realizations} realizations")
    
    all_divergences = np.array(all_divergences)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_divergences, bins=50, edgecolor='black', alpha=0.7, density=True)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Expected (0)')
    ax.set_xlabel('Divergence value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Divergence Distribution Across {n_realizations} Field Realizations\n'
                 f'({n_test_points} points each, total {len(all_divergences)} samples)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add statistics box
    stats_text = (f'Max |div|: {np.abs(all_divergences).max():.2e}\n'
                  f'Mean |div|: {np.abs(all_divergences).mean():.2e}\n'
                  f'Std: {all_divergences.std():.2e}')
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "divergence_multiple.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved to: {output_path}")
    
    plt.show()
    
    print(f"\nOverall Divergence Statistics:")
    print(f"  Max absolute: {np.abs(all_divergences).max():.2e}")
    print(f"  Mean absolute: {np.abs(all_divergences).mean():.2e}")
    print(f"  Std: {all_divergences.std():.2e}")


def compare_analytical_vs_finite_diff():
    """Compare analytical (autodiff) divergence with finite differences."""
    print("\n" + "=" * 70)
    print("ANALYTICAL vs FINITE DIFFERENCE DIVERGENCE")
    print("=" * 70)
    
    config = GridConfig.create(n_x=20, n_y=20, n_z=8)
    field = RFFGPField(
        config, d_max=2,
        sigma=1.0, lengthscale=3.0, nu=2.5, num_features=500
    )
    field.reset(jax.random.PRNGKey(42))
    
    # Get field on grid
    mean_field = field.get_mean_displacement_field()
    z_idx = config.n_z // 2
    u_field = mean_field[:, :, z_idx, 0]
    v_field = mean_field[:, :, z_idx, 1]
    
    # Finite difference divergence (du/dx + dv/dy)
    dx, dy = 1.0, 1.0
    du_dx_fd = np.gradient(u_field, dx, axis=0)
    dv_dy_fd = np.gradient(v_field, dy, axis=1)
    div_fd = du_dx_fd + dv_dy_fd
    
    # Analytical (autodiff) divergence at grid points
    div_analytical = np.zeros((config.n_x, config.n_y))
    print("\nComputing analytical divergence at grid points...")
    for i in range(config.n_x):
        for j in range(config.n_y):
            div_analytical[i, j] = compute_divergence_at_point(
                field, float(i + 1), float(j + 1), float(z_idx + 1)
            )
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    vmax_fd = max(abs(div_fd.min()), abs(div_fd.max()))
    vmax_analytical = max(abs(div_analytical.min()), abs(div_analytical.max()))
    
    im0 = axes[0].imshow(div_fd.T, origin='lower', cmap='RdBu_r',
                         vmin=-vmax_fd, vmax=vmax_fd)
    axes[0].set_title(f'Finite Difference Divergence\n(max|div|={vmax_fd:.2e})', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(div_analytical.T, origin='lower', cmap='RdBu_r',
                         vmin=-vmax_analytical, vmax=vmax_analytical)
    axes[1].set_title(f'Analytical (Autodiff) Divergence\n(max|div|={vmax_analytical:.2e})', fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])
    
    # Scatter comparison
    axes[2].scatter(div_analytical.flatten(), div_fd.flatten(), alpha=0.5, s=10)
    axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[2].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Analytical divergence', fontsize=11)
    axes[2].set_ylabel('Finite difference divergence', fontsize=11)
    axes[2].set_title('Comparison', fontsize=12)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "divergence_comparison.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved to: {output_path}")
    
    plt.show()
    
    print(f"\nComparison:")
    print(f"  Analytical max |div|: {np.abs(div_analytical).max():.2e} (should be ~0)")
    print(f"  Finite diff max |div|: {np.abs(div_fd).max():.2e} (discretization error)")
    print(f"\n  Analytical divergence is ~0 because streamfunction method")
    print(f"  guarantees div-free by construction (du/dx + dv/dy = 0 exactly)")


if __name__ == "__main__":
    visualize_divergence_single_field()
    visualize_divergence_multiple_realizations()
    compare_analytical_vs_finite_diff()
