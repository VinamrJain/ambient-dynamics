"""Statistical comparison of RFF GP Field with Cholesky sampling.

Compares:
1. Mean and variance across samples
2. Covariance structure
3. Visual field comparison
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import norm
import jax
import jax.numpy as jnp

from src.env.field import RFFGPField
from src.env.utils.types import GridConfig


def matern_kernel(r: np.ndarray, r_prime: np.ndarray, 
                  sigma: float = 1.0, lengthscale: float = 1.0, nu: float = 2.5) -> np.ndarray:
    """Compute Matern covariance matrix.
    
    Supports nu = 0.5, 1.5, 2.5, inf (RBF).
    """
    dists = cdist(r, r_prime, metric='euclidean')
    tau = dists / lengthscale
    
    if nu == 0.5:
        # Matern-1/2 (exponential)
        return sigma**2 * np.exp(-tau)
    elif nu == 1.5:
        # Matern-3/2
        sqrt3_tau = np.sqrt(3) * tau
        return sigma**2 * (1 + sqrt3_tau) * np.exp(-sqrt3_tau)
    elif nu == 2.5:
        # Matern-5/2
        sqrt5_tau = np.sqrt(5) * tau
        return sigma**2 * (1 + sqrt5_tau + (5 * tau**2) / 3) * np.exp(-sqrt5_tau)
    elif nu == np.inf:
        # RBF (squared exponential)
        return sigma**2 * np.exp(-0.5 * tau**2)
    else:
        raise ValueError(f"Unsupported nu={nu}. Use 0.5, 1.5, 2.5, or inf")


class CholeskyGPSampler:
    """Exact GP sampling via Cholesky decomposition for comparison."""
    
    def __init__(self, locations: np.ndarray, sigma: float = 1.0, 
                 lengthscale: float = 1.0, nu: float = 2.5, jitter: float = 1e-6):
        self.locations = locations
        self.sigma = sigma
        self.lengthscale = lengthscale
        self.nu = nu
        
        # Compute Cholesky factor
        K = matern_kernel(locations, locations, sigma, lengthscale, nu)
        K += jitter * np.eye(len(locations))
        self.L_chol = np.linalg.cholesky(K)
        
        self.psi = None
    
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a sample."""
        z = rng.standard_normal(len(self.locations))
        self.psi = self.L_chol @ z
        return self.psi


def compare_mean_variance(sigma: float = 1.0, lengthscale: float = 1.0, nu: float = 2.5):
    """Compare mean and variance statistics between RFF and Cholesky."""
    print("=" * 70)
    print(f"MEAN/VARIANCE COMPARISON (sigma={sigma}, lengthscale={lengthscale}, nu={nu})")
    print("=" * 70)
    
    # Create 2D grid (for Cholesky to be tractable)
    nx, ny = 15, 15
    config = GridConfig.create(n_x=nx, n_y=ny)
    
    # Grid locations for Cholesky
    x_coords = np.arange(1, nx + 1)
    y_coords = np.arange(1, ny + 1)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    locations = np.column_stack([X.ravel(), Y.ravel()])
    
    # Create samplers
    rff_field = RFFGPField(config, d_max=5, sigma=sigma, lengthscale=lengthscale, 
                           nu=nu, num_features=500, noise_std=0.0)
    chol_sampler = CholeskyGPSampler(locations, sigma, lengthscale, nu)
    
    # Generate samples
    M = 2000
    print(f"\nGenerating {M} samples from each method...")
    
    rff_samples = np.zeros((M, nx * ny))
    chol_samples = np.zeros((M, nx * ny))
    
    for i in range(M):
        # RFF sample
        rff_field.reset(jax.random.PRNGKey(i))
        rff_samples[i] = rff_field._precomputed_u.ravel()
        
        # Cholesky sample
        rng = np.random.default_rng(i + 10000)
        chol_samples[i] = chol_sampler.sample(rng)
        
        if (i + 1) % 500 == 0:
            print(f"  Completed {i + 1}/{M} samples")
    
    # Compute statistics
    rff_means = rff_samples.mean(axis=0)
    rff_vars = rff_samples.var(axis=0, ddof=1)
    chol_means = chol_samples.mean(axis=0)
    chol_vars = chol_samples.var(axis=0, ddof=1)
    
    expected_mean_std = sigma / np.sqrt(M)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram of means
    axes[0, 0].hist(chol_means, bins=40, alpha=0.6, label='Cholesky', density=True)
    axes[0, 0].hist(rff_means, bins=40, alpha=0.6, label='RFF', density=True)
    axes[0, 0].axvline(0, color='k', ls='--', lw=2, label='Target mean=0')
    x_mean = np.linspace(min(rff_means.min(), chol_means.min()), 
                         max(rff_means.max(), chol_means.max()), 100)
    axes[0, 0].plot(x_mean, norm.pdf(x_mean, 0, expected_mean_std), 
                    'r-', lw=2, label=f'Expected: N(0, {expected_mean_std:.3f}²)')
    axes[0, 0].set_xlabel('Sample mean at grid point', fontsize=11)
    axes[0, 0].set_ylabel('Density', fontsize=11)
    axes[0, 0].set_title('Distribution of Means', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Histogram of variances
    axes[0, 1].hist(chol_vars, bins=40, alpha=0.6, label='Cholesky', density=True)
    axes[0, 1].hist(rff_vars, bins=40, alpha=0.6, label='RFF', density=True)
    axes[0, 1].axvline(sigma**2, color='k', ls='--', lw=2, label=f'Target σ²={sigma**2}')
    axes[0, 1].set_xlabel('Sample variance at grid point', fontsize=11)
    axes[0, 1].set_ylabel('Density', fontsize=11)
    axes[0, 1].set_title('Distribution of Variances', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Mean field visualization
    im_rff = axes[1, 0].imshow(rff_means.reshape(nx, ny).T, origin='lower', 
                                cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    axes[1, 0].set_title(f'RFF Mean Field\n(should be ~0 everywhere)', fontsize=12)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im_rff, ax=axes[1, 0])
    
    # Variance field visualization
    vmin_var = min(rff_vars.min(), chol_vars.min())
    vmax_var = max(rff_vars.max(), chol_vars.max())
    im_var = axes[1, 1].imshow(rff_vars.reshape(nx, ny).T, origin='lower', 
                                cmap='viridis', vmin=vmin_var, vmax=vmax_var)
    axes[1, 1].set_title(f'RFF Variance Field\n(should be ~{sigma**2} everywhere)', fontsize=12)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im_var, ax=axes[1, 1])
    
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"mean_var_comparison_l{lengthscale}_nu{nu}.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved to: {output_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\nStatistics (averaged across {nx*ny} grid points):")
    print(f"  Method   | Mean of means | Std of means | Mean of vars | Target var")
    print(f"  ---------|---------------|--------------|--------------|------------")
    print(f"  RFF      | {rff_means.mean():>13.4f} | {rff_means.std():>12.4f} | {rff_vars.mean():>12.4f} | {sigma**2}")
    print(f"  Cholesky | {chol_means.mean():>13.4f} | {chol_means.std():>12.4f} | {chol_vars.mean():>12.4f} | {sigma**2}")
    print(f"  Expected | {0.0:>13.4f} | {expected_mean_std:>12.4f} | {sigma**2:>12.4f} |")


def compare_covariance_structure(sigma: float = 1.0, lengthscale: float = 1.0, nu: float = 2.5):
    """Compare empirical covariance with theoretical kernel."""
    print("\n" + "=" * 70)
    print(f"COVARIANCE STRUCTURE COMPARISON (sigma={sigma}, lengthscale={lengthscale}, nu={nu})")
    print("=" * 70)
    
    # Smaller grid for covariance matrix
    nx, ny = 12, 12
    config = GridConfig.create(n_x=nx, n_y=ny)
    
    # Grid locations
    x_coords = np.arange(1, nx + 1)
    y_coords = np.arange(1, ny + 1)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    locations = np.column_stack([X.ravel(), Y.ravel()])
    
    # Create samplers
    rff_field = RFFGPField(config, d_max=5, sigma=sigma, lengthscale=lengthscale,
                           nu=nu, num_features=500, noise_std=0.0)
    chol_sampler = CholeskyGPSampler(locations, sigma, lengthscale, nu)
    
    # Generate samples
    M = 3000
    print(f"\nGenerating {M} samples...")
    
    rff_samples = np.zeros((M, nx * ny))
    chol_samples = np.zeros((M, nx * ny))
    
    for i in range(M):
        rff_field.reset(jax.random.PRNGKey(i))
        rff_samples[i] = rff_field._precomputed_u.ravel()
        
        rng = np.random.default_rng(i + 10000)
        chol_samples[i] = chol_sampler.sample(rng)
    
    # Compute empirical covariances
    print("Computing empirical covariance matrices...")
    rff_centered = rff_samples - rff_samples.mean(axis=0, keepdims=True)
    chol_centered = chol_samples - chol_samples.mean(axis=0, keepdims=True)
    
    K_emp_rff = (rff_centered.T @ rff_centered) / (M - 1)
    K_emp_chol = (chol_centered.T @ chol_centered) / (M - 1)
    
    # Theoretical covariance
    K_theory = matern_kernel(locations, locations, sigma, lengthscale, nu)
    
    # Errors
    error_rff = K_emp_rff - K_theory
    error_chol = K_emp_chol - K_theory
    frob_rff = np.linalg.norm(error_rff, 'fro') / np.linalg.norm(K_theory, 'fro')
    frob_chol = np.linalg.norm(error_chol, 'fro') / np.linalg.norm(K_theory, 'fro')
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    vmin = min(K_theory.min(), K_emp_rff.min(), K_emp_chol.min())
    vmax = max(K_theory.max(), K_emp_rff.max(), K_emp_chol.max())
    
    im0 = axes[0, 0].imshow(K_theory, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Theoretical Covariance K', fontsize=11)
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(K_emp_rff, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'RFF Empirical Covariance', fontsize=11)
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(K_emp_chol, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'Cholesky Empirical Covariance', fontsize=11)
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Error matrices
    error_max = max(np.abs(error_rff).max(), np.abs(error_chol).max())
    im3 = axes[1, 0].imshow(np.abs(error_rff), cmap='hot', vmin=0, vmax=error_max)
    axes[1, 0].set_title(f'RFF Abs Error\n(rel Frob={frob_rff:.4f})', fontsize=11)
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(np.abs(error_chol), cmap='hot', vmin=0, vmax=error_max)
    axes[1, 1].set_title(f'Cholesky Abs Error\n(rel Frob={frob_chol:.4f})', fontsize=11)
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Scatter comparison
    sample_idx = np.random.choice(len(locations), size=min(80, len(locations)), replace=False)
    K_sample_theory = K_theory[np.ix_(sample_idx, sample_idx)]
    K_sample_rff = K_emp_rff[np.ix_(sample_idx, sample_idx)]
    K_sample_chol = K_emp_chol[np.ix_(sample_idx, sample_idx)]
    
    axes[1, 2].scatter(K_sample_theory.ravel(), K_sample_rff.ravel(), 
                       alpha=0.3, s=10, label='RFF')
    axes[1, 2].scatter(K_sample_theory.ravel(), K_sample_chol.ravel(), 
                       alpha=0.3, s=10, label='Cholesky')
    axes[1, 2].plot([vmin, vmax], [vmin, vmax], 'k--', lw=2, alpha=0.5, label='y=x')
    axes[1, 2].set_xlabel('Theoretical K[i,j]', fontsize=11)
    axes[1, 2].set_ylabel('Empirical K[i,j]', fontsize=11)
    axes[1, 2].set_title('Covariance Entry Comparison', fontsize=11)
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"cov_comparison_l{lengthscale}_nu{nu}.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved to: {output_path}")
    
    plt.show()
    
    print(f"\nRelative Frobenius norm of covariance error:")
    print(f"  RFF:      {frob_rff:.4f}")
    print(f"  Cholesky: {frob_chol:.4f}")
    print(f"  Expected: O(1/√M) = {1/np.sqrt(M):.4f}")


def compare_single_samples(sigma: float = 1.0, lengthscale: float = 2.0, nu: float = 2.5):
    """Visual comparison of single sample realizations."""
    print("\n" + "=" * 70)
    print(f"SINGLE SAMPLE COMPARISON (sigma={sigma}, lengthscale={lengthscale}, nu={nu})")
    print("=" * 70)
    
    nx, ny = 25, 25
    config = GridConfig.create(n_x=nx, n_y=ny)
    
    # Grid locations
    x_coords = np.arange(1, nx + 1)
    y_coords = np.arange(1, ny + 1)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    locations = np.column_stack([X.ravel(), Y.ravel()])
    
    # Create samplers
    rff_field = RFFGPField(config, d_max=5, sigma=sigma, lengthscale=lengthscale,
                           nu=nu, num_features=1000, noise_std=0.0)
    chol_sampler = CholeskyGPSampler(locations, sigma, lengthscale, nu)
    
    # Generate samples with different seeds
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for col, seed in enumerate([42, 123, 456]):
        # RFF sample
        rff_field.reset(jax.random.PRNGKey(seed))
        rff_sample = rff_field._precomputed_u
        
        # Cholesky sample  
        rng = np.random.default_rng(seed)
        chol_sample = chol_sampler.sample(rng).reshape(nx, ny)
        
        vmin = min(rff_sample.min(), chol_sample.min())
        vmax = max(rff_sample.max(), chol_sample.max())
        
        im0 = axes[0, col].imshow(rff_sample.T, origin='lower', cmap='RdBu_r',
                                   vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f'RFF (seed={seed})', fontsize=11)
        axes[0, col].set_xlabel('x')
        if col == 0:
            axes[0, col].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0, col])
        
        im1 = axes[1, col].imshow(chol_sample.T, origin='lower', cmap='RdBu_r',
                                   vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f'Cholesky (seed={seed})', fontsize=11)
        axes[1, col].set_xlabel('x')
        if col == 0:
            axes[1, col].set_ylabel('y')
        plt.colorbar(im1, ax=axes[1, col])
    
    fig.suptitle(f'Sample Realizations (sigma={sigma}, lengthscale={lengthscale}, nu={nu})', 
                 fontsize=13, y=1.02)
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"single_samples_l{lengthscale}_nu{nu}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    
    plt.show()
    
    print("\nNote: Different seeds = different realizations")
    print("      Both should show same spatial correlation structure")


if __name__ == "__main__":
    # Test with different parameters
    print("\n" + "=" * 70)
    print("CHOLESKY COMPARISON - Testing different parameter combinations")
    print("=" * 70)
    
    # Default parameters
    compare_single_samples(sigma=1.0, lengthscale=2.0, nu=2.5)
    compare_mean_variance(sigma=1.0, lengthscale=2.0, nu=2.5)
    compare_covariance_structure(sigma=1.0, lengthscale=2.0, nu=2.5)
    
    # Different lengthscale
    response = input("\nTest with different lengthscale (lengthscale=5.0)? [y/N]: ")
    if response.lower() == 'y':
        compare_single_samples(sigma=1.0, lengthscale=5.0, nu=2.5)
        compare_covariance_structure(sigma=1.0, lengthscale=5.0, nu=2.5)
    
    # Different nu
    response = input("\nTest with different smoothness (nu=1.5)? [y/N]: ")
    if response.lower() == 'y':
        compare_single_samples(sigma=1.0, lengthscale=2.0, nu=1.5)
        compare_covariance_structure(sigma=1.0, lengthscale=2.0, nu=1.5)
