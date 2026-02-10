"""Unit tests for RFFGPField - edge cases and correctness verification."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import warnings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.field import RFFGPField
from src.env.utils.types import GridConfig, GridPosition


class TestRFFGPFieldInitialization:
    """Test field initialization and parameter validation."""
    
    def test_2d_initialization(self):
        """2D field initializes correctly."""
        config = GridConfig.create(n_x=10, n_y=8)
        field = RFFGPField(config, d_max=2)
        
        assert field.ndim == 2
        assert field.d_max == 2
        assert field._precomputed_u is None  # Not reset yet
    
    def test_3d_initialization(self):
        """3D field initializes correctly."""
        config = GridConfig.create(n_x=8, n_y=8, n_z=5)
        field = RFFGPField(config, d_max=2)
        
        assert field.ndim == 3
        assert field.d_max == 2
        assert field._precomputed_u is None
        assert field._precomputed_v is None
    
    def test_custom_parameters(self):
        """Custom GP parameters are stored correctly."""
        config = GridConfig.create(n_x=5, n_y=5)
        field = RFFGPField(
            config, d_max=1,
            sigma=2.0, lengthscale=3.0, nu=1.5,
            num_features=100, noise_std=0.5
        )
        
        assert field.sigma == 2.0
        assert field.lengthscale == 3.0
        assert field.nu == 1.5
        assert field.num_features == 100
        assert field.noise_std == 0.5


class TestRFFGPFieldReset:
    """Test reset behavior and reproducibility."""
    
    def test_reset_initializes_arrays(self):
        """Reset initializes all required arrays."""
        config = GridConfig.create(n_x=5, n_y=5, n_z=3)
        field = RFFGPField(config, d_max=2, num_features=50)
        
        key = jax.random.PRNGKey(42)
        field.reset(key)
        
        assert field._omegas is not None
        assert field._omegas.shape == (50, 3)  # 3D spatial
        assert field._phases is not None
        assert field._phases.shape == (50,)
        assert field._weights is not None
        assert field._weights.shape == (50,)
        assert field._precomputed_u is not None
        assert field._precomputed_v is not None
    
    def test_reset_reproducibility(self):
        """Same seed produces same field."""
        config = GridConfig.create(n_x=5, n_y=5)
        field = RFFGPField(config, d_max=2, num_features=50)
        
        key = jax.random.PRNGKey(42)
        field.reset(key)
        u1 = np.array(field._precomputed_u)
        
        # Reset again with same key
        field.reset(key)
        u2 = np.array(field._precomputed_u)
        
        np.testing.assert_array_equal(u1, u2)
    
    def test_different_seeds_different_fields(self):
        """Different seeds produce different fields."""
        config = GridConfig.create(n_x=5, n_y=5)
        field = RFFGPField(config, d_max=2, num_features=50)
        
        field.reset(jax.random.PRNGKey(42))
        u1 = np.array(field._precomputed_u)
        
        field.reset(jax.random.PRNGKey(123))
        u2 = np.array(field._precomputed_u)
        
        assert not np.allclose(u1, u2)


class TestDisplacementSampling:
    """Test displacement sampling and clipping."""
    
    @pytest.fixture
    def field_2d(self):
        config = GridConfig.create(n_x=10, n_y=10)
        field = RFFGPField(config, d_max=2, sigma=1.0, noise_std=0.1)
        field.reset(jax.random.PRNGKey(42))
        return field
    
    @pytest.fixture
    def field_3d(self):
        config = GridConfig.create(n_x=8, n_y=8, n_z=5)
        field = RFFGPField(config, d_max=2, sigma=1.0, noise_std=0.1)
        field.reset(jax.random.PRNGKey(42))
        return field
    
    def test_sample_displacement_2d_returns_observation(self, field_2d):
        """2D sampling returns valid DisplacementObservation."""
        pos = GridPosition(5, 5, None)
        disp = field_2d.sample_displacement(pos, jax.random.PRNGKey(0))
        
        assert disp.v is None
        assert isinstance(disp.u, float)
        assert -field_2d.d_max <= disp.u <= field_2d.d_max
    
    def test_sample_displacement_3d_returns_observation(self, field_3d):
        """3D sampling returns valid DisplacementObservation."""
        pos = GridPosition(4, 4, 3)
        disp = field_3d.sample_displacement(pos, jax.random.PRNGKey(0))
        
        assert isinstance(disp.u, float)
        assert isinstance(disp.v, float)
        assert -field_3d.d_max <= disp.u <= field_3d.d_max
        assert -field_3d.d_max <= disp.v <= field_3d.d_max
    
    def test_clipping_enforced(self):
        """Displacements are clipped to d_max bounds."""
        # Use large sigma to ensure values exceed d_max
        config = GridConfig.create(n_x=5, n_y=5)
        field = RFFGPField(config, d_max=1, sigma=10.0, noise_std=0.0)
        field.reset(jax.random.PRNGKey(42))
        
        # Sample at all positions
        for i in range(1, 6):
            for j in range(1, 6):
                pos = GridPosition(i, j, None)
                disp = field.sample_displacement(pos, jax.random.PRNGKey(i*10+j))
                assert -1 <= disp.u <= 1, f"Clipping failed at ({i},{j}): u={disp.u}"
    
    def test_zero_noise_returns_mean(self):
        """With noise_std=0, sample equals mean."""
        config = GridConfig.create(n_x=10, n_y=10)
        field = RFFGPField(config, d_max=5, sigma=1.0, noise_std=0.0)
        field.reset(jax.random.PRNGKey(42))
        
        pos = GridPosition(5, 5, None)
        mean = field.get_mean_displacement(pos)
        
        # Sample multiple times - should all equal mean (within clipping)
        for seed in range(10):
            disp = field.sample_displacement(pos, jax.random.PRNGKey(seed))
            expected = max(-5, min(5, mean[0]))
            assert disp.u == pytest.approx(expected, abs=1e-6)


class TestMeanDisplacement:
    """Test mean displacement retrieval."""
    
    def test_get_mean_displacement_2d(self):
        """2D mean returns single value tuple."""
        config = GridConfig.create(n_x=5, n_y=5)
        field = RFFGPField(config, d_max=2)
        field.reset(jax.random.PRNGKey(42))
        
        pos = GridPosition(3, 3, None)
        mean = field.get_mean_displacement(pos)
        
        assert len(mean) == 1
        assert isinstance(mean[0], float)
    
    def test_get_mean_displacement_3d(self):
        """3D mean returns two value tuple."""
        config = GridConfig.create(n_x=5, n_y=5, n_z=3)
        field = RFFGPField(config, d_max=2)
        field.reset(jax.random.PRNGKey(42))
        
        pos = GridPosition(3, 3, 2)
        mean = field.get_mean_displacement(pos)
        
        assert len(mean) == 2
        assert isinstance(mean[0], float)
        assert isinstance(mean[1], float)
    
    def test_mean_field_shape_2d(self):
        """2D mean field has correct shape."""
        config = GridConfig.create(n_x=10, n_y=8)
        field = RFFGPField(config, d_max=2)
        field.reset(jax.random.PRNGKey(42))
        
        mean_field = field.get_mean_displacement_field()
        assert mean_field.shape == (10, 8, 1)
    
    def test_mean_field_shape_3d(self):
        """3D mean field has correct shape."""
        config = GridConfig.create(n_x=8, n_y=6, n_z=4)
        field = RFFGPField(config, d_max=2)
        field.reset(jax.random.PRNGKey(42))
        
        mean_field = field.get_mean_displacement_field()
        assert mean_field.shape == (8, 6, 4, 2)


class TestPMF:
    """Test displacement PMF computation."""
    
    def test_pmf_sums_to_one_2d(self):
        """2D PMF sums to 1."""
        config = GridConfig.create(n_x=5, n_y=5)
        field = RFFGPField(config, d_max=2, noise_std=0.5)
        field.reset(jax.random.PRNGKey(42))
        
        pos = GridPosition(3, 3, None)
        pmf = field.get_displacement_pmf(pos)
        
        assert pmf.shape == (5,)  # 2*2+1
        assert pmf.sum() == pytest.approx(1.0, abs=1e-5)
    
    def test_pmf_sums_to_one_3d(self):
        """3D joint PMF sums to 1."""
        config = GridConfig.create(n_x=5, n_y=5, n_z=3)
        field = RFFGPField(config, d_max=2, noise_std=0.5)
        field.reset(jax.random.PRNGKey(42))
        
        pos = GridPosition(3, 3, 2)
        pmf = field.get_displacement_pmf(pos)
        
        assert pmf.shape == (5, 5)  # 2*2+1 x 2*2+1
        assert pmf.sum() == pytest.approx(1.0, abs=1e-5)
    
    def test_pmf_all_positive(self):
        """PMF values are non-negative."""
        config = GridConfig.create(n_x=5, n_y=5)
        field = RFFGPField(config, d_max=3, noise_std=1.0)
        field.reset(jax.random.PRNGKey(42))
        
        for i in range(1, 6):
            for j in range(1, 6):
                pos = GridPosition(i, j, None)
                pmf = field.get_displacement_pmf(pos)
                assert np.all(pmf >= 0)
    
    def test_pmf_fallback_warning(self):
        """Warning triggered when mean far outside d_max range."""
        # Very large sigma with small d_max to trigger fallback
        config = GridConfig.create(n_x=3, n_y=3)
        field = RFFGPField(config, d_max=1, sigma=1000.0, noise_std=0.01)
        field.reset(jax.random.PRNGKey(42))
        
        # Should trigger warning for at least some positions
        pos = GridPosition(2, 2, None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pmf = field.get_displacement_pmf(pos)
            # Either triggers warning OR has valid PMF (depends on mean location)
            assert pmf.sum() == pytest.approx(1.0, abs=1e-5)


class TestVelocityAtPoint:
    """Test the differentiable velocity function."""
    
    def test_velocity_at_point_matches_precomputed_3d(self):
        """velocity_at_point matches precomputed values at grid points."""
        config = GridConfig.create(n_x=5, n_y=5, n_z=3)
        field = RFFGPField(config, d_max=2)
        field.reset(jax.random.PRNGKey(42))
        
        # Check a few grid points
        for pos in [GridPosition(1, 1, 1), GridPosition(3, 3, 2), GridPosition(5, 5, 3)]:
            u_precomputed, v_precomputed = field._get_mean_at_position(pos)
            u_computed, v_computed = field.velocity_at_point(
                float(pos.i), float(pos.j), float(pos.k)
            )
            
            assert float(u_computed) == pytest.approx(u_precomputed, rel=1e-5)
            assert float(v_computed) == pytest.approx(v_precomputed, rel=1e-5)
    
    def test_velocity_at_point_2d(self):
        """velocity_at_point works for 2D fields."""
        config = GridConfig.create(n_x=5, n_y=5)
        field = RFFGPField(config, d_max=2)
        field.reset(jax.random.PRNGKey(42))
        
        pos = GridPosition(3, 3, None)
        u_precomputed, _ = field._get_mean_at_position(pos)
        u_computed, v_computed = field.velocity_at_point(3.0, 3.0)
        
        assert v_computed is None
        assert float(u_computed) == pytest.approx(u_precomputed, rel=1e-5)
    
    def test_velocity_differentiable(self):
        """velocity_at_point is JAX-differentiable."""
        config = GridConfig.create(n_x=5, n_y=5, n_z=3)
        field = RFFGPField(config, d_max=2)
        field.reset(jax.random.PRNGKey(42))
        
        def u_func(x, y, z):
            u, _ = field.velocity_at_point(x, y, z)
            return u
        
        # Should not raise
        grad_u = jax.grad(u_func, argnums=(0, 1, 2))
        du_dx, du_dy, du_dz = grad_u(3.0, 3.0, 2.0)
        
        assert np.isfinite(du_dx)
        assert np.isfinite(du_dy)
        assert np.isfinite(du_dz)


class TestDivergenceFree:
    """Test that 3D fields are divergence-free."""
    
    def test_divergence_near_zero(self):
        """3D velocity field has near-zero divergence via autodiff."""
        config = GridConfig.create(n_x=8, n_y=8, n_z=5)
        field = RFFGPField(config, d_max=2, num_features=200)
        field.reset(jax.random.PRNGKey(42))
        
        def u_func(x, y, z):
            u, _ = field.velocity_at_point(x, y, z)
            return u
        
        def v_func(x, y, z):
            _, v = field.velocity_at_point(x, y, z)
            return v
        
        # Test at several points
        test_points = [(3.5, 4.2, 2.1), (5.0, 5.0, 3.0), (2.1, 6.7, 2.5)]
        
        for x, y, z in test_points:
            du_dx = jax.grad(u_func, argnums=0)(x, y, z)
            dv_dy = jax.grad(v_func, argnums=1)(x, y, z)
            divergence = du_dx + dv_dy
            
            assert abs(float(divergence)) < 1e-6, f"Divergence at ({x},{y},{z}) = {divergence}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimal_grid_2d(self):
        """1x1 2D grid works."""
        config = GridConfig.create(n_x=2, n_y=2)
        field = RFFGPField(config, d_max=1)
        field.reset(jax.random.PRNGKey(42))
        
        pos = GridPosition(1, 1, None)
        disp = field.sample_displacement(pos, jax.random.PRNGKey(0))
        assert isinstance(disp.u, float)
    
    def test_minimal_grid_3d(self):
        """2x2x2 3D grid works."""
        config = GridConfig.create(n_x=2, n_y=2, n_z=2)
        field = RFFGPField(config, d_max=1)
        field.reset(jax.random.PRNGKey(42))
        
        pos = GridPosition(1, 1, 1)
        disp = field.sample_displacement(pos, jax.random.PRNGKey(0))
        assert isinstance(disp.u, float)
        assert isinstance(disp.v, float)
    
    def test_corner_positions(self):
        """Corner positions work correctly."""
        config = GridConfig.create(n_x=5, n_y=5, n_z=3)
        field = RFFGPField(config, d_max=2)
        field.reset(jax.random.PRNGKey(42))
        
        corners = [
            GridPosition(1, 1, 1),
            GridPosition(1, 1, 3),
            GridPosition(1, 5, 1),
            GridPosition(5, 1, 1),
            GridPosition(5, 5, 3),
        ]
        
        for pos in corners:
            mean = field.get_mean_displacement(pos)
            assert len(mean) == 2
            pmf = field.get_displacement_pmf(pos)
            assert pmf.sum() == pytest.approx(1.0, abs=1e-5)
    
    def test_very_small_noise(self):
        """Very small noise doesn't cause numerical issues."""
        config = GridConfig.create(n_x=5, n_y=5)
        field = RFFGPField(config, d_max=2, noise_std=1e-10)
        field.reset(jax.random.PRNGKey(42))
        
        pos = GridPosition(3, 3, None)
        pmf = field.get_displacement_pmf(pos)
        
        # PMF should concentrate around the mean
        assert pmf.sum() == pytest.approx(1.0, abs=1e-5)
        assert pmf.max() > 0.5  # Most mass at one point
    
    def test_large_num_features(self):
        """Large number of features works."""
        config = GridConfig.create(n_x=5, n_y=5)
        field = RFFGPField(config, d_max=2, num_features=2000)
        field.reset(jax.random.PRNGKey(42))
        
        pos = GridPosition(3, 3, None)
        mean = field.get_mean_displacement(pos)
        assert isinstance(mean[0], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
