"""Tests for arena boundary enforcement logic.

Tests _enforce_boundaries_2d and _enforce_boundaries_3d methods in GridArena
for all three boundary modes: clip, periodic, terminal.

Key concerns:
- Off-by-one errors in 1-indexed grids
- Correct handling of ambient vs controllable axes in periodic mode
- Accurate out_of_bounds flag detection
"""

import pytest
import jax.numpy as jnp

from src.env.arena.grid_arena import GridArena
from src.env.field.simple_field import SimpleField
from src.env.actor.grid_actor import GridActor
from src.env.utils.types import GridConfig, GridPosition


# =============================================================================
# Fixtures
# =============================================================================

D_MAX_TEST = 3  # Default d_max for boundary tests


@pytest.fixture
def config_3d():
    """3D grid config: 10x10x5."""
    return GridConfig.create(n_x=10, n_y=10, n_z=5)


@pytest.fixture
def config_2d():
    """2D grid config: 10x8."""
    return GridConfig.create(n_x=10, n_y=8)


def make_arena(config: GridConfig, boundary_mode: str, d_max: int = D_MAX_TEST) -> GridArena:
    """Helper to create a GridArena for testing."""
    field = SimpleField(config, d_max=d_max)
    actor = GridActor(noise_prob=0.0)  # Deterministic for testing
    
    if config.ndim == 3:
        initial_pos = GridPosition(5, 5, 3)
    else:
        initial_pos = GridPosition(5, 4, None)
    
    return GridArena(
        field=field,
        actor=actor,
        config=config,
        initial_position=initial_pos,
        boundary_mode=boundary_mode
    )


# =============================================================================
# 3D Boundary Tests
# =============================================================================

class TestBoundaries3DClip:
    """Test clip mode for 3D grids."""
    
    def test_position_within_bounds_unchanged(self, config_3d):
        """Position within bounds should not be modified."""
        arena = make_arena(config_3d, 'clip')
        pos = GridPosition(5, 5, 3)
        
        new_pos, out_of_bounds = arena._enforce_boundaries_3d(pos)
        
        assert new_pos == pos
        assert out_of_bounds is False
    
    def test_position_at_lower_edge_unchanged(self, config_3d):
        """Position at lower edge (1, 1, 1) stays unchanged."""
        arena = make_arena(config_3d, 'clip')
        pos = GridPosition(1, 1, 1)
        
        new_pos, out_of_bounds = arena._enforce_boundaries_3d(pos)
        
        assert new_pos == pos
        assert out_of_bounds is False
    
    def test_position_at_upper_edge_unchanged(self, config_3d):
        """Position at upper edge (n_x, n_y, n_z) stays unchanged."""
        arena = make_arena(config_3d, 'clip')
        pos = GridPosition(10, 10, 5)  # Max bounds
        
        new_pos, out_of_bounds = arena._enforce_boundaries_3d(pos)
        
        assert new_pos == pos
        assert out_of_bounds is False
    
    def test_clips_below_lower_bound(self, config_3d):
        """Position below lower bound (0 or negative) clips to 1."""
        arena = make_arena(config_3d, 'clip')
        pos = GridPosition(0, -1, 0)
        
        new_pos, out_of_bounds = arena._enforce_boundaries_3d(pos)
        
        assert new_pos == GridPosition(1, 1, 1)
        assert out_of_bounds is True
    
    def test_clips_above_upper_bound(self, config_3d):
        """Position above upper bound clips to max."""
        arena = make_arena(config_3d, 'clip')
        pos = GridPosition(11, 12, 6)  # Beyond 10, 10, 5
        
        new_pos, out_of_bounds = arena._enforce_boundaries_3d(pos)
        
        assert new_pos == GridPosition(10, 10, 5)
        assert out_of_bounds is True
    
    def test_clips_single_axis_violation(self, config_3d):
        """When only one axis violates, only that axis is clipped."""
        arena = make_arena(config_3d, 'clip')
        pos = GridPosition(5, 5, 0)  # Only k violates
        
        new_pos, out_of_bounds = arena._enforce_boundaries_3d(pos)
        
        assert new_pos == GridPosition(5, 5, 1)
        assert out_of_bounds is True


class TestBoundaries3DPeriodic:
    """Test periodic mode for 3D grids."""
    
    def test_ambient_axes_wrap_around_upper(self, config_3d):
        """Ambient axes (i, j) wrap around when exceeding upper bound."""
        arena = make_arena(config_3d, 'periodic')
        pos = GridPosition(11, 11, 3)  # i=11 -> 1, j=11 -> 1
        
        new_pos, out_of_bounds = arena._enforce_boundaries_3d(pos)
        
        assert new_pos.i == 1  # Wrapped
        assert new_pos.j == 1  # Wrapped
        assert new_pos.k == 3  # Unchanged
    
    def test_ambient_axes_wrap_around_lower(self, config_3d):
        """Ambient axes (i, j) wrap around when below lower bound."""
        arena = make_arena(config_3d, 'periodic')
        pos = GridPosition(0, 0, 3)  # i=0 -> 10, j=0 -> 10
        
        new_pos, out_of_bounds = arena._enforce_boundaries_3d(pos)
        
        assert new_pos.i == 10  # Wrapped
        assert new_pos.j == 10  # Wrapped
        assert new_pos.k == 3   # Unchanged
    
    def test_controllable_axis_clips_not_wraps(self, config_3d):
        """Controllable axis (k) clips instead of wrapping."""
        arena = make_arena(config_3d, 'periodic')
        pos = GridPosition(5, 5, 0)  # k=0 should clip to 1
        
        new_pos, out_of_bounds = arena._enforce_boundaries_3d(pos)
        
        assert new_pos.k == 1  # Clipped, not wrapped
        
        # Test upper bound too
        pos = GridPosition(5, 5, 6)  # k=6 should clip to 5
        new_pos, _ = arena._enforce_boundaries_3d(pos)
        assert new_pos.k == 5


class TestBoundaries3DTerminal:
    """Test terminal mode for 3D grids."""
    
    def test_within_bounds_not_terminal(self, config_3d):
        """Position within bounds: out_of_bounds=False, position unchanged."""
        arena = make_arena(config_3d, 'terminal')
        pos = GridPosition(5, 5, 3)
        
        new_pos, out_of_bounds = arena._enforce_boundaries_3d(pos)
        
        assert new_pos == pos
        assert out_of_bounds is False
    
    def test_below_lower_bound_terminal(self, config_3d):
        """Position below lower bound: out_of_bounds=True."""
        arena = make_arena(config_3d, 'terminal')
        pos = GridPosition(0, 5, 3)
        
        new_pos, out_of_bounds = arena._enforce_boundaries_3d(pos)
        
        assert out_of_bounds is True
        # Position not clipped in terminal mode
        assert new_pos == pos
    
    def test_above_upper_bound_terminal(self, config_3d):
        """Position above upper bound: out_of_bounds=True."""
        arena = make_arena(config_3d, 'terminal')
        pos = GridPosition(5, 11, 3)
        
        new_pos, out_of_bounds = arena._enforce_boundaries_3d(pos)
        
        assert out_of_bounds is True


# =============================================================================
# 2D Boundary Tests
# =============================================================================

class TestBoundaries2DClip:
    """Test clip mode for 2D grids."""
    
    def test_position_within_bounds_unchanged(self, config_2d):
        """Position within bounds should not be modified."""
        arena = make_arena(config_2d, 'clip')
        pos = GridPosition(5, 4, None)
        
        new_pos, out_of_bounds = arena._enforce_boundaries_2d(pos)
        
        assert new_pos == pos
        assert out_of_bounds is False
    
    def test_position_at_edges_unchanged(self, config_2d):
        """Position at edges stays unchanged."""
        arena = make_arena(config_2d, 'clip')
        
        # Lower edge
        pos = GridPosition(1, 1, None)
        new_pos, out_of_bounds = arena._enforce_boundaries_2d(pos)
        assert new_pos == pos
        assert out_of_bounds is False
        
        # Upper edge
        pos = GridPosition(10, 8, None)  # n_x=10, n_y=8
        new_pos, out_of_bounds = arena._enforce_boundaries_2d(pos)
        assert new_pos == pos
        assert out_of_bounds is False
    
    def test_clips_below_lower_bound(self, config_2d):
        """Position below lower bound clips to 1."""
        arena = make_arena(config_2d, 'clip')
        pos = GridPosition(0, -1, None)
        
        new_pos, out_of_bounds = arena._enforce_boundaries_2d(pos)
        
        assert new_pos == GridPosition(1, 1, None)
        assert out_of_bounds is True
    
    def test_clips_above_upper_bound(self, config_2d):
        """Position above upper bound clips to max."""
        arena = make_arena(config_2d, 'clip')
        pos = GridPosition(11, 9, None)  # Beyond 10, 8
        
        new_pos, out_of_bounds = arena._enforce_boundaries_2d(pos)
        
        assert new_pos == GridPosition(10, 8, None)
        assert out_of_bounds is True


class TestBoundaries2DPeriodic:
    """Test periodic mode for 2D grids."""
    
    def test_ambient_axis_wraps_around(self, config_2d):
        """Ambient axis (i) wraps around."""
        arena = make_arena(config_2d, 'periodic')
        
        # Upper wrap
        pos = GridPosition(11, 4, None)  # i=11 -> 1
        new_pos, _ = arena._enforce_boundaries_2d(pos)
        assert new_pos.i == 1
        
        # Lower wrap
        pos = GridPosition(0, 4, None)  # i=0 -> 10
        new_pos, _ = arena._enforce_boundaries_2d(pos)
        assert new_pos.i == 10
    
    def test_controllable_axis_clips_not_wraps(self, config_2d):
        """Controllable axis (j) clips instead of wrapping."""
        arena = make_arena(config_2d, 'periodic')
        
        # Upper clip
        pos = GridPosition(5, 9, None)  # j=9 should clip to 8
        new_pos, _ = arena._enforce_boundaries_2d(pos)
        assert new_pos.j == 8
        
        # Lower clip
        pos = GridPosition(5, 0, None)  # j=0 should clip to 1
        new_pos, _ = arena._enforce_boundaries_2d(pos)
        assert new_pos.j == 1


class TestBoundaries2DTerminal:
    """Test terminal mode for 2D grids."""
    
    def test_within_bounds_not_terminal(self, config_2d):
        """Position within bounds: out_of_bounds=False."""
        arena = make_arena(config_2d, 'terminal')
        pos = GridPosition(5, 4, None)
        
        new_pos, out_of_bounds = arena._enforce_boundaries_2d(pos)
        
        assert new_pos == pos
        assert out_of_bounds is False
    
    def test_any_axis_violation_is_terminal(self, config_2d):
        """Any axis violation triggers terminal."""
        arena = make_arena(config_2d, 'terminal')
        
        # i violates
        pos = GridPosition(0, 4, None)
        _, out_of_bounds = arena._enforce_boundaries_2d(pos)
        assert out_of_bounds is True
        
        # j violates
        pos = GridPosition(5, 0, None)
        _, out_of_bounds = arena._enforce_boundaries_2d(pos)
        assert out_of_bounds is True
