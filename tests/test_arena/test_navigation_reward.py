"""Tests for NavigationArena reward computation.

Tests compute_reward() and _compute_distance() methods covering:
- Distance calculation for 2D and 3D
- Vicinity detection and bonus
- Distance-based penalties outside vicinity
- Exponential decay within vicinity
- Cumulative reward tracking
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from src.env.arena.navigation_arena import NavigationArena
from src.env.field.simple_field import SimpleField
from src.env.actor.grid_actor import GridActor
from src.env.utils.types import GridConfig, GridPosition


# =============================================================================
# Fixtures
# =============================================================================

D_MAX_TEST = 3  # Default d_max for reward tests


@pytest.fixture
def config_3d():
    """3D grid config: 10x10x10."""
    return GridConfig.create(n_x=10, n_y=10, n_z=10)


@pytest.fixture
def config_2d():
    """2D grid config: 10x10."""
    return GridConfig.create(n_x=10, n_y=10)


def make_navigation_arena(
    config: GridConfig,
    initial_position: GridPosition,
    target_position: GridPosition,
    vicinity_radius: float = 2.0,
    distance_reward_weight: float = -0.1,
    vicinity_bonus: float = 10.0,
    step_penalty: float = -0.5,
    use_distance_decay: bool = False,
    decay_rate: float = 0.5,
    d_max: int = D_MAX_TEST
) -> NavigationArena:
    """Helper to create a NavigationArena for testing."""
    field = SimpleField(config, d_max=d_max)
    actor = GridActor(noise_std=0.0)
    
    return NavigationArena(
        field=field,
        actor=actor,
        config=config,
        initial_position=initial_position,
        target_position=target_position,
        vicinity_radius=vicinity_radius,
        boundary_mode='clip',
        distance_reward_weight=distance_reward_weight,
        vicinity_bonus=vicinity_bonus,
        step_penalty=step_penalty,
        terminate_on_reach=False,
        use_distance_decay=use_distance_decay,
        decay_rate=decay_rate
    )


# =============================================================================
# Distance Calculation Tests
# =============================================================================

class TestDistanceCalculation:
    """Test _compute_distance for 2D and 3D."""
    
    def test_distance_3d_same_position(self, config_3d):
        """Distance between same positions is 0."""
        pos = GridPosition(5, 5, 5)
        arena = make_navigation_arena(config_3d, pos, pos)
        
        distance = arena._compute_distance(pos, pos)
        
        assert distance == 0.0
    
    def test_distance_3d_unit_displacement(self, config_3d):
        """Distance for unit displacement on each axis."""
        target = GridPosition(5, 5, 5)
        arena = make_navigation_arena(config_3d, target, target)
        
        # Unit displacement on i
        pos = GridPosition(6, 5, 5)
        assert arena._compute_distance(pos, target) == 1.0
        
        # Unit displacement on j
        pos = GridPosition(5, 6, 5)
        assert arena._compute_distance(pos, target) == 1.0
        
        # Unit displacement on k
        pos = GridPosition(5, 5, 6)
        assert arena._compute_distance(pos, target) == 1.0
    
    def test_distance_3d_diagonal(self, config_3d):
        """Distance for diagonal displacement in 3D."""
        pos1 = GridPosition(1, 1, 1)
        pos2 = GridPosition(4, 5, 6)  # Delta: (3, 4, 5)
        arena = make_navigation_arena(config_3d, pos1, pos2)
        
        # sqrt(3^2 + 4^2 + 5^2) = sqrt(9 + 16 + 25) = sqrt(50)
        expected = np.sqrt(50)
        distance = arena._compute_distance(pos1, pos2)
        
        assert np.isclose(distance, expected)
    
    def test_distance_2d_same_position(self, config_2d):
        """Distance between same positions is 0 in 2D."""
        pos = GridPosition(5, 5, None)
        arena = make_navigation_arena(config_2d, pos, pos)
        
        distance = arena._compute_distance(pos, pos)
        
        assert distance == 0.0
    
    def test_distance_2d_unit_displacement(self, config_2d):
        """Distance for unit displacement on each axis in 2D."""
        target = GridPosition(5, 5, None)
        arena = make_navigation_arena(config_2d, target, target)
        
        # Unit displacement on i
        pos = GridPosition(6, 5, None)
        assert arena._compute_distance(pos, target) == 1.0
        
        # Unit displacement on j
        pos = GridPosition(5, 6, None)
        assert arena._compute_distance(pos, target) == 1.0
    
    def test_distance_2d_diagonal(self, config_2d):
        """Distance for diagonal displacement in 2D (classic 3-4-5 triangle)."""
        pos1 = GridPosition(1, 1, None)
        pos2 = GridPosition(4, 5, None)  # Delta: (3, 4)
        arena = make_navigation_arena(config_2d, pos1, pos2)
        
        # sqrt(3^2 + 4^2) = sqrt(9 + 16) = 5
        expected = 5.0
        distance = arena._compute_distance(pos1, pos2)
        
        assert distance == expected


# =============================================================================
# Vicinity Detection Tests
# =============================================================================

class TestVicinityDetection:
    """Test reward computation based on vicinity status."""
    
    def test_inside_vicinity_gets_bonus(self, config_3d):
        """Position inside vicinity gets vicinity bonus."""
        target = GridPosition(5, 5, 5)
        initial = GridPosition(5, 5, 5)  # Start at target
        arena = make_navigation_arena(
            config_3d, initial, target,
            vicinity_radius=2.0,
            vicinity_bonus=10.0
        )
        
        # Reset and compute reward (position is at target)
        rng_key = jax.random.PRNGKey(0)
        arena.reset(rng_key)
        reward = arena.compute_reward()
        
        assert reward == 10.0
    
    def test_outside_vicinity_gets_penalty(self, config_3d):
        """Position outside vicinity gets distance + step penalty."""
        target = GridPosition(5, 5, 5)
        initial = GridPosition(1, 1, 1)  # Far from target
        arena = make_navigation_arena(
            config_3d, initial, target,
            vicinity_radius=2.0,
            distance_reward_weight=-0.1,
            step_penalty=-0.5
        )
        
        rng_key = jax.random.PRNGKey(0)
        arena.reset(rng_key)
        reward = arena.compute_reward()
        
        # Distance from (1,1,1) to (5,5,5) = sqrt(48) ~ 6.93
        distance = np.sqrt(48)
        expected = (-0.1 * distance) + (-0.5)
        
        assert np.isclose(reward, expected)
    
    def test_at_radius_boundary_is_inside(self, config_3d):
        """Position exactly at radius boundary counts as inside vicinity."""
        target = GridPosition(5, 5, 5)
        # Position at exactly radius=2.0 distance
        initial = GridPosition(7, 5, 5)  # Distance = 2.0 exactly
        arena = make_navigation_arena(
            config_3d, initial, target,
            vicinity_radius=2.0,
            vicinity_bonus=10.0
        )
        
        rng_key = jax.random.PRNGKey(0)
        arena.reset(rng_key)
        reward = arena.compute_reward()
        
        assert reward == 10.0  # Inside (distance <= radius)
    
    def test_just_outside_radius_is_outside(self, config_3d):
        """Position just outside radius gets penalty."""
        target = GridPosition(5, 5, 5)
        # Position at distance > 2.0
        initial = GridPosition(8, 5, 5)  # Distance = 3.0
        arena = make_navigation_arena(
            config_3d, initial, target,
            vicinity_radius=2.0,
            vicinity_bonus=10.0,
            distance_reward_weight=-0.1,
            step_penalty=-0.5
        )
        
        rng_key = jax.random.PRNGKey(0)
        arena.reset(rng_key)
        reward = arena.compute_reward()
        
        # Should get penalty, not bonus
        expected = (-0.1 * 3.0) + (-0.5)
        assert np.isclose(reward, expected)


# =============================================================================
# Distance Decay Tests
# =============================================================================

class TestDistanceDecay:
    """Test exponential decay of vicinity bonus."""
    
    def test_decay_at_center_gives_full_bonus(self, config_3d):
        """At center (distance=0), get full vicinity bonus."""
        target = GridPosition(5, 5, 5)
        arena = make_navigation_arena(
            config_3d, target, target,
            vicinity_radius=3.0,
            vicinity_bonus=100.0,
            use_distance_decay=True,
            decay_rate=0.5
        )
        
        rng_key = jax.random.PRNGKey(0)
        arena.reset(rng_key)
        reward = arena.compute_reward()
        
        # At center: bonus * exp(-0.5 * 0) = 100 * 1 = 100
        assert reward == 100.0
    
    def test_decay_reduces_bonus_with_distance(self, config_3d):
        """Bonus decays exponentially with distance from center."""
        target = GridPosition(5, 5, 5)
        initial = GridPosition(6, 5, 5)  # Distance = 1.0, inside radius=3
        arena = make_navigation_arena(
            config_3d, initial, target,
            vicinity_radius=3.0,
            vicinity_bonus=100.0,
            use_distance_decay=True,
            decay_rate=0.5
        )
        
        rng_key = jax.random.PRNGKey(0)
        arena.reset(rng_key)
        reward = arena.compute_reward()
        
        # At distance=1: bonus * exp(-0.5 * 1) = 100 * exp(-0.5)
        expected = 100.0 * np.exp(-0.5)
        assert np.isclose(reward, expected)
    
    def test_no_decay_gives_constant_bonus(self, config_3d):
        """Without decay, bonus is constant anywhere in vicinity."""
        target = GridPosition(5, 5, 5)
        initial = GridPosition(6, 5, 5)  # Distance = 1.0
        arena = make_navigation_arena(
            config_3d, initial, target,
            vicinity_radius=3.0,
            vicinity_bonus=100.0,
            use_distance_decay=False
        )
        
        rng_key = jax.random.PRNGKey(0)
        arena.reset(rng_key)
        reward = arena.compute_reward()
        
        assert reward == 100.0  # Full bonus regardless of position


# =============================================================================
# Cumulative Reward Tests
# =============================================================================

class TestCumulativeReward:
    """Test cumulative reward tracking."""
    
    def test_cumulative_reward_starts_at_zero(self, config_3d):
        """Cumulative reward is 0 after reset."""
        target = GridPosition(5, 5, 5)
        arena = make_navigation_arena(config_3d, target, target)
        
        rng_key = jax.random.PRNGKey(0)
        arena.reset(rng_key)
        
        assert arena.get_cumulative_reward() == 0.0
    
    def test_cumulative_reward_accumulates(self, config_3d):
        """Cumulative reward sums across multiple compute_reward calls."""
        target = GridPosition(5, 5, 5)
        arena = make_navigation_arena(
            config_3d, target, target,
            vicinity_bonus=10.0
        )
        
        rng_key = jax.random.PRNGKey(0)
        arena.reset(rng_key)
        
        # Compute reward 3 times
        arena.compute_reward()
        arena.compute_reward()
        arena.compute_reward()
        
        assert arena.get_cumulative_reward() == 30.0
    
    def test_target_reached_flag_set_on_first_entry(self, config_3d):
        """target_reached flag set when first entering vicinity."""
        target = GridPosition(5, 5, 5)
        arena = make_navigation_arena(
            config_3d, target, target,  # Start at target
            vicinity_radius=2.0
        )
        
        rng_key = jax.random.PRNGKey(0)
        arena.reset(rng_key)
        
        assert arena._target_reached is False  # Not yet computed
        
        arena.compute_reward()
        
        assert arena._target_reached is True


# =============================================================================
# 2D Reward Tests
# =============================================================================

class TestReward2D:
    """Test reward computation in 2D mode."""
    
    def test_2d_inside_vicinity_gets_bonus(self, config_2d):
        """2D: Position inside vicinity gets bonus."""
        target = GridPosition(5, 5, None)
        arena = make_navigation_arena(
            config_2d, target, target,
            vicinity_radius=2.0,
            vicinity_bonus=10.0
        )
        
        rng_key = jax.random.PRNGKey(0)
        arena.reset(rng_key)
        reward = arena.compute_reward()
        
        assert reward == 10.0
    
    def test_2d_outside_vicinity_gets_penalty(self, config_2d):
        """2D: Position outside vicinity gets penalty."""
        target = GridPosition(5, 5, None)
        initial = GridPosition(1, 1, None)  # Distance = sqrt(32) ~ 5.66
        arena = make_navigation_arena(
            config_2d, initial, target,
            vicinity_radius=2.0,
            distance_reward_weight=-0.1,
            step_penalty=-0.5
        )
        
        rng_key = jax.random.PRNGKey(0)
        arena.reset(rng_key)
        reward = arena.compute_reward()
        
        distance = np.sqrt(32)  # sqrt((5-1)^2 + (5-1)^2)
        expected = (-0.1 * distance) + (-0.5)
        
        assert np.isclose(reward, expected)
