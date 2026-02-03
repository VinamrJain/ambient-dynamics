"""End-to-end episode integration tests.

Tests full episode flow through GridEnvironment covering:
- Edge cases: minimal grids (1x1), large grids, asymmetric grids
- Scenarios: navigation, station-keeping, terminal boundaries, truncation
- State consistency: observation shapes, bounds, deterministic replay
"""

import pytest
import numpy as np
import jax

from src.env.environment import GridEnvironment
from src.env.arena.grid_arena import GridArena
from src.env.arena.navigation_arena import NavigationArena
from src.env.field.simple_field import SimpleField
from src.env.actor.grid_actor import GridActor
from src.env.utils.types import GridConfig, GridPosition


# =============================================================================
# Fixtures and Helpers
# =============================================================================

def make_navigation_env(
    config: GridConfig,
    initial_position: GridPosition,
    target_position: GridPosition,
    max_steps: int = 100,
    seed: int = 42,
    boundary_mode: str = 'clip',
    terminate_on_reach: bool = False,
    vicinity_radius: float = 2.0,
    vicinity_bonus: float = 10.0,
    distance_reward_weight: float = -0.1,
    step_penalty: float = -0.5
) -> GridEnvironment:
    """Create a NavigationArena-backed GridEnvironment."""
    field = SimpleField(config)
    actor = GridActor(noise_prob=0.0)  # Deterministic for testing
    
    arena = NavigationArena(
        field=field,
        actor=actor,
        config=config,
        initial_position=initial_position,
        target_position=target_position,
        vicinity_radius=vicinity_radius,
        boundary_mode=boundary_mode,
        distance_reward_weight=distance_reward_weight,
        vicinity_bonus=vicinity_bonus,
        step_penalty=step_penalty,
        terminate_on_reach=terminate_on_reach
    )
    
    return GridEnvironment(arena=arena, max_steps=max_steps, seed=seed)


def make_grid_env(
    config: GridConfig,
    initial_position: GridPosition,
    max_steps: int = 100,
    seed: int = 42,
    boundary_mode: str = 'clip'
) -> GridEnvironment:
    """Create a basic GridArena-backed GridEnvironment."""
    field = SimpleField(config)
    actor = GridActor(noise_prob=0.0)
    
    arena = GridArena(
        field=field,
        actor=actor,
        config=config,
        initial_position=initial_position,
        boundary_mode=boundary_mode
    )
    
    return GridEnvironment(arena=arena, max_steps=max_steps, seed=seed)


# =============================================================================
# Edge Case: Minimal Grids
# =============================================================================

class TestMinimalGrids:
    """Test edge cases with minimal grid sizes."""
    
    def test_1x1x1_grid_3d(self):
        """1x1x1 grid: agent cannot move, always at (1,1,1)."""
        # d_max must be 0 for 1x1 ambient dimensions
        config = GridConfig(n_x=1, n_y=1, n_z=1, d_max=0)
        initial = GridPosition(1, 1, 1)
        target = GridPosition(1, 1, 1)
        
        env = make_navigation_env(
            config, initial, target,
            vicinity_radius=0.5,
            max_steps=10
        )
        
        obs, info = env.reset(seed=0)
        
        # Run 5 steps - position should never change
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(action=2)  # Try to go up
            
            # Position still at (1, 1, 1)
            assert obs[0] == 1.0  # i
            assert obs[1] == 1.0  # j
            assert obs[2] == 1.0  # k
    
    def test_1x1_grid_2d(self):
        """1x1 grid in 2D: agent stuck at (1, 1)."""
        config = GridConfig(n_x=1, n_y=1, n_z=None, d_max=0)
        initial = GridPosition(1, 1, None)
        target = GridPosition(1, 1, None)
        
        env = make_navigation_env(config, initial, target, vicinity_radius=0.5)
        obs, _ = env.reset(seed=0)
        
        for _ in range(5):
            obs, _, _, _, _ = env.step(action=2)
            assert obs[0] == 1.0
            assert obs[1] == 1.0
    
    def test_1x1xN_grid_vertical_corridor(self):
        """1x1xN grid: only vertical movement possible (like a tower)."""
        config = GridConfig(n_x=1, n_y=1, n_z=10, d_max=0)
        initial = GridPosition(1, 1, 1)
        target = GridPosition(1, 1, 10)
        
        env = make_navigation_env(
            config, initial, target,
            vicinity_radius=1.5,
            terminate_on_reach=True
        )
        
        obs, _ = env.reset(seed=0)
        assert obs[2] == 1.0  # Start at k=1
        
        # Move up repeatedly
        for step in range(9):
            obs, reward, terminated, truncated, info = env.step(action=2)  # Up
            
            if terminated:
                # Should terminate when reaching target
                assert info['position'].k >= 9  # Near or at target
                break
    
    def test_Nx1_grid_horizontal_corridor_2d(self):
        """Nx1 grid in 2D: narrow horizontal corridor."""
        config = GridConfig(n_x=20, n_y=1, n_z=None, d_max=1)
        initial = GridPosition(1, 1, None)
        
        env = make_grid_env(config, initial)
        obs, _ = env.reset(seed=0)
        
        # j coordinate stuck at 1
        for _ in range(10):
            obs, _, _, _, info = env.step(action=1)  # Stay
            assert info['position'].j == 1


# =============================================================================
# Edge Case: Large Grids
# =============================================================================

class TestLargeGrids:
    """Test with large grid sizes for performance and correctness."""
    
    def test_large_3d_grid_100x100x50(self):
        """Large 3D grid: verify episode runs without errors."""
        config = GridConfig.create(n_x=100, n_y=100, d_max=5, n_z=50)
        initial = GridPosition(50, 50, 25)
        target = GridPosition(90, 90, 45)
        
        env = make_navigation_env(
            config, initial, target,
            vicinity_radius=5.0,
            max_steps=50
        )
        
        obs, _ = env.reset(seed=42)
        assert obs.shape == (5,)  # 3D observation
        
        # Run full episode
        total_reward = 0.0
        for _ in range(50):
            action = np.random.randint(0, 3)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Should have accumulated some reward (could be negative)
        assert isinstance(total_reward, float)
    
    def test_large_2d_grid_200x100(self):
        """Large 2D grid: verify correct behavior."""
        config = GridConfig.create(n_x=200, n_y=100, d_max=10)
        initial = GridPosition(100, 50, None)
        target = GridPosition(180, 90, None)
        
        env = make_navigation_env(
            config, initial, target,
            vicinity_radius=10.0,
            max_steps=30
        )
        
        obs, _ = env.reset(seed=42)
        assert obs.shape == (3,)  # 2D observation
        
        for _ in range(30):
            obs, _, terminated, truncated, _ = env.step(action=1)
            if terminated or truncated:
                break
    
    def test_asymmetric_grid_wide(self):
        """Asymmetric grid: very wide (n_x >> n_y, n_z)."""
        config = GridConfig.create(n_x=100, n_y=5, d_max=2, n_z=3)
        initial = GridPosition(1, 3, 2)
        
        env = make_grid_env(config, initial)
        obs, _ = env.reset(seed=0)
        
        # Verify bounds are respected
        for _ in range(20):
            obs, _, _, _, info = env.step(action=np.random.randint(0, 3))
            pos = info['position']
            
            assert 1 <= pos.i <= 100
            assert 1 <= pos.j <= 5
            assert 1 <= pos.k <= 3
    
    def test_asymmetric_grid_tall(self):
        """Asymmetric grid: very tall (n_z >> n_x, n_y)."""
        config = GridConfig.create(n_x=5, n_y=5, d_max=2, n_z=100)
        initial = GridPosition(3, 3, 50)
        
        env = make_grid_env(config, initial)
        obs, _ = env.reset(seed=0)
        
        for _ in range(20):
            obs, _, _, _, info = env.step(action=np.random.randint(0, 3))
            pos = info['position']
            
            assert 1 <= pos.i <= 5
            assert 1 <= pos.j <= 5
            assert 1 <= pos.k <= 100


# =============================================================================
# Observation Shape Tests
# =============================================================================

class TestObservationShapes:
    """Test observation space dimensions for 2D vs 3D."""
    
    def test_3d_observation_shape_is_5(self):
        """3D observation: [i, j, k, u, v] = shape (5,)."""
        config = GridConfig.create(n_x=10, n_y=10, d_max=2, n_z=10)
        initial = GridPosition(5, 5, 5)
        
        env = make_grid_env(config, initial)
        obs, _ = env.reset(seed=0)
        
        assert obs.shape == (5,)
        assert env.observation_space.shape == (5,)
        
        # Verify bounds
        low = env.observation_space.low
        high = env.observation_space.high
        assert low[0] == 1 and high[0] == 10  # i
        assert low[1] == 1 and high[1] == 10  # j
        assert low[2] == 1 and high[2] == 10  # k
        assert low[3] == -2 and high[3] == 2  # u
        assert low[4] == -2 and high[4] == 2  # v
    
    def test_2d_observation_shape_is_3(self):
        """2D observation: [i, j, u] = shape (3,)."""
        config = GridConfig.create(n_x=10, n_y=10, d_max=2)
        initial = GridPosition(5, 5, None)
        
        env = make_grid_env(config, initial)
        obs, _ = env.reset(seed=0)
        
        assert obs.shape == (3,)
        assert env.observation_space.shape == (3,)
        
        low = env.observation_space.low
        high = env.observation_space.high
        assert low[0] == 1 and high[0] == 10  # i
        assert low[1] == 1 and high[1] == 10  # j
        assert low[2] == -2 and high[2] == 2  # u


# =============================================================================
# Scenario: Termination Conditions
# =============================================================================

class TestTerminationConditions:
    """Test different termination scenarios."""
    
    def test_max_steps_truncation(self):
        """Episode truncates at max_steps."""
        config = GridConfig.create(n_x=10, n_y=10, d_max=1, n_z=10)
        initial = GridPosition(1, 1, 1)
        target = GridPosition(9, 9, 9)  # Far away
        
        env = make_navigation_env(
            config, initial, target,
            max_steps=10,
            terminate_on_reach=False
        )
        
        obs, _ = env.reset(seed=0)
        
        steps = 0
        truncated = False
        while not truncated:
            _, _, terminated, truncated, info = env.step(action=1)  # Stay
            steps += 1
            assert not terminated  # Should not terminate naturally
        
        assert steps == 10
        assert truncated is True
    
    def test_terminate_on_reach(self):
        """Episode terminates when target vicinity is reached."""
        config = GridConfig.create(n_x=10, n_y=10, d_max=0, n_z=10)
        initial = GridPosition(5, 5, 1)
        target = GridPosition(5, 5, 5)
        
        env = make_navigation_env(
            config, initial, target,
            vicinity_radius=1.5,
            terminate_on_reach=True,
            max_steps=100
        )
        
        obs, _ = env.reset(seed=0)
        
        # Move up toward target
        terminated = False
        steps = 0
        while not terminated and steps < 20:
            obs, _, terminated, truncated, info = env.step(action=2)  # Up
            steps += 1
            
            if info['target_reached']:
                assert terminated
                break
        
        assert terminated  # Should have reached target
    
    def test_terminal_boundary_mode(self):
        """Episode terminates when boundary is violated in terminal mode."""
        config = GridConfig.create(n_x=5, n_y=5, d_max=2, n_z=5)
        initial = GridPosition(1, 1, 1)  # At corner
        
        field = SimpleField(config)
        actor = GridActor(noise_prob=0.0)
        
        arena = GridArena(
            field=field,
            actor=actor,
            config=config,
            initial_position=initial,
            boundary_mode='terminal'
        )
        
        env = GridEnvironment(arena=arena, max_steps=100, seed=42)
        obs, _ = env.reset(seed=42)
        
        # With d_max=2, field can push us out of bounds
        # Keep stepping until we hit boundary or max_steps
        terminated = False
        for step in range(100):
            obs, _, terminated, truncated, info = env.step(action=0)  # Down
            
            if terminated:
                assert info['out_of_bounds'] is True
                break
        
        # Either terminated due to boundary or truncated
        assert terminated or truncated


# =============================================================================
# Scenario: Station-Keeping
# =============================================================================

class TestStationKeeping:
    """Test station-keeping scenario (start at target)."""
    
    def test_station_keeping_accumulates_reward(self):
        """Starting at target should accumulate positive rewards."""
        config = GridConfig.create(n_x=10, n_y=10, d_max=0, n_z=10)  # No field drift
        target = GridPosition(5, 5, 5)
        
        env = make_navigation_env(
            config, target, target,  # Start at target
            vicinity_radius=2.0,
            vicinity_bonus=10.0,
            max_steps=20
        )
        
        obs, _ = env.reset(seed=0)
        
        total_reward = 0.0
        for _ in range(10):
            obs, reward, _, _, info = env.step(action=1)  # Stay
            total_reward += reward
            assert reward == 10.0  # Should get bonus each step
        
        assert total_reward == 100.0


# =============================================================================
# Determinism and Reproducibility
# =============================================================================

class TestDeterminism:
    """Test deterministic replay with same seed."""
    
    def test_same_seed_same_trajectory(self):
        """Same seed produces identical trajectories."""
        config = GridConfig.create(n_x=10, n_y=10, d_max=2, n_z=10)
        initial = GridPosition(5, 5, 5)
        target = GridPosition(8, 8, 8)
        
        # Run episode 1
        env1 = make_navigation_env(config, initial, target, seed=12345)
        obs1, _ = env1.reset(seed=12345)
        
        trajectory1 = [obs1.copy()]
        rewards1 = []
        for step in range(20):
            obs, reward, _, _, _ = env1.step(action=step % 3)
            trajectory1.append(obs.copy())
            rewards1.append(reward)
        
        # Run episode 2 with same seed and actions
        env2 = make_navigation_env(config, initial, target, seed=12345)
        obs2, _ = env2.reset(seed=12345)
        
        trajectory2 = [obs2.copy()]
        rewards2 = []
        for step in range(20):
            obs, reward, _, _, _ = env2.step(action=step % 3)
            trajectory2.append(obs.copy())
            rewards2.append(reward)
        
        # Trajectories should be identical
        for t1, t2 in zip(trajectory1, trajectory2):
            assert np.allclose(t1, t2)
        
        for r1, r2 in zip(rewards1, rewards2):
            assert r1 == r2
    
    def test_different_seeds_different_trajectories(self):
        """Different seeds produce different trajectories (with stochastic field)."""
        config = GridConfig.create(n_x=10, n_y=10, d_max=2, n_z=10)
        initial = GridPosition(5, 5, 5)
        target = GridPosition(8, 8, 8)
        
        # Run with seed 1
        env1 = make_navigation_env(config, initial, target, seed=111)
        obs1, _ = env1.reset(seed=111)
        for _ in range(10):
            obs1, _, _, _, _ = env1.step(action=1)
        
        # Run with seed 2
        env2 = make_navigation_env(config, initial, target, seed=222)
        obs2, _ = env2.reset(seed=222)
        for _ in range(10):
            obs2, _, _, _, _ = env2.step(action=1)
        
        # Observations should differ (field applies random displacement)
        assert not np.allclose(obs1, obs2)


# =============================================================================
# State Consistency
# =============================================================================

class TestStateConsistency:
    """Test state consistency throughout episodes."""
    
    def test_step_count_increments(self):
        """step_count increments each step."""
        config = GridConfig.create(n_x=10, n_y=10, d_max=1, n_z=10)
        initial = GridPosition(5, 5, 5)
        
        env = make_grid_env(config, initial)
        obs, info = env.reset(seed=0)
        
        assert info['step_count'] == 0
        
        for expected_count in range(1, 11):
            _, _, _, _, info = env.step(action=1)
            assert info['step_count'] == expected_count
    
    def test_position_within_bounds_clip_mode(self):
        """Position always within bounds in clip mode."""
        config = GridConfig.create(n_x=5, n_y=5, d_max=2, n_z=5)
        initial = GridPosition(1, 1, 1)  # Corner
        
        env = make_grid_env(config, initial, boundary_mode='clip')
        env.reset(seed=0)
        
        # Run many steps with random actions
        for _ in range(100):
            obs, _, _, _, info = env.step(action=np.random.randint(0, 3))
            pos = info['position']
            
            assert 1 <= pos.i <= 5, f"i={pos.i} out of bounds"
            assert 1 <= pos.j <= 5, f"j={pos.j} out of bounds"
            assert 1 <= pos.k <= 5, f"k={pos.k} out of bounds"
    
    def test_position_within_bounds_2d_clip_mode(self):
        """2D position always within bounds in clip mode."""
        config = GridConfig.create(n_x=5, n_y=8, d_max=2)
        initial = GridPosition(1, 1, None)
        
        env = make_grid_env(config, initial, boundary_mode='clip')
        env.reset(seed=0)
        
        for _ in range(100):
            obs, _, _, _, info = env.step(action=np.random.randint(0, 3))
            pos = info['position']
            
            assert 1 <= pos.i <= 5, f"i={pos.i} out of bounds"
            assert 1 <= pos.j <= 8, f"j={pos.j} out of bounds"
            assert pos.k is None
    
    def test_last_displacement_updated(self):
        """last_displacement is updated each step."""
        config = GridConfig.create(n_x=10, n_y=10, d_max=2, n_z=10)
        initial = GridPosition(5, 5, 5)
        
        env = make_grid_env(config, initial)
        _, info = env.reset(seed=0)
        
        # Initial displacement should be zero
        assert info['last_displacement'].u == 0.0
        assert info['last_displacement'].v == 0.0
        
        # After step, displacement may be non-zero
        _, _, _, _, info = env.step(action=1)
        # Displacement is in range [-d_max, d_max]
        assert -2 <= info['last_displacement'].u <= 2
        assert -2 <= info['last_displacement'].v <= 2
    
    def test_reset_clears_state(self):
        """Reset clears episode state."""
        config = GridConfig.create(n_x=10, n_y=10, d_max=1, n_z=10)
        initial = GridPosition(5, 5, 5)
        target = GridPosition(8, 8, 8)
        
        env = make_navigation_env(config, initial, target)
        
        # Run some steps
        env.reset(seed=0)
        for _ in range(10):
            env.step(action=2)
        
        # Reset
        obs, info = env.reset(seed=0)
        
        assert info['step_count'] == 0
        assert info['position'] == initial
        assert info['cumulative_reward'] == 0.0
        assert info['target_reached'] is False


# =============================================================================
# Action Space
# =============================================================================

class TestActionSpace:
    """Test action space behavior."""
    
    def test_action_space_is_discrete_3(self):
        """Action space is Discrete(3): down, stay, up."""
        config = GridConfig.create(n_x=10, n_y=10, d_max=1, n_z=10)
        initial = GridPosition(5, 5, 5)
        
        env = make_grid_env(config, initial)
        
        assert env.action_space.n == 3
    
    def test_all_actions_valid(self):
        """All actions 0, 1, 2 are valid."""
        config = GridConfig.create(n_x=10, n_y=10, d_max=1, n_z=10)
        initial = GridPosition(5, 5, 5)
        
        env = make_grid_env(config, initial)
        env.reset(seed=0)
        
        # All actions should work without error
        for action in [0, 1, 2]:
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None
            assert isinstance(reward, float)
