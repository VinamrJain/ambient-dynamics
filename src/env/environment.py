"""Grid RL Environment following BLE architecture pattern."""

import time
from typing import Tuple, Dict, Any, Optional, Union
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp

from .arena.abstract_arena import AbstractArena
from .rendering.renderer import Renderer


class GridEnvironment(gym.Env):
    """Grid RL Environment - wraps Arena (following BLE pattern).
    
    This class provides the Gymnasium RL interface by wrapping an Arena
    that handles simulation and task logic. Follows the architecture pattern
    from Balloon Learning Environment.
    """
    
    def __init__(
        self,
        arena: AbstractArena,
        max_steps: int = 1000,
        seed: Optional[int] = None,
        renderer: Optional[Renderer] = None
    ):
        """Initialize grid environment.
        
        Args:
            arena: Arena instance containing simulator and task logic.
            max_steps: Maximum steps per episode before truncation.
            seed: Initial random seed (uses system time if None).
            renderer: Optional renderer for visualization.
        """
        super().__init__()
        
        self.arena = arena
        self.max_steps = max_steps
        self._renderer = renderer
        self._global_step = 0
        self._episode_step = 0
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(3)  # 0=down, 1=stay, 2=up
        self.observation_space = arena.observation_space
        
        # Set metadata for rendering
        if renderer is not None:
            self.metadata = {'render.modes': renderer.render_modes}
        
        # Initialize RNG (use time if no seed provided)
        seed = seed if seed is not None else int(time.time() * 1e6)
        self.reset(seed=seed)
    
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step (Gym API).
        
        Args:
            action: Discrete action (0=down, 1=stay, 2=up).
            
        Returns:
            observation: Flat array [i, j, k, u_obs, v_obs].
            reward: Scalar reward from arena.
            terminated: Whether episode terminated naturally.
            truncated: Whether episode was truncated (max_steps).
            info: Additional information dictionary.
        """
        # Step arena (simulator)
        observation = self.arena.step(action)
        
        # Update renderer if present
        if self._renderer is not None:
            self._renderer.step(self.arena.get_state())
        
        # Compute reward (arena-specific)
        reward = self.arena.compute_reward()
        
        # Check termination
        terminated = self.arena.is_terminal()
        self._episode_step += 1
        truncated = self._episode_step >= self.max_steps
        
        # Build info dict
        info = self._get_info()
        
        self._global_step += 1
        
        return observation, reward, terminated, truncated, info
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """Reset environment (Gym API).
        
        Args:
            seed: Random seed for episode.
            options: Additional options (unused).
            
        Returns:
            observation: Initial observation.
            info: Info dict (if return_info compatibility mode).
        """
        # Re-seed if provided
        if seed is not None:
            self.seed(seed)
        
        # Split RNG key for arena
        self._rng, arena_rng = jax.random.split(self._rng)
        
        # Reset arena
        observation = self.arena.reset(arena_rng)
        
        # Reset renderer if present
        if self._renderer is not None:
            self._renderer.reset()
            self._renderer.step(self.arena.get_state())
        
        # Reset episode counter
        self._episode_step = 0
        
        # Gym 0.26+ returns (obs, info)
        return observation, self._get_info()
    
    def render(self, mode: str = 'human') -> Union[None, np.ndarray, str]:
        """Render environment (Gym API).
        
        Args:
            mode: Rendering mode ('human', 'rgb_array', etc.).
            
        Returns:
            None, RGB array, or string depending on mode.
        """
        if self._renderer is None:
            return None
        return self._renderer.render(mode)
    
    def seed(self, seed: int) -> None:
        """Seed the environment's RNG.
        
        Args:
            seed: Random seed.
        """
        self._rng = jax.random.PRNGKey(seed)
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    @property
    def unwrapped(self) -> gym.Env:
        """Return unwrapped environment (Gym property)."""
        return self
    
    def _get_info(self) -> Dict[str, Any]:
        """Construct info dictionary.
        
        Returns:
            Dictionary with episode metadata.
        """
        state = self.arena.get_state()
        return {
            'position': state.position,
            'last_displacement': state.last_displacement,
            'step_count': state.step_count,
            'episode_step': self._episode_step,
            'is_terminal': self.arena.is_terminal(),
            'out_of_bounds': state.out_of_bounds
        }