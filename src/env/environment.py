import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable

from .utils.types import GridPosition, Displacement, VerticalAction, GridConfig
from .field.abstract_field import AbstractField
from .actor.abstract_actor import AbstractActor
from .rendering.simple_renderer import SimpleRenderer

class GridEnvironment(gym.Env):
    """Gymnasium environment for discrete grid world with stochastic fields."""
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 field: AbstractField,
                 actor: AbstractActor,
                 config: GridConfig,
                 reward_fn: Optional[Callable[[Dict[str, Any], int], float]] = None,
                 max_steps: int = 1000):
        """
        Initialize grid environment.

        Args:
            field: Environmental field instance
            actor: Actor instance
            config: Grid configuration
            reward_fn: Optional reward function with signature (state: Dict[str, Any], action: int) -> float.
                      If None, uses default reward (always 0.0)
            max_steps: Maximum episode length
        """
        super().__init__()
        
        self.field = field
        self.actor = actor
        self.config = config
        self.reward_fn = reward_fn or self._default_reward
        self.max_steps = max_steps

        # Initialize renderer
        self.renderer = SimpleRenderer(config)

        # Validate initial actor position
        if not self.actor.is_valid_position(self.actor.position):
            raise ValueError(f"Invalid initial actor position: {self.actor.position}")
        
        # Action space: {-1, 0, +1} for vertical control
        self.action_space = spaces.Discrete(3)
        
        # Observation space: position + local displacement observation
        self.observation_space = spaces.Dict({
            'position': spaces.Box(
                low=np.array([1, 1, 1]), 
                high=np.array([config.n_x, config.n_y, config.n_z]),
                dtype=np.int32
            ),
            'local_displacement': spaces.Box(
                low=np.array([-config.d_max, -config.d_max]),
                high=np.array([config.d_max, config.d_max]),
                dtype=np.int32
            )
        })
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset field and actor
        self.field.reset(seed=seed)
        self.actor.reset()
        
        # Reset episode counters
        self.step_count = 0
        self.done = False
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        if self.done:
            raise RuntimeError(
                f"Environment episode is finished (step {self.step_count}). "
                "Call reset() to start a new episode."
            )

        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}. Must be in range [0, {self.action_space.n-1}]"
            )

        # Convert action to vertical action
        try:
            vertical_action = VerticalAction(action - 1)  # Map {0,1,2} -> {-1,0,+1}
        except Exception as e:
            raise ValueError(f"Failed to convert action {action} to VerticalAction: {e}")
        
        # Sample horizontal displacement from field
        horizontal_displacement = self.field.sample_displacement(self.actor.position)
        
        # Apply horizontal displacement (with boundary enforcement)
        new_position_horizontal = self.field.enforce_boundaries(
            self.actor.position, horizontal_displacement)
        
        # Update actor position horizontally
        self.actor.position = GridPosition(
            new_position_horizontal.i, 
            new_position_horizontal.j, 
            self.actor.position.k
        )
        
        # Execute vertical action
        self.actor.position = self.actor.step_vertical(vertical_action)

        # Validate final position (should not happen with proper boundary enforcement)
        if not self.actor.is_valid_position(self.actor.position):
            raise RuntimeError(f"Actor moved to invalid position: {self.actor.position}")

        # Compute reward
        try:
            reward = self.reward_fn(self._get_state(), action)
            if not isinstance(reward, (int, float)):
                raise TypeError(f"Reward function returned {type(reward)}, expected int or float")
        except Exception as e:
            raise RuntimeError(f"Error computing reward: {e}")

        # Store for rendering
        self._last_reward = reward
        self._last_displacement = horizontal_displacement

        # Update episode counters
        self.step_count += 1
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps
        self.done = terminated or truncated
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info['horizontal_displacement'] = horizontal_displacement
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, Any]:
        """Construct observation from current state."""
        # Sample displacement observation (what the agent observes)
        local_displacement = self.field.sample_displacement(self.actor.position)
        
        return {
            'position': np.array([self.actor.position.i, 
                                 self.actor.position.j, 
                                 self.actor.position.k]),
            'local_displacement': np.array([local_displacement.u, 
                                          local_displacement.v])
        }
    
    def _get_state(self) -> Dict[str, Any]:
        """Return complete environment state."""
        return {
            'actor': self.actor.get_state(),
            'field': self.field.get_field_state(),
            'step_count': self.step_count
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Return additional information."""
        return {
            'step_count': self.step_count,
            'position': self.actor.position,
            'is_terminal': self._is_terminated()
        }
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Basic termination conditions:
        # 1. Actor goes out of bounds (shouldn't happen with boundary enforcement, but safety check)
        # 2. Could add goal-reaching condition here in the future

        position = self.actor.position
        out_of_bounds = (
            position.i < 1 or position.i > self.config.n_x or
            position.j < 1 or position.j > self.config.n_y or
            position.k < 1 or position.k > self.config.n_z
        )

        return out_of_bounds
    
    def _default_reward(self, state: Dict[str, Any], action: int) -> float:
        """Default reward function (override for specific objectives)."""
        return 0.0
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            # Use simple text-based renderer
            info = {
                'reward': getattr(self, '_last_reward', 0.0),
                'horizontal_displacement': getattr(self, '_last_displacement', None)
            }
            self.renderer.render_grid(
                self.actor.position,
                self.step_count,
                target_position=None,  # Can be set based on reward function
                info=info
            )
        elif mode == 'rgb_array':
            # Return RGB array representation (implement as needed)
            pass
    
    def close(self):
        """Clean up environment resources."""
        pass