"""Abstract arena interface following BLE pattern.

The arena encapsulates the simulator and task-specific logic (rewards, termination,
boundaries). The environment wraps the arena to provide the Gym RL interface.
"""

from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
import jax.numpy as jnp

from ..utils.types import ArenaState


class AbstractArena(ABC):
    """Abstract arena interface (simulator + task logic).
    
    Following BLE's BalloonArenaInterface pattern, the arena contains:
    - Physics/dynamics simulation (field + actor interactions)
    - Task-specific logic (rewards, termination conditions)
    - State management and checkpointing
    
    The arena is decoupled from the RL environment wrapper.
    """
    
    @abstractmethod
    def reset(self, rng_key: jnp.ndarray) -> np.ndarray:
        """Reset arena to initial state.
        
        Args:
            rng_key: JAX PRNG key for reproducible initialization.
            
        Returns:
            Initial observation as flat numpy array [i, j, k, u_obs, v_obs].
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> np.ndarray:
        """Execute one simulation step.
        
        Args:
            action: Discrete action (0=down, 1=stay, 2=up).
            
        Returns:
            Observation as flat numpy array [i, j, k, u_obs, v_obs].
        """
        pass
    
    @abstractmethod
    def get_state(self) -> ArenaState:
        """Get complete arena state for checkpointing/analysis.
        
        Returns:
            Current arena state including position, history, metadata.
        """
        pass
    
    @abstractmethod
    def set_state(self, state: ArenaState) -> None:
        """Restore arena from state (for checkpointing).
        
        Args:
            state: Arena state to restore.
        """
        pass
    
    @abstractmethod
    def compute_reward(self) -> float:
        """Compute reward for current state (task-specific).
        
        Returns:
            Scalar reward value.
        """
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if current state is terminal (task-specific).
        
        Returns:
            True if episode should terminate, False otherwise.
        """
        pass
    
    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        """Define the observation space.
        
        Returns:
            Gym space object defining valid observations.
        """
        pass

