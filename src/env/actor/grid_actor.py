"""Basic grid actor implementation with stochastic controllable axis dynamics."""

import numpy as np
import jax
import jax.numpy as jnp

from .abstract_actor import AbstractActor
from ..utils.types import GridPosition


class GridActor(AbstractActor):
    """Basic grid actor with noisy controllable axis dynamics.
    
    Supports both 2D and 3D settings:
    - 3D: Controls z-axis (k coordinate)
    - 2D: Controls y-axis (j coordinate)
    
    
    - Intended displacement: probability (1 - eps)
    - One level above intended: probability eps/2
    - One level below intended: probability eps/2
    
    Where eps = noise_prob (total noise probability).
    """
    
    def __init__(self, noise_prob: float = 0.1):
        """Initialize grid actor.
        
        Args:
            noise_prob: Total probability of noise (eps).
                       Split equally between adjacent levels (eps/2 each).
        """
        super().__init__()
        self.noise_prob = float(noise_prob)
        
        if not 0.0 <= noise_prob <= 1.0:
            raise ValueError(f"noise_prob must be in [0, 1], got {noise_prob}")
    
    def step_controllable(
        self, position: GridPosition, action: int, rng_key: jnp.ndarray
    ) -> GridPosition:
        """Apply action on controllable axis with noise (functional/stateless).
        
        Args:
            position: Current position.
            action: Controllable axis action (0=decrease, 1=stay, 2=increase).
            rng_key: JAX PRNG key for stochastic dynamics.
            
        Returns:
            New position after action (may be outside bounds).
        """
        # Map action to intended displacement: 0→-1, 1→0, 2→+1
        intended_displacement = action - 1
        
        # Sample outcome: 0=below, 1=intended, 2=above
        # Probabilities: [eps/2, 1-eps, eps/2]
        eps_half = self.noise_prob / 2.0
        probs = jnp.array([eps_half, 1.0 - self.noise_prob, eps_half])
        outcome = jax.random.choice(rng_key, jnp.array([0, 1, 2]), p=probs)
        
        # Map outcome to displacement relative to intended
        # 0→intended-1, 1→intended, 2→intended+1
        displacement = int(intended_displacement + (outcome - 1))
        
        # Apply displacement to controllable axis
        new_controllable = position.controllable + displacement
        
        # Return new position based on dimensionality
        if position.ndim == 3:
            return GridPosition(position.i, position.j, new_controllable)
        else:
            return GridPosition(position.i, new_controllable, None)
    
    def get_controllable_displacement_pmf(self) -> np.ndarray:
        """Get full PMF over controllable axis displacements for all actions.
        
        Following formulation: intended displacement gets (1-eps), adjacent levels get eps/2 each.
        
        Returns:
            PMF array of shape (3, 5) where entry [a, j] is P(displacement = j-c_max | action=a). 
            where c_max is the maximum controllable displacement magnitude.
            
            For c_max=2, displacements are {-2, -1, 0, +1, +2}:
            - Action 0 (decrease, intended=-1): nonzero for {-2, -1, 0}
            - Action 1 (stay, intended=0): nonzero for {-1, 0, +1}
            - Action 2 (increase, intended=+1): nonzero for {0, +1, +2}
            
        Example with eps=0.2:
            pmf[0, :] = [0.1, 0.8, 0.1, 0.0, 0.0]  # action=decrease
            pmf[1, :] = [0.0, 0.1, 0.8, 0.1, 0.0]  # action=stay
            pmf[2, :] = [0.0, 0.0, 0.1, 0.8, 0.1]  # action=increase
        """
        c_max = 2  # TODO: Could be made configurable or inferred (max displacement magnitude). Right now the function logic is hardcoded for c_max=2.
        n_displacements = 2 * c_max + 1  # 5 displacements: {-2, -1, 0, +1, +2}
        n_actions = 3  # {0=decrease, 1=stay, 2=increase}
        
        # Initialize PMF array
        pmf = np.zeros((n_actions, n_displacements), dtype=np.float32)
        
        eps_half = self.noise_prob / 2.0  # Split noise equally between adjacent levels
        
        # For each action
        for action in range(n_actions):
            # Map action to intended displacement: 0→-1, 1→0, 2→+1
            intended_displacement = action - 1
            
            # Actual displacements: intended ± 1
            # Convert to indices: displacement d maps to index j = d + c_max
            below_idx = (intended_displacement - 1) + c_max  # intended - 1
            intended_idx = intended_displacement + c_max      # intended
            above_idx = (intended_displacement + 1) + c_max   # intended + 1
            
            # Set probabilities: eps/2 for adjacent, 1-eps for intended
            pmf[action, below_idx] = eps_half            # eps/2 for below
            pmf[action, intended_idx] = 1.0 - self.noise_prob  # 1-eps for intended
            pmf[action, above_idx] = eps_half            # eps/2 for above
        
        return pmf