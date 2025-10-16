"""Basic grid actor implementation with stochastic vertical dynamics."""

import numpy as np
import jax
import jax.numpy as jnp

from .abstract_actor import AbstractActor
from ..utils.types import GridPosition, GridConfig


class GridActor(AbstractActor):
    """Basic grid actor with noisy vertical dynamics.
    
    Vertical actions are executed with some noise probability:
    - With probability (1 - noise_prob): execute intended action
    - With probability noise_prob: execute random action from {-1, 0, +1}
    """
    
    def __init__(self, config: GridConfig, noise_prob: float = 0.1):
        """Initialize grid actor.
        
        Args:
            config: Grid configuration.
            noise_prob: Probability of random action noise in [0, 1].
        """
        super().__init__(config)
        self.noise_prob = float(noise_prob)
        
        if not 0.0 <= noise_prob <= 1.0:
            raise ValueError(f"noise_prob must be in [0, 1], got {noise_prob}")
    
    def step_vertical(
        self, position: GridPosition, action: int, rng_key: jnp.ndarray
    ) -> GridPosition:
        """Apply vertical action with noise (functional/stateless).
        
        Args:
            position: Current position.
            action: Vertical action (0=down, 1=stay, 2=up).
            rng_key: JAX PRNG key for stochastic dynamics.
            
        Returns:
            New position after vertical action (may be outside bounds).
        """
        # Map action to displacement: 0→-1, 1→0, 2→+1
        intended_displacement = action - 1
        
        # Determine if noise occurs
        noise_key, disp_key = jax.random.split(rng_key)
        use_noise = jax.random.bernoulli(noise_key, self.noise_prob)
        
        # Sample random displacement if noise
        random_displacement = jax.random.randint(
            disp_key, shape=(), minval=-1, maxval=2
        )
        
        # Select displacement based on noise
        z_displacement = jax.lax.select(
            use_noise,
            random_displacement,
            intended_displacement
        )
        
        # Apply vertical displacement (no boundary enforcement here)
        new_k = position.k + int(z_displacement)
        
        return GridPosition(position.i, position.j, new_k)
    
    def get_vertical_displacement_pmf(self, action: int) -> np.ndarray:
        """Get PMF over vertical displacements for analysis.
        
        Args:
            action: Vertical action (0=down, 1=stay, 2=up).
            
        Returns:
            PMF array of length 3 for displacements {-1, 0, +1}.
        """
        # Map action to intended displacement
        intended_displacement = action - 1
        
        # Initialize PMF for {-1, 0, +1}
        pmf = np.zeros(3, dtype=np.float32)
        
        # Noise contributes uniform probability to all displacements
        noise_prob_each = self.noise_prob / 3.0
        pmf += noise_prob_each
        
        # Intended action gets remaining probability
        intended_idx = intended_displacement + 1  # Map {-1,0,+1} → {0,1,2}
        pmf[intended_idx] += (1.0 - self.noise_prob)
        
        return pmf