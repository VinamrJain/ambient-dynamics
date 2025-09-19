import numpy as np
from typing import Dict, Any
from .abstract_actor import AbstractActor
from ..utils.types import GridPosition, VerticalAction, GridConfig

class GridActor(AbstractActor):
    """Basic grid actor with simple vertical dynamics."""
    
    def __init__(self, config: GridConfig, initial_position: GridPosition,
                 noise_prob: float = 0.1, seed: int = None):
        """
        Initialize grid actor.

        Args:
            config: Grid configuration
            initial_position: Starting position
            noise_prob: Probability of action noise
            seed: Random seed for reproducibility
        """
        super().__init__(config, initial_position)
        self.noise_prob = noise_prob
        self.seed = seed
        self._rng = np.random.RandomState(seed)
    
    def step_vertical(self, action: VerticalAction) -> GridPosition:
        """Execute vertical action with noise."""
        # Determine vertical displacement
        if self._rng.random() < self.noise_prob:
            # Add noise: random displacement in {-1, 0, +1}
            z_displacement = self._rng.choice([-1, 0, 1])
        else:
            # Execute intended action
            z_displacement = action.action
        
        # Apply displacement with boundary enforcement
        new_k = self.enforce_vertical_boundaries(self.position.k + z_displacement)
        
        # Update position
        self.position = GridPosition(self.position.i, self.position.j, new_k)
        return self.position
    
    def get_vertical_displacement_pmf(self, action: VerticalAction) -> np.ndarray:
        """Return PMF over vertical displacements."""
        pmf = np.zeros(3)  # For displacements {-1, 0, +1}
        
        # Intended action probability
        intended_idx = action.action + 1  # Map {-1,0,+1} -> {0,1,2}
        pmf[intended_idx] = 1 - self.noise_prob
        
        # Noise probability distributed uniformly
        noise_prob_each = self.noise_prob / 3
        pmf += noise_prob_each
        
        return pmf