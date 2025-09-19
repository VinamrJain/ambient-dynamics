import numpy as np
from typing import Dict, Any
from .abstract_field import AbstractField
from ..utils.types import GridPosition, Displacement, GridConfig

class SimpleField(AbstractField):
    """Simple field implementation with uniform random displacements."""
    
    def __init__(self, config: GridConfig, seed: int = None):
        super().__init__(config, seed)
        self.reset()
    
    def reset(self, seed: int = None) -> None:
        """Reset field to new random configuration."""
        if seed is not None:
            self._rng = np.random.RandomState(seed)
    
    def sample_displacement(self, position: GridPosition) -> Displacement:
        """Sample uniform random displacement."""
        u = self._rng.randint(-self.config.d_max, self.config.d_max + 1)
        v = self._rng.randint(-self.config.d_max, self.config.d_max + 1)
        return Displacement(u, v)
    
    def get_field_state(self) -> Dict[str, Any]:
        """Return field state."""
        return {
            'type': 'simple',
            'config': self.config,
            'seed': self.seed
        }
    
    def get_displacement_pmf(self, position: GridPosition) -> np.ndarray:
        """Return uniform PMF over displacement space."""
        size = 2 * self.config.d_max + 1
        return np.ones((size, size)) / (size ** 2)