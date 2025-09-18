from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np
from ..utils.types import GridPosition, Displacement, GridConfig

class AbstractField(ABC):
    """Abstract base class for environmental fields."""
    
    def __init__(self, config: GridConfig, seed: int = None):
        """Initialize field with configuration and random seed."""
        self.config = config
        self.seed = seed
        self._rng = np.random.RandomState(seed)
    
    @abstractmethod
    def reset(self, seed: int = None) -> None:
        """Reset/regenerate the field configuration."""
        pass
    
    @abstractmethod
    def sample_displacement(self, position: GridPosition) -> Displacement:
        """Sample horizontal displacement at given position."""
        pass
    
    @abstractmethod
    def get_field_state(self) -> Dict[str, Any]:
        """Return complete field state (for analysis/debugging)."""
        pass
    
    @abstractmethod
    def get_displacement_pmf(self, position: GridPosition) -> np.ndarray:
        """Return displacement PMF at given position."""
        pass
    
    def is_valid_position(self, position: GridPosition) -> bool:
        """Check if position is within grid bounds."""
        return (1 <= position.i <= self.config.n_x and
                1 <= position.j <= self.config.n_y and
                1 <= position.k <= self.config.n_z)
    
    def enforce_boundaries(self, position: GridPosition, 
                          displacement: Displacement) -> GridPosition:
        """Apply displacement with boundary enforcement."""
        new_i = max(1, min(position.i + displacement.u, self.config.n_x))
        new_j = max(1, min(position.j + displacement.v, self.config.n_y))
        return GridPosition(new_i, new_j, position.k)