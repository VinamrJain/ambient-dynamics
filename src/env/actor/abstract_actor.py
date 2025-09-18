from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np
from ..utils.types import GridPosition, VerticalAction, GridConfig

class AbstractActor(ABC):
    """Abstract base class for actors in the grid environment."""
    
    def __init__(self, config: GridConfig, initial_position: GridPosition):
        """Initialize actor with configuration and starting position."""
        self.config = config
        self.position = initial_position
        self._initial_position = initial_position
    
    @abstractmethod
    def step_vertical(self, action: VerticalAction) -> GridPosition:
        """Execute vertical action and return new position."""
        pass
    
    @abstractmethod
    def get_vertical_displacement_pmf(self, action: VerticalAction) -> np.ndarray:
        """Return PMF over vertical displacements for given action."""
        pass
    
    def reset(self, position: GridPosition = None) -> GridPosition:
        """Reset actor to initial or specified position."""
        self.position = position or self._initial_position
        return self.position
    
    def is_valid_position(self, position: GridPosition) -> bool:
        """Check if position is within grid bounds."""
        return (1 <= position.i <= self.config.n_x and
                1 <= position.j <= self.config.n_y and
                1 <= position.k <= self.config.n_z)
    
    def enforce_vertical_boundaries(self, new_k: int) -> int:
        """Enforce vertical boundary constraints."""
        return max(1, min(new_k, self.config.n_z))
    
    def get_state(self) -> Dict[str, Any]:
        """Return current actor state."""
        return {
            'position': self.position,
            'initial_position': self._initial_position
        }