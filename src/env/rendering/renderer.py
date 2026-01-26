"""Abstract renderer interface following BLE pattern."""

from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np

from ..utils.types import ArenaState


class Renderer(ABC):
    """Abstract renderer for visualizing arena states.
    
    Following BLE's Renderer pattern, renderers accumulate state history
    during an episode and render on demand in different modes.
    """
    
    @abstractmethod
    def reset(self) -> None:
        """Reset renderer for new episode.
        
        Clears accumulated state history and prepares for new episode.
        """
        pass
    
    @abstractmethod
    def step(self, state: ArenaState) -> None:
        """Accumulate arena state for rendering.
        
        Args:
            state: Current arena state to record.
        """
        pass
    
    @abstractmethod
    def render(self, mode: str) -> Union[None, np.ndarray, str]:
        """Render accumulated states.
        
        Args:
            mode: Rendering mode. Standard modes:
                - 'human': Display to screen (returns None)
                - 'rgb_array': Return RGB array (H, W, 3)
                - 'ansi': Return text representation (string)
        
        Returns:
            None, numpy array, or string depending on mode.
        
        Raises:
            ValueError: If mode is not supported.
        """
        pass
    
    @property
    @abstractmethod
    def render_modes(self) -> List[str]:
        """List of supported rendering modes.
        
        Returns:
            List of mode strings (e.g., ['human', 'rgb_array']).
        """
        pass

