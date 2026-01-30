"""Rendering components for visualization."""

from .renderer import Renderer
from .navigation_renderer import NavigationRenderer
from .navigation_renderer_2d import NavigationRenderer2D
from ..utils.types import GridConfig


def create_navigation_renderer(config: GridConfig, **kwargs):
    """Factory function to create appropriate renderer based on dimensionality.
    
    Args:
        config: Grid configuration (determines 2D vs 3D).
        **kwargs: Additional arguments passed to renderer constructor.
        
    Returns:
        NavigationRenderer (3D) or NavigationRenderer2D (2D).
    """
    if config.ndim == 3:
        return NavigationRenderer(config, **kwargs)
    else:
        return NavigationRenderer2D(config, **kwargs)


__all__ = [
    'Renderer',
    'NavigationRenderer',
    'NavigationRenderer2D',
    'create_navigation_renderer',
]