"""Pure utility functions for rendering configuration.

These functions are stateless and can be reused across different renderers.
"""

from typing import Dict, Any
from io import BytesIO

import numpy as np
import plotly.graph_objects as go
from PIL import Image

from ..utils.types import GridConfig


def compute_scaling(config: GridConfig) -> Dict[str, float]:
    """Compute marker sizes based on grid dimensions.
    
    Args:
        config: Grid configuration.
        
    Returns:
        Dict with keys: grid_point_size, actor_size, target_size, trajectory_width
    """
    if config.ndim == 3:
        max_dim = max(config.n_x, config.n_y, config.n_z)
    else:
        max_dim = max(config.n_x, config.n_y)
    
    return {
        'grid_point_size': max(2, 60 / max_dim),
        'actor_size': max(10, 200 / max_dim),
        'target_size': max(8, 150 / max_dim),
        'trajectory_width': max(2, 25 / max_dim),
    }


def get_layout_config_2d(
    config: GridConfig,
    title_text: str,
    width: int,
    height: int
) -> Dict[str, Any]:
    """Get layout configuration dict for 2D figures.
    
    Args:
        config: Grid configuration.
        title_text: Title text for the figure.
        width: Figure width in pixels.
        height: Figure height in pixels.
        
    Returns:
        Layout configuration dict for fig.update_layout().
    """
    return dict(
        title=dict(
            text=title_text,
            font=dict(size=18),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        xaxis=dict(
            title=dict(text='X (ambient)', font=dict(size=16)),
            tickfont=dict(size=12),
            range=[0.5, config.n_x + 0.5],
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            title=dict(text='Y (controllable)', font=dict(size=16)),
            tickfont=dict(size=12),
            range=[0.5, config.n_y + 0.5]
        ),
        width=width,
        height=height,
        showlegend=True,
        legend=dict(
            x=0.01, y=0.99,
            font=dict(size=11),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        margin=dict(l=60, r=20, t=60, b=60)
    )


def get_layout_config_3d(
    config: GridConfig,
    title_text: str,
    width: int,
    height: int,
    camera_eye: Dict[str, float]
) -> Dict[str, Any]:
    """Get layout configuration dict for 3D figures.
    
    Args:
        config: Grid configuration.
        title_text: Title text for the figure.
        width: Figure width in pixels.
        height: Figure height in pixels.
        camera_eye: Camera position dict with x, y, z keys.
        
    Returns:
        Layout configuration dict for fig.update_layout().
    """
    camera = dict(
        eye=camera_eye,
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    )
    
    return dict(
        title=dict(
            text=title_text,
            font=dict(size=22),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text='X (i)', font=dict(size=20)),
                tickfont=dict(size=14),
                range=[0.5, config.n_x + 0.5]
            ),
            yaxis=dict(
                title=dict(text='Y (ambient)', font=dict(size=20)),
                tickfont=dict(size=14),
                range=[0.5, config.n_y + 0.5]
            ),
            zaxis=dict(
                title=dict(text='Z (controllable)', font=dict(size=20)),
                tickfont=dict(size=14),
                range=[0.5, config.n_z + 0.5]
            ),
            aspectmode='data',
            camera=camera
        ),
        width=width,
        height=height,
        showlegend=True,
        legend=dict(
            x=0.01, y=0.99,
            font=dict(size=12),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        margin=dict(l=10, r=10, t=60, b=10)
    )


def get_animated_layout_2d(
    config: GridConfig,
    width: int,
    height: int
) -> go.Layout:
    """Get layout for 2D animated figures.
    
    Args:
        config: Grid configuration.
        width: Figure width in pixels.
        height: Figure height in pixels (will add 100 for controls).
        
    Returns:
        Plotly Layout object.
    """
    return go.Layout(
        xaxis=dict(
            title=dict(text='X (ambient)', font=dict(size=14)),
            tickfont=dict(size=11),
            range=[0.5, config.n_x + 0.5],
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            title=dict(text='Y (controllable)', font=dict(size=14)),
            tickfont=dict(size=11),
            range=[0.5, config.n_y + 0.5]
        ),
        width=width,
        height=height + 100,
        showlegend=True,
        legend=dict(
            x=0.01, y=0.99,
            font=dict(size=11),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        margin=dict(l=60, r=20, t=60, b=80)
    )


def get_animated_layout_3d(
    config: GridConfig,
    width: int,
    height: int,
    camera_eye: Dict[str, float]
) -> go.Layout:
    """Get layout for 3D animated figures.
    
    Args:
        config: Grid configuration.
        width: Figure width in pixels.
        height: Figure height in pixels (will add 100 for controls).
        camera_eye: Camera position dict with x, y, z keys.
        
    Returns:
        Plotly Layout object.
    """
    camera = dict(
        eye=camera_eye,
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    )
    
    return go.Layout(
        scene=dict(
            xaxis=dict(
                title=dict(text='X (ambient)', font=dict(size=14)),
                tickfont=dict(size=11),
                range=[0.5, config.n_x + 0.5]
            ),
            yaxis=dict(
                title=dict(text='Y (ambient)', font=dict(size=14)),
                tickfont=dict(size=11),
                range=[0.5, config.n_y + 0.5]
            ),
            zaxis=dict(
                title=dict(text='Z (controllable)', font=dict(size=14)),
                tickfont=dict(size=11),
                range=[0.5, config.n_z + 0.5]
            ),
            aspectmode='data',
            camera=camera
        ),
        width=width,
        height=height + 100,
        showlegend=True,
        legend=dict(
            x=0.01, y=0.99,
            font=dict(size=12),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        margin=dict(l=10, r=10, t=60, b=80)
    )


def fig_to_array(fig: go.Figure, width: int, height: int) -> np.ndarray:
    """Convert Plotly figure to numpy RGB array.
    
    Args:
        fig: Plotly figure to convert.
        width: Output width in pixels.
        height: Output height in pixels.
        
    Returns:
        Numpy array of shape (height, width, 3) with RGB values.
        
    Raises:
        ValueError: If kaleido is not installed.
    """
    try:
        img_bytes = fig.to_image(format='png', width=width, height=height)
        
        img = Image.open(BytesIO(img_bytes))
        img_array = np.array(img)
        
        # Ensure RGB (remove alpha if present)
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        return img_array
    
    except ValueError as e:
        if 'kaleido' in str(e).lower():
            print("ERROR: rgb_array mode requires 'kaleido' package.")
            print("   Install with: pip install kaleido")
        raise
