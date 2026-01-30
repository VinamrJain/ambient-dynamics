"""HTML export functionality for navigation renderer.

Uses renderer's trace methods to avoid code duplication.
Supports both 2D and 3D based on renderer.ndim.
"""

import os
from typing import TYPE_CHECKING
import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..navigation_renderer import NavigationRenderer


def save_html(
    renderer: 'NavigationRenderer',
    output_path: str
) -> None:
    """Export final state as static interactive HTML.
    
    Creates a standalone HTML file with interactive plot (final state only).
    
    Args:
        renderer: NavigationRenderer instance with recorded states.
        output_path: Path to save HTML file.
    
    Example:
        >>> renderer = NavigationRenderer(config)
        >>> # ... run episode ...
        >>> renderer.save_html('episode_static.html')
    """
    if not renderer.states:
        print("WARNING: No states recorded. Run episode first.")
        return
    
    print(f"Exporting static HTML...")
    
    # Create figure from renderer (works for both 2D and 3D)
    fig = renderer._create_figure()
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save
    fig.write_html(output_path)
    
    dim_str = "3D" if renderer.ndim == 3 else "2D"
    print(f"Static HTML saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"   Features: {dim_str} interactive (zoom, pan)")


def save_animated_html(
    renderer: 'NavigationRenderer',
    output_path: str,
    fps: int = 10
) -> None:
    """Export episode as animated interactive HTML.
    
    Creates a standalone HTML file with:
    - Play/Pause controls
    - Frame scrubber (slider)
    - Full interactivity at each frame (2D or 3D based on renderer)
    
    Args:
        renderer: NavigationRenderer instance with recorded states.
        output_path: Path to save HTML file.
        fps: Frames per second (default: 10).
    
    Example:
        >>> renderer = NavigationRenderer(config)
        >>> # ... run episode ...
        >>> renderer.save_animated_html('episode_animated.html', fps=10)
    """
    if not renderer.states:
        print("WARNING: No states recorded. Run episode first.")
        return
    
    print(f"Creating animated HTML with {len(renderer.states)} frames...")
    print("(This may take a moment...)")
    
    # Create frames for animation using renderer's trace methods
    frames = []
    frame_duration = int(1000 / fps)
    
    for i, state in enumerate(renderer.states):
        # Build frame data using renderer's trace methods (no duplication)
        frame_data = []
        
        if renderer.show_grid_points:
            frame_data.append(renderer._get_grid_points_trace())
        
        frame_data.append(renderer._get_target_vicinity_trace(state))
        frame_data.append(renderer._get_target_trace(state))
        frame_data.append(renderer._get_initial_position_trace(state))
        frame_data.append(renderer._get_trajectory_trace(up_to_idx=i))
        frame_data.append(renderer._get_actor_trace(state))
        
        # Create frame with title
        action_names = ['DEC', 'STAY', 'INC']
        action_name = action_names[state.last_action] if state.last_action is not None else 'N/A'
        
        frame = go.Frame(
            data=frame_data,
            name=f'frame{i}',
            layout=go.Layout(
                title=dict(
                    text=(
                        f"Step: {state.step_count} | "
                        f"Action: {action_name} | "
                        f"Reward: {state.last_reward:+.2f} | "
                        f"Cumulative: {state.cumulative_reward:+.2f}"
                    ),
                    font=dict(size=18),
                    x=0.5,
                    xanchor='center'
                )
            )
        )
        frames.append(frame)
    
    # Create initial figure using renderer's layout method
    fig = go.Figure(
        data=frames[0].data if frames else [],
        layout=renderer._get_animated_layout(),
        frames=frames
    )
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                x=0.1,
                y=0.0,
                xanchor='left',
                yanchor='bottom',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': frame_duration, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'y': 0.0,
            'xanchor': 'left',
            'x': 0.25,
            'currentvalue': {
                'prefix': 'Frame: ',
                'visible': True,
                'xanchor': 'left'
            },
            'pad': {'b': 10, 't': 10},
            'len': 0.7,
            'steps': [
                {
                    'args': [[f'frame{i}'], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'method': 'animate',
                    'label': str(i)
                }
                for i in range(len(frames))
            ]
        }]
    )
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save to file
    fig.write_html(output_path)
    
    dim_str = "3D" if renderer.ndim == 3 else "2D"
    print(f"Animated HTML saved to: {output_path}")
    print(f"   Frames: {len(frames)}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"   Features: Play/Pause, Scrubber, {dim_str} controls")

