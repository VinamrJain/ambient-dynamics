"""HTML export functionality for navigation renderer.""" ## FIXME: Layout configuration from navigation_renderer.py, and code redundancy can be reduced.

import os
from typing import TYPE_CHECKING
import numpy as np
import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..navigation_renderer import NavigationRenderer


def save_html(
    renderer: 'NavigationRenderer',
    output_path: str
) -> None:
    """Export final state as static interactive HTML.
    
    Creates a standalone HTML file with interactive 3D plot (final state only).
    You can rotate, zoom, and pan the 3D view.
    
    Args:
        renderer: NavigationRenderer instance with recorded states.
        output_path: Path to save HTML file.
    
    Example:
        >>> renderer = NavigationRenderer(config)
        >>> # ... run episode ...
        >>> renderer.save_html('episode_static.html')
    """
    if not renderer.states:
        print("⚠️  No states recorded. Run episode first.")
        return
    
    print(f"Exporting static HTML...")
    
    # Create figure from renderer
    fig = renderer._create_figure()
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save
    fig.write_html(output_path)
    
    print(f"✅ Static HTML saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"   Features: 3D interactive (rotate, zoom, pan)")


def save_animated_html(
    renderer: 'NavigationRenderer',
    output_path: str,
    fps: int = 10
) -> None:
    """Export episode as animated interactive HTML.
    
    Creates a standalone HTML file with:
    - Play/Pause controls
    - Frame scrubber (slider)
    - Full 3D interactivity at each frame
    
    This is the "best of both worlds" - animation + interactivity!
    
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
        print("⚠️  No states recorded. Run episode first.")
        return
    
    print(f"Creating animated HTML with {len(renderer.states)} frames...")
    print("(This may take a moment...)")
    
    # Create frames for animation
    frames = []
    frame_duration = int(1000 / fps)  # Convert fps to milliseconds
    
    for i, state in enumerate(renderer.states):
        # Create data for this frame
        frame_data = []
        
        # Add all visualization elements for this frame
        if renderer.show_grid_points:
            frame_data.extend(_get_grid_points_data(renderer))
        
        frame_data.extend(_get_target_vicinity_data(renderer, state))
        frame_data.extend(_get_target_data(renderer, state))
        frame_data.extend(_get_initial_position_data(renderer, state))
        frame_data.extend(_get_trajectory_data_up_to(renderer, i))
        frame_data.extend(_get_actor_data(renderer, state))
        
        # Create frame
        action_names = ['DOWN', 'STAY', 'UP']
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
    
    # Create initial figure with first frame's data
    fig = go.Figure(
        data=frames[0].data if frames else [],
        layout=_get_animated_layout(renderer, renderer.states[0] if renderer.states else None),
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
                        label='▶ Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': frame_duration, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    ),
                    dict(
                        label='⏸ Pause',
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
    
    print(f"✅ Animated HTML saved to: {output_path}")
    print(f"   Frames: {len(frames)}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"   Features: Play/Pause, Scrubber, Full 3D controls")


# ============================================================================
# Helper functions for building Plotly traces
# ============================================================================

def _get_animated_layout(renderer, state):
    """Get layout for animated figure."""
    camera = dict(
        eye=renderer.camera_eye,
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    )
    
    return go.Layout(
        scene=dict(
            xaxis=dict(
                title=dict(text='X (i)', font=dict(size=14)),
                tickfont=dict(size=11),
                range=[0.5, renderer.config.n_x + 0.5]
            ),
            yaxis=dict(
                title=dict(text='Y (j)', font=dict(size=14)),
                tickfont=dict(size=11),
                range=[0.5, renderer.config.n_y + 0.5]
            ),
            zaxis=dict(
                title=dict(text='Z (k)', font=dict(size=14)),
                tickfont=dict(size=11),
                range=[0.5, renderer.config.n_z + 0.5]
            ),
            aspectmode='data',
            camera=camera
        ),
        width=renderer.width,
        height=renderer.height + 100,  # Extra space for controls
        showlegend=True,
        legend=dict(
            x=0.01, 
            y=0.99, 
            font=dict(size=12),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        margin=dict(l=10, r=10, t=60, b=80)  # Extra bottom margin for controls
    )


def _get_grid_points_data(renderer):
    """Get grid points as Plotly trace data."""
    i_range = range(1, renderer.config.n_x + 1, renderer.grid_subsample)
    j_range = range(1, renderer.config.n_y + 1, renderer.grid_subsample)
    k_range = range(1, renderer.config.n_z + 1, renderer.grid_subsample)
    
    i_coords, j_coords, k_coords = [], [], []
    for i in i_range:
        for j in j_range:
            for k in k_range:
                i_coords.append(i)
                j_coords.append(j)
                k_coords.append(k)
    
    return [go.Scatter3d(
        x=i_coords, y=j_coords, z=k_coords,
        mode='markers',
        marker=dict(size=renderer.grid_point_size, color='gray', opacity=0.4),
        name='Grid',
        showlegend=False,
        hoverinfo='skip'
    )]


def _get_target_vicinity_data(renderer, state):
    """Get target vicinity cylinder as Plotly trace data."""
    theta = np.linspace(0, 2 * np.pi, 40)
    z_levels = np.linspace(1, renderer.config.n_z, 30)
    theta_grid, z_grid = np.meshgrid(theta, z_levels)
    x_cylinder = state.target_position.i + state.vicinity_radius * np.cos(theta_grid)
    y_cylinder = state.target_position.j + state.vicinity_radius * np.sin(theta_grid)
    
    return [go.Surface(
        x=x_cylinder, y=y_cylinder, z=z_grid,
        colorscale=[[0, 'lightgreen'], [1, 'lightgreen']],
        opacity=0.25,
        showscale=False,
        showlegend=False,
        hoverinfo='skip',
        name='Vicinity'
    )]


def _get_target_data(renderer, state):
    """Get target marker as Plotly trace data."""
    return [go.Scatter3d(
        x=[state.target_position.i],
        y=[state.target_position.j],
        z=[state.target_position.k],
        mode='markers',
        marker=dict(
            size=renderer.target_size * 0.5,
            color='green',
            symbol='x',
            line=dict(color='darkgreen', width=2)
        ),
        name='Target',
        showlegend=True
    )]


def _get_initial_position_data(renderer, state):
    """Get initial position marker as Plotly trace data."""
    return [go.Scatter3d(
        x=[state.initial_position.i],
        y=[state.initial_position.j],
        z=[state.initial_position.k],
        mode='markers',
        marker=dict(
            size=renderer.target_size * 0.8,
            color='orange',
            symbol='circle',
            line=dict(color='darkorange', width=2),
            opacity=0.8
        ),
        name='Start',
        showlegend=True
    )]


def _get_trajectory_data_up_to(renderer, frame_idx):
    """Get trajectory up to given frame as Plotly trace data."""
    positions = [s.position for s in renderer.states[:frame_idx + 1]]
    traj_array = np.array([[p.i, p.j, p.k] for p in positions])
    
    return [go.Scatter3d(
        x=traj_array[:, 0],
        y=traj_array[:, 1],
        z=traj_array[:, 2],
        mode='lines+markers',
        line=dict(color='royalblue', width=renderer.trajectory_width * 2.5),
        marker=dict(size=renderer.trajectory_width * 3, color='steelblue', opacity=0.7),
        name='Trajectory',
        showlegend=True,
        opacity=0.8
    )]


def _get_actor_data(renderer, state):
    """Get current actor position as Plotly trace data."""
    return [go.Scatter3d(
        x=[state.position.i],
        y=[state.position.j],
        z=[state.position.k],
        mode='markers',
        marker=dict(
            size=renderer.actor_size * 0.8,
            color='red',
            symbol='diamond',
            line=dict(color='darkred', width=2.5)
        ),
        name='Actor',
        showlegend=True
    )]

