"""Navigation arena renderer using Plotly backend.

Supports both 2D and 3D navigation tasks:
- 2D: Uses Scatter plots, vicinity shown as circle
- 3D: Uses Scatter3d plots, vicinity shown as cylinder

Core visualization logic for navigation tasks. Supports:
- Standard Gymnasium modes: 'human' (interactive), 'rgb_array' (numpy array)
- Export methods: save_gif(), save_mp4(), save_html(), save_animated_html()
"""

from typing import List, Union, Optional
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from PIL import Image

from .renderer import Renderer
from ..utils.types import GridConfig, NavigationArenaState


class NavigationRenderer(Renderer):
    """Renderer for navigation tasks (Plotly backend).
    
    Extracts all visualization info from NavigationArenaState.
    
    Features:
    - Interactive 2D/3D visualization
    - Grid points (with smart sub-sampling for large grids)
    - Actor position and trajectory
    - Target position with vicinity region (circle for 2D, cylinder for 3D)
    - Episode info (step, action, reward, cumulative reward)
    """
    
    def __init__(
        self,
        config: GridConfig,
        width: int = 1024,
        height: int = 768,
        show_grid_points: bool = True,
        grid_subsample: Optional[int] = None,
        backend: str = 'plotly',
        camera_eye: Optional[dict] = None
    ):
        """Initialize navigation renderer.
        
        Args:
            config: Grid configuration.
            width: Figure width in pixels.
            height: Figure height in pixels.
            show_grid_points: Whether to show grid points.
            grid_subsample: Subsample factor for grid points (auto if None).
            backend: Rendering backend ('plotly').
            camera_eye: Camera position for 3D (ignored for 2D).
                       Defaults to (1.5, -1.5, 1.0) for 45 degree perspective.
                       Examples:
                         - Top-down: {'x': 0, 'y': 0, 'z': 2.5}
                         - Side view: {'x': 2.5, 'y': 0, 'z': 0}
                         - Isometric: {'x': 1.5, 'y': -1.5, 'z': 1.0}
        """
        self.config = config
        self.width = width
        self.height = height
        self.show_grid_points = show_grid_points
        self.backend = backend
        self.camera_eye = camera_eye or {'x': 1.5, 'y': -1.5, 'z': 1.0}
        
        # Compute smart scaling based on grid size
        self._compute_scaling()
        
        # Determine grid subsampling #TODO: Improve grid subsampling logic.
        if self.ndim == 3:
            total_points = config.n_x * config.n_y * config.n_z
            if grid_subsample is None:
                if total_points > 10000:
                    grid_subsample = max(2, int(np.cbrt(total_points / 1000)))
                elif total_points > 1000:
                    grid_subsample = 2
                else:
                    grid_subsample = 1
        else:
            total_points = config.n_x * config.n_y
            if grid_subsample is None:
                if total_points > 1000:
                    grid_subsample = max(2, int(np.sqrt(total_points / 500)))
                elif total_points > 200:
                    grid_subsample = 2
                else:
                    grid_subsample = 1
        self.grid_subsample = grid_subsample
        
        # Episode data
        self.states: List[NavigationArenaState] = []
    
    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return self.config.ndim
    
    def reset(self) -> None:
        """Reset renderer for new episode."""
        self.states = []
    
    def step(self, state: NavigationArenaState) -> None:
        """Record navigation arena state for visualization."""
        self.states.append(state)
    
    def render(self, mode: str) -> Union[None, np.ndarray, str]:
        """Render the visualization.
        
        Args:
            mode: 'human' (show in browser) or 'rgb_array' (return numpy array).
            
        Returns:
            None for 'human', numpy array (H, W, 3) for 'rgb_array'.
        """
        if mode not in self.render_modes:
            raise ValueError(f"Unsupported render mode: {mode}. Use one of {self.render_modes}")
        
        fig = self._create_figure()
        
        if mode == 'human':
            fig.show()
            return None
        elif mode == 'rgb_array':
            return self._fig_to_array(fig)

    @property
    def render_modes(self) -> List[str]:
        """Supported render modes."""
        return ['human', 'rgb_array']
    
    # ========================================================================
    # Export Methods
    # ========================================================================
    
    def save_gif(self, output_path: str, fps: int = 10, subsample: int = 1) -> None:
        from .exporters import save_gif
        save_gif(self, output_path, fps, subsample)
    
    def save_mp4(self, output_path: str, fps: int = 15, subsample: int = 1) -> None:
        from .exporters import save_mp4
        save_mp4(self, output_path, fps, subsample=subsample)
    
    def save_html(self, output_path: str) -> None:
        from .exporters import save_html
        save_html(self, output_path)
    
    def save_animated_html(self, output_path: str, fps: int = 10) -> None:
        from .exporters import save_animated_html
        save_animated_html(self, output_path, fps)
    
    # ========================================================================
    # Internal visualization methods
    # ========================================================================
    
    def _create_figure(self) -> go.Figure:
        """Create Plotly figure with all visualization elements."""
        fig = go.Figure()
        
        if not self.states:
            return fig  # Empty figure if no states
        
        # Extract info from latest state
        current_state = self.states[-1]
        
        # Add visualization elements
        if self.show_grid_points:
            self._add_grid_points(fig)
        
        self._add_target_vicinity(fig, current_state)
        self._add_target(fig, current_state)
        self._add_initial_position(fig, current_state)
        self._add_trajectory(fig)
        self._add_actor(fig, current_state)
        self._configure_layout(fig, current_state)
        
        return fig

    def _configure_layout(self, fig: go.Figure, state: NavigationArenaState):
        """Configure figure layout."""
        action_names = ['DEC', 'STAY', 'INC']
        action_name = action_names[state.last_action] if state.last_action is not None else 'N/A'
        
        title_text = (
            f"<b>Step: {state.step_count} | "
            f"Action: {action_name} | "
            f"Reward: {state.last_reward:+.2f} | "
            f"Cumulative: {state.cumulative_reward:+.2f}</b>"
        )
        
        if self.ndim == 3:
            self._configure_layout_3d(fig, title_text)
        else:
            self._configure_layout_2d(fig, title_text)
    
    def _configure_layout_3d(self, fig: go.Figure, title_text: str):
        """Configure 3D layout."""
        camera = dict(
            eye=self.camera_eye,  # Camera position
            center=dict(x=0, y=0, z=0),  # Look at center
            up=dict(x=0, y=0, z=1)  # Z-axis is up
        )
        
        fig.update_layout(
            title=dict(
                text=title_text, 
                font=dict(size=22),
                x=0.5,  # Center title
                xanchor='center',
                y=0.98,  # Near top
                yanchor='top'
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text='X (i)', font=dict(size=20)),
                    tickfont=dict(size=14),
                    range=[0.5, self.config.n_x + 0.5]
                ),
                yaxis=dict(
                    title=dict(text='Y (ambient)', font=dict(size=20)),
                    tickfont=dict(size=14),
                    range=[0.5, self.config.n_y + 0.5]
                ),
                zaxis=dict(
                    title=dict(text='Z (controllable)', font=dict(size=20)),
                    tickfont=dict(size=14),
                    range=[0.5, self.config.n_z + 0.5]
                ),
                aspectmode='data',
                camera=camera
            ),
            width=self.width,
            height=self.height,
            showlegend=True,
            legend=dict(
                x=0.01, y=0.99, 
                font=dict(size=12),
                bgcolor='rgba(255, 255, 255, 0.8)',  # Semi-transparent background
                bordercolor='gray',
                borderwidth=1
            ),
            # Adjust margins to prevent clipping
            margin=dict(l=10, r=10, t=60, b=10)  # Increased top margin for title
        )
    
    def _configure_layout_2d(self, fig: go.Figure, title_text: str):
        """Configure 2D layout."""
        fig.update_layout(
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
                range=[0.5, self.config.n_x + 0.5],
                scaleanchor='y',
                scaleratio=1
            ),
            yaxis=dict(
                title=dict(text='Y (controllable)', font=dict(size=16)),
                tickfont=dict(size=12),
                range=[0.5, self.config.n_y + 0.5]
            ),
            width=self.width,
            height=self.height,
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
    
    def _fig_to_array(self, fig: go.Figure) -> np.ndarray:
        """Convert Plotly figure to numpy array."""
        try:
            # Try using kaleido (preferred)
            img_bytes = fig.to_image(format='png', width=self.width, height=self.height)
            
            # Load with PIL and convert to numpy
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
            else:
                raise
    
    def _compute_scaling(self):
        """Compute marker sizes based on grid dimensions."""
        if self.ndim == 3:
            max_dim = max(self.config.n_x, self.config.n_y, self.config.n_z)
        else:
            max_dim = max(self.config.n_x, self.config.n_y)
        
        self.grid_point_size = max(2, 60 / max_dim)
        self.actor_size = max(10, 200 / max_dim)
        self.target_size = max(8, 150 / max_dim)
        self.trajectory_width = max(2, 25 / max_dim)
    
    # ========================================================================
    # Visualization elements (dimension-aware)
    # ========================================================================
    
    def _add_grid_points(self, fig: go.Figure):
        """Add subsampled grid points."""
        i_range = range(1, self.config.n_x + 1, self.grid_subsample)
        j_range = range(1, self.config.n_y + 1, self.grid_subsample)
        
        if self.ndim == 3:
            k_range = range(1, self.config.n_z + 1, self.grid_subsample)
            i_coords, j_coords, k_coords = [], [], []
            for i in i_range:
                for j in j_range:
                    for k in k_range:
                        i_coords.append(i)
                        j_coords.append(j)
                        k_coords.append(k)
            
            fig.add_trace(go.Scatter3d(
                x=i_coords, y=j_coords, z=k_coords,
                mode='markers',
                marker=dict(size=self.grid_point_size, color='gray', opacity=0.4),
                name='Grid',
                showlegend=False,
                hoverinfo='skip'
            ))
        else:
            i_coords, j_coords = [], []
            for i in i_range:
                for j in j_range:
                    i_coords.append(i)
                    j_coords.append(j)
            
            fig.add_trace(go.Scatter(
                x=i_coords, y=j_coords,
                mode='markers',
                marker=dict(size=self.grid_point_size, color='gray', opacity=0.4),
                name='Grid',
                showlegend=False,
                hoverinfo='skip'
            ))
    
    def _add_target_vicinity(self, fig: go.Figure, state: NavigationArenaState):
        """Add target vicinity region (cylinder for 3D, circle for 2D)."""
        if self.ndim == 3:
            # 3D: Cylinder
            theta = np.linspace(0, 2 * np.pi, 40)
            z_levels = np.linspace(1, self.config.n_z, 30)
            theta_grid, z_grid = np.meshgrid(theta, z_levels)
            x_cylinder = state.target_position.i + state.vicinity_radius * np.cos(theta_grid)
            y_cylinder = state.target_position.j + state.vicinity_radius * np.sin(theta_grid)
            
            fig.add_trace(go.Surface(
                x=x_cylinder, y=y_cylinder, z=z_grid,
                colorscale=[[0, 'lightgreen'], [1, 'lightgreen']],
                opacity=0.25,
                showscale=False,
                showlegend=False,
                hoverinfo='skip',
                name='Vicinity'
            ))
        else:
            # 2D: Circle
            theta = np.linspace(0, 2 * np.pi, 60)
            x_circle = state.target_position.i + state.vicinity_radius * np.cos(theta)
            y_circle = state.target_position.j + state.vicinity_radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=x_circle, y=y_circle,
                mode='lines',
                fill='toself',
                fillcolor='rgba(144, 238, 144, 0.3)',
                line=dict(color='lightgreen', width=2),
                name='Vicinity',
                showlegend=False,
                hoverinfo='skip'
            ))
    
    def _add_target(self, fig: go.Figure, state: NavigationArenaState):
        """Add target marker."""
        if self.ndim == 3:
            fig.add_trace(go.Scatter3d(
                x=[state.target_position.i],
                y=[state.target_position.j],
                z=[state.target_position.k],
                mode='markers',
                marker=dict(
                    size=self.target_size * 0.5,
                    color='green',
                    symbol='x',
                    line=dict(color='darkgreen', width=2)
                ),
                name='Target',
                showlegend=True
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[state.target_position.i],
                y=[state.target_position.j],
                mode='markers',
                marker=dict(
                    size=self.target_size,
                    color='green',
                    symbol='x',
                    line=dict(color='darkgreen', width=2)
                ),
                name='Target',
                showlegend=True
            ))
    
    def _add_initial_position(self, fig: go.Figure, state: NavigationArenaState):
        """Add initial position marker."""
        if self.ndim == 3:
            fig.add_trace(go.Scatter3d(
                x=[state.initial_position.i],
                y=[state.initial_position.j],
                z=[state.initial_position.k],
                mode='markers',
                marker=dict(
                    size=self.target_size * 0.8,
                    color='orange',
                    symbol='circle',
                    line=dict(color='darkorange', width=2),
                    opacity=0.8
                ),
                name='Start',
                showlegend=True
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[state.initial_position.i],
                y=[state.initial_position.j],
                mode='markers',
                marker=dict(
                    size=self.target_size * 0.8,
                    color='orange',
                    symbol='circle',
                    line=dict(color='darkorange', width=2),
                    opacity=0.8
                ),
                name='Start',
                showlegend=True
            ))
    
    def _add_trajectory(self, fig: go.Figure):
        """Add trajectory line."""
        positions = [s.position for s in self.states]
        
        if self.ndim == 3:
            traj_array = np.array([[p.i, p.j, p.k] for p in positions])
            fig.add_trace(go.Scatter3d(
                x=traj_array[:, 0],
                y=traj_array[:, 1],
                z=traj_array[:, 2],
                mode='lines+markers',
                line=dict(color='royalblue', width=self.trajectory_width * 2.5),
                marker=dict(size=self.trajectory_width * 3, color='steelblue', opacity=0.7),
                name='Trajectory',
                showlegend=True,
                opacity=0.8
            ))
        else:
            traj_array = np.array([[p.i, p.j] for p in positions])
            fig.add_trace(go.Scatter(
                x=traj_array[:, 0],
                y=traj_array[:, 1],
                mode='lines+markers',
                line=dict(color='royalblue', width=self.trajectory_width),
                marker=dict(size=self.trajectory_width * 1.5, color='steelblue', opacity=0.7),
                name='Trajectory',
                showlegend=True,
                opacity=0.8
            ))
    
    def _add_actor(self, fig: go.Figure, state: NavigationArenaState):
        """Add current actor position."""
        pos = state.position
        
        if self.ndim == 3:
            fig.add_trace(go.Scatter3d(
                x=[pos.i],
                y=[pos.j],
                z=[pos.k],
                mode='markers',
                marker=dict(
                    size=self.actor_size * 0.8,
                    color='red',
                    symbol='diamond',
                    line=dict(color='darkred', width=2.5)
                ),
                name='Actor',
                showlegend=True
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[pos.i],
                y=[pos.j],
                mode='markers',
                marker=dict(
                    size=self.actor_size,
                    color='red',
                    symbol='diamond',
                    line=dict(color='darkred', width=2)
                ),
                name='Actor',
                showlegend=True
            ))
