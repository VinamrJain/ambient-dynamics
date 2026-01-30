"""2D Navigation arena renderer using Plotly backend.

Visualization for 2D navigation tasks where:
- X-axis: Ambient axis (field-controlled)
- Y-axis: Controllable axis (agent-controlled)

Supports:
- Standard Gymnasium modes: 'human' (interactive), 'rgb_array' (numpy array)
- Export methods: save_gif(), save_mp4(), save_html()
"""

from typing import List, Union, Optional
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from PIL import Image

from .renderer import Renderer
from ..utils.types import GridConfig, NavigationArenaState


class NavigationRenderer2D(Renderer):
    """2D Renderer for navigation tasks (Plotly backend).
    
    Extracts all visualization info from NavigationArenaState.
    
    Features:
    - Interactive 2D visualization
    - Grid points (with smart sub-sampling for large grids)
    - Actor position and trajectory
    - Target position with vicinity circle
    - Episode info (step, action, reward, cumulative reward)
    """
    
    def __init__(
        self,
        config: GridConfig,
        width: int = 800,
        height: int = 600,
        show_grid_points: bool = True,
        grid_subsample: Optional[int] = None,
        backend: str = 'plotly'
    ):
        """Initialize 2D navigation renderer.
        
        Args:
            config: Grid configuration (must be 2D: n_z is None).
            width: Figure width in pixels.
            height: Figure height in pixels.
            show_grid_points: Whether to show grid points.
            grid_subsample: Subsample factor for grid points (auto if None).
            backend: Rendering backend ('plotly').
        """
        if config.ndim != 2:
            raise ValueError(
                f"NavigationRenderer2D requires 2D config (n_z=None), got ndim={config.ndim}"
            )
        
        self.config = config
        self.width = width
        self.height = height
        self.show_grid_points = show_grid_points
        self.backend = backend
        
        # Compute smart scaling based on grid size
        self._compute_scaling()
        
        # Determine grid subsampling
        total_points = config.n_x * config.n_y
        if grid_subsample is None:
            if total_points > 1000:
                grid_subsample = max(2, int(np.sqrt(total_points / 500)))
            elif total_points > 200:
                grid_subsample = 2
            else:
                grid_subsample = 1
        self.grid_subsample = grid_subsample
        
        # Episode data (extracted from NavigationArenaState)
        self.states: List[NavigationArenaState] = []
    
    def reset(self) -> None:
        """Reset renderer for new episode."""
        self.states = []
    
    def step(self, state: NavigationArenaState) -> None:
        """Record navigation arena state for visualization.
        
        Args:
            state: Complete navigation arena state containing all episode info.
        """
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
    
    # ========================================================================
    # Internal visualization methods
    # ========================================================================
    
    def _create_figure(self) -> go.Figure:
        """Create Plotly figure with all 2D visualization elements."""
        fig = go.Figure()
        
        if not self.states:
            return fig
        
        current_state = self.states[-1]
        
        # 1. Grid points (optional, subsampled)
        if self.show_grid_points:
            self._add_grid_points(fig)
        
        # 2. Target vicinity circle
        self._add_target_vicinity(fig, current_state)
        
        # 3. Target marker
        self._add_target(fig, current_state)
        
        # 4. Initial position marker
        self._add_initial_position(fig, current_state)
        
        # 5. Trajectory
        self._add_trajectory(fig)
        
        # 6. Current actor position
        self._add_actor(fig, current_state)
        
        # 7. Configure layout
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
                x=0.01, 
                y=0.99, 
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
            img_bytes = fig.to_image(format='png', width=self.width, height=self.height)
            img = Image.open(BytesIO(img_bytes))
            img_array = np.array(img)
            
            if img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            return img_array
        
        except ValueError as e:
            if 'kaleido' in str(e).lower():
                print("\nERROR: rgb_array mode requires 'kaleido' package.")
                print("   Install with: pip install kaleido")
            else:
                raise
    
    def _compute_scaling(self):
        """Compute marker sizes based on grid dimensions."""
        max_dim = max(self.config.n_x, self.config.n_y)
        
        self.grid_point_size = max(3, 80 / max_dim)
        self.actor_size = max(12, 250 / max_dim)
        self.target_size = max(10, 200 / max_dim)
        self.trajectory_width = max(2, 30 / max_dim)
    
    def _add_grid_points(self, fig: go.Figure):
        """Add subsampled grid points."""
        i_range = range(1, self.config.n_x + 1, self.grid_subsample)
        j_range = range(1, self.config.n_y + 1, self.grid_subsample)
        
        i_coords, j_coords = [], []
        for i in i_range:
            for j in j_range:
                i_coords.append(i)
                j_coords.append(j)
        
        fig.add_trace(go.Scatter(
            x=i_coords, y=j_coords,
            mode='markers',
            marker=dict(
                size=self.grid_point_size,
                color='gray',
                opacity=0.4
            ),
            name='Grid',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    def _add_target_vicinity(self, fig: go.Figure, state: NavigationArenaState):
        """Add circle representing target vicinity."""
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
        current_pos = state.position

        fig.add_trace(go.Scatter(
            x=[current_pos.i],
            y=[current_pos.j],
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
