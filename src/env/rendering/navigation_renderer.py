"""Navigation arena renderer using Plotly backend."""

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
    Backend-agnostic design allows switching to matplotlib or other libraries.
    
    Features:
    - Interactive 3D visualization
    - Grid points (with smart sub-sampling for large grids)
    - Actor position and trajectory
    - Target position with vicinity cylinder
    - Episode info (step, action, reward, cumulative reward)
    """
    
    def __init__(
        self,
        config: GridConfig,
        width: int = 1024,
        height: int = 768,
        show_grid_points: bool = True,
        grid_subsample: Optional[int] = None,
        backend: str = 'plotly'
    ):
        """Initialize navigation renderer.
        
        Args:
            config: Grid configuration.
            width: Figure width in pixels.
            height: Figure height in pixels.
            show_grid_points: Whether to show grid points.
            grid_subsample: Subsample factor for grid points (auto if None).
            backend: Rendering backend ('plotly' or 'matplotlib' - future).
        """
        self.config = config
        self.width = width
        self.height = height
        self.show_grid_points = show_grid_points
        self.backend = backend
        
        # Compute smart scaling based on grid size
        self._compute_scaling()
        
        # Determine grid subsampling
        total_points = config.n_x * config.n_y * config.n_z
        if grid_subsample is None:
            # Auto-subsample for large grids
            if total_points > 10000:
                grid_subsample = max(2, int(np.cbrt(total_points / 1000)))
            elif total_points > 1000:
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
        
        # Create figure
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
    
    def _create_figure(self) -> go.Figure:
        """Create Plotly figure with all visualization elements."""
        fig = go.Figure()
        
        if not self.states:
            return fig  # Empty figure if no states
        
        # Extract info from latest state
        current_state = self.states[-1]
        
        # 1. Grid points (optional, subsampled)
        if self.show_grid_points:
            self._add_grid_points(fig)
        
        # 2. Target vicinity cylinder
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
        """Configure figure layout and camera."""
        # Extract info from state
        action_names = ['DOWN', 'STAY', 'UP']
        action_name = action_names[state.last_action] if state.last_action is not None else 'N/A'
        
        # Title with episode info
        title_text = (
            f"Step: {state.step_count} | "
            f"Action: {action_name} | "
            f"Reward: {state.last_reward:+.2f} | "
            f"Cumulative: {state.cumulative_reward:+.2f}"
        )
        
        fig.update_layout(
            title=dict(text=title_text, font=dict(size=24, weight='bold')),
            scene=dict(
                xaxis=dict(
                    title=dict(text='X (i)', font=dict(size=20)),
                    tickfont=dict(size=16),
                    range=[0.5, self.config.n_x + 0.5]
                ),
                yaxis=dict(
                    title=dict(text='Y (j)', font=dict(size=20)),
                    tickfont=dict(size=16),
                    range=[0.5, self.config.n_y + 0.5]
                ),
                zaxis=dict(
                    title=dict(text='Z (k)', font=dict(size=20)),
                    tickfont=dict(size=16),
                    range=[0.5, self.config.n_z + 0.5]
                ),
                aspectmode='data'
            ),
            width=self.width,
            height=self.height,
            showlegend=True,
            legend=dict(x=0.02, y=0.98, font=dict(size=16)),
            margin=dict(l=0, r=0, t=40, b=0)
        )
    
    def _fig_to_array(self, fig: go.Figure) -> np.ndarray:
        """Convert Plotly figure to numpy array.
        
        Args:
            fig: Plotly figure.
            
        Returns:
            RGB array of shape (height, width, 3).
        """
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
                print("\n⚠️  Note: rgb_array mode requires 'kaleido' package.")
                print("   Install with: pip install kaleido")
                print("   For now, use mode='human' to view in browser.")
                print("   Returning placeholder array.\n")
                
                # Return placeholder array
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                raise
    
    def _compute_scaling(self):
            """Compute marker sizes based on grid dimensions."""
            # Scale markers inversely with grid size
            max_dim = max(self.config.n_x, self.config.n_y, self.config.n_z)
            
            # Grid point size
            self.grid_point_size = max(1, 50 / max_dim)
            
            # Actor marker size
            self.actor_size = max(10, 200 / max_dim)
            
            # Target marker size
            self.target_size = max(8, 150 / max_dim)
            
            # Trajectory line width
            self.trajectory_width = max(2, 20 / max_dim)
    
    def _add_grid_points(self, fig: go.Figure):
        """Add subsampled grid points."""
        # Generate subsampled grid
        i_range = range(1, self.config.n_x + 1, self.grid_subsample)
        j_range = range(1, self.config.n_y + 1, self.grid_subsample)
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
        """Add translucent cylinder representing target vicinity."""
        # Create cylinder mesh using Mesh3d
        theta = np.linspace(0, 2 * np.pi, 40)
        z_levels = np.linspace(1, self.config.n_z, 30)

        # Create cylinder surface
        theta_grid, z_grid = np.meshgrid(theta, z_levels)
        x_cylinder = state.target_position.i + state.vicinity_radius * np.cos(theta_grid)
        y_cylinder = state.target_position.j + state.vicinity_radius * np.sin(theta_grid)

        # Add cylinder as surface
        fig.add_trace(go.Surface(
            x=x_cylinder,
            y=y_cylinder,
            z=z_grid,
            colorscale=[[0, 'lightgreen'], [1, 'lightgreen']],
            opacity=0.25,
            showscale=False,
            showlegend=False,
            hoverinfo='skip',
            name='Vicinity'
        ))
    
    def _add_target(self, fig: go.Figure, state: NavigationArenaState):
        """Add target marker."""
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
            # text=['★'],
            # textfont=dict(size=self.target_size * 1.5, color='darkgreen'),
            # textposition='middle center',
            name='Target',
            showlegend=True
        ))
    
    def _add_initial_position(self, fig: go.Figure, state: NavigationArenaState):
        """Add initial position marker."""
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
    
    def _add_trajectory(self, fig: go.Figure):
        """Add trajectory line."""
        # Extract positions from states
        positions = [s.position for s in self.states]
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
    
    def _add_actor(self, fig: go.Figure, state: NavigationArenaState):
        """Add current actor position."""
        current_pos = state.position

        fig.add_trace(go.Scatter3d(
            x=[current_pos.i],
            y=[current_pos.j],
            z=[current_pos.k],
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
