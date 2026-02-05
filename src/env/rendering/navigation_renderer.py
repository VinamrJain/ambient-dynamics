from typing import List, Union, Optional, TYPE_CHECKING
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

from .renderer import Renderer
from .rendering_utils import (
    compute_scaling,
    get_layout_config_2d,
    get_layout_config_3d,
    get_animated_layout_2d,
    get_animated_layout_3d,
    fig_to_array,
)
from ..utils.types import GridConfig, NavigationArenaState, GridPosition

if TYPE_CHECKING:
    from ..field.abstract_field import AbstractField


class NavigationRenderer(Renderer):
    """Navigation arena renderer using Plotly backend.

    Supports both 2D and 3D navigation tasks:
    - 2D: Uses Scatter plots, vicinity shown as circle
    - 3D: Uses Scatter3d plots, vicinity shown as cylinder

    Core visualization logic for navigation tasks. Supports:
    - Standard Gymnasium modes: 'human' (interactive), 'rgb_array' (numpy array)
    - Export methods: save_gif(), save_mp4(), save_html(), save_animated_html()
    """
    
    def __init__(
        self,
        config: GridConfig,
        width: int = 1024,
        height: int = 768,
        show_grid_points: bool = True,
        grid_subsample: Optional[int] = None,
        backend: str = 'plotly',
        camera_eye: Optional[dict] = None,
        field: Optional['AbstractField'] = None,
        show_field: bool = False
    ):
        """Initialize navigation renderer.
        
        Args:
            config: Grid configuration.
            width: Figure width in pixels.
            height: Figure height in pixels.
            show_grid_points: Whether to show grid points.
            grid_subsample: Subsample factor for grid points and field arrows (auto if None).
            backend: Rendering backend ('plotly').
            camera_eye: Camera position for 3D (ignored for 2D).
                       Defaults to (1.5, -1.5, 1.0) for 45 degree perspective.
                       Examples:
                         - Top-down: {'x': 0, 'y': 0, 'z': 2.5}
                         - Side view: {'x': 2.5, 'y': 0, 'z': 0}
                         - Isometric: {'x': 1.5, 'y': -1.5, 'z': 1.0}
            field: Optional field for visualizing mean displacements.
            show_field: Whether to show field arrows (requires field with get_mean_displacement).
        """
        self.config = config
        self.width = width
        self.height = height
        self.show_grid_points = show_grid_points
        self.backend = backend
        self.camera_eye = camera_eye or {'x': 1.5, 'y': -1.5, 'z': 1.0}
        self.field = field
        self.show_field = show_field
        
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
            return fig_to_array(fig, self.width, self.height)

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
        
        if self.show_field:
            self._add_field(fig)
        
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
            layout_config = get_layout_config_3d(
                self.config, title_text, self.width, self.height, self.camera_eye
            )
        else:
            layout_config = get_layout_config_2d(
                self.config, title_text, self.width, self.height
            )
        fig.update_layout(**layout_config)
    
    def _compute_scaling(self):
        """Compute marker sizes based on grid dimensions."""
        scaling = compute_scaling(self.config)
        self.grid_point_size = scaling['grid_point_size']
        self.actor_size = scaling['actor_size']
        self.target_size = scaling['target_size']
        self.trajectory_width = scaling['trajectory_width']
    
    # ========================================================================
    # Visualization elements (dimension-aware)
    # ========================================================================
    
    def _add_grid_points(self, fig: go.Figure):
        """Add subsampled grid points."""
        fig.add_trace(self._get_grid_points_trace())
    
    def _add_field(self, fig: go.Figure):
        """Add field mean displacement arrows."""
        trace = self._get_field_trace()
        if trace is not None:
            if isinstance(trace, list):
                for t in trace:
                    fig.add_trace(t)
            else:
                fig.add_trace(trace)
    
    def _add_target_vicinity(self, fig: go.Figure, state: NavigationArenaState):
        """Add target vicinity region (cylinder for 3D, circle for 2D)."""
        fig.add_trace(self._get_target_vicinity_trace(state))
    
    def _add_target(self, fig: go.Figure, state: NavigationArenaState):
        """Add target marker."""
        fig.add_trace(self._get_target_trace(state))
    
    def _add_initial_position(self, fig: go.Figure, state: NavigationArenaState):
        """Add initial position marker."""
        fig.add_trace(self._get_initial_position_trace(state))
    
    def _add_trajectory(self, fig: go.Figure):
        """Add trajectory line."""
        fig.add_trace(self._get_trajectory_trace())
    
    def _add_actor(self, fig: go.Figure, state: NavigationArenaState):
        """Add current actor position."""
        fig.add_trace(self._get_actor_trace(state))
    
    # ========================================================================
    # Trace data methods (for use by exporters)
    # ========================================================================
    
    def _get_grid_points_trace(self):
        """Get grid points as Plotly trace."""
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
            
            return go.Scatter3d(
                x=i_coords, y=j_coords, z=k_coords,
                mode='markers',
                marker=dict(size=self.grid_point_size, color='gray', opacity=0.4),
                name='Grid',
                showlegend=False,
                hoverinfo='skip'
            )
        else:
            i_coords, j_coords = [], []
            for i in i_range:
                for j in j_range:
                    i_coords.append(i)
                    j_coords.append(j)
            
            return go.Scatter(
                x=i_coords, y=j_coords,
                mode='markers',
                marker=dict(size=self.grid_point_size, color='gray', opacity=0.4),
                name='Grid',
                showlegend=False,
                hoverinfo='skip'
            )
    
    def _get_field_trace(self):
        """Get field mean displacement arrows as Plotly trace(s).
        
        Returns:
            - 2D: List of traces from create_quiver (arrows in x direction only)
            - 3D: go.Cone trace at all subsampled z-slices (arrows in x-y plane, w=0)
            - None if field not available, doesn't support get_mean_displacement,
              or all arrows are zero
        """
        if self.field is None:
            return None
        
        # Check if field supports mean displacement
        if not hasattr(self.field, 'get_mean_displacement'):
            return None
        
        i_range = range(1, self.config.n_x + 1, self.grid_subsample)
        j_range = range(1, self.config.n_y + 1, self.grid_subsample)
        
        if self.ndim == 3:
            # 3D: Cones showing (u, v, 0) displacement at all subsampled z-slices
            k_range = range(1, self.config.n_z + 1, self.grid_subsample)
            
            x_coords, y_coords, z_coords = [], [], []
            u_vals, v_vals, w_vals = [], [], []
            
            for i in i_range:
                for j in j_range:
                    for k in k_range:
                        pos = GridPosition(i, j, k)
                        mean = self.field.get_mean_displacement(pos)
                        if mean is not None:
                            x_coords.append(i)
                            y_coords.append(j)
                            z_coords.append(k)
                            u_vals.append(mean[0])
                            v_vals.append(mean[1] if len(mean) > 1 else 0.0)
                            w_vals.append(0.0)  # No z-component (field only affects ambient)
            
            if not x_coords:
                return None
            
            # Check if all arrows are zero (e.g., SimpleField)
            max_mag = max(np.sqrt(u**2 + v**2) for u, v in zip(u_vals, v_vals)) if u_vals else 0.0
            if max_mag < 1e-6:
                # All arrows are essentially zero - skip visualization
                return None
            
            # Scale for visibility
            sizeref = max(0.5, max_mag) if max_mag > 0 else 0.5
            
            return go.Cone(
                x=x_coords, y=y_coords, z=z_coords,
                u=u_vals, v=v_vals, w=w_vals,
                sizemode='absolute',
                sizeref=sizeref,
                anchor='tail',
                colorscale='Blues',
                opacity=0.6,
                showscale=False,
                name='Field',
                showlegend=True
            )
        else:
            # 2D: Quiver arrows showing (u, 0) displacement
            # In 2D, field only affects x (ambient), arrows point in x direction
            x_coords, y_coords = [], []
            u_vals, v_vals = [], []  # v is always 0 for 2D (field doesn't affect y)
            
            for i in i_range:
                for j in j_range:
                    pos = GridPosition(i, j, None)
                    mean = self.field.get_mean_displacement(pos)
                    if mean is not None:
                        x_coords.append(float(i))
                        y_coords.append(float(j))
                        u_vals.append(mean[0])
                        v_vals.append(0.0)  # Field only affects ambient (x), not controllable (y)
            
            if not x_coords:
                return None
            
            # Check if all arrows are zero (e.g., SimpleField)
            max_mag = max(abs(u) for u in u_vals) if u_vals else 0.0
            if max_mag < 1e-6:
                # All arrows are essentially zero - skip visualization
                return None
            
            # Scale arrows for visibility
            scale = 0.3 * self.grid_subsample / max_mag if max_mag > 0 else 0.3
            
            # Create quiver plot
            quiver_fig = ff.create_quiver(
                x_coords, y_coords, u_vals, v_vals,
                scale=scale,
                arrow_scale=0.3,
                line=dict(color='steelblue', width=1.5),
                name='Field'
            )
            
            # Return the traces (quiver creates multiple traces)
            traces = list(quiver_fig.data)
            # Only show legend for first trace
            for i, trace in enumerate(traces):
                trace.showlegend = (i == 0)
                trace.name = 'Field' if i == 0 else None
            
            return traces
    
    def _get_target_vicinity_trace(self, state: NavigationArenaState):
        """Get target vicinity region as Plotly trace."""
        if self.ndim == 3:
            theta = np.linspace(0, 2 * np.pi, 40)
            z_levels = np.linspace(1, self.config.n_z, 30)
            theta_grid, z_grid = np.meshgrid(theta, z_levels)
            x_cylinder = state.target_position.i + state.vicinity_radius * np.cos(theta_grid)
            y_cylinder = state.target_position.j + state.vicinity_radius * np.sin(theta_grid)
            
            return go.Surface(
                x=x_cylinder, y=y_cylinder, z=z_grid,
                colorscale=[[0, 'lightgreen'], [1, 'lightgreen']],
                opacity=0.25,
                showscale=False,
                showlegend=False,
                hoverinfo='skip',
                name='Vicinity'
            )
        else:
            theta = np.linspace(0, 2 * np.pi, 60)
            x_circle = state.target_position.i + state.vicinity_radius * np.cos(theta)
            y_circle = state.target_position.j + state.vicinity_radius * np.sin(theta)
            
            return go.Scatter(
                x=x_circle, y=y_circle,
                mode='lines',
                fill='toself',
                fillcolor='rgba(144, 238, 144, 0.3)',
                line=dict(color='lightgreen', width=2),
                name='Vicinity',
                showlegend=False,
                hoverinfo='skip'
            )
    
    def _get_target_trace(self, state: NavigationArenaState):
        """Get target marker as Plotly trace."""
        if self.ndim == 3:
            return go.Scatter3d(
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
            )
        else:
            return go.Scatter(
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
            )
    
    def _get_initial_position_trace(self, state: NavigationArenaState):
        """Get initial position marker as Plotly trace."""
        if self.ndim == 3:
            return go.Scatter3d(
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
            )
        else:
            return go.Scatter(
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
            )
    
    def _get_trajectory_trace(self, up_to_idx: int = None):
        """Get trajectory as Plotly trace.
        
        Args:
            up_to_idx: If provided, only include states up to this index.
        """
        if up_to_idx is not None:
            positions = [s.position for s in self.states[:up_to_idx + 1]]
        else:
            positions = [s.position for s in self.states]
        
        if self.ndim == 3:
            traj_array = np.array([[p.i, p.j, p.k] for p in positions])
            return go.Scatter3d(
                x=traj_array[:, 0],
                y=traj_array[:, 1],
                z=traj_array[:, 2],
                mode='lines+markers',
                line=dict(color='royalblue', width=self.trajectory_width * 2.5),
                marker=dict(size=self.trajectory_width * 3, color='steelblue', opacity=0.7),
                name='Trajectory',
                showlegend=True,
                opacity=0.8
            )
        else:
            traj_array = np.array([[p.i, p.j] for p in positions])
            return go.Scatter(
                x=traj_array[:, 0],
                y=traj_array[:, 1],
                mode='lines+markers',
                line=dict(color='royalblue', width=self.trajectory_width),
                marker=dict(size=self.trajectory_width * 1.5, color='steelblue', opacity=0.7),
                name='Trajectory',
                showlegend=True,
                opacity=0.8
            )
    
    def _get_actor_trace(self, state: NavigationArenaState):
        """Get actor marker as Plotly trace."""
        pos = state.position
        
        if self.ndim == 3:
            return go.Scatter3d(
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
            )
        else:
            return go.Scatter(
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
            )
    
    def _get_animated_layout(self) -> go.Layout:
        """Get layout for animated figure (used by html exporter)."""
        if self.ndim == 3:
            return get_animated_layout_3d(
                self.config, self.width, self.height, self.camera_eye
            )
        else:
            return get_animated_layout_2d(self.config, self.width, self.height)
