"""3D visualization renderer for grid environment."""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

from ..utils.types import GridPosition, GridConfig
from .visualization_recorder import VisualizationRecorder


class Grid3DRenderer:
    """Renders 3D visualizations of the grid environment."""
    
    def __init__(self, config: GridConfig, figsize: Tuple[float, float] = (12, 10)):
        """
        Initialize 3D renderer.
        
        Args:
            config: Grid configuration
            figsize: Figure size (width, height) in inches
        """
        self.config = config
        self.figsize = figsize
        
        # Calculate scaling factors based on grid size
        self._compute_scaling_factors()
        
    def _compute_scaling_factors(self) -> None:
        """Compute scaling factors for arrows and markers based on grid size."""
        # Base scaling on the maximum grid dimension
        max_dim = max(self.config.n_x, self.config.n_y, self.config.n_z)
        
        # Marker size (for grid points, actor, target)
        # Smaller grids get larger markers
        self.marker_size = max(20, 200 / max_dim)
        self.actor_marker_size = self.marker_size * 3
        self.target_marker_size = self.marker_size * 2.5
        
        # Arrow length scale (proportion of grid spacing)
        # Arrows should not overwhelm the visualization
        self.arrow_length_scale = 0.3 if max_dim < 10 else 0.2
        
        # Arrow width
        self.arrow_width = 0.015 if max_dim < 10 else 0.01
        
        # Grid point visibility threshold
        # For large grids, we might want to hide grid points to reduce clutter
        self.show_grid_points = max_dim <= 15
        
    def create_figure(self) -> Tuple[Figure, plt.Axes]:
        """
        Create a new 3D figure and axes.
        
        Returns:
            Tuple of (figure, axes)
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set labels
        ax.set_xlabel('X (i)', fontsize=10)
        ax.set_ylabel('Y (j)', fontsize=10)
        ax.set_zlabel('Z (k)', fontsize=10)
        
        # Set limits
        ax.set_xlim(0.5, self.config.n_x + 0.5)
        ax.set_ylim(0.5, self.config.n_y + 0.5)
        ax.set_zlim(0.5, self.config.n_z + 0.5)
        
        # Set grid
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def plot_grid_points(self, ax: plt.Axes, alpha: float = 0.3) -> None:
        """
        Plot grid points as small markers.
        
        Args:
            ax: Matplotlib 3D axes
            alpha: Transparency of grid points
        """
        if not self.show_grid_points:
            return
            
        # Create grid of all points
        i_coords, j_coords, k_coords = np.meshgrid(
            range(1, self.config.n_x + 1),
            range(1, self.config.n_y + 1),
            range(1, self.config.n_z + 1),
            indexing='ij'
        )
        
        ax.scatter(i_coords.flatten(), j_coords.flatten(), k_coords.flatten(),
                  c='gray', s=self.marker_size, alpha=alpha, marker='o')
    
    def plot_wind_field(self, ax: plt.Axes, wind_field: np.ndarray,
                       subsample: Optional[int] = None) -> None:
        """
        Plot wind field as arrows.
        
        Args:
            ax: Matplotlib 3D axes
            wind_field: Wind field array of shape (n_x, n_y, n_z, 2)
                       where last dimension is (u, v) displacement
            subsample: If provided, only plot every nth arrow in each dimension
                      Useful for large grids to reduce clutter
        """
        # Auto-subsample for large grids
        if subsample is None:
            max_dim = max(self.config.n_x, self.config.n_y, self.config.n_z)
            subsample = max(1, max_dim // 12)
        
        # Create grid indices
        i_range = range(1, self.config.n_x + 1, subsample)
        j_range = range(1, self.config.n_y + 1, subsample)
        k_range = range(1, self.config.n_z + 1, subsample)
        
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    # Get displacement at this grid point (convert to 0-indexed)
                    u, v = wind_field[i-1, j-1, k-1, 0], wind_field[i-1, j-1, k-1, 1]
                    
                    # Skip if displacement is zero
                    magnitude = np.sqrt(u**2 + v**2)
                    if magnitude < 1e-6:
                        continue
                    
                    # Scale arrow length
                    scale = self.arrow_length_scale / (1.0 + 0.1 * magnitude)
                    u_scaled = u * scale
                    v_scaled = v * scale
                    
                    # Plot arrow
                    ax.quiver(i, j, k, u_scaled, v_scaled, 0,
                            color='blue', alpha=0.5, arrow_length_ratio=0.3,
                            linewidth=1.5, pivot='middle')
    
    def plot_actor(self, ax: plt.Axes, position: GridPosition,
                  color: str = 'red', label: str = 'Actor') -> None:
        """
        Plot actor as a star marker.
        
        Args:
            ax: Matplotlib 3D axes
            position: Actor position
            color: Marker color
            label: Legend label
        """
        ax.scatter([position.i], [position.j], [position.k],
                  c=color, s=self.actor_marker_size, marker='*',
                  edgecolors='black', linewidths=1.5, label=label, zorder=10)
    
    def plot_target(self, ax: plt.Axes, position: GridPosition,
                   vicinity_radius: Optional[float] = None,
                   color: str = 'green') -> None:
        """
        Plot target as a circle with optional vicinity region.
        
        Args:
            ax: Matplotlib 3D axes
            position: Target position
            vicinity_radius: Radius of vicinity region (semi-transparent)
            color: Target color
        """
        # Plot target as circle marker
        ax.scatter([position.i], [position.j], [position.k],
                  c=color, s=self.target_marker_size, marker='o',
                  edgecolors='black', linewidths=1.5, label='Target', zorder=9)
        
        # Plot vicinity region if specified
        if vicinity_radius is not None and vicinity_radius > 0:
            # Create a sphere or circle at the target position
            # For 3D, we'll draw circles at the target z-level and adjacent levels
            theta = np.linspace(0, 2*np.pi, 30)
            
            # Draw vicinity circle at target level
            x_circle = position.i + vicinity_radius * np.cos(theta)
            y_circle = position.j + vicinity_radius * np.sin(theta)
            z_circle = np.full_like(x_circle, position.k)
            
            ax.plot(x_circle, y_circle, z_circle,
                   color='cyan', alpha=0.4, linewidth=2, linestyle='--')
            
            # Fill the circle (approximate with triangulation)
            # Draw a semi-transparent patch in 3D
            circle = Circle((position.i, position.j), vicinity_radius,
                          color='cyan', alpha=0.15)
            ax.add_patch(circle)
            art3d.pathpatch_2d_to_3d(circle, z=position.k, zdir="z")
    
    def plot_trajectory(self, ax: plt.Axes, trajectory: np.ndarray,
                       color: str = 'orange', linewidth: float = 2.5) -> None:
        """
        Plot actor trajectory as a line.
        
        Args:
            ax: Matplotlib 3D axes
            trajectory: Array of shape (n_steps, 3) with (i, j, k) coordinates
            color: Line color
            linewidth: Line width
        """
        if len(trajectory) < 2:
            return
            
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
               color=color, linewidth=linewidth, alpha=0.7,
               label='Trajectory', linestyle='-', marker='o', markersize=4)
    
    def render_static(self, recorder: VisualizationRecorder,
                     step_idx: Optional[int] = None,
                     title: Optional[str] = None,
                     show_wind: bool = True,
                     show_trajectory: bool = True,
                     show_grid: bool = True) -> Figure:
        """
        Render a static 3D visualization of the environment.
        
        Args:
            recorder: VisualizationRecorder with episode data
            step_idx: Which step to visualize (None = last step)
            title: Plot title
            show_wind: Whether to show wind field arrows
            show_trajectory: Whether to show actor trajectory
            show_grid: Whether to show grid points
            
        Returns:
            Matplotlib figure
        """
        fig, ax = self.create_figure()
        
        # Determine which step to show
        if step_idx is None:
            step_idx = len(recorder) - 1
        step_idx = max(0, min(step_idx, len(recorder) - 1))
        
        # Plot grid points
        if show_grid:
            self.plot_grid_points(ax)
        
        # Plot wind field
        if show_wind and recorder.get_wind_field() is not None:
            self.plot_wind_field(ax, recorder.get_wind_field())
        
        # Plot target
        if recorder.target_position is not None:
            self.plot_target(ax, recorder.target_position,
                           vicinity_radius=recorder.target_vicinity_radius)
        
        # Plot trajectory up to current step
        if show_trajectory and len(recorder) > 0:
            traj = recorder.get_trajectory_array()[:step_idx+1]
            if len(traj) > 1:
                self.plot_trajectory(ax, traj)
        
        # Plot actor at current position
        if len(recorder) > 0:
            current_pos = recorder.trajectory[step_idx]
            self.plot_actor(ax, current_pos)
        
        # Set title
        if title is None:
            title = f'Grid Environment - Step {step_idx}/{len(recorder)-1}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend
        ax.legend(loc='upper left', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def render_frame(self, recorder: VisualizationRecorder, step_idx: int,
                    show_wind: bool = True, show_trajectory: bool = True,
                    show_grid: bool = True) -> Figure:
        """
        Render a single frame for animation.
        
        Args:
            recorder: VisualizationRecorder with episode data
            step_idx: Which step to render
            show_wind: Whether to show wind field
            show_trajectory: Whether to show trajectory up to this point
            show_grid: Whether to show grid points
            
        Returns:
            Matplotlib figure
        """
        summary = recorder.get_episode_summary()
        title = f"Step {step_idx}/{summary['num_steps']-1} | Reward: {recorder.rewards[step_idx]:.2f}"
        
        return self.render_static(recorder, step_idx=step_idx, title=title,
                                show_wind=show_wind, show_trajectory=show_trajectory,
                                show_grid=show_grid)
    
    def save_figure(self, fig: Figure, filepath: str, dpi: int = 150) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib figure
            filepath: Output file path
            dpi: Resolution in dots per inch
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

