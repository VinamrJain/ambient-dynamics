"""Advanced visualization using matplotlib/plotly for the grid environment."""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from ..utils.types import GridPosition, GridConfig

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class AdvancedRenderer:
    """Advanced renderer with matplotlib and plotly support."""

    def __init__(self, config: GridConfig, backend: str = "matplotlib"):
        """
        Initialize advanced renderer.

        Args:
            config: Grid configuration
            backend: "matplotlib" or "plotly"
        """
        self.config = config
        self.backend = backend.lower()

        if self.backend == "matplotlib" and not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for advanced rendering")
        elif self.backend == "plotly" and not HAS_PLOTLY:
            raise ImportError("plotly is required for advanced rendering")

    def render_environment_state(self, actor_position: GridPosition,
                                field_state: Optional[Dict[str, Any]] = None,
                                target_position: Optional[GridPosition] = None,
                                step_count: int = 0,
                                save_path: Optional[str] = None) -> None:
        """
        Render the complete environment state including wind field.

        Args:
            actor_position: Current actor position
            field_state: Field state for wind visualization
            target_position: Optional target position
            step_count: Current step number
            save_path: Optional path to save the figure
        """
        if self.backend == "matplotlib":
            self._render_matplotlib(actor_position, field_state, target_position,
                                  step_count, save_path)
        elif self.backend == "plotly":
            self._render_plotly(actor_position, field_state, target_position,
                              step_count, save_path)

    def _render_matplotlib(self, actor_position: GridPosition,
                          field_state: Optional[Dict[str, Any]],
                          target_position: Optional[GridPosition],
                          step_count: int,
                          save_path: Optional[str]) -> None:
        """Render using matplotlib."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Install with: pip install matplotlib")
            return

        # Create figure with subplots for each layer
        n_layers = self.config.n_z
        fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
        if n_layers == 1:
            axes = [axes]

        fig.suptitle(f"Grid Environment - Step {step_count}", fontsize=14)

        for layer_idx, ax in enumerate(axes):
            layer = layer_idx + 1  # 1-indexed
            self._render_layer_matplotlib(ax, layer, actor_position,
                                        field_state, target_position)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def _render_layer_matplotlib(self, ax, layer: int, actor_position: GridPosition,
                                field_state: Optional[Dict[str, Any]],
                                target_position: Optional[GridPosition]) -> None:
        """Render a single layer using matplotlib."""
        ax.set_xlim(0.5, self.config.n_x + 0.5)
        ax.set_ylim(0.5, self.config.n_y + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Layer {layer}")

        # Draw grid cells
        for i in range(1, self.config.n_x + 1):
            for j in range(1, self.config.n_y + 1):
                rect = patches.Rectangle((i-0.4, j-0.4), 0.8, 0.8,
                                       linewidth=1, edgecolor='gray',
                                       facecolor='lightblue', alpha=0.3)
                ax.add_patch(rect)

        # Draw wind field if available
        if field_state and hasattr(self, '_get_wind_vectors'):
            self._draw_wind_field_matplotlib(ax, layer, field_state)

        # Draw target if on this layer
        if (target_position and target_position.k == layer):
            target_circle = patches.Circle((target_position.i, target_position.j),
                                         0.3, color='red', alpha=0.7, label='Target')
            ax.add_patch(target_circle)

        # Draw actor if on this layer
        if actor_position.k == layer:
            actor_circle = patches.Circle((actor_position.i, actor_position.j),
                                        0.2, color='blue', label='Actor')
            ax.add_patch(actor_circle)

        ax.set_xticks(range(1, self.config.n_x + 1))
        ax.set_yticks(range(1, self.config.n_y + 1))

    def _render_plotly(self, actor_position: GridPosition,
                      field_state: Optional[Dict[str, Any]],
                      target_position: Optional[GridPosition],
                      step_count: int,
                      save_path: Optional[str]) -> None:
        """Render using plotly."""
        if not HAS_PLOTLY:
            print("Plotly not available. Install with: pip install plotly")
            return

        # Create subplots for each layer
        n_layers = self.config.n_z
        fig = make_subplots(
            rows=1, cols=n_layers,
            subplot_titles=[f"Layer {i+1}" for i in range(n_layers)],
            specs=[[{"type": "scatter"}] * n_layers]
        )

        for layer in range(1, n_layers + 1):
            col = layer
            self._render_layer_plotly(fig, layer, col, actor_position,
                                    field_state, target_position)

        fig.update_layout(
            title=f"Grid Environment - Step {step_count}",
            showlegend=True,
            height=400,
            width=300 * n_layers
        )

        if save_path:
            fig.write_html(save_path)

        fig.show()

    def _render_layer_plotly(self, fig, layer: int, col: int,
                           actor_position: GridPosition,
                           field_state: Optional[Dict[str, Any]],
                           target_position: Optional[GridPosition]) -> None:
        """Render a single layer using plotly."""
        # Grid background
        x_grid, y_grid = np.meshgrid(range(1, self.config.n_x + 1),
                                   range(1, self.config.n_y + 1))

        fig.add_trace(
            go.Scatter(
                x=x_grid.flatten(),
                y=y_grid.flatten(),
                mode='markers',
                marker=dict(size=20, color='lightblue', opacity=0.3),
                name='Grid',
                showlegend=(col == 1)
            ),
            row=1, col=col
        )

        # Target position
        if target_position and target_position.k == layer:
            fig.add_trace(
                go.Scatter(
                    x=[target_position.i],
                    y=[target_position.j],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Target',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )

        # Actor position
        if actor_position.k == layer:
            fig.add_trace(
                go.Scatter(
                    x=[actor_position.i],
                    y=[actor_position.j],
                    mode='markers',
                    marker=dict(size=12, color='blue', symbol='circle'),
                    name='Actor',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )

        # Update axes
        fig.update_xaxes(range=[0.5, self.config.n_x + 0.5], row=1, col=col)
        fig.update_yaxes(range=[0.5, self.config.n_y + 0.5], row=1, col=col)

    def animate_trajectory(self, trajectory: List[Tuple[GridPosition, Dict[str, Any]]],
                          target_position: Optional[GridPosition] = None,
                          save_path: Optional[str] = None) -> None:
        """
        Create an animated visualization of the trajectory.

        Args:
            trajectory: List of (position, info) tuples
            target_position: Optional target position
            save_path: Optional path to save animation
        """
        if not trajectory:
            return

        if self.backend == "matplotlib" and HAS_MATPLOTLIB:
            self._animate_matplotlib(trajectory, target_position, save_path)
        elif self.backend == "plotly" and HAS_PLOTLY:
            print("Plotly animation not implemented yet. Use matplotlib backend.")
        else:
            print(f"Animation not available for backend: {self.backend}")

    def _animate_matplotlib(self, trajectory: List[Tuple[GridPosition, Dict[str, Any]]],
                           target_position: Optional[GridPosition],
                           save_path: Optional[str]) -> None:
        """Create matplotlib animation."""
        if not HAS_MATPLOTLIB:
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0.5, self.config.n_x + 0.5)
        ax.set_ylim(0.5, self.config.n_y + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Draw static elements
        for i in range(1, self.config.n_x + 1):
            for j in range(1, self.config.n_y + 1):
                rect = patches.Rectangle((i-0.4, j-0.4), 0.8, 0.8,
                                       linewidth=1, edgecolor='gray',
                                       facecolor='lightblue', alpha=0.3)
                ax.add_patch(rect)

        if target_position:
            target_circle = patches.Circle((target_position.i, target_position.j),
                                         0.3, color='red', alpha=0.7, label='Target')
            ax.add_patch(target_circle)

        # Actor and trajectory elements
        actor_circle = patches.Circle((0, 0), 0.2, color='blue', label='Actor')
        ax.add_patch(actor_circle)

        trajectory_line, = ax.plot([], [], 'b--', alpha=0.5, linewidth=2, label='Path')

        ax.legend()
        ax.set_title("Trajectory Animation")

        def animate(frame):
            if frame < len(trajectory):
                pos, info = trajectory[frame]
                actor_circle.center = (pos.i, pos.j)

                # Update trajectory line
                x_data = [p.i for p, _ in trajectory[:frame+1]]
                y_data = [p.j for p, _ in trajectory[:frame+1]]
                trajectory_line.set_data(x_data, y_data)

                ax.set_title(f"Step {frame} - Layer {pos.k}")

            return actor_circle, trajectory_line

        anim = FuncAnimation(fig, animate, frames=len(trajectory),
                           interval=500, blit=False, repeat=True)

        if save_path:
            anim.save(save_path, writer='pillow', fps=2)

        plt.show()

    def _get_wind_vectors(self, field_state: Dict[str, Any], layer: int):
        """Extract wind vectors for visualization (placeholder)."""
        # This would need to be implemented based on the specific field implementation
        # For now, return empty to avoid errors
        return [], [], [], []