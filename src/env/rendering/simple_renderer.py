"""Simple text-based rendering for the grid environment."""

from typing import Dict, Any, Optional
from ..utils.types import GridPosition, GridConfig


class SimpleRenderer:
    """Simple text-based renderer for debugging and basic visualization."""

    def __init__(self, config: GridConfig):
        """Initialize renderer with grid configuration."""
        self.config = config

    def render_grid(self, actor_position: GridPosition, step_count: int = 0,
                   target_position: Optional[GridPosition] = None,
                   info: Optional[Dict[str, Any]] = None) -> None:
        """
        Render a simple 2D text representation of the current state.

        Args:
            actor_position: Current actor position
            step_count: Current step number
            target_position: Optional target position to display
            info: Optional additional info to display
        """
        print(f"\n=== Step {step_count} - Layer {actor_position.k}/{self.config.n_z} ===")

        # Print grid from top to bottom (y-axis reversed for display)
        for j in range(self.config.n_y, 0, -1):
            row = f"{j:2d}|"
            for i in range(1, self.config.n_x + 1):
                symbol = self._get_cell_symbol(i, j, actor_position, target_position)
                row += f" {symbol} "
            print(row)

        # Print column numbers
        col_labels = "   " + "".join(f"{i:3d}" for i in range(1, self.config.n_x + 1))
        print(col_labels)

        # Print additional info
        if info:
            print(f"Reward: {info.get('reward', 'N/A')}")
            if 'horizontal_displacement' in info and info['horizontal_displacement'] is not None:
                disp = info['horizontal_displacement']
                print(f"Wind: ({disp.u:+d}, {disp.v:+d})")
        print()

    def _get_cell_symbol(self, i: int, j: int, actor_position: GridPosition,
                        target_position: Optional[GridPosition]) -> str:
        """Get the symbol to display for a grid cell."""
        # Check if actor is at this position (on current layer)
        if (i == actor_position.i and j == actor_position.j and
            actor_position.k == actor_position.k):
            return "A"

        # Check if target is at this position (if provided)
        if (target_position and i == target_position.i and
            j == target_position.j and actor_position.k == target_position.k):
            return "T"

        # Check if target is at this (i,j) but different layer
        if (target_position and i == target_position.i and
            j == target_position.j and actor_position.k != target_position.k):
            return "t"

        return "."

    def render_trajectory(self, trajectory: list, config: GridConfig) -> None:
        """
        Render a trajectory showing the path taken by the actor.

        Args:
            trajectory: List of (position, info) tuples
            config: Grid configuration
        """
        if not trajectory:
            return

        print(f"\n=== Trajectory ({len(trajectory)} steps) ===")

        # Show path on each layer
        layers = set(pos.k for pos, _ in trajectory)

        for layer in sorted(layers):
            print(f"\nLayer {layer}:")
            layer_positions = [(pos, step_num) for step_num, (pos, _) in enumerate(trajectory)
                             if pos.k == layer]

            if not layer_positions:
                continue

            # Create grid for this layer
            for j in range(config.n_y, 0, -1):
                row = f"{j:2d}|"
                for i in range(1, config.n_x + 1):
                    symbol = "."
                    # Find if actor was at this position and when
                    for pos, step_num in layer_positions:
                        if pos.i == i and pos.j == j:
                            if step_num == 0:
                                symbol = "S"  # Start
                            elif step_num == len(trajectory) - 1:
                                symbol = "E"  # End
                            else:
                                symbol = str(step_num % 10)  # Step number (mod 10)
                            break
                    row += f" {symbol} "
                print(row)

            # Print column numbers
            col_labels = "   " + "".join(f"{i:3d}" for i in range(1, config.n_x + 1))
            print(col_labels)