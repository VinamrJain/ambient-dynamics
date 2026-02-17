"""Navigation arena for target-reaching tasks."""

import numpy as np

from .grid_arena import GridArena
from .reward import RewardFunction
from ..field.abstract_field import AbstractField
from ..actor.abstract_actor import AbstractActor
from ..utils.types import (
    GridPosition, GridConfig, ArenaState, GridArenaState, NavigationArenaState
)


class NavigationArena(GridArena):
    """Arena for navigation and station-keeping tasks.

    Supports both 2D and 3D settings (inherits from GridArena).

    Reward is provided by the injected reward_fn (RewardFunction).
    """

    def __init__(
        self,
        field: AbstractField,
        actor: AbstractActor,
        config: GridConfig,
        initial_position: GridPosition,
        target_position: GridPosition,
        vicinity_radius: float,
        boundary_mode: str = 'terminal',
        *,
        reward_fn: RewardFunction,
        terminate_on_reach: bool = False,
    ):
        """Initialize navigation arena.

        Args:
            field: Environmental field providing ambient displacements.
            actor: Actor with controllable axis dynamics.
            config: Grid configuration.
            initial_position: Starting position.
            target_position: Goal position to reach.
            vicinity_radius: Radius around target that counts as "reached".
            boundary_mode: Boundary handling ('clip', 'periodic', 'terminal').
            reward_fn: Reward function (e.g. NavigationReward); caller constructs with desired params.
            terminate_on_reach: If True, episode ends when target is first reached.
        """
        super().__init__(
            field=field,
            actor=actor,
            config=config,
            initial_position=initial_position,
            boundary_mode=boundary_mode
        )

        if vicinity_radius <= 0.0:
            raise ValueError(f"vicinity_radius must be positive, got {vicinity_radius}")

        if not (1 <= target_position.i <= config.n_x and
                1 <= target_position.j <= config.n_y):
            raise ValueError(
                f"target_position {target_position} is outside grid "
                f"({config.n_x}, {config.n_y}, {config.n_z})"
            )
        if config.ndim == 3 and not (1 <= target_position.k <= config.n_z):
            raise ValueError(
                f"target_position.k={target_position.k} is outside grid "
                f"[1, {config.n_z}]"
            )

        self.target_position = target_position
        self.vicinity_radius = vicinity_radius
        self.terminate_on_reach = terminate_on_reach
        self.reward_fn = reward_fn

        self._target_reached = False
        self._cumulative_reward = 0.0

    def reset(self, rng_key):
        """Reset arena and navigation state."""
        obs = super().reset(rng_key)
        self._target_reached = False
        self._cumulative_reward = 0.0
        return obs

    def _compute_distance(self, pos1: GridPosition, pos2: GridPosition) -> float:
        """Compute Euclidean distance between two positions (handles 2D and 3D)."""
        if self.ndim == 3:
            return np.sqrt(
                (pos1.i - pos2.i) ** 2 +
                (pos1.j - pos2.j) ** 2 +
                (pos1.k - pos2.k) ** 2
            )
        return np.sqrt(
            (pos1.i - pos2.i) ** 2 +
            (pos1.j - pos2.j) ** 2
        )

    def compute_reward(self) -> float:
        """Compute reward (proximity minus step cost)."""
        reward = self.reward_fn.compute_scalar(self.position)

        distance_to_target = self._compute_distance(self.position, self.target_position)
        if distance_to_target <= self.vicinity_radius and not self._target_reached:
            self._target_reached = True

        self._cumulative_reward += reward
        self._last_reward = reward

        return reward

    def is_terminal(self) -> bool:
        """Check if episode should terminate."""
        if self.terminate_on_reach and self._target_reached:
            return True
        return super().is_terminal()

    def get_cumulative_reward(self) -> float:
        """Get cumulative reward for current episode."""
        return self._cumulative_reward

    def get_state(self) -> NavigationArenaState:
        """Get complete navigation arena state."""
        base_state = super().get_state()
        # Create extended navigation state with full config
        return NavigationArenaState(
            # Universal state
            step_count=base_state.step_count,
            last_action=base_state.last_action,
            last_reward=base_state.last_reward,
            rng_key=base_state.rng_key,
            # Grid state
            position=base_state.position,
            last_position=base_state.last_position,
            last_displacement=base_state.last_displacement,
            out_of_bounds=base_state.out_of_bounds,
            initial_position=base_state.initial_position,
            # Navigation dynamic state
            cumulative_reward=self._cumulative_reward,
            target_reached=self._target_reached,
            # Navigation static config
            target_position=self.target_position,
            vicinity_radius=self.vicinity_radius,
        )

    def set_state(self, state: ArenaState) -> None:
        """Restore navigation arena state."""
        super().set_state(state)
        if isinstance(state, NavigationArenaState):
            self._cumulative_reward = state.cumulative_reward
            self._target_reached = state.target_reached
        else:
            self._cumulative_reward = 0.0
            self._target_reached = False
