"""Navigation arena for target-reaching tasks."""

import numpy as np

from .grid_arena import GridArena
from ..field.abstract_field import AbstractField
from ..actor.abstract_actor import AbstractActor
from ..utils.types import (
    GridPosition, GridConfig, ArenaState, GridArenaState, NavigationArenaState
)


class NavigationArena(GridArena):
    """Arena for navigation and station-keeping tasks.
    
    Reward structure:
    - Inside vicinity: Continuous vicinity_bonus (station-keeping reward)
        - Optionally decays exponentially from center to edge
    - Outside vicinity: distance_penalty + step_penalty

    This structure handles both:
        - Navigation: Starting outside → penalties until reaching vicinity → bonus
        - Station-keeping: Starting at target → continuous bonus → penalties if drift away

    Termination:
    - Optional: Can terminate when target is first reached, or continue indefinitely (station-keeping)
    """
    
    def __init__(
        self,
        field: AbstractField,
        actor: AbstractActor,
        config: GridConfig,
        initial_position: GridPosition,
        target_position: GridPosition,
        vicinity_radius: float,
        boundary_mode: str = 'clip',
        distance_reward_weight: float = -0.1,
        vicinity_bonus: float = 100.0,
        step_penalty: float = -0.1,
        terminate_on_reach: bool = False,
        use_distance_decay: bool = False,
        decay_rate: float = 0.5
    ):
        """Initialize navigation arena.
        
        Args:
            field: Environmental field.
            actor: Actor with vertical dynamics.
            config: Grid configuration.
            initial_position: Starting position.
            target_position: Goal position to reach.
            vicinity_radius: Radius around target that counts as "reached".
            boundary_mode: Boundary handling ('clip', 'periodic', 'terminal').
            distance_reward_weight: Weight for distance-based penalty outside vicinity (typically negative).
            vicinity_bonus: Continuous reward for staying in vicinity.
            step_penalty: Penalty per step outside vicinity (typically negative).
            terminate_on_reach: If True, episode ends when target is first reached.
            use_distance_decay: If True, vicinity bonus decays exponentially from center.
            decay_rate: Rate of exponential decay for vicinity bonus (higher = faster decay).
        """
        super().__init__(
            field=field,
            actor=actor,
            config=config,
            initial_position=initial_position,
            boundary_mode=boundary_mode
        )
        
        self.target_position = target_position
        self.vicinity_radius = vicinity_radius
        self.distance_reward_weight = distance_reward_weight
        self.vicinity_bonus = vicinity_bonus
        self.step_penalty = step_penalty
        self.terminate_on_reach = terminate_on_reach
        self.use_distance_decay = use_distance_decay
        self.decay_rate = decay_rate
        
        # Track if target has been reached (for termination)
        self._target_reached = False
        self._cumulative_reward = 0.0
    
    def reset(self, rng_key):
        """Reset arena and navigation state."""
        obs = super().reset(rng_key)
        self._target_reached = False
        self._cumulative_reward = 0.0
        return obs
    
    def compute_reward(self) -> float:
        """Compute reward for station-keeping and navigation.
        """
        # Calculate Euclidean distance to target
        distance_to_target = np.sqrt(
            (self.position.i - self.target_position.i) ** 2 +
            (self.position.j - self.target_position.j) ** 2 +
            (self.position.k - self.target_position.k) ** 2
        )
        # Check if in target vicinity
        in_vicinity = distance_to_target <= self.vicinity_radius
        
        if in_vicinity:
            # Inside vicinity: Get station-keeping reward
            if self.use_distance_decay:
                # Exponential decay from center to edge
                # At center (distance=0): full bonus
                # At edge (distance=radius): bonus * exp(-decay_rate * radius)
                decay_factor = np.exp(-self.decay_rate * distance_to_target)
                reward = self.vicinity_bonus * decay_factor
            else:
                # Constant reward for being anywhere in vicinity
                reward = self.vicinity_bonus
            
            # Mark as reached (for optional termination)
            if not self._target_reached:
                self._target_reached = True
        else:
            # Outside vicinity: Penalties to encourage navigation
            distance_penalty = self.distance_reward_weight * distance_to_target
            reward = distance_penalty + self.step_penalty
        
        # Update cumulative reward and store last reward
        self._cumulative_reward += reward
        self._last_reward = reward
        
        return reward
    
    def is_terminal(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if target reached (if enabled)
        if self.terminate_on_reach and self._target_reached:
            return True
        
        # Check parent class termination (e.g., out of bounds)
        return super().is_terminal()
    
    def get_cumulative_reward(self) -> float:
        """Get cumulative reward for current episode."""
        return self._cumulative_reward
    
    def get_state(self) -> NavigationArenaState:
        """Get complete navigation arena state."""
        # Get base grid arena state
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
            # Navigation static config (for visualization/analysis)
            target_position=self.target_position,
            vicinity_radius=self.vicinity_radius,
            distance_reward_weight=self.distance_reward_weight,
            vicinity_bonus=self.vicinity_bonus,
            step_penalty=self.step_penalty,
            use_distance_decay=self.use_distance_decay,
            decay_rate=self.decay_rate
        )
    
    def set_state(self, state: ArenaState) -> None:
        """Restore navigation arena state."""
        # Let parent restore base and grid fields
        super().set_state(state)
        
        # Restore navigation-specific dynamic state if available
        if isinstance(state, NavigationArenaState):
            self._cumulative_reward = state.cumulative_reward
            self._target_reached = state.target_reached
            # Note: Task config (target_position, vicinity_radius, weights) are static
            # and set during __init__, not restored from state
        else:
            # Reset navigation state if given only base/grid state
            self._cumulative_reward = 0.0
            self._target_reached = False

