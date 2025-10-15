"""Episode data recorder for visualization purposes."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ..utils.types import GridPosition, GridConfig


class VisualizationRecorder:
    """Records episode data for visualization and analysis."""
    
    def __init__(self, config: GridConfig, target_position: Optional[GridPosition] = None,
                 target_vicinity_radius: float = 1.5):
        """
        Initialize visualization recorder.
        
        Args:
            config: Grid configuration
            target_position: Optional target position to visualize
            target_vicinity_radius: Radius around target for vicinity visualization
        """
        self.config = config
        self.target_position = target_position
        self.target_vicinity_radius = target_vicinity_radius
        
        # Episode data
        self.trajectory: List[GridPosition] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.step_info: List[Dict[str, Any]] = []
        
        # Wind field data (cached for efficiency)
        self._wind_field_cached: Optional[np.ndarray] = None
        self._wind_field_timestamp: Optional[int] = None
        
    def reset(self) -> None:
        """Reset recorder for new episode."""
        self.trajectory = []
        self.actions = []
        self.rewards = []
        self.step_info = []
        self._wind_field_cached = None
        self._wind_field_timestamp = None
        
    def record_step(self, position: GridPosition, action: Optional[int] = None,
                   reward: float = 0.0, info: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a single step of the episode.
        
        Args:
            position: Actor position at this step
            action: Action taken (None for initial position)
            reward: Reward received
            info: Additional step information
        """
        self.trajectory.append(position)
        if action is not None:
            self.actions.append(action)
        self.rewards.append(reward)
        if info is not None:
            self.step_info.append(info)
        
    def cache_wind_field(self, wind_field: np.ndarray, timestamp: int = 0) -> None:
        """
        Cache wind field data for visualization.
        
        Args:
            wind_field: Wind field array of shape (n_x, n_y, n_z, 2) 
                       where last dimension is (u, v) displacement
            timestamp: Episode step when this field was captured
        """
        self._wind_field_cached = wind_field.copy()
        self._wind_field_timestamp = timestamp
        
    def get_wind_field(self) -> Optional[np.ndarray]:
        """Get cached wind field data."""
        return self._wind_field_cached
    
    def get_trajectory_array(self) -> np.ndarray:
        """
        Get trajectory as numpy array.
        
        Returns:
            Array of shape (n_steps, 3) with (i, j, k) coordinates
        """
        if not self.trajectory:
            return np.array([]).reshape(0, 3)
        return np.array([(pos.i, pos.j, pos.k) for pos in self.trajectory])
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics of recorded episode."""
        return {
            'num_steps': len(self.trajectory),
            'total_reward': sum(self.rewards),
            'mean_reward': np.mean(self.rewards) if self.rewards else 0.0,
            'start_position': self.trajectory[0] if self.trajectory else None,
            'end_position': self.trajectory[-1] if self.trajectory else None,
            'target_position': self.target_position,
        }
    
    def __len__(self) -> int:
        """Return number of recorded steps."""
        return len(self.trajectory)

