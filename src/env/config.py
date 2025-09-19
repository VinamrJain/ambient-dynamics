"""Configuration management for the grid environment."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .utils.types import GridConfig, GridPosition


class EnvironmentConfig:
    """Configuration manager for grid environment."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default config.
        """
        if config_path is None:
            # Use default config
            config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            'grid': {'n_x': 5, 'n_y': 5, 'n_z': 3, 'd_max': 1},
            'environment': {'max_steps': 100, 'seed': 42},
            'field': {'type': 'simple', 'seed': 42},
            'actor': {
                'type': 'grid',
                'initial_position': [3, 3, 2],
                'noise_prob': 0.1,
                'seed': 42
            },
            'reward': {
                'type': 'distance_to_target',
                'target_position': [3, 3, 2],
                'step_penalty': 0.1,
                'goal_reward': 10.0
            },
            'rendering': {
                'mode': 'simple',
                'show_wind': True,
                'show_target': True
            }
        }

    @property
    def grid_config(self) -> GridConfig:
        """Get grid configuration."""
        grid_cfg = self.config['grid']
        return GridConfig(
            n_x=grid_cfg['n_x'],
            n_y=grid_cfg['n_y'],
            n_z=grid_cfg['n_z'],
            d_max=grid_cfg['d_max']
        )

    @property
    def initial_position(self) -> GridPosition:
        """Get initial actor position."""
        pos = self.config['actor']['initial_position']
        return GridPosition(i=pos[0], j=pos[1], k=pos[2])

    @property
    def target_position(self) -> Optional[GridPosition]:
        """Get target position if defined."""
        if 'target_position' in self.config['reward']:
            pos = self.config['reward']['target_position']
            return GridPosition(i=pos[0], j=pos[1], k=pos[2])
        return None

    @property
    def max_steps(self) -> int:
        """Get maximum steps per episode."""
        return self.config['environment']['max_steps']

    @property
    def environment_seed(self) -> int:
        """Get environment seed."""
        return self.config['environment']['seed']

    @property
    def field_seed(self) -> int:
        """Get field seed."""
        return self.config['field']['seed']

    @property
    def actor_seed(self) -> int:
        """Get actor seed."""
        return self.config['actor']['seed']

    @property
    def noise_prob(self) -> float:
        """Get actor noise probability."""
        return self.config['actor']['noise_prob']

    @property
    def step_penalty(self) -> float:
        """Get step penalty for reward function."""
        return self.config['reward']['step_penalty']

    @property
    def goal_reward(self) -> float:
        """Get goal reward for reward function."""
        return self.config['reward']['goal_reward']

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by nested key (e.g., 'actor.noise_prob')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def update(self, key: str, value: Any) -> None:
        """Update configuration value by nested key."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to YAML file."""
        save_path = Path(path) if path else self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)