from .environment import GridEnvironment
from .field.simple_field import SimpleField
from .actor.grid_actor import GridActor
from .utils.types import GridConfig, GridPosition

# Note: Environment registration moved to separate function to avoid import issues
# Call register_environments() after installing the package

__all__ = [
    'GridEnvironment',
    'SimpleField',
    'GridActor',
    'GridConfig',
    'GridPosition',
    'register_environments'
]

def register_environments():
    """Register gymnasium environments. Call this after package installation."""
    try:
        from gymnasium.envs.registration import register

        register(
            id='GridEnv-v0',
            entry_point='src.env:GridEnvironment',
            max_episode_steps=1000,
            kwargs={
                'field': None,  # Must be provided when creating environment
                'actor': None,  # Must be provided when creating environment
                'config': GridConfig(5, 5, 3, 1),  # Default config
            }
        )

        register(
            id='GridEnv-Simple-v0',
            entry_point='src.env:GridEnvironment',
            max_episode_steps=100,
            kwargs={
                'field': None,
                'actor': None,
                'config': GridConfig(5, 5, 3, 1),
            }
        )

        print("Grid environments registered successfully!")

    except ImportError:
        print("Warning: gymnasium not available. Environment registration skipped.")