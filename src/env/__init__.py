from gymnasium.envs.registration import register
from .environment import GridEnvironment
from .field.implementations.simple_field import SimpleField
from .actor.grid_actor import GridActor
from .utils.types import GridConfig, GridPosition

# Register environment variants
register(
    id='GridEnv-v0',
    entry_point='grid_env:GridEnvironment',
    max_episode_steps=1000,
)

__all__ = [
    'GridEnvironment',
    'SimpleField', 
    'GridActor',
    'GridConfig',
    'GridPosition'
]