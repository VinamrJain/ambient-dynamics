"""YAML-based experiment configuration.

An experiment config specifies:
- environment parameters (grid, field, actor, arena, reward)
- agent class + hyperparameters
- evaluation parameters (episodes, seeds, horizon)

Configs are plain dicts loaded from YAML.  Helper functions construct
the environment and agent from a config dict so that the runner and
launcher don't need to know the concrete classes.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from ..env import (
    GridEnvironment,
    NavigationArena,
    NavigationReward,
    GridActor,
    GridConfig,
    GridPosition,
)
from ..env.field import RFFGPField  # not re-exported from env.__init__
from ..agents import (
    Agent,
    AgentConfig,
    RandomAgent,
    DPAgent,
    DPAgentConfig,
    DQNAgent,
    DQNConfig,
    PPOAgent,
    PPOConfig,
)


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

_AGENT_REGISTRY: dict[str, tuple[type[Agent], type[AgentConfig]]] = {
    "random": (RandomAgent, AgentConfig),
    "dp": (DPAgent, DPAgentConfig),
    "dqn": (DQNAgent, DQNConfig),
    "ppo": (PPOAgent, PPOConfig),
}


def register_agent(
    name: str, agent_cls: type[Agent], config_cls: type[AgentConfig],
) -> None:
    """Register a custom agent so it can be referenced by name in YAML configs."""
    _AGENT_REGISTRY[name] = (agent_cls, config_cls)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config file."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Factory: build env from config dict
# ---------------------------------------------------------------------------

def build_env(
    cfg: dict[str, Any], seed: int = 0,
) -> tuple[GridEnvironment, NavigationArena]:
    """Construct a GridEnvironment + NavigationArena from a config dict.

    Expected keys under ``cfg["env"]``::

        grid:     {n_x, n_y, n_z (optional)}
        field:    {d_max, sigma, lengthscale, nu, num_features, noise_std}
        actor:    {scale, noise_std, z_max}
        arena:    {initial_position, target_position, vicinity_radius,
                   boundary_mode, terminate_on_reach}
        reward:   {peak_reward, step_cost, proximity_scale}
        horizon:  int
    """
    e = cfg["env"]

    # Grid
    gc = e["grid"]
    grid_config = GridConfig.create(gc["n_x"], gc["n_y"], gc.get("n_z"))

    # Field
    fc = e["field"]
    field = RFFGPField(grid_config, **fc)

    # Actor
    actor = GridActor(**e["actor"])

    # Reward
    rc = e["reward"]
    ac = e["arena"]
    target = _parse_position(ac["target_position"], grid_config.ndim)
    reward_fn = NavigationReward(
        target_position=target,
        vicinity_radius=ac["vicinity_radius"],
        **rc,
    )

    # Arena
    initial = _parse_position(ac["initial_position"], grid_config.ndim)
    arena = NavigationArena(
        field=field,
        actor=actor,
        config=grid_config,
        initial_position=initial,
        target_position=target,
        vicinity_radius=ac["vicinity_radius"],
        boundary_mode=ac.get("boundary_mode", "terminal"),
        reward_fn=reward_fn,
        terminate_on_reach=ac.get("terminate_on_reach", False),
    )

    horizon = e.get("horizon", 100)
    env = GridEnvironment(arena=arena, max_steps=horizon, seed=seed)
    return env, arena


def _parse_position(pos, ndim: int) -> GridPosition:
    """Parse a list [i, j] or [i, j, k] into a GridPosition."""
    if ndim == 3:
        return GridPosition(pos[0], pos[1], pos[2])
    return GridPosition(pos[0], pos[1], None)


# ---------------------------------------------------------------------------
# Factory: build agent from config dict
# ---------------------------------------------------------------------------

def build_agent(
    cfg: dict[str, Any],
    num_actions: int,
    obs_shape: tuple[int, ...],
) -> Agent:
    """Construct an Agent from the ``cfg["agent"]`` section.

    Expected keys::

        name:    str  (key into agent registry, e.g. "dqn")
        params:  dict (passed as kwargs to the agent config dataclass)
    """
    ac = cfg["agent"]
    name = ac["name"]
    params = ac.get("params", {})

    if name not in _AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent '{name}'. Registered: {list(_AGENT_REGISTRY)}"
        )

    agent_cls, config_cls = _AGENT_REGISTRY[name]
    agent_config = config_cls(**params)
    return agent_cls(config=agent_config, num_actions=num_actions, obs_shape=obs_shape)


# ---------------------------------------------------------------------------
# Seed utilities
# ---------------------------------------------------------------------------

def derive_seed(master_seed: int, episode_idx: int) -> int:
    """Deterministically derive a per-episode seed from a master seed.

    Uses a hash so that seeds are well-distributed even for adjacent indices.
    """
    h = hashlib.sha256(f"{master_seed}:{episode_idx}".encode()).hexdigest()
    return int(h[:8], 16)  # 32-bit seed
