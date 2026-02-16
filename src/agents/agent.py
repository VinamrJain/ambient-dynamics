"""Agent abstractions for ambient-dynamics.

Defines the core Agent ABC, AgentConfig, AgentMode, Logger protocol,
and a RandomAgent baseline. All RL agents in this project inherit from Agent.
"""

import abc
import enum
from dataclasses import dataclass, asdict
from typing import Any, Protocol, runtime_checkable

import jax
import numpy as np


# ---------------------------------------------------------------------------
# Agent mode
# ---------------------------------------------------------------------------

class AgentMode(enum.Enum):
    """Whether the agent is training (may update params) or evaluating (frozen)."""
    TRAIN = "train"
    EVAL = "eval"


# ---------------------------------------------------------------------------
# Logger protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Logger(Protocol):
    """Minimal logging interface that W&B, TensorBoard, or a no-op can satisfy."""

    def log_scalar(self, key: str, value: float, step: int) -> None: ...
    def log_dict(self, data: dict[str, Any], step: int) -> None: ...
    def log_config(self, config: dict[str, Any]) -> None: ...


class NoOpLogger:
    """Default silent logger -- used when no backend is configured."""

    def log_scalar(self, key: str, value: float, step: int) -> None:
        pass

    def log_dict(self, data: dict[str, Any], step: int) -> None:
        pass

    def log_config(self, config: dict[str, Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Base configuration -- subclasses extend with algorithm-specific fields.

    Every config is a plain dataclass so it can be serialized to/from YAML
    and logged as a W&B artifact.
    """
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Agent ABC
# ---------------------------------------------------------------------------

class Agent(abc.ABC):
    """Abstract base class for all agents.

    Lifecycle managed by the evaluation runner:

        obs, info = env.reset(seed=seed)
        action = agent.begin_episode(obs)
        while not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if not done:
                action = agent.step(reward, obs)
        agent.end_episode(reward, terminal=terminated)
    """

    def __init__(
        self,
        config: AgentConfig,
        num_actions: int,
        obs_shape: tuple[int, ...],
    ) -> None:
        self._config = config
        self._num_actions = num_actions
        self._obs_shape = obs_shape
        self._mode = AgentMode.TRAIN
        self._logger: Logger = NoOpLogger()
        self._rng = jax.random.PRNGKey(config.seed)

    # -- properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def mode(self) -> AgentMode:
        return self._mode

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return self._obs_shape

    # -- core RL loop --------------------------------------------------------

    @abc.abstractmethod
    def begin_episode(self, observation: np.ndarray) -> int:
        """Receive the first observation, return the first action."""

    @abc.abstractmethod
    def step(self, reward: float, observation: np.ndarray) -> int:
        """Receive (reward, observation) pair, return next action."""

    @abc.abstractmethod
    def end_episode(self, reward: float, terminal: bool) -> None:
        """Signal end of episode with final reward and termination flag."""

    # -- mode ----------------------------------------------------------------

    def set_mode(self, mode: AgentMode) -> None:
        """Switch between TRAIN and EVAL.

        Subclasses may override to disable exploration, freeze params, etc.
        """
        self._mode = mode

    # -- logging -------------------------------------------------------------

    def set_logger(self, logger: Logger) -> None:
        """Attach a logging backend (W&B, TensorBoard, etc.)."""
        self._logger = logger

    # -- seeding -------------------------------------------------------------

    def seed(self, rng_key: jax.Array) -> None:
        """Reset the agent's JAX PRNG state."""
        self._rng = rng_key

    def _split_key(self) -> jax.Array:
        """Split the internal PRNG key and return a fresh sub-key."""
        self._rng, subkey = jax.random.split(self._rng)
        return subkey

    # -- checkpointing -------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save agent state to *path*. No-op by default."""

    def load_checkpoint(self, path: str) -> None:
        """Load agent state from *path*. No-op by default."""


# ---------------------------------------------------------------------------
# Random agent
# ---------------------------------------------------------------------------

class RandomAgent(Agent):
    """Uniformly random action selection -- useful as a lower-bound baseline."""

    def begin_episode(self, observation: np.ndarray) -> int:
        return self._random_action()

    def step(self, reward: float, observation: np.ndarray) -> int:
        return self._random_action()

    def end_episode(self, reward: float, terminal: bool) -> None:
        pass

    def _random_action(self) -> int:
        key = self._split_key()
        return int(jax.random.randint(key, shape=(), minval=0, maxval=self._num_actions))
