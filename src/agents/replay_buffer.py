"""Simple NumPy-backed replay buffer for off-policy RL agents.

Replaces the stable-baselines3 ReplayBuffer dependency used by CleanRL,
keeping only what DQN (and later TD3/SAC) actually needs.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Fixed-size circular replay buffer stored in NumPy arrays.

    Stores (obs, next_obs, action, reward, done) transitions.
    Sampling returns plain NumPy arrays ready to be passed to JAX.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions.
    obs_shape : tuple[int, ...]
        Shape of a single observation.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self._ptr = 0       # next write position
        self._size = 0      # current number of stored transitions

        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((capacity,), dtype=np.int32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)

    # -- storage -------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self._obs[self._ptr] = obs
        self._next_obs[self._ptr] = next_obs
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # -- sampling ------------------------------------------------------------

    def sample(
        self, batch_size: int, rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random minibatch.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.
        rng : np.random.Generator, optional
            NumPy random generator for reproducibility.
            Falls back to ``np.random.default_rng()`` if not provided.

        Returns
        -------
        observations, next_observations, actions, rewards, dones
            Each is a NumPy array with leading dimension ``batch_size``.
        """
        if rng is None:
            rng = np.random.default_rng()
        idxs = rng.integers(0, self._size, size=batch_size)
        return (
            self._obs[idxs],
            self._next_obs[idxs],
            self._actions[idxs],
            self._rewards[idxs],
            self._dones[idxs],
        )

    # -- properties ----------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of transitions currently stored."""
        return self._size

    def __len__(self) -> int:
        return self._size
