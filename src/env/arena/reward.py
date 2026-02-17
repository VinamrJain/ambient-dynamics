"""Reward function abstractions for arena tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import jax.numpy as jnp

from ..utils.types import GridPosition, GridConfig


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class RewardFunction(ABC):
    """JAX-compatible vectorized reward function.

    Subclasses must implement both a scalar query (used by the arena at
    runtime) and a vectorized grid query (used by planning agents).
    """

    @abstractmethod
    def compute_scalar(self, position: GridPosition) -> float:
        """Return the reward for a single grid position."""

    @abstractmethod
    def compute_grid(self, grid_config: GridConfig) -> jnp.ndarray:
        """Return rewards for every cell in the grid (JAX array).

        Shape ``(n_x, n_y)`` (2D) or ``(n_x, n_y, n_z)`` (3D).
        Entry at 0-indexed ``[i, j, ...]`` = reward at 1-indexed ``(i+1, j+1, ...)``.
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _distance_to_target(position: GridPosition, target: GridPosition) -> float:
    """Euclidean distance from position to target (2D or 3D)."""
    if position.k is not None and target.k is not None:
        return float(np.sqrt(
            (position.i - target.i) ** 2
            + (position.j - target.j) ** 2
            + (position.k - target.k) ** 2
        ))
    return float(np.sqrt(
        (position.i - target.i) ** 2
        + (position.j - target.j) ** 2
    ))


def _distance_grid_jax(grid_config: GridConfig, target: GridPosition) -> jnp.ndarray:
    """Vectorized Euclidean distance to target over the grid (JAX)."""
    if grid_config.ndim == 2:
        ii = jnp.arange(1, grid_config.n_x + 1, dtype=jnp.float32)
        jj = jnp.arange(1, grid_config.n_y + 1, dtype=jnp.float32)
        I, J = jnp.meshgrid(ii, jj, indexing="ij")
        return jnp.sqrt((I - target.i) ** 2 + (J - target.j) ** 2)
    ii = jnp.arange(1, grid_config.n_x + 1, dtype=jnp.float32)
    jj = jnp.arange(1, grid_config.n_y + 1, dtype=jnp.float32)
    kk = jnp.arange(1, grid_config.n_z + 1, dtype=jnp.float32)
    I, J, K = jnp.meshgrid(ii, jj, kk, indexing="ij")
    return jnp.sqrt(
        (I - target.i) ** 2
        + (J - target.j) ** 2
        + (K - target.k) ** 2
    )


# ---------------------------------------------------------------------------
# Navigation reward (proximity + step cost)
# ---------------------------------------------------------------------------

class NavigationReward(RewardFunction):
    """Proximity reward: non-negative, step cost, inverse-linear in distance.

    r(D) = (peak_reward - step_cost) / (1 + proximity_scale * D).
    See env-formulation/navigation-reward.md for derivation.
    """

    def __init__(
        self,
        target_position: GridPosition,
        vicinity_radius: float,
        peak_reward: float = 10.0,
        step_cost: float = 0.1,
        proximity_scale: float = 0.1,
    ) -> None:
        self.target_position = target_position
        self.vicinity_radius = vicinity_radius
        self.peak_reward = peak_reward
        self.step_cost = step_cost
        self.proximity_scale = proximity_scale

        if peak_reward <= step_cost:
            raise ValueError(
                f"peak_reward must be > step_cost, got peak_reward={peak_reward}, step_cost={step_cost}"
            )
        if step_cost < 0.0 or proximity_scale <= 0.0:
            raise ValueError("step_cost >= 0 and proximity_scale > 0 required")

    def compute_scalar(self, position: GridPosition) -> float:
        dist = _distance_to_target(position, self.target_position)
        bonus = self.step_cost + (self.peak_reward - self.step_cost) / (
            1.0 + self.proximity_scale * dist
        )
        return float(bonus - self.step_cost)

    def compute_grid(self, grid_config: GridConfig) -> jnp.ndarray:
        dist = _distance_grid_jax(grid_config, self.target_position)
        bonus = self.step_cost + (self.peak_reward - self.step_cost) / (
            1.0 + self.proximity_scale * dist
        )
        return bonus - self.step_cost
