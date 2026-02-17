"""DQN agent -- CleanRL-style single-file JAX/Flax implementation.

Ported from CleanRL ``dqn_jax.py`` and adapted to our Agent ABC.

Key components retained from CleanRL:
- QNetwork (Flax module)
- TrainState with target_params
- JIT-compiled MSE update step
- Linear epsilon schedule
- Target network soft update via optax.incremental_update

Key adaptations:
- Wrapped in Agent ABC (begin_episode / step / end_episode)
- Own NumPy-backed ReplayBuffer (no SB3 dependency)
- Configurable MLP widths for our low-dim observations
- Logging via self._logger protocol
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState as _FlaxTrainState

from .agent import Agent, AgentConfig, AgentMode
from .replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DQNConfig(AgentConfig):
    """DQN hyperparameters."""

    # Network
    hidden_dims: tuple[int, ...] = (64, 64)

    # Optimiser
    learning_rate: float = 2.5e-4

    # Replay buffer
    buffer_size: int = 10_000
    batch_size: int = 128
    learning_starts: int = 500

    # Training cadence
    train_frequency: int = 4
    target_update_frequency: int = 500
    tau: float = 1.0          # 1.0 = hard update; <1 for Polyak averaging

    # Discount
    gamma: float = 0.99

    # Exploration (linear epsilon schedule)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_fraction: float = 0.5  # fraction of total_timesteps for annealing
    total_timesteps: int = 50_000  # used to compute schedule duration


# ---------------------------------------------------------------------------
# Flax network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Simple MLP Q-network: obs -> Q(s, a) for each action."""
    action_dim: int
    hidden_dims: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


# ---------------------------------------------------------------------------
# Train state (extends Flax TrainState with target params)
# ---------------------------------------------------------------------------

class TrainState(_FlaxTrainState):
    target_params: flax.core.FrozenDict


# ---------------------------------------------------------------------------
# Pure functions (JIT-compiled)
# ---------------------------------------------------------------------------

@jax.jit
def _update(
    q_state: TrainState,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float,
) -> tuple[jnp.ndarray, jnp.ndarray, TrainState]:
    """One gradient step on the DQN MSE loss."""
    q_next_target = q_state.apply_fn(
        q_state.target_params, next_observations,
    )
    q_next_target = jnp.max(q_next_target, axis=-1)
    td_target = rewards + (1.0 - dones) * gamma * q_next_target

    def mse_loss(params):
        q_pred = q_state.apply_fn(params, observations)
        q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions]
        return ((q_pred - td_target) ** 2).mean(), q_pred

    (loss, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(
        q_state.params,
    )
    q_state = q_state.apply_gradients(grads=grads)
    return loss, q_pred, q_state


def _linear_schedule(
    start: float, end: float, duration: int, t: int,
) -> float:
    slope = (end - start) / max(duration, 1)
    return max(slope * t + start, end)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent(Agent):
    """DQN with experience replay and target network.

    Wraps a CleanRL-style DQN inside the Agent ABC so it can be driven
    by the evaluation runner's ``begin_episode`` / ``step`` / ``end_episode``
    loop.  Training happens *online* during ``step()`` when in TRAIN mode.
    """

    def __init__(
        self,
        config: DQNConfig,
        num_actions: int,
        obs_shape: tuple[int, ...],
    ) -> None:
        super().__init__(config, num_actions, obs_shape)
        self._cfg: DQNConfig = config

        # ---- Network + optimiser -------------------------------------------
        self._q_network = QNetwork(
            action_dim=num_actions,
            hidden_dims=config.hidden_dims,
        )
        dummy_obs = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
        init_key = self._split_key()
        params = self._q_network.init(init_key, dummy_obs)

        self._q_state = TrainState.create(
            apply_fn=self._q_network.apply,
            params=params,
            target_params=params,
            tx=optax.adam(learning_rate=config.learning_rate),
        )

        # ---- Replay buffer -------------------------------------------------
        self._rb = ReplayBuffer(config.buffer_size, obs_shape)
        self._np_rng = np.random.default_rng(config.seed)

        # ---- Counters ------------------------------------------------------
        self._global_step: int = 0
        self._episode_reward: float = 0.0

        # ---- Epsilon schedule ----------------------------------------------
        self._eps_duration = int(
            config.epsilon_fraction * config.total_timesteps
        )

        # ---- Transition bookkeeping ----------------------------------------
        self._last_obs: np.ndarray | None = None
        self._last_action: int | None = None

    # ------------------------------------------------------------------
    # Epsilon
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        return _linear_schedule(
            self._cfg.epsilon_start,
            self._cfg.epsilon_end,
            self._eps_duration,
            self._global_step,
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _select_action(self, observation: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        eps = self.epsilon if self._mode == AgentMode.TRAIN else self._cfg.epsilon_end
        if self._np_rng.random() < eps:
            return int(self._np_rng.integers(self._num_actions))
        # Greedy from Q-network
        obs_jnp = jnp.asarray(observation[None], dtype=jnp.float32)
        q_values = self._q_network.apply(self._q_state.params, obs_jnp)
        return int(jnp.argmax(q_values, axis=-1)[0])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _maybe_train(self) -> None:
        """Run one gradient step if conditions are met."""
        if self._mode != AgentMode.TRAIN:
            return
        if self._global_step < self._cfg.learning_starts:
            return
        if self._global_step % self._cfg.train_frequency != 0:
            return

        obs, next_obs, actions, rewards, dones = self._rb.sample(
            self._cfg.batch_size, rng=self._np_rng,
        )

        loss, q_pred, self._q_state = _update(
            self._q_state,
            jnp.asarray(obs),
            jnp.asarray(actions),
            jnp.asarray(next_obs),
            jnp.asarray(rewards),
            jnp.asarray(dones),
            self._cfg.gamma,
        )

        # Logging
        if self._global_step % 100 == 0:
            self._logger.log_dict({
                "losses/td_loss": float(loss),
                "losses/q_values": float(jnp.mean(q_pred)),
                "charts/epsilon": self.epsilon,
            }, step=self._global_step)

        # Target network update
        if self._global_step % self._cfg.target_update_frequency == 0:
            self._q_state = self._q_state.replace(
                target_params=optax.incremental_update(
                    self._q_state.params,
                    self._q_state.target_params,
                    self._cfg.tau,
                ),
            )

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def begin_episode(self, observation: np.ndarray) -> int:
        self._episode_reward = 0.0
        action = self._select_action(observation)
        self._last_obs = observation.copy()
        self._last_action = action
        return action

    def step(self, reward: float, observation: np.ndarray) -> int:
        self._episode_reward += reward

        # Store transition from previous step
        if self._last_obs is not None and self._mode == AgentMode.TRAIN:
            self._rb.add(
                self._last_obs,
                observation,
                self._last_action,
                reward,
                done=False,
            )

        self._global_step += 1
        self._maybe_train()

        # Select next action
        action = self._select_action(observation)
        self._last_obs = observation.copy()
        self._last_action = action
        return action

    def end_episode(self, reward: float, terminal: bool) -> None:
        self._episode_reward += reward

        # Store the final transition
        if self._last_obs is not None and self._mode == AgentMode.TRAIN:
            # For the terminal transition, next_obs doesn't matter
            # (masked by done=True in the TD target).
            self._rb.add(
                self._last_obs,
                self._last_obs,  # placeholder next_obs
                self._last_action,
                reward,
                done=True,
            )
            self._global_step += 1
            self._maybe_train()

        self._logger.log_scalar(
            "charts/episodic_return", self._episode_reward, self._global_step,
        )

        self._last_obs = None
        self._last_action = None

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        import pickle
        from pathlib import Path
        ckpt = {
            "params": self._q_state.params,
            "target_params": self._q_state.target_params,
            "opt_state": self._q_state.opt_state,
            "step": self._q_state.step,
            "global_step": self._global_step,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)

    def load_checkpoint(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self._q_state = self._q_state.replace(
            params=ckpt["params"],
            target_params=ckpt["target_params"],
            opt_state=ckpt["opt_state"],
            step=ckpt["step"],
        )
        self._global_step = ckpt["global_step"]
