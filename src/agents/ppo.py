"""PPO agent -- CleanRL-style single-file JAX/Flax implementation.

Ported from CleanRL ``ppo_atari_envpool_xla_jax_scan.py`` and adapted
to our Agent ABC with the following changes:

- Stripped envpool, Atari, CNN, XLA env handles
- Replaced conv feature extractor with configurable MLP
- Rollout collected step-by-step via Agent ABC (not vectorised)
- GAE and PPO loss update retained from CleanRL (jax.lax.scan)
- Single-env, non-vectorised: num_envs == 1 always
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from .agent import Agent, AgentConfig, AgentMode


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig(AgentConfig):
    """PPO hyperparameters."""

    # Network
    hidden_dims: tuple[int, ...] = (64, 64)

    # Optimiser
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    max_grad_norm: float = 0.5

    # Rollout
    num_steps: int = 128          # transitions per rollout before update

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO update
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5

    # Schedule (for LR annealing)
    total_timesteps: int = 50_000


# ---------------------------------------------------------------------------
# Flax networks
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Shared-trunk MLP with separate actor (logits) and critic (value) heads."""
    action_dim: int
    hidden_dims: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Shared trunk
        for h in self.hidden_dims:
            x = nn.Dense(h, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
            x = nn.tanh(x)
        # Actor head
        logits = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
        )(x)
        # Critic head
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
        )(x)
        return logits, value.squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout storage  (plain NumPy, converted to JAX at update time)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-size buffer collecting one rollout of ``num_steps`` transitions."""

    def __init__(self, num_steps: int, obs_shape: tuple[int, ...]) -> None:
        self.num_steps = num_steps
        self.obs = np.zeros((num_steps, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(num_steps, dtype=np.int32)
        self.logprobs = np.zeros(num_steps, dtype=np.float32)
        self.rewards = np.zeros(num_steps, dtype=np.float32)
        self.dones = np.zeros(num_steps, dtype=np.float32)
        self.values = np.zeros(num_steps, dtype=np.float32)
        self.ptr = 0

    def add(
        self, obs: np.ndarray, action: int, logprob: float,
        reward: float, done: bool, value: float,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.ptr += 1

    @property
    def full(self) -> bool:
        return self.ptr >= self.num_steps

    def reset(self) -> None:
        self.ptr = 0

    def to_jax(self):
        """Return JAX arrays for the stored data."""
        return (
            jnp.asarray(self.obs),
            jnp.asarray(self.actions),
            jnp.asarray(self.logprobs),
            jnp.asarray(self.rewards),
            jnp.asarray(self.dones),
            jnp.asarray(self.values),
        )


# ---------------------------------------------------------------------------
# Pure functions (JIT-compiled)
# ---------------------------------------------------------------------------

def _compute_gae(
    rewards: jnp.ndarray,     # (T,)
    values: jnp.ndarray,      # (T,)
    dones: jnp.ndarray,       # (T,)
    last_value: float,
    last_done: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE advantages and returns for a single-env rollout."""
    T = rewards.shape[0]
    # Append bootstrap value
    next_values = jnp.concatenate([values[1:], jnp.array([last_value])])
    next_dones = jnp.concatenate([dones[1:], jnp.array([last_done])])

    def _step(advantage, t):
        # Scan from T-1 down to 0 (reversed)
        idx = T - 1 - t
        delta = rewards[idx] + gamma * next_values[idx] * (1.0 - next_dones[idx]) - values[idx]
        advantage = delta + gamma * gae_lambda * (1.0 - next_dones[idx]) * advantage
        return advantage, advantage

    _, advantages_rev = jax.lax.scan(_step, jnp.float32(0.0), jnp.arange(T))
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


def _make_ppo_update_fn(
    apply_fn,
    num_steps: int,
    num_minibatches: int,
    update_epochs: int,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    norm_adv: bool,
    max_grad_norm: float,
):
    """Build a JIT-compiled PPO update function (closure over hyperparams)."""

    minibatch_size = num_steps // num_minibatches

    def _ppo_loss(params, obs, actions, old_logprobs, advantages, returns):
        logits, values = apply_fn(params, obs)
        # Log-probs under current policy
        log_probs_all = jax.nn.log_softmax(logits)
        new_logprob = log_probs_all[jnp.arange(actions.shape[0]), actions]
        # Entropy
        probs = jax.nn.softmax(logits)
        entropy = -(probs * log_probs_all).sum(-1).mean()
        # Ratio
        logratio = new_logprob - old_logprobs
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1.0) - logratio).mean()
        # Normalise advantages
        mb_adv = advantages
        if norm_adv:
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
        # Clipped surrogate
        pg_loss1 = -mb_adv * ratio
        pg_loss2 = -mb_adv * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        # Value loss
        v_loss = 0.5 * ((values - returns) ** 2).mean()
        loss = pg_loss - ent_coef * entropy + vf_coef * v_loss
        return loss, (pg_loss, v_loss, entropy, approx_kl)

    ppo_loss_grad = jax.value_and_grad(_ppo_loss, has_aux=True)

    @jax.jit
    def update(
        train_state: TrainState,
        obs: jnp.ndarray,        # (T, *obs_shape)
        actions: jnp.ndarray,    # (T,)
        logprobs: jnp.ndarray,   # (T,)
        advantages: jnp.ndarray, # (T,)
        returns: jnp.ndarray,    # (T,)
        key: jax.Array,
    ):
        def _epoch(carry, _):
            ts, k = carry
            k, subkey = jax.random.split(k)
            perm = jax.random.permutation(subkey, num_steps)
            # Reshape into minibatches
            mb_obs = obs[perm].reshape(num_minibatches, minibatch_size, *obs.shape[1:])
            mb_act = actions[perm].reshape(num_minibatches, minibatch_size)
            mb_lp = logprobs[perm].reshape(num_minibatches, minibatch_size)
            mb_adv = advantages[perm].reshape(num_minibatches, minibatch_size)
            mb_ret = returns[perm].reshape(num_minibatches, minibatch_size)

            def _minibatch(ts, data):
                o, a, lp, adv, ret = data
                (loss, aux), grads = ppo_loss_grad(ts.params, o, a, lp, adv, ret)
                ts = ts.apply_gradients(grads=grads)
                return ts, (loss, *aux)

            ts, metrics = jax.lax.scan(
                _minibatch, ts, (mb_obs, mb_act, mb_lp, mb_adv, mb_ret),
            )
            return (ts, k), metrics

        (train_state, key), all_metrics = jax.lax.scan(
            _epoch, (train_state, key), None, length=update_epochs,
        )
        # all_metrics: (update_epochs, num_minibatches, ...)
        # Take last epoch, last minibatch for logging
        loss = all_metrics[0][-1, -1]
        pg_loss = all_metrics[1][-1, -1]
        v_loss = all_metrics[2][-1, -1]
        entropy = all_metrics[3][-1, -1]
        approx_kl = all_metrics[4][-1, -1]
        return train_state, key, loss, pg_loss, v_loss, entropy, approx_kl

    return update


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent(Agent):
    """PPO with clipped surrogate objective, GAE, and shared actor-critic MLP.

    Collects transitions step-by-step via the Agent ABC interface.
    When the rollout buffer is full (every ``num_steps`` steps), runs the
    PPO update (GAE + minibatch SGD with ``jax.lax.scan``).
    """

    def __init__(
        self,
        config: PPOConfig,
        num_actions: int,
        obs_shape: tuple[int, ...],
    ) -> None:
        super().__init__(config, num_actions, obs_shape)
        self._cfg: PPOConfig = config

        if config.num_steps % config.num_minibatches != 0:
            raise ValueError(
                f"num_steps ({config.num_steps}) must be divisible by "
                f"num_minibatches ({config.num_minibatches})"
            )

        # ---- Network -------------------------------------------------------
        self._net = ActorCritic(
            action_dim=num_actions,
            hidden_dims=config.hidden_dims,
        )
        dummy_obs = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
        init_key = self._split_key()
        params = self._net.init(init_key, dummy_obs)

        # ---- Optimiser (with optional LR annealing) ------------------------
        num_updates = config.total_timesteps // config.num_steps
        if config.anneal_lr:
            def lr_schedule(count):
                frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / max(num_updates, 1)
                return config.learning_rate * jnp.maximum(frac, 0.0)
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=config.learning_rate, eps=1e-5),
            )

        self._train_state = TrainState.create(
            apply_fn=self._net.apply, params=params, tx=tx,
        )

        # ---- JIT-compiled update -------------------------------------------
        self._update_fn = _make_ppo_update_fn(
            apply_fn=self._net.apply,
            num_steps=config.num_steps,
            num_minibatches=config.num_minibatches,
            update_epochs=config.update_epochs,
            clip_coef=config.clip_coef,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            norm_adv=config.norm_adv,
            max_grad_norm=config.max_grad_norm,
        )

        # ---- Rollout buffer ------------------------------------------------
        self._buf = RolloutBuffer(config.num_steps, obs_shape)

        # ---- Counters and bookkeeping --------------------------------------
        self._global_step: int = 0
        self._update_count: int = 0
        self._episode_reward: float = 0.0

        self._last_obs: np.ndarray | None = None
        self._last_action: int | None = None
        self._last_logprob: float = 0.0
        self._last_value: float = 0.0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _get_action_and_value(self, obs: np.ndarray) -> tuple[int, float, float]:
        """Sample action from policy, return (action, logprob, value)."""
        obs_jnp = jnp.asarray(obs[None], dtype=jnp.float32)
        logits, value = self._net.apply(self._train_state.params, obs_jnp)
        logits = logits[0]  # unbatch
        value = value[0]

        if self._mode == AgentMode.EVAL:
            action = int(jnp.argmax(logits))
            logprob = float(jax.nn.log_softmax(logits)[action])
        else:
            key = self._split_key()
            # Gumbel-softmax trick (same as CleanRL)
            u = jax.random.uniform(key, shape=logits.shape)
            action = int(jnp.argmax(logits - jnp.log(-jnp.log(u))))
            logprob = float(jax.nn.log_softmax(logits)[action])

        return action, logprob, float(value)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _run_update(self, last_value: float, last_done: float) -> None:
        """Run GAE + PPO update on the collected rollout."""
        obs, actions, logprobs, rewards, dones, values = self._buf.to_jax()

        advantages, returns = _compute_gae(
            rewards, values, dones,
            last_value, last_done,
            self._cfg.gamma, self._cfg.gae_lambda,
        )

        key = self._split_key()
        (self._train_state, _key, loss, pg_loss, v_loss,
         entropy, approx_kl) = self._update_fn(
            self._train_state,
            obs, actions, logprobs, advantages, returns, key,
        )
        self._rng = _key  # propagate key state

        self._update_count += 1
        self._logger.log_dict({
            "losses/total": float(loss),
            "losses/policy": float(pg_loss),
            "losses/value": float(v_loss),
            "losses/entropy": float(entropy),
            "losses/approx_kl": float(approx_kl),
        }, step=self._global_step)

        self._buf.reset()

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def begin_episode(self, observation: np.ndarray) -> int:
        self._episode_reward = 0.0
        action, logprob, value = self._get_action_and_value(observation)
        self._last_obs = observation.copy()
        self._last_action = action
        self._last_logprob = logprob
        self._last_value = value
        return action

    def step(self, reward: float, observation: np.ndarray) -> int:
        self._episode_reward += reward

        # Store previous transition
        if self._last_obs is not None and self._mode == AgentMode.TRAIN:
            self._buf.add(
                self._last_obs, self._last_action, self._last_logprob,
                reward, done=False, value=self._last_value,
            )
            self._global_step += 1

            # If rollout buffer is full, do PPO update
            if self._buf.full:
                # Bootstrap value from current observation
                _, _, bootstrap_v = self._get_action_and_value(observation)
                self._run_update(last_value=bootstrap_v, last_done=0.0)

        # Select next action
        action, logprob, value = self._get_action_and_value(observation)
        self._last_obs = observation.copy()
        self._last_action = action
        self._last_logprob = logprob
        self._last_value = value
        return action

    def end_episode(self, reward: float, terminal: bool) -> None:
        self._episode_reward += reward

        # Store final transition
        if self._last_obs is not None and self._mode == AgentMode.TRAIN:
            self._buf.add(
                self._last_obs, self._last_action, self._last_logprob,
                reward, done=True, value=self._last_value,
            )
            self._global_step += 1

            # If buffer is full (or close), run update with terminal bootstrap
            if self._buf.full:
                self._run_update(last_value=0.0, last_done=1.0)

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
            "params": self._train_state.params,
            "opt_state": self._train_state.opt_state,
            "step": self._train_state.step,
            "global_step": self._global_step,
            "update_count": self._update_count,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)

    def load_checkpoint(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self._train_state = self._train_state.replace(
            params=ckpt["params"],
            opt_state=ckpt["opt_state"],
            step=ckpt["step"],
        )
        self._global_step = ckpt["global_step"]
        self._update_count = ckpt["update_count"]
