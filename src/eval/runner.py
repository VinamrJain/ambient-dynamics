"""Episode runner -- the core train/eval loop.

Provides ``run_episode`` (single episode) and ``run_experiment`` (multi-episode
evaluation over a suite of seeds), wiring together the Agent ABC lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..agents.agent import Agent, AgentMode
from ..env.environment import GridEnvironment

from .experiment_config import build_env, build_agent, derive_seed


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Structured result from a single episode."""
    seed: int
    total_reward: float
    episode_length: int
    terminated: bool          # True if the env hit a terminal state
    target_reached: bool
    rewards: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "terminated": self.terminated,
            "target_reached": self.target_reached,
        }


@dataclass
class ExperimentResult:
    """Aggregated results from multiple episodes."""
    agent_name: str
    episodes: list[EpisodeResult]

    @property
    def mean_reward(self) -> float:
        return float(np.mean([e.total_reward for e in self.episodes]))

    @property
    def std_reward(self) -> float:
        return float(np.std([e.total_reward for e in self.episodes]))

    @property
    def mean_length(self) -> float:
        return float(np.mean([e.episode_length for e in self.episodes]))

    @property
    def reach_rate(self) -> float:
        return float(np.mean([e.target_reached for e in self.episodes]))

    def summary(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "n_episodes": len(self.episodes),
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_length": self.mean_length,
            "reach_rate": self.reach_rate,
        }


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: GridEnvironment,
    agent: Agent,
    seed: int,
    *,
    train: bool = False,
) -> EpisodeResult:
    """Run one episode, returning structured results.

    Parameters
    ----------
    env : GridEnvironment
    agent : Agent
    seed : int
        Seed for ``env.reset()``.
    train : bool
        If True the agent is in TRAIN mode; otherwise EVAL.
    """
    agent.set_mode(AgentMode.TRAIN if train else AgentMode.EVAL)

    obs, info = env.reset(seed=seed)
    agent.prepare_episode(env)

    action = agent.begin_episode(obs)

    rewards: list[float] = []
    total_reward = 0.0
    steps = 0
    terminated = False

    for _ in range(env.max_steps):
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward)
        total_reward += reward
        steps += 1
        terminated = term

        if term or trunc:
            agent.end_episode(reward, terminal=term)
            break
        action = agent.step(reward, obs)
    else:
        # Loop finished without break (max_steps reached without term/trunc
        # from env.step -- shouldn't normally happen since env sets truncated)
        agent.end_episode(0.0, terminal=False)

    target_reached = info.get("target_reached", False)

    return EpisodeResult(
        seed=seed,
        total_reward=total_reward,
        episode_length=steps,
        terminated=terminated,
        target_reached=target_reached,
        rewards=rewards,
    )


# ---------------------------------------------------------------------------
# Multi-episode experiment
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: dict,
    *,
    num_episodes: int = 10,
    master_seed: int = 0,
    train_episodes: int = 0,
) -> ExperimentResult:
    """Run a full experiment from a config dict.

    Parameters
    ----------
    cfg : dict
        Experiment config (env + agent sections).
    num_episodes : int
        Number of *evaluation* episodes.
    master_seed : int
        Master seed; per-episode seeds derived deterministically.
    train_episodes : int
        Number of training episodes to run before evaluation.
        Set to 0 for non-learning agents (random, DP).

    Returns
    -------
    ExperimentResult
    """
    # Build a template env to get obs_shape / num_actions
    template_env, _ = build_env(cfg, seed=0)
    obs_shape = template_env.observation_space.shape
    num_actions = template_env.action_space.n
    template_env.close()

    agent = build_agent(cfg, num_actions=num_actions, obs_shape=obs_shape)

    # ---- Training phase ----------------------------------------------------
    if train_episodes > 0:
        agent.set_mode(AgentMode.TRAIN)
        for ep_idx in range(train_episodes):
            ep_seed = derive_seed(master_seed, ep_idx)
            env, _ = build_env(cfg, seed=ep_seed)
            run_episode(env, agent, seed=ep_seed, train=True)
            env.close()

    # ---- Evaluation phase --------------------------------------------------
    agent.set_mode(AgentMode.EVAL)
    results: list[EpisodeResult] = []
    for ep_idx in range(num_episodes):
        ep_seed = derive_seed(master_seed + 1_000_000, ep_idx)
        env, _ = build_env(cfg, seed=ep_seed)
        result = run_episode(env, agent, seed=ep_seed, train=False)
        results.append(result)
        env.close()

    return ExperimentResult(agent_name=agent.name, episodes=results)
