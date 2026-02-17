"""Smoke tests for the PPO agent on a small 2D grid."""

import numpy as np
from src.env import (
    GridEnvironment, NavigationArena, NavigationReward,
    GridActor, GridConfig, GridPosition,
)
from src.env.field import RFFGPField
from src.agents import PPOAgent, PPOConfig, RandomAgent, AgentConfig
from src.agents.agent import AgentMode


def make_small_2d_env(seed=42):
    config = GridConfig.create(n_x=50, n_y=50)
    field = RFFGPField(
        config, d_max=10, sigma=2.0, lengthscale=5.0, nu=2.5,
        num_features=500, noise_std=0.5,
    )
    actor = GridActor(noise_std=0.5, scale=2.0, z_max=10)
    target = GridPosition(40, 40, None)
    reward_fn = NavigationReward(
        target_position=target,
        vicinity_radius=5.0,
        peak_reward=10.0,
        step_cost=0.1,
        proximity_scale=1.0,
    )
    arena = NavigationArena(
        field=field, actor=actor, config=config,
        initial_position=GridPosition(2, 2, None),
        target_position=target,
        vicinity_radius=5.0,
        boundary_mode="terminal",
        reward_fn=reward_fn,
        terminate_on_reach=False,
    )
    env = GridEnvironment(arena=arena, max_steps=50, seed=seed)
    return env


def run_episode(env, agent, seed):
    """Run a single episode, return total reward."""
    obs, info = env.reset(seed=seed)
    action = agent.begin_episode(obs)
    total = 0.0
    for _ in range(env.max_steps - 1):
        obs, r, term, trunc, info = env.step(action)
        total += r
        if term or trunc:
            agent.end_episode(r, terminal=term)
            return total
        action = agent.step(r, obs)
    agent.end_episode(0.0, terminal=False)
    return total


def test_ppo_constructs_and_acts():
    """PPO agent can be created and produce valid actions."""
    cfg = PPOConfig(
        seed=0, hidden_dims=(32, 32),
        num_steps=32, num_minibatches=4, update_epochs=2,
        total_timesteps=2000,
    )
    agent = PPOAgent(cfg, num_actions=3, obs_shape=(3,))
    env = make_small_2d_env()

    obs, _ = env.reset(seed=0)
    action = agent.begin_episode(obs)
    assert 0 <= action <= 2
    for _ in range(15):
        obs, r, term, trunc, _ = env.step(action)
        if term or trunc:
            break
        action = agent.step(r, obs)
        assert 0 <= action <= 2
    agent.end_episode(0.0, terminal=False)
    env.close()
    print("[OK] PPO constructs and produces valid actions")


def test_ppo_trains():
    """PPO training loop runs without errors over multiple episodes."""
    cfg = PPOConfig(
        seed=0, hidden_dims=(32, 32),
        num_steps=32, num_minibatches=4, update_epochs=2,
        total_timesteps=5000,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_coef=0.2, ent_coef=0.01, vf_coef=0.5,
    )
    agent = PPOAgent(cfg, num_actions=3, obs_shape=(3,))
    agent.set_mode(AgentMode.TRAIN)

    n_train = 40
    train_rewards = []
    for ep in range(n_train):
        env = make_small_2d_env(seed=ep)
        r = run_episode(env, agent, seed=ep)
        train_rewards.append(r)
        env.close()

    # Evaluate (greedy)
    agent.set_mode(AgentMode.EVAL)
    n_eval = 10
    eval_rewards = []
    for ep in range(n_eval):
        env = make_small_2d_env(seed=1000 + ep)
        r = run_episode(env, agent, seed=1000 + ep)
        eval_rewards.append(r)
        env.close()

    # Random baseline
    rng_agent = RandomAgent(AgentConfig(seed=42), num_actions=3, obs_shape=(3,))
    rng_rewards = []
    for ep in range(n_eval):
        env = make_small_2d_env(seed=1000 + ep)
        r = run_episode(env, rng_agent, seed=1000 + ep)
        rng_rewards.append(r)
        env.close()

    train_mean = np.mean(train_rewards[-10:])
    eval_mean = np.mean(eval_rewards)
    rng_mean = np.mean(rng_rewards)

    print(f"[PPO] Train (last 10): {train_mean:.2f}  "
          f"| Eval: {eval_mean:.2f}  "
          f"| Random: {rng_mean:.2f}")
    print("[OK] PPO training loop completed without errors")


if __name__ == "__main__":
    print("=" * 60)
    print("PPO Agent Smoke Tests")
    print("=" * 60)

    print("\n--- PPO construction + action ---")
    test_ppo_constructs_and_acts()

    print("\n--- PPO training loop ---")
    test_ppo_trains()

    print("\nAll PPO tests passed.")
