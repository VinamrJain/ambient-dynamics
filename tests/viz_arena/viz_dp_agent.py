"""Visualize DP agent navigation on 2D and 3D grids.

Runs the DP (oracle) agent alongside a random baseline and exports
animated HTML visualizations via NavigationRenderer.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from src.env import (
    GridEnvironment,
    NavigationArena,
    NavigationReward,
    GridActor,
    NavigationRenderer,
    GridConfig,
    GridPosition,
)
from src.env.field import RFFGPField
from src.agents import DPAgent, DPAgentConfig, RandomAgent, AgentConfig


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_2d_env(seed=42, renderer=None):
    """2D navigation environment."""
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
        initial_position=GridPosition(15, 15, None),
        target_position=target,
        vicinity_radius=5.0,
        boundary_mode="terminal",
        reward_fn=reward_fn,
        terminate_on_reach=False,
    )
    env = GridEnvironment(arena=arena, max_steps=50, seed=seed, renderer=renderer)
    return env, arena, config


def make_3d_env(seed=42, renderer=None):
    """3D navigation environment."""
    config = GridConfig.create(n_x=25, n_y=25, n_z=15)
    field = RFFGPField(
        config, d_max=10, sigma=1.0, lengthscale=3.0, nu=2.5,
        num_features=500, noise_std=0.5,
    )
    actor = GridActor(noise_std=0.1, scale=0.5, z_max=5)
    target = GridPosition(14, 14, 7)
    reward_fn = NavigationReward(
        target_position=target,
        vicinity_radius=2.0,
        peak_reward=10.0,
        step_cost=0.1,
        proximity_scale=1.0,
    )
    arena = NavigationArena(
        field=field, actor=actor, config=config,
        initial_position=GridPosition(10, 10, 8),
        target_position=target,
        vicinity_radius=2.0,
        boundary_mode="terminal",
        reward_fn=reward_fn,
        terminate_on_reach=False,
    )
    env = GridEnvironment(arena=arena, max_steps=50, seed=seed, renderer=renderer)
    return env, arena, config


# ---------------------------------------------------------------------------
# Episode runner (returns trajectory + reward)
# ---------------------------------------------------------------------------

def run_episode(env, arena, agent, horizon, *, is_dp=False, seed=42):
    """Run a single episode and return (total_reward, n_steps, info).

    For DPAgent, calls plan() before acting.
    """
    obs, info = env.reset(seed=seed)
    if is_dp:
        agent.plan(arena, horizon)

    action = agent.begin_episode(obs)
    total_reward = 0.0
    steps = 0

    for _ in range(horizon - 1):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
        action = agent.step(reward, obs)

    return total_reward, steps, info


# ---------------------------------------------------------------------------
# 2D visualization
# ---------------------------------------------------------------------------

def run_dp_2d_viz():
    """DP agent on 2D grid with animated HTML output."""
    print("=" * 70)
    print("2D DP AGENT NAVIGATION")
    print("=" * 70)

    # --- DP agent (with renderer) ---
    env, arena, config = make_2d_env(seed=42)
    renderer = NavigationRenderer(
        config=config,
        show_grid_points=True,
        width=900, height=900,
        field=arena.field,
        show_field=True,
    )
    env, arena, config = make_2d_env(seed=42, renderer=renderer)
    horizon = env.max_steps

    dp_agent = DPAgent(num_actions=3, obs_shape=(3,))
    dp_reward, dp_steps, dp_info = run_episode(
        env, arena, dp_agent, horizon, is_dp=True,
    )
    print(f"\n  [DP]     reward={dp_reward:+.2f}  steps={dp_steps}"
          f"  target_reached={dp_info['target_reached']}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "2d")
    os.makedirs(output_dir, exist_ok=True)
    dp_html = os.path.join(output_dir, "dp_agent_2d.html")
    renderer.save_animated_html(dp_html)
    print(f"  Saved: {dp_html}")
    env.close()

    # --- Random agent (with renderer) ---
    rng_renderer = NavigationRenderer(
        config=config,
        show_grid_points=True,
        width=900, height=900,
        field=arena.field,
        show_field=True,
    )
    env, arena, config = make_2d_env(seed=42, renderer=rng_renderer)
    rng_agent = RandomAgent(AgentConfig(seed=42), num_actions=3, obs_shape=(3,))
    rng_reward, rng_steps, rng_info = run_episode(
        env, arena, rng_agent, horizon, is_dp=False,
    )
    print(f"  [Random] reward={rng_reward:+.2f}  steps={rng_steps}"
          f"  target_reached={rng_info['target_reached']}")

    rng_html = os.path.join(output_dir, "random_agent_2d.html")
    rng_renderer.save_animated_html(rng_html)
    print(f"  Saved: {rng_html}")
    env.close()


# ---------------------------------------------------------------------------
# 3D visualization
# ---------------------------------------------------------------------------

def run_dp_3d_viz():
    """DP agent on 3D grid with animated HTML output."""
    print("\n" + "=" * 70)
    print("3D DP AGENT NAVIGATION")
    print("=" * 70)

    # --- DP agent (with renderer) ---
    env, arena, config = make_3d_env(seed=42)
    renderer = NavigationRenderer(
        config=config,
        show_grid_points=True,
        width=900, height=900,
        field=arena.field,
        show_field=True,
    )
    env, arena, config = make_3d_env(seed=42, renderer=renderer)
    horizon = env.max_steps

    dp_agent = DPAgent(num_actions=3, obs_shape=(5,))
    dp_reward, dp_steps, dp_info = run_episode(
        env, arena, dp_agent, horizon, is_dp=True,
        seed=102,
    )
    print(f"\n  [DP]     reward={dp_reward:+.2f}  steps={dp_steps}"
          f"  target_reached={dp_info['target_reached']}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "3d")
    os.makedirs(output_dir, exist_ok=True)
    dp_html = os.path.join(output_dir, "dp_agent_3d.html")
    renderer.save_animated_html(dp_html)
    print(f"  Saved: {dp_html}")
    env.close()

    # --- Random agent (with renderer) ---
    rng_renderer = NavigationRenderer(
        config=config,
        show_grid_points=True,
        width=900, height=900,
        field=arena.field,
        show_field=True,
    )
    env, arena, config = make_3d_env(seed=42, renderer=rng_renderer)
    rng_agent = RandomAgent(AgentConfig(seed=42), num_actions=3, obs_shape=(5,))
    rng_reward, rng_steps, rng_info = run_episode(
        env, arena, rng_agent, horizon, is_dp=False,
    )
    print(f"  [Random] reward={rng_reward:+.2f}  steps={rng_steps}"
          f"  target_reached={rng_info['target_reached']}")

    rng_html = os.path.join(output_dir, "random_agent_3d.html")
    rng_renderer.save_animated_html(rng_html)
    print(f"  Saved: {rng_html}")
    env.close()


# ---------------------------------------------------------------------------
# DP vs Random comparison (numerical, no rendering)
# ---------------------------------------------------------------------------

def run_dp_vs_random(dim=2):
    """Compare DP and random agents over multiple episodes.
    Args:
        dim: int, 2 for 2D environment, 3 for 3D environment.
    """
    print("\n" + "=" * 70)
    if dim == 2:
        print("DP vs RANDOM -- 5-episode comparison (2D)")
    elif dim == 3:
        print("DP vs RANDOM -- 5-episode comparison (3D)")
    else:
        raise ValueError("dim must be 2 or 3")
    print("=" * 70)

    n_episodes = 5
    dp_rewards = []
    rng_rewards = []

    if dim == 2:
        env, arena, config = make_2d_env()
        num_actions = 3
        obs_shape = (3,)
    else:
        env, arena, config = make_3d_env()
        num_actions = 3
        obs_shape = (5,)

    horizon = env.max_steps
    dp_agent = DPAgent(num_actions=num_actions, obs_shape=obs_shape)
    rng_agent = RandomAgent(AgentConfig(), num_actions=num_actions, obs_shape=obs_shape)

    for ep in range(n_episodes):
        seed = 100 + ep

        dp_r, dp_steps, dp_info = run_episode(env, arena, dp_agent, horizon, is_dp=True, seed=seed)
        dp_rewards.append(dp_r)
        print(f"  [DP]     reward={dp_r:+.2f}  steps={dp_steps}"
          f"  target_reached={dp_info['target_reached']}")
        rng_r, rng_steps, rng_info = run_episode(env, arena, rng_agent, horizon, is_dp=False, seed=seed)
        rng_rewards.append(rng_r)
        print(f"  [Random] reward={rng_r:+.2f}  steps={rng_steps}"
          f"  target_reached={rng_info['target_reached']}")

    env.close()

    dp_mean = np.mean(dp_rewards)
    rng_mean = np.mean(rng_rewards)
    print(f"\n  DP  mean reward: {dp_mean:+.2f}  (per-ep: {[f'{r:+.1f}' for r in dp_rewards]})")
    print(f"  Rng mean reward: {rng_mean:+.2f}  (per-ep: {[f'{r:+.1f}' for r in rng_rewards]})")
    print(f"  DP advantage:    {dp_mean - rng_mean:+.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    #run_dp_2d_viz()
    run_dp_3d_viz()
    #run_dp_vs_random(dim=3)
    print("\nDone.")
