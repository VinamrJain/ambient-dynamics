"""End-to-end tests for the evaluation harness."""

import os

import numpy as np

from src.eval import (
    load_config,
    build_env,
    build_agent,
    run_episode,
    run_experiment,
    derive_seed,
    launch_suite,
    PrintLogger,
)
from src.agents.agent import AgentMode


SUITE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments", "configs", "smoke_suite.yaml",
)


def test_derive_seed_deterministic():
    """Same inputs always produce the same seed."""
    s1 = derive_seed(42, 0)
    s2 = derive_seed(42, 0)
    s3 = derive_seed(42, 1)
    assert s1 == s2
    assert s1 != s3
    print(f"[OK] derive_seed deterministic: seed(42,0)={s1}, seed(42,1)={s3}")


def test_build_env_and_agent():
    """Config-driven env and agent construction."""
    cfg = load_config(SUITE_PATH)
    defaults = cfg["defaults"]
    # Build with random agent
    run_cfg = {**defaults, "agent": {"name": "random", "params": {"seed": 0}}}
    env, arena = build_env(run_cfg, seed=0)
    agent = build_agent(run_cfg, num_actions=3, obs_shape=env.observation_space.shape)
    assert agent.name == "RandomAgent"
    env.close()
    print("[OK] build_env and build_agent from config")


def test_run_episode_random():
    """run_episode with RandomAgent returns valid results."""
    cfg = load_config(SUITE_PATH)
    defaults = cfg["defaults"]
    run_cfg = {**defaults, "agent": {"name": "random", "params": {"seed": 0}}}
    env, arena = build_env(run_cfg, seed=42)
    agent = build_agent(run_cfg, num_actions=3, obs_shape=env.observation_space.shape)
    result = run_episode(env, agent, seed=42)
    assert result.episode_length > 0
    assert len(result.rewards) == result.episode_length
    env.close()
    print(f"[OK] run_episode random: reward={result.total_reward:.2f}, "
          f"steps={result.episode_length}")


def test_run_episode_dp():
    """run_episode with DPAgent (auto plan)."""
    cfg = load_config(SUITE_PATH)
    defaults = cfg["defaults"]
    run_cfg = {**defaults, "agent": {"name": "dp", "params": {}}}
    env, arena = build_env(run_cfg, seed=42)
    agent = build_agent(run_cfg, num_actions=3, obs_shape=env.observation_space.shape)
    result = run_episode(env, agent, seed=42)
    assert result.episode_length > 0
    env.close()
    print(f"[OK] run_episode DP: reward={result.total_reward:.2f}, "
          f"steps={result.episode_length}")


def test_run_experiment():
    """run_experiment with DQN (train + eval)."""
    cfg = load_config(SUITE_PATH)
    defaults = cfg["defaults"]
    dqn_run = cfg["runs"][2]  # DQN entry
    from src.eval.launcher import _deep_merge
    run_cfg = _deep_merge(defaults, dqn_run)
    result = run_experiment(
        run_cfg, num_episodes=3, master_seed=42, train_episodes=10,
    )
    assert len(result.episodes) == 3
    print(f"[OK] run_experiment DQN: mean_reward={result.mean_reward:.2f}, "
          f"reach_rate={result.reach_rate:.0%}")


def test_launch_suite_serial():
    """launch_suite with workers=1 (serial) on the smoke config."""
    cfg = load_config(SUITE_PATH)
    results = launch_suite(cfg, max_workers=1)
    assert len(results) == 4
    for r in results:
        assert "error" not in r, f"Run {r['run_id']} failed: {r.get('error')}"
        print(f"  {r['run_id']:>8s}  |  reward={r['mean_reward']:.2f} ± {r['std_reward']:.2f}")
    print("[OK] launch_suite serial completed")


if __name__ == "__main__":
    print("=" * 60)
    print("Eval Harness Tests")
    print("=" * 60)

    print("\n--- Seed derivation ---")
    test_derive_seed_deterministic()

    print("\n--- Build env + agent ---")
    test_build_env_and_agent()

    print("\n--- run_episode (random) ---")
    test_run_episode_random()

    print("\n--- run_episode (DP) ---")
    test_run_episode_dp()

    print("\n--- run_experiment (DQN) ---")
    test_run_experiment()

    print("\n--- launch_suite (serial) ---")
    test_launch_suite_serial()

    print("\nAll eval harness tests passed.")
