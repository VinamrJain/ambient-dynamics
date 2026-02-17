"""Shared reward factory and contract for arena tests.

To switch the reward under test (e.g. to a new reward class), change only
this file: update make_reward_fn and (if needed) expected_reward_at_distance.
Tests in test_navigation_reward.py and test_episode_integration.py stay
reward-agnostic and use these helpers.
"""

from __future__ import annotations

from src.env.arena.reward import RewardFunction, NavigationReward
from src.env.utils.types import GridPosition


# Default reward params used when tests don't specify (NavigationReward).
DEFAULT_REWARD_KWARGS = {
    "peak_reward": 10.0,
    "step_cost": 0.1,
    "proximity_scale": 0.1,
}


def make_reward_fn(
    target_position: GridPosition,
    vicinity_radius: float,
    **kwargs,
) -> RewardFunction:
    """Build the reward function used by arena tests.

    Change this function to use a different reward class (e.g. a new
    RewardFunction subclass); all tests that build an arena via this helper
    will then use the new reward without further edits.
    """
    merged = {**DEFAULT_REWARD_KWARGS, **kwargs}
    return NavigationReward(
        target_position=target_position,
        vicinity_radius=vicinity_radius,
        **{k: v for k, v in merged.items() if k in ("peak_reward", "step_cost", "proximity_scale")},
    )


def expected_reward_at_distance(dist: float, **kwargs) -> float:
    """Expected reward at distance D for the reward currently returned by make_reward_fn.

    Used by tests that assert exact reward values (e.g. formula contracts).
    Update this when you switch make_reward_fn to a different reward class.
    """
    merged = {**DEFAULT_REWARD_KWARGS, **kwargs}
    peak = merged.get("peak_reward", DEFAULT_REWARD_KWARGS["peak_reward"])
    cost = merged.get("step_cost", DEFAULT_REWARD_KWARGS["step_cost"])
    scale = merged.get("proximity_scale", DEFAULT_REWARD_KWARGS["proximity_scale"])
    # NavigationReward: r(D) = (peak - cost) / (1 + scale * D)
    return (peak - cost) / (1.0 + scale * dist)
