"""Tests for the AP-SSP agent."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from src.agents.ap_ssp_agent import APSSPAgent, APSSPAgentConfig
from src.env.actor.grid_actor import GridActor
from src.env.arena.navigation_arena import NavigationArena
from src.env.arena.reward import StepCostReward
from src.env.field.rff_gp_field import RFFGPField
from src.env.utils.types import GridConfig, GridPosition


def _build_arena(seed: int = 0) -> NavigationArena:
    cfg = GridConfig.create(n_x=15, n_y=15)
    field = RFFGPField(
        cfg, d_max=3, sigma=1.0, lengthscale=3.0, nu=2.5,
        num_features=200, noise_std=0.0,
    )
    actor = GridActor(noise_std=0.0, scale=1.0, z_max=2)
    target = GridPosition(8, 8, None)
    reward_fn = StepCostReward(
        target_position=target, vicinity_radius=1.0, step_cost=1.0,
    )
    arena = NavigationArena(
        field=field, actor=actor, config=cfg,
        initial_position=GridPosition(2, 2, None),
        target_position=target,
        vicinity_radius=1.0,
        boundary_mode="terminal",
        vicinity_metric="euclidean",
        reward_fn=reward_fn,
    )
    arena.reset(jr.PRNGKey(seed))
    return arena


def test_warmstart_matches_cold_start():
    """Warmstart from a converged H reaches the same fixed point as cold start."""
    arena = _build_arena(seed=0)

    cold = APSSPAgent(APSSPAgentConfig(max_iters=2000, rel_tol=1e-5, warmstart=False))
    cold.plan(arena)
    H_cold = np.array(cold._cost_table)

    warm = APSSPAgent(APSSPAgentConfig(max_iters=2000, rel_tol=1e-5, warmstart=True))
    warm.plan(arena)  # cold plan (no prior table)
    warm.plan(arena)  # warm replan on identical arena
    H_warm = np.array(warm._cost_table)

    assert H_cold.shape == H_warm.shape
    assert np.max(np.abs(H_cold - H_warm)) < 1e-4
