"""Visualize AP-SSP agent navigation across multiple targets.

Demonstrates the AP-SSP agent's ability to plan once for all (state, goal)
pairs and then navigate a sequence of waypoints — including a hard
start-position reset — without replanning.

Uses DynamicSGArena (no GridEnvironment wrapper) and MultiSegmentRenderer
for a multi-colored interactive HTML output.
"""

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import jax

from src.env.arena.dynamic_sg_arena import DynamicSGArena
from src.env.arena.reward import StepCostReward
from src.env.rendering.multi_segment_renderer import MultiSegmentRenderer
from src.env.field import RFFGPField
from src.env.actor.grid_actor import GridActor
from src.env.utils.types import GridConfig, GridPosition
from src.agents.ap_ssp_agent import APSSPAgent, APSSPAgentConfig


# ── Configuration ──────────────────────────────────────────────────────────

SEED = 42
GRID_NX, GRID_NY = 50, 50
MAX_GLOBAL_STEPS = 2000
MAX_SEGMENT_STEPS = 400

FIELD_PARAMS = dict(
    d_max=10,
    sigma=2.0,
    lengthscale=5.0,
    nu=2.5,
    num_features=500,
    noise_std=0.2,
)
ACTOR_PARAMS = dict(noise_std=0.1, scale=1.0, z_max=3)

VICINITY_RADIUS = 2.0
STEP_COST = 1.0
BOUNDARY_MODE = "terminal"

INITIAL_POSITION = GridPosition(25, 25, None)
INITIAL_TARGET = GridPosition(40, 40, None)

# (new_start_or_None, target)
SEGMENT_SEQUENCE = [
    (None, GridPosition(20, 20, None)),
    (None, GridPosition(25, 25, None)),
    (None, GridPosition(30, 30, None)),
    (GridPosition(15, 15, None), GridPosition(35, 35, None)),
]


# ── Main ───────────────────────────────────────────────────────────────────


def run_multi_target_viz():
    print("=" * 70)
    print("2D AP-SSP AGENT — MULTI-TARGET NAVIGATION")
    print("=" * 70)

    # 1. Build components
    config = GridConfig.create(n_x=GRID_NX, n_y=GRID_NY)
    field = RFFGPField(config, **FIELD_PARAMS)
    actor = GridActor(**ACTOR_PARAMS)

    reward_fn = StepCostReward(
        target_position=INITIAL_TARGET,
        vicinity_radius=VICINITY_RADIUS,
        step_cost=STEP_COST,
    )

    arena = DynamicSGArena(
        field=field,
        actor=actor,
        config=config,
        initial_position=INITIAL_POSITION,
        target_position=INITIAL_TARGET,
        vicinity_radius=VICINITY_RADIUS,
        boundary_mode=BOUNDARY_MODE,
        vicinity_metric="euclidean",
        reward_fn=reward_fn,
    )

    renderer = MultiSegmentRenderer(
        config=config,
        show_grid_points=True,
        width=900,
        height=900,
        field=field,
        show_field=True,
    )

    # 2. Full reset (samples the field)
    rng = jax.random.PRNGKey(SEED)
    rng, reset_key = jax.random.split(rng)
    arena.reset(reset_key)

    # 3. Plan once
    print("\nPlanning AP-SSP for all state-goal pairs …")
    agent = APSSPAgent(
        APSSPAgentConfig(max_iters=200, tol=1e-3),
        num_actions=3,
        obs_shape=(3,),
    )
    agent.plan(arena)
    print("Planning complete!\n")

    # 4. Segment loop
    for seg_idx, (new_start, new_target) in enumerate(SEGMENT_SEQUENCE):
        print(f"--- Segment {seg_idx + 1} ---")

        # Finalize previous segment in renderer (no-op on first iteration)
        renderer.new_segment()

        # Optionally hard-reset start position
        if new_start is not None:
            arena.set_position(new_start)
            print(f"  Hard-reset start → {new_start}")

        # Update target
        arena.set_target(new_target)
        agent.set_target(new_target)

        # Soft-reset segment counters (field untouched)
        obs = arena.soft_reset()

        # Record segment's initial state
        renderer.step(arena.get_state())

        expected_cost = agent.get_expected_cost(arena.position, new_target)
        print(f"  Target: {new_target}  |  Start: {arena.position}")
        print(f"  Expected cost (steps): {expected_cost:.2f}")

        # Step loop
        action = agent.begin_episode(obs)
        while (
            arena._segment_step_count < MAX_SEGMENT_STEPS
            and arena._global_step_count < MAX_GLOBAL_STEPS
        ):
            obs = arena.step(action)
            reward = arena.compute_reward()
            state = arena.get_state()
            renderer.step(state)

            if state.target_reached:
                break
            if arena.is_terminal():
                print("  !! Out of bounds — terminated.")
                break

            action = agent.step(reward, obs)

        state = arena.get_state()
        print(
            f"  Segment steps: {state.segment_step_count}  |  "
            f"Seg reward: {state.segment_cumulative_reward:+.2f}  |  "
            f"Global steps: {state.global_step_count}  |  "
            f"Global reward: {state.global_cumulative_reward:+.2f}  |  "
            f"Reached: {state.target_reached}"
        )
        print()

    # 5. Export
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "ap_ssp"
    )
    os.makedirs(output_dir, exist_ok=True)

    # animated_path = os.path.join(output_dir, "multi_target_navigation_animated.html")
    # print("Rendering animated HTML …")
    # renderer.save_animated_html(animated_path, fps=10)

    gif_path = os.path.join(output_dir, "multi_target_navigation.gif")
    print("Rendering GIF …")
    renderer.save_gif(gif_path, fps=10)

    # mp4_path = os.path.join(output_dir, "multi_target_navigation.mp4")
    # print("Rendering MP4 …")
    # renderer.save_mp4(mp4_path, fps=15)

    print("\nDone!")


if __name__ == "__main__":
    run_multi_target_viz()
