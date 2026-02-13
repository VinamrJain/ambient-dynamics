"""Parameterized contract tests for NavigationArena reward computation.

Covers distance calculation, vicinity detection, decay, cumulative tracking,
and constructor validation — for both 2D and 3D across randomized scenarios.
"""

import pytest
import numpy as np
import jax

from src.env.arena.navigation_arena import NavigationArena
from src.env.field.simple_field import SimpleField
from src.env.actor.grid_actor import GridActor
from src.env.utils.types import GridConfig, GridPosition

RNG = np.random.default_rng(9999)


# =============================================================================
# Helpers
# =============================================================================

def _make_arena(
    ndim: int, n: int, initial: GridPosition, target: GridPosition, *,
    vicinity_radius: float = 2.0, vicinity_bonus: float = 10.0,
    step_penalty: float = -0.5, distance_reward_weight: float = -0.1,
    use_distance_decay: bool = False, decay_rate: float = 0.5,
    d_max: int = 1
) -> NavigationArena:
    if ndim == 3:
        config = GridConfig.create(n, n, n)
    else:
        config = GridConfig.create(n, n)
    field = SimpleField(config, d_max=d_max)
    actor = GridActor(noise_std=0.0)
    return NavigationArena(
        field=field, actor=actor, config=config,
        initial_position=initial, target_position=target,
        vicinity_radius=vicinity_radius, boundary_mode="clip",
        distance_reward_weight=distance_reward_weight,
        vicinity_bonus=vicinity_bonus, step_penalty=step_penalty,
        terminate_on_reach=False,
        use_distance_decay=use_distance_decay, decay_rate=decay_rate,
    )


# =============================================================================
# Parameterized distance + reward contract
# =============================================================================

REWARD_CASES = [
    # (ndim, initial, target, vicinity_radius, description)
    (2, (5, 5), (5, 5), 2.0, "2d-at-target"),
    (2, (1, 1), (5, 5), 2.0, "2d-far-from-target"),
    (2, (7, 5), (5, 5), 2.0, "2d-on-boundary"),       # dist = 2.0 exactly
    (2, (8, 5), (5, 5), 2.0, "2d-just-outside"),       # dist = 3.0
    (2, (1, 1), (10, 10), 1.0, "2d-max-distance"),
    (3, (5, 5, 5), (5, 5, 5), 2.0, "3d-at-target"),
    (3, (1, 1, 1), (5, 5, 5), 2.0, "3d-far-from-target"),
    (3, (7, 5, 5), (5, 5, 5), 2.0, "3d-on-boundary"),  # dist = 2.0
    (3, (8, 5, 5), (5, 5, 5), 2.0, "3d-just-outside"),
    (3, (1, 1, 1), (10, 10, 10), 0.5, "3d-max-distance"),
]


@pytest.mark.parametrize(
    ("ndim", "init_tup", "targ_tup", "radius", "desc"), REWARD_CASES,
    ids=lambda *a: a[-1] if isinstance(a[-1], str) else str(a)
)
def test_reward_contracts(ndim, init_tup, targ_tup, radius, desc):
    """Reward contracts: distance, vicinity detection, penalty/bonus signs."""
    n = 10
    if ndim == 2:
        initial = GridPosition(init_tup[0], init_tup[1], None)
        target = GridPosition(targ_tup[0], targ_tup[1], None)
    else:
        initial = GridPosition(*init_tup)
        target = GridPosition(*targ_tup)

    bonus = 50.0
    weight = -0.2
    penalty = -1.0
    arena = _make_arena(ndim, n, initial, target,
                        vicinity_radius=radius, vicinity_bonus=bonus,
                        distance_reward_weight=weight, step_penalty=penalty,
                        d_max=0)  # d_max=0 so field can't move us

    arena.reset(jax.random.PRNGKey(0))

    # Compute expected distance
    if ndim == 2:
        dist = np.sqrt((init_tup[0] - targ_tup[0])**2 + (init_tup[1] - targ_tup[1])**2)
    else:
        dist = np.sqrt(sum((a - b)**2 for a, b in zip(init_tup, targ_tup)))

    assert arena._compute_distance(initial, target) == pytest.approx(dist, abs=1e-10)

    reward = arena.compute_reward()

    in_vicinity = dist <= radius
    if in_vicinity:
        assert reward == pytest.approx(bonus, rel=1e-6)
        assert arena._target_reached is True
    else:
        expected = weight * dist + penalty
        assert reward == pytest.approx(expected, rel=1e-6)
        assert reward < 0  # penalties are negative


# =============================================================================
# Distance decay contract
# =============================================================================

DECAY_CASES = [
    # (ndim, initial, target, decay_rate, description)
    (2, (5, 5), (5, 5), 0.5, "2d-center"),
    (2, (6, 5), (5, 5), 0.5, "2d-dist1"),
    (2, (6, 5), (5, 5), 2.0, "2d-dist1-fast-decay"),
    (3, (5, 5, 5), (5, 5, 5), 0.5, "3d-center"),
    (3, (6, 5, 5), (5, 5, 5), 0.5, "3d-dist1"),
    (3, (6, 5, 5), (5, 5, 5), 0.0, "3d-zero-decay"),  # exp(0) = 1
]


@pytest.mark.parametrize(
    ("ndim", "init_tup", "targ_tup", "decay_rate", "desc"), DECAY_CASES,
    ids=lambda *a: a[-1] if isinstance(a[-1], str) else str(a)
)
def test_decay_contracts(ndim, init_tup, targ_tup, decay_rate, desc):
    """Vicinity bonus with exponential decay."""
    n = 10
    bonus = 100.0
    if ndim == 2:
        initial = GridPosition(init_tup[0], init_tup[1], None)
        target = GridPosition(targ_tup[0], targ_tup[1], None)
        dist = np.sqrt((init_tup[0]-targ_tup[0])**2 + (init_tup[1]-targ_tup[1])**2)
    else:
        initial = GridPosition(*init_tup)
        target = GridPosition(*targ_tup)
        dist = np.sqrt(sum((a-b)**2 for a, b in zip(init_tup, targ_tup)))

    arena = _make_arena(ndim, n, initial, target,
                        vicinity_radius=3.0, vicinity_bonus=bonus,
                        use_distance_decay=True, decay_rate=decay_rate, d_max=0)
    arena.reset(jax.random.PRNGKey(0))
    reward = arena.compute_reward()

    expected = bonus * np.exp(-decay_rate * dist)
    assert reward == pytest.approx(expected, rel=1e-6)


# =============================================================================
# Cumulative reward + reset contract
# =============================================================================

@pytest.mark.parametrize("ndim", [2, 3])
def test_cumulative_reward_and_reset(ndim):
    """Cumulative reward accumulates then resets to zero."""
    n = 10
    if ndim == 2:
        pos = GridPosition(5, 5, None)
    else:
        pos = GridPosition(5, 5, 5)

    arena = _make_arena(ndim, n, pos, pos, vicinity_bonus=7.0, d_max=0)
    arena.reset(jax.random.PRNGKey(0))

    for k in range(1, 6):
        arena.compute_reward()
        assert arena.get_cumulative_reward() == pytest.approx(7.0 * k)

    arena.reset(jax.random.PRNGKey(1))
    assert arena.get_cumulative_reward() == 0.0
    assert arena._target_reached is False


# =============================================================================
# Constructor validation
# =============================================================================

@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"vicinity_radius": 0.0}, "vicinity_radius must be positive"),
        ({"vicinity_radius": -1.0}, "vicinity_radius must be positive"),
        ({"vicinity_bonus": 0.0}, "vicinity_bonus must be positive"),
        ({"vicinity_bonus": -5.0}, "vicinity_bonus must be positive"),
        ({"step_penalty": 0.1}, "step_penalty must be non-positive"),
        ({"step_penalty": 10.0}, "step_penalty must be non-positive"),
        ({"distance_reward_weight": 0.1}, "distance_reward_weight must be non-positive"),
        ({"decay_rate": -0.1}, "decay_rate must be non-negative"),
    ],
)
def test_navigation_arena_rejects_invalid_params(kwargs, match):
    config = GridConfig.create(n_x=10, n_y=10)
    field = SimpleField(config, d_max=1)
    actor = GridActor(noise_std=0.0)
    defaults = dict(
        field=field, actor=actor, config=config,
        initial_position=GridPosition(5, 5, None),
        target_position=GridPosition(5, 5, None),
        vicinity_radius=2.0, vicinity_bonus=10.0,
        step_penalty=-0.5, distance_reward_weight=-0.1,
        decay_rate=0.5,
    )
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        NavigationArena(**defaults)


@pytest.mark.parametrize(
    ("target", "ndim", "match"),
    [
        (GridPosition(0, 5, None), 2, "target_position.*outside grid"),
        (GridPosition(11, 5, None), 2, "target_position.*outside grid"),
        (GridPosition(5, 0, None), 2, "target_position.*outside grid"),
        (GridPosition(0, 5, 5), 3, "target_position.*outside grid"),
        (GridPosition(5, 5, 0), 3, "target_position.k.*outside grid"),
        (GridPosition(5, 5, 11), 3, "target_position.k.*outside grid"),
    ],
)
def test_navigation_arena_rejects_oob_target(target, ndim, match):
    if ndim == 2:
        config = GridConfig.create(n_x=10, n_y=10)
        init = GridPosition(5, 5, None)
    else:
        config = GridConfig.create(n_x=10, n_y=10, n_z=10)
        init = GridPosition(5, 5, 5)
    field = SimpleField(config, d_max=1)
    actor = GridActor(noise_std=0.0)
    with pytest.raises(ValueError, match=match):
        NavigationArena(field=field, actor=actor, config=config,
                        initial_position=init, target_position=target,
                        vicinity_radius=2.0)
