"""Comprehensive contract tests for RFFGPField."""

import os
import sys
import warnings

import jax
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.field import RFFGPField
from src.env.utils.types import GridConfig, GridPosition

RNG = np.random.default_rng(2026)


def _sample_case(ndim: int, seed: int) -> dict:
    """Build one randomized-yet-safe initialization case."""
    if ndim == 2:
        n_x = int(RNG.integers(1, 128))
        n_y = int(RNG.integers(1, 128))
        n_z = None
        d_bound = n_x - 1
    else:
        n_x = int(RNG.integers(1, 128))
        n_y = int(RNG.integers(1, 128))
        n_z = int(RNG.integers(1, 64))
        d_bound = min(n_x, n_y) - 1

    return {
        "ndim": ndim,
        "seed": seed,
        "n_x": n_x,
        "n_y": n_y,
        "n_z": n_z,
        "d_max": int(RNG.integers(1, max(2, d_bound + 1))),
        "sigma": float(10 ** RNG.uniform(-1, 1)),
        "lengthscale": float(10 ** RNG.uniform(-2, 2)),
        "nu": float(RNG.choice([0.5, 1.5, 2.5, 4.5])),
        "num_features": int(RNG.integers(32, 160)),
        "noise_std": float(10 ** RNG.uniform(-4, 0)),
    }


RANDOM_CASES = (
    [_sample_case(2, seed) for seed in range(10)]
    + [_sample_case(3, 100 + seed) for seed in range(10)]
)


def _build_field(case: dict, noise_std: float | None = None) -> RFFGPField:
    config = GridConfig.create(case["n_x"], case["n_y"], case["n_z"])
    return RFFGPField(
        config=config,
        d_max=case["d_max"],
        sigma=case["sigma"],
        lengthscale=case["lengthscale"],
        nu=case["nu"],
        num_features=case["num_features"],
        noise_std=case["noise_std"] if noise_std is None else noise_std,
    )


def _probe_positions(case: dict) -> list[GridPosition]:
    center_i = (case["n_x"] + 1) // 2
    center_j = (case["n_y"] + 1) // 2
    if case["ndim"] == 2:
        return [
            GridPosition(1, 1, None),
            GridPosition(center_i, center_j, None),
            GridPosition(case["n_x"], case["n_y"], None),
        ]
    center_k = (case["n_z"] + 1) // 2
    return [
        GridPosition(1, 1, 1),
        GridPosition(center_i, center_j, center_k),
        GridPosition(case["n_x"], case["n_y"], case["n_z"]),
    ]


@pytest.mark.parametrize("case", RANDOM_CASES, ids=lambda c: f'{c["ndim"]}d-seed{c["seed"]}')
def test_randomized_field_contracts(case: dict):
    """Single comprehensive test: init + reset + mean + sample + pmf + velocity."""
    field = _build_field(case)
    spatial_dim = 2 if case["ndim"] == 2 else 3

    # Initialization contract
    assert field.ndim == case["ndim"]
    assert field.d_max == case["d_max"]
    assert field._precomputed_u is None
    if case["ndim"] == 3:
        assert field._precomputed_v is None

    # Reset and shape contract
    field.reset(jax.random.PRNGKey(case["seed"]))
    assert field._omegas.shape == (case["num_features"], spatial_dim)
    assert field._phases.shape == (case["num_features"],)
    assert field._weights.shape == (case["num_features"],)

    mean_field = field.get_mean_displacement_field()
    if case["ndim"] == 2:
        assert mean_field.shape == (case["n_x"], case["n_y"], 1)
    else:
        assert mean_field.shape == (case["n_x"], case["n_y"], case["n_z"], 2)

    for idx, pos in enumerate(_probe_positions(case)):
        mean = field.get_mean_displacement(pos)
        sample = field.sample_displacement(pos, jax.random.PRNGKey(case["seed"] + idx + 1))
        pmf = field.get_displacement_pmf(pos)

        # Mean shape/value and consistency with precomputed field
        if case["ndim"] == 2:
            assert len(mean) == 1
            assert isinstance(mean[0], float)
            i_idx, j_idx = pos.i - 1, pos.j - 1
            assert mean[0] == pytest.approx(float(mean_field[i_idx, j_idx, 0]), rel=1e-5, abs=1e-6)
        else:
            assert len(mean) == 2
            assert isinstance(mean[0], float)
            assert isinstance(mean[1], float)
            i_idx, j_idx, k_idx = pos.i - 1, pos.j - 1, pos.k - 1
            assert mean[0] == pytest.approx(float(mean_field[i_idx, j_idx, k_idx, 0]), rel=1e-5, abs=1e-6)
            assert mean[1] == pytest.approx(float(mean_field[i_idx, j_idx, k_idx, 1]), rel=1e-5, abs=1e-6)

        # Sample clipping contract
        assert -field.d_max <= sample.u <= field.d_max
        if case["ndim"] == 2:
            assert sample.v is None
        else:
            assert isinstance(sample.v, float)
            assert -field.d_max <= sample.v <= field.d_max

        # PMF positivity + normalization contract
        expected_size = 2 * field.d_max + 1
        if case["ndim"] == 2:
            assert pmf.shape == (expected_size,)
        else:
            assert pmf.shape == (expected_size, expected_size)
        assert np.all(np.isfinite(pmf))
        assert np.all(pmf >= 0.0)
        assert float(np.sum(pmf)) == pytest.approx(1.0, abs=1e-5)

        # velocity_at_point matches precomputed mean at integer grid points
        if case["ndim"] == 2:
            u, v = field.velocity_at_point(float(pos.i), float(pos.j))
            assert v is None
            assert float(u) == pytest.approx(mean[0], rel=1e-4, abs=1e-6)
        else:
            u, v = field.velocity_at_point(float(pos.i), float(pos.j), float(pos.k))
            assert float(u) == pytest.approx(mean[0], rel=1e-4, abs=1e-6)
            assert float(v) == pytest.approx(mean[1], rel=1e-4, abs=1e-6)


@pytest.mark.parametrize("case", RANDOM_CASES[:3] + RANDOM_CASES[10:13], ids=lambda c: f'{c["ndim"]}d-seed{c["seed"]}')
def test_reset_reproducibility_and_seed_change(case: dict):
    """Same key reproduces exactly, different key changes sampled field."""
    field = _build_field(case)
    key = jax.random.PRNGKey(case["seed"])

    field.reset(key)
    mean_1 = field.get_mean_displacement_field().copy()

    field.reset(key)
    mean_2 = field.get_mean_displacement_field().copy()

    np.testing.assert_array_equal(mean_1, mean_2)

    field.reset(jax.random.PRNGKey(case["seed"] + 10_000))
    mean_3 = field.get_mean_displacement_field().copy()
    assert not np.allclose(mean_1, mean_3)


@pytest.mark.parametrize("case", RANDOM_CASES[:3] + RANDOM_CASES[10:13], ids=lambda c: f'{c["ndim"]}d-seed{c["seed"]}')
def test_zero_noise_sample_matches_clipped_mean(case: dict):
    """With zero observation noise, sample equals clipped mean exactly."""
    field = _build_field(case, noise_std=0.0)
    field.reset(jax.random.PRNGKey(case["seed"]))

    for pos in _probe_positions(case):
        mean = field.get_mean_displacement(pos)
        sample = field.sample_displacement(pos, jax.random.PRNGKey(case["seed"] + 55))
        assert sample.u == pytest.approx(np.clip(mean[0], -field.d_max, field.d_max), abs=1e-7)
        if case["ndim"] == 2:
            assert sample.v is None
        else:
            assert sample.v == pytest.approx(np.clip(mean[1], -field.d_max, field.d_max), abs=1e-7)


@pytest.mark.parametrize("case", RANDOM_CASES[10:13], ids=lambda c: f'3d-seed{c["seed"]}')
def test_divergence_free_streamfunction(case: dict):
    """3D streamfunction parameterization is divergence-free: du/dx + dv/dy ~= 0."""
    field = _build_field(case)
    field.reset(jax.random.PRNGKey(case["seed"]))

    def u_fn(x, y, z):
        u, _ = field.velocity_at_point(x, y, z)
        return u

    def v_fn(x, y, z):
        _, v = field.velocity_at_point(x, y, z)
        return v

    points = [
        (1.25, 1.75, 1.5),
        (case["n_x"] * 0.6, case["n_y"] * 0.4, case["n_z"] * 0.7),
    ]
    for x, y, z in points:
        divergence = jax.grad(u_fn, argnums=0)(x, y, z) + jax.grad(v_fn, argnums=1)(x, y, z)
        assert abs(float(divergence)) < 1e-3


def test_pmf_fallback_warning_or_valid_normalization():
    """Extreme setup either warns on fallback or still returns normalized PMF."""
    config = GridConfig.create(n_x=3, n_y=3)
    field = RFFGPField(config, d_max=1, sigma=1_000.0, noise_std=0.01, num_features=128)
    field.reset(jax.random.PRNGKey(42))

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pmf = field.get_displacement_pmf(GridPosition(2, 2, None))
        assert pmf.shape == (3,)
        assert float(np.sum(pmf)) == pytest.approx(1.0, abs=1e-5)
        assert np.all(pmf >= 0.0)
        assert captured is not None
