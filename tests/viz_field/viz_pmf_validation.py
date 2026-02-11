"""# %% [markdown]
# Empirical vs Analytical PMF Validation for RFFGPField (Interactive)
#
# This notebook-style script validates whether empirical displacement outcomes
# from `sample_displacement` match analytical probabilities from
# `get_displacement_pmf`.
#
# Experiments included:
# 1. 2D interior regime (mean away from clipping boundaries)
# 2. 2D boundary regime (mean near +d_max to test tail mass accumulation)
# 3. 2D extreme clipping regime (mean far beyond +d_max)
# 4. 2D deterministic regime (noise_std = 0)
# 5. 3D joint PMF validation (u, v) with heatmaps
# """

# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import jax


from _viz_common import add_project_root_to_path, format_params, save_figure

add_project_root_to_path()

from src.env.field import RFFGPField
from src.env.utils.types import GridConfig, GridPosition


# %% [markdown]
# ## Shared Helpers

# %%
def _build_2d_field(
    d_max: int,
    noise_std: float,
    sigma: float = 1.0,
    lengthscale: float = 3.0,
    nu: float = 2.5,
    num_features: int = 800,
    grid_shape: tuple[int, int] = (15, 15),
    seed: int = 0,
) -> RFFGPField:
    """Create a 2D field and reset once."""
    n_x, n_y = grid_shape
    config = GridConfig.create(n_x=n_x, n_y=n_y)
    field = RFFGPField(
        config=config,
        d_max=d_max,
        sigma=sigma,
        lengthscale=lengthscale,
        nu=nu,
        num_features=num_features,
        noise_std=noise_std,
    )
    field.reset(jax.random.PRNGKey(seed))
    return field


def _build_3d_field(
    d_max: int,
    noise_std: float,
    sigma: float = 1.0,
    lengthscale: float = 3.0,
    nu: float = 2.5,
    num_features: int = 800,
    grid_shape: tuple[int, int, int] = (11, 11, 5),
    seed: int = 0,
) -> RFFGPField:
    """Create a 3D field and reset once."""
    n_x, n_y, n_z = grid_shape
    config = GridConfig.create(n_x=n_x, n_y=n_y, n_z=n_z)
    field = RFFGPField(
        config=config,
        d_max=d_max,
        sigma=sigma,
        lengthscale=lengthscale,
        nu=nu,
        num_features=num_features,
        noise_std=noise_std,
    )
    field.reset(jax.random.PRNGKey(seed))
    return field


def _set_local_means_2d(field: RFFGPField, position: GridPosition, u_mean: float) -> None:
    """Override local 2D mean for controlled PMF experiments."""
    i_idx, j_idx = position.i - 1, position.j - 1
    field._precomputed_u = field._precomputed_u.at[i_idx, j_idx].set(u_mean)


def _set_local_means_3d(field: RFFGPField, position: GridPosition, u_mean: float, v_mean: float) -> None:
    """Override local 3D means for controlled PMF experiments."""
    i_idx, j_idx, k_idx = position.i - 1, position.j - 1, position.k - 1
    field._precomputed_u = field._precomputed_u.at[i_idx, j_idx, k_idx].set(u_mean)
    field._precomputed_v = field._precomputed_v.at[i_idx, j_idx, k_idx].set(v_mean)


def _sample_empirical_1d(
    field: RFFGPField,
    position: GridPosition,
    n_samples: int,
    seed: int = 123,
) -> np.ndarray:
    """Sample empirical 1D displacement distribution from simulator."""
    d_max = field.d_max
    counts = np.zeros(2 * d_max + 1, dtype=np.int64)
    key = jax.random.PRNGKey(seed)
    for _ in range(n_samples):
        key, sample_key = jax.random.split(key)
        obs = field.sample_displacement(position, sample_key)
        counts[obs.u_int + d_max] += 1
    return counts / float(n_samples)


def _sample_empirical_2d_joint(
    field: RFFGPField,
    position: GridPosition,
    n_samples: int,
    seed: int = 123,
) -> np.ndarray:
    """Sample empirical joint (u, v) displacement distribution from simulator."""
    d_max = field.d_max
    counts = np.zeros((2 * d_max + 1, 2 * d_max + 1), dtype=np.int64)
    key = jax.random.PRNGKey(seed)
    for _ in range(n_samples):
        key, sample_key = jax.random.split(key)
        obs = field.sample_displacement(position, sample_key)
        counts[obs.u_int + d_max, obs.v_int + d_max] += 1
    return counts / float(n_samples)


def _summarize_fit(predicted: np.ndarray, empirical: np.ndarray) -> dict[str, float]:
    """Return compact fit metrics between empirical and predicted distributions."""
    abs_diff = np.abs(empirical - predicted)
    return {
        "tv_distance": float(0.5 * np.sum(abs_diff)),
        "max_abs_error": float(np.max(abs_diff)),
        "l2_error": float(np.sqrt(np.sum((empirical - predicted) ** 2))),
    }


# %% [markdown]
# ## 1-4) 2D Controlled PMF Experiments
#
# **Mathematical Setup:**
# Let $U_{\text{obs}} = \mu + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$.
# The observed displacement is $U = \text{round}(\text{clip}(U_{\text{obs}}, -d_{\max}, d_{\max}))$.
#
# **Analytical PMF** $P(U = k)$ for $k \in \{-d_{\max}, \ldots, d_{\max}\}$:
# - Left boundary: $P(U = -d_{\max}) = \Phi\left(\frac{-d_{\max} + 0.5 - \mu}{\sigma}\right)$
# - Interior: $P(U = k) = \Phi\left(\frac{k + 0.5 - \mu}{\sigma}\right) - \Phi\left(\frac{k - 0.5 - \mu}{\sigma}\right)$
# - Right boundary: $P(U = d_{\max}) = 1 - \Phi\left(\frac{d_{\max} - 0.5 - \mu}{\sigma}\right)$
#
# **Experiment Construction:**
# We explicitly override `field._precomputed_u` at a fixed grid position to control $\mu$,
# then compare Monte Carlo samples from `sample_displacement` against `get_displacement_pmf`.
#
# **Test Conditions (with $d_{\max}=4$):**
# 1. **Interior** ($\mu=0.2$): Mean near center; PMF should be near-Gaussian, symmetric-ish.
# 2. **Near +boundary** ($\mu=d_{\max}-0.25=3.75$): Mean close to upper clip; expect visible
#    mass accumulation at $+d_{\max}$ bin from right-tail truncation.
# 3. **Far +boundary** ($\mu=2.5 \cdot d_{\max}=10$): Mean far beyond clip range; nearly all
#    mass should collapse into the $+d_{\max}$ bin (one-hot-like).
# 4. **Zero noise** ($\sigma=0$): Deterministic case; PMF becomes a one-hot at $\text{round}(\text{clip}(\mu))$.
#
# **Expected Plot Behavior:**
# - Left panel: Empirical histogram (bars) should closely overlay analytical PMF (line).
# - Right panel: Pointwise error should be small (< 0.01) with TV distance near zero.
# - Boundary cases show asymmetric PMFs with mass piling at $\pm d_{\max}$.

# %%
def validate_pmf_2d_suite(
    n_samples: int = 120_000,
    d_max: int = 4,
    grid_shape: tuple[int, int] = (17, 17),
) -> None:
    """Run multiple controlled 2D experiments and compare empirical vs analytical PMF."""
    position = GridPosition((grid_shape[0] + 1) // 2, (grid_shape[1] + 1) // 2, None)
    experiments = [
        {
            "name": "interior",
            "u_mean": 0.2,
            "noise_std": 0.9,
            "seed": 1,
            "n_samples": n_samples,
            "note": "Mean away from boundaries; should look near-Gaussian around center bins.",
        },
        {
            "name": "near_+dmax_boundary",
            "u_mean": d_max - 0.25,
            "noise_std": 0.9,
            "seed": 2,
            "n_samples": n_samples,
            "note": "Tests accumulation of right-tail mass at +d_max.",
        },
        {
            "name": "far_+dmax_tail",
            "u_mean": 2.5 * d_max,
            "noise_std": 1.0,
            "seed": 3,
            "n_samples": n_samples,
            "note": "Most mass should collapse at +d_max due to clipping.",
        },
        {
            "name": "zero_noise_deterministic",
            "u_mean": 1.3,
            "noise_std": 0.0,
            "seed": 4,
            "n_samples": max(10_000, n_samples // 6),
            "note": "Should become a one-hot PMF at round(clip(mu)).",
        },
    ]

    d_values = np.arange(-d_max, d_max + 1)
    fig, axes = plt.subplots(len(experiments), 2, figsize=(14, 4.2 * len(experiments)))
    if len(experiments) == 1:
        axes = np.asarray([axes])

    for row, exp in enumerate(experiments):
        field = _build_2d_field(
            d_max=d_max,
            noise_std=exp["noise_std"],
            grid_shape=grid_shape,
            seed=exp["seed"],
        )
        _set_local_means_2d(field, position, exp["u_mean"])

        predicted = field.get_displacement_pmf(position)
        empirical = _sample_empirical_1d(
            field,
            position=position,
            n_samples=exp["n_samples"],
            seed=10_000 + exp["seed"],
        )
        metrics = _summarize_fit(predicted, empirical)

        # Left: predicted vs empirical bars
        ax0 = axes[row, 0]
        width = 0.42
        ax0.bar(d_values - width / 2, predicted, width=width, alpha=0.8, label="Analytical PMF")
        ax0.bar(d_values + width / 2, empirical, width=width, alpha=0.65, label="Empirical freq")
        ax0.set_title(
            f"{exp['name']} | TV={metrics['tv_distance']:.4f}, "
            f"max|err|={metrics['max_abs_error']:.4f}"
        )
        ax0.set_xlabel("discrete displacement k")
        ax0.set_ylabel("probability")
        ax0.grid(alpha=0.3)
        ax0.legend()

        # Right: signed residual
        ax1 = axes[row, 1]
        residual = empirical - predicted
        ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
        ax1.bar(d_values, residual, alpha=0.75, color="tab:purple")
        ax1.set_title(
            f"Empirical - Analytical residuals ({exp['name']})\n"
            f"mu={exp['u_mean']:.2f}, noise_std={exp['noise_std']:.2f}, n={exp['n_samples']}"
        )
        ax1.set_xlabel("discrete displacement k")
        ax1.set_ylabel("residual")
        ax1.grid(alpha=0.3)
        ax1.text(
            0.02,
            0.95,
            exp["note"],
            transform=ax1.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        print(
            f"[2D:{exp['name']}] "
            f"TV={metrics['tv_distance']:.5f}, "
            f"max_abs_error={metrics['max_abs_error']:.5f}, "
            f"L2={metrics['l2_error']:.5f}"
        )

    subtitle = format_params(
        {
            "d_max": d_max,
            "grid": f"{grid_shape[0]}x{grid_shape[1]}",
            "position": f"({position.i},{position.j})",
        }
    )
    fig.suptitle(
        "Empirical vs Analytical PMF (2D Controlled Experiments)\n"
        f"{subtitle}",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    out = save_figure(fig, "pmf_validation_2d_suite.png", bbox_inches="tight")
    print(f"Saved to: {out}")
    plt.show()


# %%
validate_pmf_2d_suite(n_samples=120_000, d_max=4, grid_shape=(17, 17))


# %% [markdown]
# ## 5) 3D Joint PMF Validation
#
# **Mathematical Setup:**
# In 3D mode, the streamfunction GP yields two displacement components:
# $U_{\text{obs}} = \mu_u + \epsilon_u$ and $V_{\text{obs}} = \mu_v + \epsilon_v$
# where $\epsilon_u, \epsilon_v \sim \mathcal{N}(0, \sigma^2)$ are independent.
#
# **Analytical Joint PMF:**
# Since $U$ and $V$ are conditionally independent given the field means,
# $P(U=i, V=j) = P(U=i) \cdot P(V=j)$ (outer product of marginal PMFs).
#
# **Experiment Construction:**
# We override `field._precomputed_u` and `field._precomputed_v` at a fixed 3D position
# to control $(\mu_u, \mu_v)$, then compare Monte Carlo samples against `get_displacement_pmf`.
#
# **Expected Plot Behavior:**
# - Left panel: Analytical joint PMF heatmap centered near $(\mu_u, \mu_v)$.
# - Middle panel: Empirical joint histogram should closely match analytical.
# - Right panel: Difference heatmap should show small residuals (near zero).

# %%
def validate_pmf_3d_joint(
    n_samples: int = 180_000,
    d_max: int = 3,
    grid_shape: tuple[int, int, int] = (11, 11, 5),
    u_mean: float = 1.4,
    v_mean: float = -1.1,
    noise_std: float = 0.85,
) -> None:
    """Validate analytical joint PMF in 3D by empirical sampling."""
    position = GridPosition((grid_shape[0] + 1) // 2, (grid_shape[1] + 1) // 2, (grid_shape[2] + 1) // 2)
    field = _build_3d_field(
        d_max=d_max,
        noise_std=noise_std,
        grid_shape=grid_shape,
        seed=77,
    )
    _set_local_means_3d(field, position, u_mean=u_mean, v_mean=v_mean)

    predicted = field.get_displacement_pmf(position)
    empirical = _sample_empirical_2d_joint(field, position, n_samples=n_samples, seed=2026)
    diff = empirical - predicted
    metrics = _summarize_fit(predicted.ravel(), empirical.ravel())

    d_values = np.arange(-d_max, d_max + 1)
    vmax_prob = float(max(predicted.max(), empirical.max()))
    vmax_diff = float(np.max(np.abs(diff)))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    im0 = axes[0].imshow(
        predicted.T,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax_prob,
        extent=[-d_max - 0.5, d_max + 0.5, -d_max - 0.5, d_max + 0.5],
    )
    axes[0].set_title("Analytical joint PMF")
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("v")
    axes[0].set_xticks(d_values)
    axes[0].set_yticks(d_values)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(
        empirical.T,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax_prob,
        extent=[-d_max - 0.5, d_max + 0.5, -d_max - 0.5, d_max + 0.5],
    )
    axes[1].set_title(f"Empirical joint freq (n={n_samples})")
    axes[1].set_xlabel("u")
    axes[1].set_ylabel("v")
    axes[1].set_xticks(d_values)
    axes[1].set_yticks(d_values)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(
        diff.T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax_diff,
        vmax=vmax_diff,
        extent=[-d_max - 0.5, d_max + 0.5, -d_max - 0.5, d_max + 0.5],
    )
    axes[2].set_title(
        "Empirical - Analytical\n"
        f"TV={metrics['tv_distance']:.4f}, max|err|={metrics['max_abs_error']:.4f}"
    )
    axes[2].set_xlabel("u")
    axes[2].set_ylabel("v")
    axes[2].set_xticks(d_values)
    axes[2].set_yticks(d_values)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    subtitle = format_params(
        {
            "mu_u": f"{u_mean:.2f}",
            "mu_v": f"{v_mean:.2f}",
            "noise_std": f"{noise_std:.2f}",
            "d_max": d_max,
            "grid": f"{grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]}",
            "position": f"({position.i},{position.j},{position.k})",
        }
    )
    fig.suptitle(f"3D Joint PMF Validation\n{subtitle}", fontsize=13, y=1.02)
    plt.tight_layout()
    out = save_figure(fig, "pmf_validation_3d_joint.png", bbox_inches="tight")
    print(f"Saved to: {out}")
    print(
        f"[3D:joint] TV={metrics['tv_distance']:.5f}, "
        f"max_abs_error={metrics['max_abs_error']:.5f}, L2={metrics['l2_error']:.5f}"
    )
    plt.show()


# %%
validate_pmf_3d_joint(
    n_samples=180_000,
    d_max=3,
    grid_shape=(11, 11, 5),
    u_mean=1.4,
    v_mean=-1.1,
    noise_std=0.85,
)


# %% [markdown]
# ## 6) 2D Multi-Position Validation (Natural Field Means)
#
# This experiment validates PMF consistency across **heterogeneous local means** arising
# naturally from a single GP field realization. Unlike previous experiments that override
# $\mu_{\mathbf{r}}$, here we use the actual GP sample values.
#
# **Setup.** Given a 2D RFF-GP field with fixed realization, we select a grid of positions
# $\{\mathbf{r}_1, \ldots, \mathbf{r}_K\}$ spanning the domain. At each position $\mathbf{r}_k$,
# the local mean $\mu_k = U(\mathbf{r}_k)$ is determined by the GP sample.
#
# **Analytical PMF.** For each position, the clipped PMF follows the standard form:
# $$p_{\mathbf{r}_k}(d) = \begin{cases}
# \Phi\left(\frac{-d_{\max} + 0.5 - \mu_k}{\sigma_{\text{noise}}}\right), & d = -d_{\max} \\
# \Phi\left(\frac{d + 0.5 - \mu_k}{\sigma_{\text{noise}}}\right) - \Phi\left(\frac{d - 0.5 - \mu_k}{\sigma_{\text{noise}}}\right), & -d_{\max} < d < d_{\max} \\
# 1 - \Phi\left(\frac{d_{\max} - 0.5 - \mu_k}{\sigma_{\text{noise}}}\right), & d = d_{\max}
# \end{cases}$$
#
# **Validation.** We draw $N$ Monte Carlo samples at each position and compute TV distance,
# max absolute error, and L2 error. This verifies that `get_displacement_pmf` correctly
# adapts to spatially varying means across the field.
#
# **Visualizations (4 subplots).**
#
# 1. **TV Distance Heatmap (top-left):** Spatial map of TV distance $d_{\text{TV}}(p_k, \hat{p}_k)$
#    over the $(i, j)$ grid. Color intensity indicates PMF estimation error at each position.
#    Uniform low values indicate consistent accuracy across the domain.
#
# 2. **Error Distribution Histogram (top-right):** Overlaid histograms of TV distance and
#    max absolute error across all $K$ positions. Shows the empirical distribution of
#    estimation errors. Expect both distributions to be concentrated near zero with
#    small spread, indicating uniformly good agreement across positions.
#
# 3. **Error vs. Local |μ| (bottom-left):** Scatter plot of TV distance and max absolute
#    error against $|\mu_k|$ for all positions. Tests whether PMF accuracy degrades for
#    extreme means (near $\pm d_{\max}$) where clipping is more prevalent. Expect no
#    systematic trend if the analytical PMF correctly handles all mean values.
#
# 4. **Error vs. Clipping Mass (bottom-right):** Scatter plot of TV distance and max
#    absolute error against boundary mass $p_k(-d_{\max}) + p_k(d_{\max})$. High clipping
#    mass indicates the distribution is truncated significantly. Tests whether positions
#    with more clipping exhibit higher Monte Carlo variance or systematic bias.
#
# **Expected results.** For sufficient $N$, TV distances should be uniformly small
# ($\lesssim 0.02$) across all positions regardless of local mean or clipping mass,
# confirming that the analytical PMF correctly handles the full range of $\mu_k$ values
# produced by the GP. No systematic correlation should appear in the scatter plots.

# %%
def validate_pmf_2d_multi_position(
    n_samples_per_position: int = 35_000,
    d_max: int = 4,
    grid_shape: tuple[int, int] = (21, 21),
    noise_std: float = 0.85,
    n_positions_x: int = 6,
    n_positions_y: int = 6,
    seed: int = 314,
) -> None:
    """Validate PMF agreement across multiple natural grid positions in one realization."""
    field = _build_2d_field(
        d_max=d_max,
        noise_std=noise_std,
        grid_shape=grid_shape,
        seed=seed,
    )

    n_x, n_y = grid_shape
    i_candidates = np.unique(np.round(np.linspace(1, n_x, n_positions_x)).astype(int))
    j_candidates = np.unique(np.round(np.linspace(1, n_y, n_positions_y)).astype(int))
    positions = [GridPosition(int(i), int(j), None) for i in i_candidates for j in j_candidates]

    tv_list: list[float] = []
    max_err_list: list[float] = []
    l2_list: list[float] = []
    mu_list: list[float] = []
    clip_mass_list: list[float] = []

    tv_map = np.full((n_x, n_y), np.nan, dtype=np.float32)
    max_err_map = np.full((n_x, n_y), np.nan, dtype=np.float32)

    print(f"Evaluating {len(positions)} positions, {n_samples_per_position} samples each...")
    for idx, pos in enumerate(positions):
        predicted = field.get_displacement_pmf(pos)
        empirical = _sample_empirical_1d(
            field,
            position=pos,
            n_samples=n_samples_per_position,
            seed=20_000 + idx,
        )
        metrics = _summarize_fit(predicted, empirical)
        mu = field.get_mean_displacement(pos)[0]
        clipping_mass = float(predicted[0] + predicted[-1])

        tv_list.append(metrics["tv_distance"])
        max_err_list.append(metrics["max_abs_error"])
        l2_list.append(metrics["l2_error"])
        mu_list.append(float(mu))
        clip_mass_list.append(clipping_mass)

        tv_map[pos.i - 1, pos.j - 1] = metrics["tv_distance"]
        max_err_map[pos.i - 1, pos.j - 1] = metrics["max_abs_error"]

    tv_arr = np.asarray(tv_list)
    max_err_arr = np.asarray(max_err_list)
    mu_arr = np.asarray(mu_list)
    clip_mass_arr = np.asarray(clip_mass_list)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # TV distance map over selected positions
    im0 = axes[0, 0].imshow(tv_map.T, origin="lower", cmap="magma")
    axes[0, 0].set_title("TV distance map (selected positions)")
    axes[0, 0].set_xlabel("i")
    axes[0, 0].set_ylabel("j")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # Histogram of errors
    axes[0, 1].hist(tv_arr, bins=16, alpha=0.7, edgecolor="black", label="TV distance")
    axes[0, 1].hist(max_err_arr, bins=16, alpha=0.55, edgecolor="black", label="max abs error")
    axes[0, 1].set_title("Error distribution across positions")
    axes[0, 1].set_xlabel("error")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Error vs |mu|
    axes[1, 0].scatter(np.abs(mu_arr), tv_arr, alpha=0.8, s=35, label="TV distance")
    axes[1, 0].scatter(np.abs(mu_arr), max_err_arr, alpha=0.8, s=35, label="max abs error")
    axes[1, 0].set_title("Error vs local |mu|")
    axes[1, 0].set_xlabel("|local mean displacement|")
    axes[1, 0].set_ylabel("error")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Error vs predicted clipping mass
    axes[1, 1].scatter(clip_mass_arr, tv_arr, alpha=0.8, s=35, label="TV distance")
    axes[1, 1].scatter(clip_mass_arr, max_err_arr, alpha=0.8, s=35, label="max abs error")
    axes[1, 1].set_title("Error vs predicted clipping mass")
    axes[1, 1].set_xlabel("predicted clipping mass P(k=-dmax)+P(k=+dmax)")
    axes[1, 1].set_ylabel("error")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    subtitle = format_params(
        {
            "grid": f"{n_x}x{n_y}",
            "d_max": d_max,
            "noise_std": f"{noise_std:.2f}",
            "positions": len(positions),
            "samples_per_position": n_samples_per_position,
            "seed": seed,
        }
    )
    fig.suptitle(
        "Multi-Position PMF Validation (2D, Natural Means)\n"
        f"{subtitle}",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    out = save_figure(fig, "pmf_validation_2d_multi_position.png", bbox_inches="tight")
    print(f"Saved to: {out}")
    print(
        f"[2D:multi-position] "
        f"TV mean={tv_arr.mean():.5f}, TV p95={np.quantile(tv_arr, 0.95):.5f}, TV max={tv_arr.max():.5f} | "
        f"max_abs mean={max_err_arr.mean():.5f}, max_abs max={max_err_arr.max():.5f}"
    )

    # Print top-5 worst positions by TV distance for targeted inspection.
    rank_idx = np.argsort(-tv_arr)
    print("Top-5 positions by TV distance:")
    for r in rank_idx[:5]:
        pos = positions[int(r)]
        print(
            f"  pos=({pos.i},{pos.j}) "
            f"mu={mu_arr[r]:.3f}, clip_mass={clip_mass_arr[r]:.3f}, "
            f"TV={tv_arr[r]:.5f}, max_abs={max_err_arr[r]:.5f}"
        )

    plt.show()


# %%
validate_pmf_2d_multi_position(
    n_samples_per_position=3_000,
    d_max=4,
    grid_shape=(21, 21),
    noise_std=0.85,
    n_positions_x=6,
    n_positions_y=6,
    seed=314,
)

