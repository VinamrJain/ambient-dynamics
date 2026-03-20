# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
"""Passive Field Estimation with Uniform Random Sampling."""

import sys
from pathlib import Path


def add_project_root_to_path() -> Path:
    project_root = Path.cwd()
    for _ in range(10):
        if (project_root / "src").is_dir() and (project_root / "pixi.toml").exists():
            break
        parent = project_root.parent
        if parent == project_root:
            raise FileNotFoundError(
                "Project root (directory with src/ and pixi.toml) not found."
            )
        project_root = parent
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root


add_project_root_to_path()

import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import gpjax as gpx
import optax as ox

from src.env.field.rff_gp_field import RFFGPField
from src.env.utils.types import GridConfig
from experiments.field_estimation.utils.evaluation import compute_field_rmse

# %%
# 1. Setup True Field
# ===================
grid_size = 50
sigma_true = 5
lengthscale_true = 7.5
nu_true = 2.5
noise_std_true = 0.0

seed = 42
key = jr.PRNGKey(seed)
key, field_key = jr.split(key)

grid_config = GridConfig.create(n_x=grid_size, n_y=grid_size)
field = RFFGPField(
    config=grid_config,
    d_max=4 * sigma_true,  # change this according to the field parameters
    sigma=sigma_true,
    lengthscale=lengthscale_true,
    nu=nu_true,
    noise_std=noise_std_true,
)
field.reset(field_key)

true_u = field._precomputed_u.squeeze()  # shape (grid_size, grid_size)

# %%
# 2. Collect Uniform Random Observations
# ======================================
M = 500
key, sample_key = jr.split(key)

# Sample M coordinates uniformly
i_idx = jr.randint(sample_key, (M,), 1, grid_size + 1)
sample_key, _ = jr.split(sample_key)
j_idx = jr.randint(sample_key, (M,), 1, grid_size + 1)

X_train = jnp.column_stack([i_idx, j_idx]).astype(jnp.float64)

# Get observations with noise
y_train = jnp.zeros((M, 1))
key, noise_key = jr.split(key)
noise_keys = jr.split(noise_key, M)

from src.env.utils.types import GridPosition

for idx in range(M):
    pos = GridPosition(int(X_train[idx, 0]), int(X_train[idx, 1]))
    disp = field.sample_displacement(pos, noise_keys[idx])
    y_train = y_train.at[idx, 0].set(disp.u)

dataset = gpx.Dataset(X=X_train, y=y_train)

# Setup Test Grid
x_coords = jnp.arange(1, grid_size + 1)
y_coords = jnp.arange(1, grid_size + 1)
Xm, Ym = jnp.meshgrid(x_coords, y_coords, indexing="ij")
X_test = jnp.column_stack([Xm.ravel(), Ym.ravel()]).astype(jnp.float64)

# %%
# 3. Experiment 1: GP with True Parameters
# ========================================
kernel_true = gpx.kernels.Matern52(variance=sigma_true**2, lengthscale=lengthscale_true)
prior_true = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel_true)
likelihood_true = gpx.likelihoods.Gaussian(
    num_datapoints=dataset.n, obs_stddev=jnp.array([noise_std_true])
)
posterior_true = prior_true * likelihood_true

latent_dist_1 = posterior_true.predict(X_test, train_data=dataset)
mu_1 = latent_dist_1.mean.reshape(grid_size, grid_size)
var_1 = latent_dist_1.variance.reshape(grid_size, grid_size)

rmse_1 = compute_field_rmse(true_u, mu_1)
print(f"Exp 1 (True Params) RMSE: {rmse_1:.4f}")

# %%
# 4. Experiment 2: GP with True Nu, Optimized Hyperparameters (Adam)
# ===========================================================
kernel_opt1 = gpx.kernels.Matern52(variance=1.0, lengthscale=1.0)
prior_opt1 = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel_opt1)
likelihood_opt1 = gpx.likelihoods.Gaussian(
    num_datapoints=dataset.n, obs_stddev=jnp.array([0.5])
)
posterior_opt1 = prior_opt1 * likelihood_opt1

nmll = lambda p, d: -gpx.objectives.conjugate_mll(p, d)
print("\nOptimizing Matern52 (Exp 2) with Adam...")
optim = ox.adam(learning_rate=0.05)
opt_posterior_opt1, history_1 = gpx.fit(
    model=posterior_opt1, objective=nmll, train_data=dataset, optim=optim, num_iters=500
)

v_opt1 = float(opt_posterior_opt1.prior.kernel.variance[...])
l_opt1 = float(opt_posterior_opt1.prior.kernel.lengthscale[...])
n_opt1 = float(opt_posterior_opt1.likelihood.obs_stddev[...][0])
plt.plot(history_1)
plt.title("Adam Optimization History")
plt.xlabel("Iteration")
plt.ylabel("Negative Marginal Log Likelihood")
plt.show()

print(
    f"True Params: Sigma^2={sigma_true**2:.4f}, L={lengthscale_true:.4f}, Noise={noise_std_true:.4f}"
)
print(f"Opt Params : Sigma^2={v_opt1:.4f}, L={l_opt1:.4f}, Noise={n_opt1:.4f}")

latent_dist_2 = opt_posterior_opt1.predict(X_test, train_data=dataset)
mu_2 = latent_dist_2.mean.reshape(grid_size, grid_size)
var_2 = latent_dist_2.variance.reshape(grid_size, grid_size)

rmse_2 = compute_field_rmse(true_u, mu_2)
print(f"Exp 2 (Opt Params, True Nu) RMSE: {rmse_2:.4f}")

# %%
# 5. Experiment 3: GP with Different Nu (Matern32), Optimized
# ===========================================================
kernel_opt2 = gpx.kernels.Matern32(variance=1.0, lengthscale=1.0)
prior_opt2 = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel_opt2)
likelihood_opt2 = gpx.likelihoods.Gaussian(
    num_datapoints=dataset.n, obs_stddev=jnp.array([0.5])
)
posterior_opt2 = prior_opt2 * likelihood_opt2

print("\nOptimizing Matern32 (Exp 3) with Scipy L-BFGS-B...")
opt_posterior_opt2, history_2 = gpx.fit_scipy(
    model=posterior_opt2, objective=nmll, train_data=dataset
)

v_opt2 = float(opt_posterior_opt2.prior.kernel.variance[...])
l_opt2 = float(opt_posterior_opt2.prior.kernel.lengthscale[...])
n_opt2 = float(opt_posterior_opt2.likelihood.obs_stddev[...][0])

print(
    f"Opt Params (Matern32): Sigma^2={v_opt2:.4f}, L={l_opt2:.4f}, Noise={n_opt2:.4f}"
)

latent_dist_3 = opt_posterior_opt2.predict(X_test, train_data=dataset)
mu_3 = latent_dist_3.mean.reshape(grid_size, grid_size)
var_3 = latent_dist_3.variance.reshape(grid_size, grid_size)

rmse_3 = compute_field_rmse(true_u, mu_3)
print(f"Exp 3 (Opt Params, Diff Nu) RMSE: {rmse_3:.4f}")

# %%
# 6. Plotting
# ===========
# Figure 1: True field and observations (1x2)
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes1[0].imshow(
    true_u.T,
    origin="lower",
    cmap="viridis",
    extent=[0.5, grid_size + 0.5, 0.5, grid_size + 0.5],
)
axes1[0].set_title("True Field")
plt.colorbar(im0, ax=axes1[0])

axes1[1].scatter(
    X_train[:, 0], X_train[:, 1], c=y_train[:, 0], cmap="viridis", edgecolors="k", s=50
)
axes1[1].set_xlim(0.5, grid_size + 0.5)
axes1[1].set_ylim(0.5, grid_size + 0.5)
axes1[1].set_title(f"Sample Heatmap (N={M})")
axes1[1].set_aspect("equal")
plt.tight_layout()
plt.show()

# Figure 2: GP Predictions, Errors, and Variances (3x3)
fig2, axes2 = plt.subplots(3, 3, figsize=(18, 15))

# Exp 1 (True Params)
im1_0 = axes2[0, 0].imshow(
    mu_1.T,
    origin="lower",
    cmap="viridis",
    extent=[0.5, grid_size + 0.5, 0.5, grid_size + 0.5],
)
axes2[0, 0].set_title("Exp1 Predicted Mean")
plt.colorbar(im1_0, ax=axes2[0, 0])

err_1 = jnp.abs(true_u - mu_1)
im1_1 = axes2[0, 1].imshow(
    err_1.T,
    origin="lower",
    cmap="Reds",
    extent=[0.5, grid_size + 0.5, 0.5, grid_size + 0.5],
)
axes2[0, 1].set_title(f"Exp1 (True Params) Abs Error\nRMSE: {rmse_1:.4f}")
plt.colorbar(im1_1, ax=axes2[0, 1])

im1_2 = axes2[0, 2].imshow(
    var_1.T,
    origin="lower",
    cmap="plasma",
    extent=[0.5, grid_size + 0.5, 0.5, grid_size + 0.5],
)
axes2[0, 2].set_title("Exp1 Variance")
plt.colorbar(im1_2, ax=axes2[0, 2])

# Exp 2 (Optimized, True Nu)
im2_0 = axes2[1, 0].imshow(
    mu_2.T,
    origin="lower",
    cmap="viridis",
    extent=[0.5, grid_size + 0.5, 0.5, grid_size + 0.5],
)
axes2[1, 0].set_title("Exp2 Predicted Mean")
plt.colorbar(im2_0, ax=axes2[1, 0])

err_2 = jnp.abs(true_u - mu_2)
im2_1 = axes2[1, 1].imshow(
    err_2.T,
    origin="lower",
    cmap="Reds",
    extent=[0.5, grid_size + 0.5, 0.5, grid_size + 0.5],
)
axes2[1, 1].set_title(f"Exp2 (Opt Params, True Nu) Abs Error\nRMSE: {rmse_2:.4f}")
plt.colorbar(im2_1, ax=axes2[1, 1])

im2_2 = axes2[1, 2].imshow(
    var_2.T,
    origin="lower",
    cmap="plasma",
    extent=[0.5, grid_size + 0.5, 0.5, grid_size + 0.5],
)
axes2[1, 2].set_title("Exp2 Variance")
plt.colorbar(im2_2, ax=axes2[1, 2])

# Exp 3 (Optimized, Diff Nu)
im3_0 = axes2[2, 0].imshow(
    mu_3.T,
    origin="lower",
    cmap="viridis",
    extent=[0.5, grid_size + 0.5, 0.5, grid_size + 0.5],
)
axes2[2, 0].set_title("Exp3 Predicted Mean")
plt.colorbar(im3_0, ax=axes2[2, 0])

err_3 = jnp.abs(true_u - mu_3)
im3_1 = axes2[2, 1].imshow(
    err_3.T,
    origin="lower",
    cmap="Reds",
    extent=[0.5, grid_size + 0.5, 0.5, grid_size + 0.5],
)
axes2[2, 1].set_title(f"Exp3 (Opt Params, Diff Nu) Abs Error\nRMSE: {rmse_3:.4f}")
plt.colorbar(im3_1, ax=axes2[2, 1])

im3_2 = axes2[2, 2].imshow(
    var_3.T,
    origin="lower",
    cmap="plasma",
    extent=[0.5, grid_size + 0.5, 0.5, grid_size + 0.5],
)
axes2[2, 2].set_title("Exp3 Variance")
plt.colorbar(im3_2, ax=axes2[2, 2])

plt.tight_layout()
plt.show()

# %%
# 7. Experiment 4: RMSE Scaling with Sample Size
# ==============================================
import time

sample_sizes = [10, 20, 50, 100, 250, 500, 1000, 2500]
sample_sizes.sort()
N_max = max(sample_sizes)

print(f"\n--- Running Experiment 4: RMSE Scaling (Max Samples = {N_max}) ---")

# Prepare N_max samples
key, sample_key = jr.split(key)
i_idx_scale = jr.randint(sample_key, (N_max,), 1, grid_size + 1)
sample_key, _ = jr.split(sample_key)
j_idx_scale = jr.randint(sample_key, (N_max,), 1, grid_size + 1)

X_train_scale = jnp.column_stack([i_idx_scale, j_idx_scale]).astype(jnp.float64)

y_train_scale = jnp.zeros((N_max, 1))
key, noise_key = jr.split(key)
noise_keys_scale = jr.split(noise_key, N_max)

from src.env.utils.types import GridPosition

for idx in range(N_max):
    pos = GridPosition(int(X_train_scale[idx, 0]), int(X_train_scale[idx, 1]))
    disp = field.sample_displacement(pos, noise_keys_scale[idx])
    y_train_scale = y_train_scale.at[idx, 0].set(disp.u)

rmse_results_true = []
rmse_results_opt = []


def evaluate_subset_gp(X_sub, y_sub, optimize=False):
    """Compute RMSE for a subset, optionally optimizing parameters (Matern52)."""
    sub_dataset = gpx.Dataset(X=X_sub, y=y_sub)

    if optimize:
        kernel = gpx.kernels.Matern52(variance=1.0, lengthscale=1.0)
        prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=sub_dataset.n, obs_stddev=jnp.array([0.5])
        )
        posterior = prior * likelihood

        nmll = lambda p, d: -gpx.objectives.conjugate_mll(p, d)
        optim_scale = ox.adam(learning_rate=0.05)
        # Suppress verbose output for the loop
        opt_posterior, _ = gpx.fit(
            model=posterior,
            objective=nmll,
            train_data=sub_dataset,
            optim=optim_scale,
            num_iters=300,
        )

        v_opt = float(opt_posterior.prior.kernel.variance[...])
        l_opt = float(opt_posterior.prior.kernel.lengthscale[...])
        n_opt = float(opt_posterior.likelihood.obs_stddev[...][0])

        latent_dist = opt_posterior.predict(X_test, train_data=sub_dataset)
        mu_pred = latent_dist.mean.reshape(grid_size, grid_size)
        rmse_val = jnp.sqrt(jnp.mean((true_u - mu_pred) ** 2))

        return float(rmse_val), v_opt, l_opt, n_opt

    else:
        # Use True Parameters
        kernel_true = gpx.kernels.Matern52(
            variance=sigma_true**2, lengthscale=lengthscale_true
        )
        prior = gpx.gps.Prior(
            mean_function=gpx.mean_functions.Zero(), kernel=kernel_true
        )
        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=sub_dataset.n, obs_stddev=jnp.array([noise_std_true])
        )
        posterior = prior * likelihood
        latent_dist = posterior.predict(X_test, train_data=sub_dataset)
        mu_pred = latent_dist.mean.reshape(grid_size, grid_size)
        rmse_val = jnp.sqrt(jnp.mean((true_u - mu_pred) ** 2))

        return float(rmse_val), sigma_true**2, lengthscale_true, noise_std_true


for n in sample_sizes:
    t0 = time.time()
    X_sub = X_train_scale[:n]
    y_sub = y_train_scale[:n]

    # True Parameters
    rmse_true, v_t, l_t, n_t = evaluate_subset_gp(X_sub, y_sub, optimize=False)
    rmse_results_true.append(rmse_true)

    # Optimized Parameters
    # (Comment out the following line and related opt lines if it takes too long!)
    rmse_opt, v_o, l_o, n_o = evaluate_subset_gp(X_sub, y_sub, optimize=True)
    rmse_results_opt.append(rmse_opt)

    print(f"N = {n:<4} | Time = {time.time() - t0:.2f}s")
    print(
        f"  [True] RMSE = {rmse_true:.4f} | Sigma^2={v_t:.2f}, L={l_t:.2f}, Noise={n_t:.2f}"
    )
    print(
        f"  [Opt ] RMSE = {rmse_opt:.4f} | Sigma^2={v_o:.2f}, L={l_o:.2f}, Noise={n_o:.2f}"
    )

# Plot RMSE Scaling
plt.figure(figsize=(8, 5))
plt.plot(
    sample_sizes,
    rmse_results_true,
    marker="o",
    linestyle="-",
    linewidth=2,
    color="b",
    label="True Params",
)
plt.plot(
    sample_sizes,
    rmse_results_opt,
    marker="s",
    linestyle="--",
    linewidth=2,
    color="r",
    label="Optimized Params",
)
plt.xlabel("Number of Samples (N)")
plt.ylabel("RMSE")
plt.title("RMSE Scaling vs. Sample Size")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# %%
# 8. Multi-field RMSE (mean / std over independent field seeds)
# ==============================================================

n_fields = 10
base_seed_multi = (
    1000  # base seed for field_idx -> jr.PRNGKey(base_seed_multi + field_idx)
)
M_multi = M  # number of uniform observations per field (edit freely)
optimize_multi = True  # False: use true kernel hyperparameters (no Adam)
adam_lr_multi = 0.05
adam_iters_multi = 300

x_coords_m = jnp.arange(1, grid_size + 1)
y_coords_m = jnp.arange(1, grid_size + 1)
Xm_m, Ym_m = jnp.meshgrid(x_coords_m, y_coords_m, indexing="ij")
X_test_multi = jnp.column_stack([Xm_m.ravel(), Ym_m.ravel()]).astype(jnp.float64)


def _extract_scalar(x):
    # Accept scalar, 0d array, or size-1 array, else raise
    arr = jnp.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    elif arr.shape == (1,):
        return float(arr[0])
    else:
        raise ValueError(
            f"Expected scalar or shape-(1,) array for hyperparameter, got shape {arr.shape}"
        )


def rmse_gp_on_field(
    true_u_field: jnp.ndarray,
    X_tr: jnp.ndarray,
    y_tr: jnp.ndarray,
    *,
    optimize: bool,
):
    """Fit GP on (X_tr, y_tr), predict full grid, RMSE vs true_u_field.
    Returns (rmse, variance, lengthscale, noise_std) if optimize, else only rmse."""
    sub_ds = gpx.Dataset(X=X_tr, y=y_tr)
    if optimize:
        kernel = gpx.kernels.Matern52(variance=1.0, lengthscale=1.0)
        prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=sub_ds.n, obs_stddev=jnp.array([0.5])
        )
        posterior = prior * likelihood

        def nmll_m(p, d):
            return -gpx.objectives.conjugate_mll(p, d)

        opt_posterior, state = gpx.fit(
            model=posterior,
            objective=nmll_m,
            train_data=sub_ds,
            optim=ox.adam(learning_rate=adam_lr_multi),
            num_iters=adam_iters_multi,
        )
        latent_dist = opt_posterior.predict(X_test_multi, train_data=sub_ds)
        # Extract learned kernel and noise hyperparameters as safe Python floats
        learned_kernel = opt_posterior.prior.kernel
        learned_variance = (
            learned_kernel.variance if hasattr(learned_kernel, "variance") else None
        )
        learned_lengthscale = (
            learned_kernel.lengthscale
            if hasattr(learned_kernel, "lengthscale")
            else None
        )
        learned_noise = (
            opt_posterior.likelihood.obs_stddev
            if hasattr(opt_posterior.likelihood, "obs_stddev")
            else None
        )
        mu_pred = latent_dist.mean.reshape(grid_size, grid_size)
        rmse = compute_field_rmse(true_u_field, mu_pred)
        # Convert possibly shape-(1,) jax arrays to Python floats safely
        learned_variance = (
            _extract_scalar(learned_variance) if learned_variance is not None else None
        )
        learned_lengthscale = (
            _extract_scalar(learned_lengthscale)
            if learned_lengthscale is not None
            else None
        )
        learned_noise = (
            _extract_scalar(learned_noise) if learned_noise is not None else None
        )
        return (
            rmse,
            learned_variance,
            learned_lengthscale,
            learned_noise,
        )
    else:
        kernel_true_m = gpx.kernels.Matern52(
            variance=sigma_true**2, lengthscale=lengthscale_true
        )
        prior = gpx.gps.Prior(
            mean_function=gpx.mean_functions.Zero(), kernel=kernel_true_m
        )
        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=sub_ds.n, obs_stddev=jnp.array([noise_std_true])
        )
        posterior = prior * likelihood
        latent_dist = posterior.predict(X_test_multi, train_data=sub_ds)
        mu_pred = latent_dist.mean.reshape(grid_size, grid_size)
        rmse = compute_field_rmse(true_u_field, mu_pred)
        return rmse, sigma_true**2, lengthscale_true, noise_std_true


print(
    f"\n--- Multi-field RMSE ({n_fields} fields, M={M_multi}, optimize={optimize_multi}) ---"
)

rmses_multi = []
variances_multi = []
lengthscales_multi = []
noises_multi = []

for field_idx in range(n_fields):
    seed_f = base_seed_multi + field_idx
    key_f = jr.PRNGKey(seed_f)
    key_f, field_key_f = jr.split(key_f)

    grid_cfg_m = GridConfig.create(n_x=grid_size, n_y=grid_size)
    field_m = RFFGPField(
        config=grid_cfg_m,
        d_max=4 * sigma_true,
        sigma=sigma_true,
        lengthscale=lengthscale_true,
        nu=nu_true,
        noise_std=noise_std_true,
    )
    field_m.reset(field_key_f)
    true_u_f = field_m._precomputed_u.squeeze()

    key_f, sk_i = jr.split(key_f)
    i_idx_m = jr.randint(sk_i, (M_multi,), 1, grid_size + 1)
    key_f, sk_j = jr.split(key_f)
    j_idx_m = jr.randint(sk_j, (M_multi,), 1, grid_size + 1)
    X_tr_m = jnp.column_stack([i_idx_m, j_idx_m]).astype(jnp.float64)

    y_tr_m = jnp.zeros((M_multi, 1))
    key_f, nk_m = jr.split(key_f)
    noise_keys_m = jr.split(nk_m, M_multi)
    for idx_m in range(M_multi):
        pos_m = GridPosition(int(X_tr_m[idx_m, 0]), int(X_tr_m[idx_m, 1]))
        disp_m = field_m.sample_displacement(pos_m, noise_keys_m[idx_m])
        y_tr_m = y_tr_m.at[idx_m, 0].set(disp_m.u)

    result = rmse_gp_on_field(true_u_f, X_tr_m, y_tr_m, optimize=optimize_multi)
    r_m, v_m, l_m, n_m = result
    rmses_multi.append(r_m)
    variances_multi.append(v_m)
    lengthscales_multi.append(l_m)
    noises_multi.append(n_m)
    print(
        f"  field seed {seed_f}: RMSE = {r_m:.4f} | Sigma^2={v_m:.2f}, L={l_m:.2f}, Noise={n_m:.2f}"
    )

rmse_arr_m = jnp.asarray(rmses_multi)
var_arr_m = jnp.asarray(variances_multi)
len_arr_m = jnp.asarray(lengthscales_multi)
noise_arr_m = jnp.asarray(noises_multi)

print(
    f"\nAggregate: RMSE mean = {float(jnp.mean(rmse_arr_m)):.4f}, "
    f"std = {float(jnp.std(rmse_arr_m)):.4f}"
)

print(
    f"Optimized Hyperparameters (mean ± std over {n_fields} fields):\n"
    f"  Sigma^2: {float(jnp.mean(var_arr_m)):.3f} ± {float(jnp.std(var_arr_m)):.3f}\n"
    f"  Lengthscale: {float(jnp.mean(len_arr_m)):.3f} ± {float(jnp.std(len_arr_m)):.3f}\n"
    f"  Noise: {float(jnp.mean(noise_arr_m)):.4f} ± {float(jnp.std(noise_arr_m)):.4f}"
)
