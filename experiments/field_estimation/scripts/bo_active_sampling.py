# %%
"""Bayesian Optimization Active Sampling with AP-SSP Planning."""

import sys
import os
from pathlib import Path


def add_project_root_to_path() -> Path:
    project_root = Path(__file__).resolve().parents[3]
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
import jax.scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import gpjax as gpx
import optax as ox

from src.env.field.rff_gp_field import RFFGPField
from src.env.actor.grid_actor import GridActor
from src.env.arena.dynamic_sg_arena import DynamicSGArena
from src.env.arena.reward import StepCostReward
from src.env.utils.types import GridConfig, GridPosition
from src.agents.ap_ssp_agent import APSSPAgent, APSSPAgentConfig

# %%
# 1. Setup True Field and Arena
# =============================
grid_size = 40
sigma_true = 5.0
lengthscale_true = 7.5
nu_true = 2.5
noise_std_true = 0.2
seed = 42
boundary_mode = "periodic"

key = jr.PRNGKey(seed)
key, field_key = jr.split(key)

grid_config = GridConfig.create(n_x=grid_size, n_y=grid_size)
field = RFFGPField(
    config=grid_config,
    d_max=10,
    sigma=sigma_true,
    lengthscale=lengthscale_true,
    nu=nu_true,
    noise_std=noise_std_true,
)
field.reset(field_key)
true_u = field._precomputed_u.squeeze()

# Build Actor & Arena
actor = GridActor(noise_std=0.0)
vicinity_radius = 2.0
step_cost = 1.0

initial_pos = GridPosition(int(grid_size // 2), int(grid_size // 2))
reward_fn = StepCostReward(
    target_position=initial_pos,
    vicinity_radius=vicinity_radius,
    step_cost=step_cost,
)

arena = DynamicSGArena(
    field=field,
    actor=actor,
    config=grid_config,
    initial_position=initial_pos,
    target_position=initial_pos,
    vicinity_radius=vicinity_radius,
    boundary_mode=boundary_mode,
    vicinity_metric="euclidean",
    reward_fn=reward_fn,
)

key, reset_key = jr.split(key)
arena.reset(reset_key)

# Plan AP-SSP
print("Planning AP-SSP for all state-goal pairs ...")
agent = APSSPAgent(APSSPAgentConfig(max_iters=500, tol=1e-3))
agent.plan(arena)
print("Planning complete!")

# Setup GP Test Grid
x_coords = jnp.arange(1, grid_size + 1)
y_coords = jnp.arange(1, grid_size + 1)
Xm, Ym = jnp.meshgrid(x_coords, y_coords, indexing="ij")
X_test = jnp.column_stack([Xm.ravel(), Ym.ravel()]).astype(jnp.float64)

var_init = sigma_true**2
lengthscale_init = lengthscale_true


# %%
# 2. General Evaluation and Acquisition Functions
# ===============================================
def evaluate_gp(X_tr, y_tr, optimize=False):
    dataset = gpx.Dataset(X=X_tr, y=y_tr)

    if optimize:
        kernel = gpx.kernels.Matern52(variance=var_init, lengthscale=lengthscale_init)
        prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=dataset.n, obs_stddev=jnp.array([0.5])
        )
        posterior = prior * likelihood

        nmll = lambda p, d: -gpx.objectives.conjugate_mll(p, d)
        optim = ox.adam(learning_rate=0.05)
        opt_posterior, _ = gpx.fit(
            model=posterior,
            objective=nmll,
            train_data=dataset,
            optim=optim,
            num_iters=300,
        )
        latent_dist = opt_posterior.predict(X_test, train_data=dataset)
    else:
        kernel = gpx.kernels.Matern52(variance=var_init, lengthscale=lengthscale_init)
        prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=dataset.n, obs_stddev=jnp.array([noise_std_true])
        )
        posterior = prior * likelihood
        latent_dist = posterior.predict(X_test, train_data=dataset)

    mu = latent_dist.mean.reshape(grid_size, grid_size)
    var = latent_dist.variance.reshape(grid_size, grid_size)
    rmse_val = jnp.sqrt(jnp.mean((true_u - mu) ** 2))

    return mu, var, float(rmse_val)


def ei_acq(mu_flat, var_flat, y_train):
    f_best = jnp.max(y_train)
    sigma = jnp.sqrt(jnp.maximum(var_flat, 1e-9))
    z = (mu_flat - f_best) / sigma
    phi = jax.scipy.stats.norm.pdf(z)
    Phi = jax.scipy.stats.norm.cdf(z)
    ei = (mu_flat - f_best) * Phi + sigma * phi
    return jnp.where(sigma > 1e-6, ei, 0.0)


# %%
# 3. Trajectory Loop Function
# ===========================
def run_trajectory_loops(acq_type, num_loops=5, max_steps_per_loop=100):
    global key
    print(f"\n--- Running Strategy: {acq_type} ---")

    # Reset
    arena.set_position(initial_pos)
    obs = arena.soft_reset()

    X_obs = []
    y_obs = []

    # Initial sample
    key, noise_key = jr.split(key)
    disp = field.sample_displacement(initial_pos, noise_key)
    X_obs.append([initial_pos.i, initial_pos.j])
    y_obs.append([disp.u])

    rmse_history = []

    for h in range(num_loops):
        X_tr = jnp.array(X_obs, dtype=jnp.float64)
        y_tr = jnp.array(y_obs, dtype=jnp.float64)

        mu, var, rmse = evaluate_gp(X_tr, y_tr, optimize=False)
        rmse_history.append(rmse)
        print(f"Loop {h + 1}/{num_loops} | Samples: {len(X_obs)} | RMSE: {rmse:.4f}")

        mu_flat = mu.ravel()
        var_flat = var.ravel()

        if acq_type == "random":
            key, subkey = jr.split(key)
            best_idx = jr.randint(subkey, (), 0, len(X_test))
            best_idx = int(best_idx)
        elif acq_type == "max_variance":
            best_idx = int(jnp.argmax(var_flat))
        elif acq_type == "ei":
            acq = ei_acq(mu_flat, var_flat, y_tr)
            best_idx = int(jnp.argmax(acq))
        elif acq_type == "cost_aware_ei":
            acq = ei_acq(mu_flat, var_flat, y_tr)
            current_pos = arena.position
            costs = agent._cost_table[
                current_pos.i - 1, current_pos.j - 1, :, :
            ].ravel()
            costs = jnp.maximum(costs, 1.0)
            acq_cost_aware = acq / costs
            best_idx = int(jnp.argmax(acq_cost_aware))
        else:
            raise ValueError(f"Unknown acq_type: {acq_type}")

        target_coords = X_test[best_idx]
        target_pos = GridPosition(int(target_coords[0]), int(target_coords[1]))

        print(f"  -> Target: {target_pos} from {arena.position}")

        # Navigate to target
        arena.set_target(target_pos)
        agent.set_target(target_pos)
        obs = arena.soft_reset()

        action = agent.begin_episode(obs)
        for step in range(max_steps_per_loop):
            obs = arena.step(action)
            reward = arena.compute_reward()
            state = arena.get_state()

            # Record observation
            key, noise_key = jr.split(key)
            disp = field.sample_displacement(state.last_position, noise_key)
            X_obs.append([state.last_position.i, state.last_position.j])
            y_obs.append([disp.u])

            if state.target_reached:
                print(f"  -> Reached in {step + 1} steps.")
                break
            if arena.is_terminal():
                print(f"  -> Out of bounds.")
                break

            action = agent.step(reward, obs)

    # Final eval
    X_tr = jnp.array(X_obs, dtype=jnp.float64)
    y_tr = jnp.array(y_obs, dtype=jnp.float64)
    mu, var, rmse = evaluate_gp(X_tr, y_tr, optimize=False)
    rmse_history.append(rmse)

    print(f"Final | Samples: {len(X_obs)} | RMSE: {rmse:.4f}")
    return mu, var, X_tr, y_tr, rmse_history


# %%
# 4. Run Experiments
# ==================
h_loops = 5
max_steps = 100

results = {}
strategies = ["random", "max_variance", "ei", "cost_aware_ei"]
labels = ["Random Target", "Max Variance", "Expected Improvement (EI)", "Cost-Aware EI"]

for strat in strategies:
    results[strat] = run_trajectory_loops(
        strat, num_loops=h_loops, max_steps_per_loop=max_steps
    )

# %%
# 5. Plotting
# ===========
print("\nGenerating plots...")
os.makedirs("plots/bo_active", exist_ok=True)

fig, axes = plt.subplots(5, 3, figsize=(15, 25))

# Row 0: True Field
im0 = axes[0, 0].imshow(
    true_u.T,
    origin="lower",
    cmap="viridis",
    extent=[0.5, grid_size + 0.5, 0.5, grid_size + 0.5],
)
axes[0, 0].set_title("True Field")
plt.colorbar(im0, ax=axes[0, 0])
axes[0, 1].axis("off")
axes[0, 2].axis("off")

for i, strat in enumerate(strategies):
    r = i + 1
    mu, var, X_tr, y_tr, rmse_hist = results[strat]

    # Col 0: Mean + Samples
    im_m = axes[r, 0].imshow(
        mu.T,
        origin="lower",
        cmap="viridis",
        extent=[0.5, grid_size + 0.5, 0.5, grid_size + 0.5],
    )
    axes[r, 0].plot(X_tr[:, 0], X_tr[:, 1], color="white", alpha=0.5, linewidth=1)
    axes[r, 0].scatter(X_tr[:, 0], X_tr[:, 1], c="red", s=10, edgecolors="k")
    axes[r, 0].set_title(f"{labels[i]}: Predicted Mean\n({len(X_tr)} samples)")
    plt.colorbar(im_m, ax=axes[r, 0])

    # Col 1: Error
    err = jnp.abs(true_u - mu)
    im_e = axes[r, 1].imshow(
        err.T,
        origin="lower",
        cmap="Reds",
        extent=[0.5, grid_size + 0.5, 0.5, grid_size + 0.5],
    )
    axes[r, 1].set_title(f"{labels[i]}: Abs Error\nFinal RMSE: {rmse_hist[-1]:.4f}")
    plt.colorbar(im_e, ax=axes[r, 1])

    # Col 2: Variance
    im_v = axes[r, 2].imshow(
        var.T,
        origin="lower",
        cmap="plasma",
        extent=[0.5, grid_size + 0.5, 0.5, grid_size + 0.5],
    )
    axes[r, 2].set_title(f"{labels[i]}: Variance")
    plt.colorbar(im_v, ax=axes[r, 2])

plt.tight_layout()
posteriors_path = "plots/bo_active/bo_posteriors.png"
plt.savefig(posteriors_path)
print(f"Saved posteriors plot to {posteriors_path}")
plt.show()

# Figure: RMSE Scaling
plt.figure(figsize=(10, 6))
colors = ["b", "g", "r", "purple"]
markers = ["o", "s", "^", "D"]

for i, strat in enumerate(strategies):
    rmse_hist = results[strat][4]
    loops = np.arange(len(rmse_hist))
    plt.plot(
        loops,
        rmse_hist,
        marker=markers[i],
        color=colors[i],
        label=labels[i],
        linewidth=2,
    )

plt.xlabel("Trajectory Loop Iteration")
plt.ylabel("RMSE")
plt.title("RMSE vs Trajectory Loops (Online Active Learning)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
rmse_path = "plots/bo_active/bo_rmse_scaling.png"
plt.savefig(rmse_path)
print(f"Saved RMSE scaling plot to {rmse_path}")
plt.show()

print("Done!")
