"""Dynamic Programming All-Pairs Stochastic Shortest Path Agent.

Computes the optimal policy and expected step cost for a stochastic shortest 
path formulation to any goal on the grid, via Batch Value Iteration.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .agent import Agent, AgentConfig
from .dp_agent import _build_field_pmf_table, _get_actor_pmf
from ..env.arena.navigation_arena import NavigationArena
from ..env.utils.types import GridConfig, GridPosition

@dataclass
class APSSPAgentConfig(AgentConfig):
    """Configuration for the AP-SSP agent."""
    max_iters: int = 2000
    tol: float = 1e-3

def _ap_ssp_value_iteration_2d(
    field_pmf: jnp.ndarray,
    actor_pmf: jnp.ndarray,
    vicinity_radius: float,
    vicinity_metric: str,
    d_max: int,
    z_max: int,
    boundary_mode: str,
    max_iters: int = 2000,
    tol: float = 1e-3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute All-Pairs Stochastic Shortest Path (AP-SSP) tables for 2D.
    
    Returns
    -------
    H : jnp.ndarray shape (n_x, n_y, n_x, n_y) 
        Expected Cost (number of steps).
    Pi : jnp.ndarray shape (n_x, n_y, n_x, n_y)
        Optimal policy.
    """
    n_x, n_y, _ = field_pmf.shape
    
    all_i = jnp.arange(n_x)
    all_j = jnp.arange(n_y)
    
    # Grid coordinates for distance calculation
    I = all_i[:, None, None, None]
    J = all_j[None, :, None, None]
    GI = all_i[None, None, :, None]
    GJ = all_j[None, None, None, :]
    
    # Distance metrics
    if vicinity_metric == 'euclidean':
        dist = jnp.sqrt((I - GI)**2 + (J - GJ)**2)
    elif vicinity_metric == 'l1':
        dist = jnp.abs(I - GI) + jnp.abs(J - GJ)
    elif vicinity_metric == 'linf':
        dist = jnp.maximum(jnp.abs(I - GI), jnp.abs(J - GJ))
    else:
        # Fallback to euclidean if somehow wrong string passed
        dist = jnp.sqrt((I - GI)**2 + (J - GJ)**2)

    reached_mask = dist <= vicinity_radius
    
    # Initialize H. Reached states have 0 cost. Others have high cost.
    max_cost = 10000.0
    H_init = jnp.where(reached_mask, 0.0, max_cost)
    
    u_offsets = jnp.arange(-d_max, d_max + 1)
    v_offsets = jnp.arange(-z_max, z_max + 1)
    
    def _apply_boundary_i(idx):
        if boundary_mode == "clip": return jnp.clip(idx, 0, n_x - 1)
        elif boundary_mode == "periodic": return idx % n_x
        else: return jnp.clip(idx, 0, n_x - 1)
        
    def _apply_boundary_j(idx):
        return jnp.clip(idx, 0, n_y - 1)
        
    def _oob_mask_i(idx): return (idx < 0) | (idx >= n_x)
    def _oob_mask_j(idx): return (idx < 0) | (idx >= n_y)
    
    raw_next_i = all_i[:, None] + u_offsets[None, :]
    next_i = _apply_boundary_i(raw_next_i)
    oob_i = _oob_mask_i(raw_next_i) if boundary_mode == "terminal" else None
    
    raw_next_j = all_j[:, None] + v_offsets[None, :]
    next_j = _apply_boundary_j(raw_next_j)
    oob_j = _oob_mask_j(raw_next_j) if boundary_mode == "terminal" else None
    
    @jax.jit
    def step_fn(val):
        iter_count, H, Pi, max_diff = val
        
        # Build indexing for H lookup
        idx_i = next_i[:, None, :, None]
        idx_j = next_j[None, :, None, :]
        
        # H_lookup shape: (n_x, n_y, 2d+1, 2z+1, n_x, n_y)
        # Advanced indexing in JAX:
        H_lookup = H[idx_i, idx_j]
        
        if boundary_mode == "terminal":
            oob_mask = oob_i[:, None, :, None] | oob_j[None, :, None, :]
            # H_lookup should be max_cost for out-of-bounds transitions
            H_lookup = jnp.where(oob_mask[..., None, None], max_cost, H_lookup)
            
        # Expectation over ambient field
        # W shape: (n_x, n_y, 2z+1, n_x, n_y)
        W = jnp.einsum("iju,ijuvgh->ijvgh", field_pmf, H_lookup)
        
        # Expectation over actor (actions)
        # Q shape: (n_x, n_y, n_act, n_x, n_y)
        Q = 1.0 + jnp.einsum("av,ijvgh->ijagh", actor_pmf, W)
        
        H_new = jnp.min(Q, axis=2)
        Pi_new = jnp.argmin(Q, axis=2).astype(jnp.int32)
        
        # Apply boundary/goal condition
        H_new = jnp.where(reached_mask, 0.0, H_new)
        
        diff = jnp.max(jnp.abs(H - H_new))
        return (iter_count + 1, H_new, Pi_new, diff)
        
    init_val = (0, H_init, jnp.zeros_like(H_init, dtype=jnp.int32), jnp.array(1e6, dtype=jnp.float32))
    val = init_val
    
    from tqdm import tqdm
    with tqdm(total=max_iters, desc="AP-SSP Value Iteration") as pbar:
        while val[0] < max_iters and val[3] > tol:
            val = step_fn(val)
            pbar.update(1)
            pbar.set_postfix({"max_diff": f"{float(val[3]):.4f}"})
            
    return val[1], val[2]


class APSSPAgent(Agent):
    """Agent that plans stochastic shortest paths to any goal."""

    def __init__(
        self,
        config: APSSPAgentConfig | None = None,
        num_actions: int = 3,
        obs_shape: tuple[int, ...] = (3,),
    ) -> None:
        if config is None:
            config = APSSPAgentConfig()
        super().__init__(config, num_actions, obs_shape)

        self._cost_table: Optional[jnp.ndarray] = None
        self._policy_table: Optional[jnp.ndarray] = None
        self._ndim: Optional[int] = None
        self.target_position: Optional[GridPosition] = None

    def prepare_episode(self, env) -> None:
        """Run batch value iteration for the current field."""
        self.plan(env.arena)
        
    def plan(self, arena: NavigationArena) -> None:
        """Run AP-SSP Value Iteration."""
        cfg: GridConfig = arena.config
        if cfg.ndim != 2:
            raise NotImplementedError("APSSPAgent currently only supports 2D environments.")
            
        d_max = arena.field.d_max
        z_max = arena.actor.z_max
        self._ndim = cfg.ndim

        field_pmf = _build_field_pmf_table(arena)
        actor_pmf = _get_actor_pmf(arena)
        
        vicinity_radius = arena.vicinity_radius
        vicinity_metric = getattr(arena, 'vicinity_metric', 'euclidean')
        boundary_mode = arena.boundary_mode

        H, Pi = _ap_ssp_value_iteration_2d(
            field_pmf, actor_pmf, vicinity_radius, vicinity_metric,
            d_max, z_max, boundary_mode,
            max_iters=self.config.max_iters, tol=self.config.tol
        )
        self._cost_table = np.array(H)
        self._policy_table = np.array(Pi)

    def get_expected_cost(self, pos: GridPosition, target: GridPosition) -> float:
        """Look up the expected cost to reach target from pos."""
        if self._cost_table is None:
            raise RuntimeError("Agent must be planned first.")
        return float(self._cost_table[pos.i - 1, pos.j - 1, target.i - 1, target.j - 1])

    def set_target(self, target: GridPosition) -> None:
        """Set the target position for the inner routine."""
        self.target_position = target

    def _action_from_obs(self, observation: np.ndarray) -> int:
        if self._policy_table is None:
            raise RuntimeError("Agent must be planned first.")
        if self.target_position is None:
            raise RuntimeError("Target position must be set before stepping.")
            
        i_idx = int(observation[0]) - 1
        j_idx = int(observation[1]) - 1
        g_i = self.target_position.i - 1
        g_j = self.target_position.j - 1
        
        return int(self._policy_table[i_idx, j_idx, g_i, g_j])

    def begin_episode(self, observation: np.ndarray) -> int:
        return self._action_from_obs(observation)

    def step(self, reward: float, observation: np.ndarray) -> int:
        return self._action_from_obs(observation)

    def end_episode(self, reward: float, terminal: bool) -> None:
        pass
