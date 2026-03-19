"""Evaluation metrics for field estimation."""

import jax
import jax.numpy as jnp

def kl_divergence(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """Compute KL-Divergence between two discrete distributions.
    
    Args:
        p: shape (..., n_bins) - true distribution.
        q: shape (..., n_bins) - estimated distribution.
        
    Returns:
        KL divergence(p || q)
    """
    # Clip to avoid log(0)
    p = jnp.clip(p, 1e-12, 1.0)
    q = jnp.clip(q, 1e-12, 1.0)
    # Re-normalize
    p /= jnp.sum(p, axis=-1, keepdims=True)
    q /= jnp.sum(q, axis=-1, keepdims=True)
    
    return jnp.sum(p * jnp.log(p / q), axis=-1)

def tv_distance(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """Compute Total Variation distance between two discrete distributions.
    
    Args:
        p: shape (..., n_bins) - true distribution.
        q: shape (..., n_bins) - estimated distribution.
        
    Returns:
        TV distance = 0.5 * sum(|p - q|)
    """
    return 0.5 * jnp.sum(jnp.abs(p - q), axis=-1)

def compute_field_rmse(true_mean: jnp.ndarray, estimated_mean: jnp.ndarray) -> float:
    """Compute RMSE between true field mean and estimated mean.
    
    Args:
        true_mean: shape (..., D)
        estimated_mean: shape (..., D)
    """
    return float(jnp.sqrt(jnp.mean((true_mean - estimated_mean)**2)))
