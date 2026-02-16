"""2D Navigation Arena with RFF GP Field Visualization."""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax

from src.env import (
    GridEnvironment,
    NavigationArena,
    GridActor,
    NavigationRenderer,
    GridConfig,
    GridPosition,
)
from src.env.field import RFFGPField


def run_2d_visualization_rff():
    """Run 2D navigation with RFF GP Field."""
    
    print("=" * 70)
    print("2D NAVIGATION - RFF GP FIELD")
    print("=" * 70)
    
    # Configuration: 2D grid
    config = GridConfig.create(n_x=100, n_y=80)
    d_max = 20
    
    print(f"\nGrid configuration:")
    print(f"  Dimensions: {config.ndim}D")
    print(f"  Size: {config.n_x} x {config.n_y}")
    print(f"  Max displacement: {d_max}")
    
    # GP Field parameters
    sigma = 5
    lengthscale = 10
    nu = 2.5
    
    print(f"\nRFF GP Field parameters:")
    print(f"  sigma: {sigma} (amplitude)")
    print(f"  lengthscale: {lengthscale} (correlation)")
    print(f"  nu: {nu} (smoothness)")
    
    # Positions
    initial_position = GridPosition(20, 20, None)
    target_position = GridPosition(80, 60, None)
    vicinity_radius = 15.0
    
    print(f"\nNavigation task:")
    print(f"  Start: ({initial_position.i}, {initial_position.j})")
    print(f"  Target: ({target_position.i}, {target_position.j})")
    
    # Create RFF GP field (2D = scalar field for single ambient axis)
    field = RFFGPField(
        config, d_max=d_max,
        sigma=sigma, lengthscale=lengthscale, nu=nu,
        num_features=500, noise_std=0.5
    )
    actor = GridActor(noise_std=0.1)
    
    arena = NavigationArena(
        field=field,
        actor=actor,
        config=config,
        initial_position=initial_position,
        target_position=target_position,
        vicinity_radius=vicinity_radius,
        boundary_mode='clip',
        distance_reward_weight=-0.1,
        vicinity_bonus=5.0,
        step_penalty=-0.1,
        terminate_on_reach=False,
        use_distance_decay=True,
        decay_rate=0.3
    )
    
    renderer = NavigationRenderer(
        config=config,
        show_grid_points=True,
        width=900,
        height=700,
        field=field,
        show_field=True
    )
    
    env = GridEnvironment(
        arena=arena,
        max_steps=100,
        seed=42,
        renderer=renderer
    )
    
    # Run episode
    print("\n" + "-" * 70)
    print("Running episode with random policy (50 steps)...")
    
    obs, info = env.reset(seed=42)
    
    total_reward = 0.0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}")
            break
    
    print(f"  Final position: ({info['position'].i}, {info['position'].j})")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Target reached: {info['target_reached']}")
    print("-" * 70)
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "2d")
    os.makedirs(output_dir, exist_ok=True)
    animated_html_path = os.path.join(output_dir, "viz_2d_rff_output_animated.html")
    
    renderer.save_animated_html(animated_html_path)
    print(f"\nSaved to: {animated_html_path}")
    
    env.close()


if __name__ == "__main__":
    run_2d_visualization_rff()
