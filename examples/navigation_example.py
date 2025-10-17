"""Example demonstrating navigation arena with Plotly visualization."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import (
    GridEnvironment,
    NavigationArena,
    SimpleField,
    GridActor,
    NavigationRenderer,
    GridConfig,
    GridPosition,
)
import numpy as np


def main():
    print("=" * 70)
    print("GRID ENVIRONMENT - NAVIGATION EXAMPLE")
    print("=" * 70)
    
    # 1. Create configuration
    config = GridConfig(n_x=10, n_y=10, n_z=5, d_max=1)
    print(f"\nGrid size: {config.n_x} x {config.n_y} x {config.n_z}")
    
    # 2. Create field and actor
    field = SimpleField(config)
    actor = GridActor(noise_prob=0.1)
    
    # 3. Define start and target positions
    initial_position = GridPosition(2, 2, 2)
    target_position = GridPosition(8, 8, 4)
    vicinity_radius = 1.5
    
    print(f"Start position: ({initial_position.i}, {initial_position.j}, {initial_position.k})")
    print(f"Target position: ({target_position.i}, {target_position.j}, {target_position.k})")
    print(f"Vicinity radius: {vicinity_radius}")
    
    # 4. Create navigation arena
    arena = NavigationArena(
        field=field,
        actor=actor,
        config=config,
        initial_position=initial_position,
        target_position=target_position,
        vicinity_radius=vicinity_radius,
        boundary_mode='clip',
        distance_reward_weight=-0.1,   # Penalty per unit distance (outside vicinity)
        vicinity_bonus=1.0,            # Continuous reward for staying in vicinity
        step_penalty=-0.1,             # Penalty per step (outside vicinity)
        terminate_on_reach=False,      # Continue for station-keeping
        use_distance_decay=True,       # Bonus decays from center to edge
        decay_rate=0.3                 # Exponential decay rate
    )
    
    # 5. Create renderer (extracts all info from arena state)
    renderer = NavigationRenderer(
        config=config,
        width=1200,
        height=900,
        show_grid_points=True,
        backend='plotly'
    )
    
    print(f"\nRenderer settings:")
    print(f"  Backend: {renderer.backend}")
    print(f"  Grid subsample: {renderer.grid_subsample}")
    print(f"  Grid point size: {renderer.grid_point_size:.2f}")
    print(f"  Actor marker size: {renderer.actor_size:.2f}")
    
    # 6. Create environment
    env = GridEnvironment(
        arena=arena,
        max_steps=100,
        seed=42,
        renderer=renderer
    )
    
    print(f"\nMax steps: {env.max_steps}")
    print("\n" + "=" * 70)
    
    # 7. Run episode (renderer handles all logging internally)
    obs, info = env.reset(seed=42)
    
    print("\nRunning episode with random policy...")
    print("-" * 70)
    
    done = False
    step = 0
    
    while not done and step < 50:  # Limit for demo
        # Step environment (renderer automatically records state)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
    
    print(f"Episode finished after {step} steps")
    print("-" * 70)
    
    # 8. Show final visualization
    print("\n" + "=" * 70)
    print("Showing final visualization in browser...")
    print("(An interactive 3D plot will open in your default browser)")
    print("=" * 70)
    
    try:
        env.render(mode='human')
        print("\n✅ Visualization displayed!")
        print("\nNote: For GIF/video export, install kaleido:")
        print("      pip install kaleido")
    except Exception as e:
        print(f"⚠️  Could not show visualization: {e}")
    
    env.close()
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

