"""Basic usage example for the new Grid Environment architecture.

Demonstrates:
- Creating field, actor, and arena components
- Wrapping arena with GridEnvironment (Gym API)
- Running episodes with flat observations
- JAX-based random seeding
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import (
    GridEnvironment,
    GridArena,
    SimpleField,
    GridActor,
    GridConfig,
    GridPosition,
)


def main():
    # 1. Create configuration
    config = GridConfig(n_x=5, n_y=5, n_z=3, d_max=1)
    
    # 2. Create field (environmental dynamics)
    field = SimpleField(config)
    
    # 3. Create actor (vertical dynamics)
    actor = GridActor(noise_prob=0.1)
    
    # 4. Create arena (simulator + task logic)
    arena = GridArena(
        field=field,
        actor=actor,
        config=config,
        initial_position=GridPosition(3, 3, 2),
        boundary_mode='clip'  # Options: 'clip', 'periodic', 'terminal'
    )
    
    # 5. Create environment (Gym RL interface)
    env = GridEnvironment(
        arena=arena,
        max_steps=100,
        seed=42
    )
    
    # 6. Run episode
    print("=" * 60)
    print("GRID ENVIRONMENT - BASIC USAGE EXAMPLE")
    print("=" * 60)
    print(f"\nGrid size: {config.n_x} x {config.n_y} x {config.n_z}")
    print(f"Max displacement: {config.d_max}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("\n" + "=" * 60)
    
    # Reset environment
    observation, info = env.reset(seed=42)
    print(f"\nInitial observation: {observation}")
    print(f"  Position: ({int(observation[0])}, {int(observation[1])}, {int(observation[2])})")
    print(f"  Last displacement: ({observation[3]:.1f}, {observation[4]:.1f})")
    
    # Run episode with random actions
    done = False
    total_reward = 0.0
    step = 0
    
    print("\nRunning episode...")
    print("-" * 60)
    
    while not done and step < 10:  # Limit to 10 steps for demo
        # Random action: 0=down, 1=stay, 2=up
        action = env.action_space.sample()
        action_names = ['DOWN', 'STAY', 'UP']
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Print step info
        pos = info['position']
        last_disp = info['last_displacement']
        print(f"Step {step+1}:")
        print(f"  Action: {action_names[action]}")
        print(f"  Position: ({pos.i}, {pos.j}, {pos.k})")
        print(f"  Displacement: ({last_disp.u:.1f}, {last_disp.v:.1f})")
        print(f"  Reward: {reward:.2f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print()
        
        step += 1
    
    print("-" * 60)
    print(f"\nEpisode finished after {step} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final position: {info['position']}")
    
    env.close()
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

