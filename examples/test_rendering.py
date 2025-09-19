"""Test the new rendering system."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import GridEnvironment, SimpleField, GridActor, GridConfig, GridPosition

def simple_reward_function(state, action):
    """Simple reward function for testing."""
    actor_state = state['actor']
    position = actor_state['position']
    target_i, target_j, target_k = 3, 3, 2
    distance = abs(position.i - target_i) + abs(position.j - target_j) + abs(position.k - target_k)
    return -distance - 0.1

def main():
    # Create configuration
    config = GridConfig(n_x=5, n_y=5, n_z=3, d_max=1)

    # Create field and actor
    field = SimpleField(config, seed=42)
    actor = GridActor(config, GridPosition(1, 1, 1), seed=42)

    # Create environment with reward function
    env = GridEnvironment(
        field=field,
        actor=actor,
        config=config,
        reward_fn=simple_reward_function,
        max_steps=10
    )

    # Run a few steps and show rendering
    obs, info = env.reset()
    print("=== Testing New Rendering System ===")
    env.render()  # Initial state

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n--- After action {action} ---")
        env.render()

        if terminated or truncated:
            break

    env.close()

if __name__ == "__main__":
    main()