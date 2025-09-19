"""Example of using the configuration system."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import GridEnvironment, SimpleField, GridActor
from src.env.config import EnvironmentConfig


def distance_based_reward(state, action, config):
    """Distance-based reward function using configuration."""
    actor_state = state['actor']
    position = actor_state['position']

    target = config.target_position
    if target is None:
        return -config.step_penalty

    # Distance-based reward
    distance = abs(position.i - target.i) + abs(position.j - target.j) + abs(position.k - target.k)

    if distance == 0:
        return config.goal_reward  # Reached target
    else:
        return -distance - config.step_penalty


def main():
    # Load configuration
    config = EnvironmentConfig()  # Uses default config

    print("Configuration loaded:")
    print(f"Grid: {config.grid_config}")
    print(f"Max steps: {config.max_steps}")
    print(f"Target: {config.target_position}")
    print()

    # Create components using configuration
    field = SimpleField(config.grid_config, seed=config.field_seed)
    actor = GridActor(
        config.grid_config,
        config.initial_position,
        noise_prob=config.noise_prob,
        seed=config.actor_seed
    )

    # Create reward function with config
    def reward_fn(state, action):
        return distance_based_reward(state, action, config)

    # Create environment
    env = GridEnvironment(
        field=field,
        actor=actor,
        config=config.grid_config,
        reward_fn=reward_fn,
        max_steps=config.max_steps
    )

    # Run episode
    obs, info = env.reset(seed=config.environment_seed)
    done = False
    step = 0

    print("Starting episode...")
    env.render()

    while not done and step < 10:  # Limit to 10 steps for demo
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        print(f"\nStep {step}:")
        print(f"Action: {action}, Reward: {reward:.1f}")
        env.render()

        if terminated:
            print("Episode terminated!")
        elif truncated:
            print("Episode truncated!")

    env.close()

    # Example of updating configuration
    print("\nUpdating configuration...")
    config.update('actor.noise_prob', 0.2)
    config.update('reward.step_penalty', 0.05)
    print(f"New noise prob: {config.noise_prob}")
    print(f"New step penalty: {config.step_penalty}")


if __name__ == "__main__":
    main()