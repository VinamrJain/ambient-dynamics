import gymnasium as gym
import grid_env
from grid_env import SimpleField, GridActor, GridConfig, GridPosition

def main():
    # Create configuration
    config = GridConfig(n_x=5, n_y=5, n_z=3, d_max=1)
    
    # Create field and actor
    field = SimpleField(config, seed=42)
    actor = GridActor(config, GridPosition(3, 3, 2))
    
    # Create environment
    env = grid_env.GridEnvironment(
        field=field,
        actor=actor, 
        config=config,
        max_steps=100
    )
    
    # Run episode
    obs, info = env.reset()
    done = False
    
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        print(f"Position: {info['position']}, Reward: {reward}")
    
    env.close()

if __name__ == "__main__":
    main()