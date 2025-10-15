"""Example of using the 3D visualization system for grid environment."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

from src.env import GridEnvironment, SimpleField, GridActor, GridConfig, GridPosition
from src.env.rendering import VisualizationRecorder, Grid3DRenderer, AnimationGenerator


def create_reward_function(target_position: GridPosition):
    """Create a distance-based reward function."""
    def reward_fn(state, action):
        actor_state = state['actor']
        position = actor_state['position']
        
        # Distance-based reward (negative distance from target)
        distance = (abs(position.i - target_position.i) + 
                   abs(position.j - target_position.j) + 
                   abs(position.k - target_position.k))
        
        # Reward structure: closer is better, penalty for each step
        reward = -distance - 0.1
        
        # Bonus for reaching target
        if distance == 0:
            reward += 10.0
            
        return reward
    
    return reward_fn


def run_episode_with_recording(env, recorder, max_steps=50):
    """Run an episode and record data for visualization."""
    obs, info = env.reset()
    recorder.reset()
    
    # Record initial position
    recorder.record_step(env.actor.position, action=None, reward=0.0, info=info)
    
    # Cache wind field for visualization (computed once)
    wind_field = env.field.get_mean_displacement_field()
    recorder.cache_wind_field(wind_field)
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Take action (random for this example)
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        
        # Record step
        recorder.record_step(env.actor.position, action=action, 
                           reward=reward, info=info)
        
        # Print progress
        print(f"Step {step}: Position {env.actor.position}, Reward {reward:.2f}")
    
    # Print episode summary
    summary = recorder.get_episode_summary()
    print("\n" + "="*50)
    print("Episode Summary:")
    print(f"  Total steps: {summary['num_steps']}")
    print(f"  Total reward: {summary['total_reward']:.2f}")
    print(f"  Mean reward: {summary['mean_reward']:.2f}")
    print(f"  Start: {summary['start_position']}")
    print(f"  End: {summary['end_position']}")
    print(f"  Target: {summary['target_position']}")
    print("="*50 + "\n")
    
    return recorder


def main():
    print("="*60)
    print("3D Grid Environment Visualization Example")
    print("="*60 + "\n")
    
    # Configuration
    config = GridConfig(n_x=8, n_y=8, n_z=5, d_max=1)
    target_position = GridPosition(7, 7, 4)
    initial_position = GridPosition(2, 2, 2)
    
    print(f"Grid size: {config.n_x} x {config.n_y} x {config.n_z}")
    print(f"Initial position: {initial_position}")
    print(f"Target position: {target_position}")
    print(f"Max displacement: {config.d_max}\n")
    
    # Create environment components
    field = SimpleField(config, seed=42)
    actor = GridActor(config, initial_position, noise_prob=0.1, seed=42)
    
    # Create reward function
    reward_fn = create_reward_function(target_position)
    
    # Create environment
    env = GridEnvironment(
        field=field,
        actor=actor,
        config=config,
        reward_fn=reward_fn,
        max_steps=100
    )
    
    # Create visualization recorder
    recorder = VisualizationRecorder(
        config=config,
        target_position=target_position,
        target_vicinity_radius=1.5
    )
    
    # Run episode with recording
    print("Running episode...")
    recorder = run_episode_with_recording(env, recorder, max_steps=30)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "output" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...\n")
    
    # 1. Create static visualization of final state
    print("1. Generating static plot of final state...")
    renderer = Grid3DRenderer(config, figsize=(14, 12))
    fig = renderer.render_static(
        recorder,
        step_idx=None,  # Last step
        title="Grid Environment - Final State",
        show_wind=True,
        show_trajectory=True,
        show_grid=True
    )
    static_path = output_dir / "final_state.png"
    renderer.save_figure(fig, str(static_path), dpi=150)
    print(f"   Saved to: {static_path}")
    
    # 2. Create static visualization of intermediate state
    print("2. Generating static plot of intermediate state...")
    mid_step = len(recorder) // 2
    fig = renderer.render_static(
        recorder,
        step_idx=mid_step,
        title=f"Grid Environment - Step {mid_step}",
        show_wind=True,
        show_trajectory=True,
        show_grid=True
    )
    static_mid_path = output_dir / "intermediate_state.png"
    renderer.save_figure(fig, str(static_mid_path), dpi=150)
    print(f"   Saved to: {static_mid_path}")
    
    # 3. Generate animation (GIF)
    print("3. Generating animation (this may take a moment)...")
    anim_generator = AnimationGenerator(config, figsize=(14, 12))
    gif_path = output_dir / "episode_animation.gif"
    
    try:
        anim_generator.generate_gif(
            recorder,
            str(gif_path),
            fps=5,
            show_wind=True,
            show_trajectory=True,
            show_grid=True,
            step_interval=1  # Render every step
        )
        print(f"   Saved to: {gif_path}")
    except Exception as e:
        print(f"   Warning: Failed to generate GIF: {e}")
        print("   (You may need to install pillow: pip install pillow)")
    
    # 4. Generate individual frames (optional)
    print("4. Generating individual frames...")
    frames_dir = output_dir / "frames"
    try:
        anim_generator.generate_frames(
            recorder,
            str(frames_dir),
            show_wind=True,
            show_trajectory=True,
            show_grid=True,
            step_interval=2,  # Every 2nd step to save space
            dpi=100
        )
    except Exception as e:
        print(f"   Warning: Failed to generate frames: {e}")
    
    print("\n" + "="*60)
    print("Visualization complete! Check the output directory:")
    print(f"  {output_dir}")
    print("="*60)
    
    env.close()


if __name__ == "__main__":
    main()

