"""Quick demo of 3D visualization - simplified example."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.env import GridEnvironment, SimpleField, GridActor, GridConfig, GridPosition
from src.env.rendering import VisualizationRecorder, Grid3DRenderer


def main():
    # Small grid for quick visualization
    config = GridConfig(n_x=5, n_y=5, n_z=3, d_max=1)
    target = GridPosition(4, 4, 3)
    start = GridPosition(2, 2, 1)
    
    # Setup environment
    field = SimpleField(config, seed=42)
    actor = GridActor(config, start, noise_prob=0.1, seed=42)
    env = GridEnvironment(field=field, actor=actor, config=config, max_steps=20)
    
    # Setup recorder
    recorder = VisualizationRecorder(config, target_position=target, target_vicinity_radius=1.5)
    
    # Run short episode
    obs, info = env.reset()
    recorder.reset()
    recorder.record_step(env.actor.position, reward=0.0)
    recorder.cache_wind_field(env.field.get_mean_displacement_field())
    
    print("Running 15 steps...")
    for i in range(15):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        recorder.record_step(env.actor.position, action=action, reward=reward)
        
        if terminated or truncated:
            break
    
    print(f"Episode finished with {len(recorder)} steps\n")
    
    # Generate visualization
    print("Creating 3D visualization...")
    renderer = Grid3DRenderer(config, figsize=(12, 10))
    fig = renderer.render_static(recorder, show_wind=True, show_trajectory=True)
    
    # Save to output
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "quick_viz.png"
    renderer.save_figure(fig, str(output_path), dpi=150)
    
    print(f"Visualization saved to: {output_path}")
    
    env.close()


if __name__ == "__main__":
    main()

