"""Simple demo of the clean export API.

Shows how easy it is to export episodes in different formats
with the new modular architecture.
"""

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


def main():
    print("=" * 70)
    print("SIMPLE EXPORT DEMO")
    print("=" * 70)
    print("\nDemonstrates the clean export API")
    print("=" * 70 + "\n")
    
    # Setup environment
    config = GridConfig(n_x=10, n_y=10, n_z=5, d_max=1)
    field = SimpleField(config)
    actor = GridActor(noise_prob=0.1)
    
    arena = NavigationArena(
        field=field,
        actor=actor,
        config=config,
        initial_position=GridPosition(2, 2, 2),
        target_position=GridPosition(8, 8, 4),
        vicinity_radius=1.5,
        distance_reward_weight=-0.1,
        vicinity_bonus=1.0,
        step_penalty=-0.1,
        terminate_on_reach=False,
        use_distance_decay=True,
        decay_rate=0.3
    )
    
    # Create renderer with custom camera angle
    renderer = NavigationRenderer(
        config=config, 
        width=1000, 
        height=800,
        camera_eye={'x': 1.5, 'y': -1.5, 'z': 1.0}  # Isometric view
    )
    
    env = GridEnvironment(arena=arena, max_steps=100, seed=42, renderer=renderer)
    
    # Run episode
    print("Running episode...")
    obs, info = env.reset(seed=42)
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    print(f"Episode complete: {len(renderer.states)} steps\n")
    
    # Export in multiple formats - all with one line each!
    print("Exporting episode in multiple formats...\n")
    
    # 1. GIF
    print("1. Exporting GIF...")
    renderer.save_gif('output/demo/episode.gif', fps=10, subsample=2)
    
    # 2. MP4
    print("\n2. Exporting MP4...")
    renderer.save_mp4('output/demo/episode.mp4', fps=15)
    
    # 3. Static HTML
    print("\n3. Exporting static HTML...")
    renderer.save_html('output/demo/episode_static.html')
    
    # 4. Animated HTML
    print("\n4. Exporting animated HTML...")
    renderer.save_animated_html('output/demo/episode_animated.html', fps=10)
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nAll files saved to: output/demo/")
    print("\nTry opening episode_animated.html in your browser!")
    print("You can play/pause and interact with the 3D view at each frame.")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    main()

