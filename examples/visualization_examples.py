"""Examples of different visualization approaches for navigation arena.

Demonstrates:
1. Live interactive visualization ('human' mode)
2. GIF generation from episode
3. Video generation (MP4)
4. Static frame capture
5. HTML export for sharing
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
import numpy as np
import time


def example_1_live_interactive():
    """Example 1: Live interactive visualization in browser.
    
    This is the simplest approach - opens an interactive 3D plot in your browser.
    Best for: Quick debugging, manual inspection, understanding behavior.
    """
    print("=" * 70)
    print("EXAMPLE 1: Live Interactive Visualization")
    print("=" * 70)
    
    # Setup
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
    
    renderer = NavigationRenderer(config=config, width=1200, height=900)
    env = GridEnvironment(arena=arena, max_steps=100, seed=42, renderer=renderer)
    
    # Run episode
    print("\nRunning episode...")
    obs, info = env.reset(seed=42)
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    # Visualize
    print("Opening interactive visualization in browser...")
    print("(You can rotate, zoom, pan the 3D plot)")
    env.render(mode='human')
    
    env.close()
    print("\n✅ Example 1 complete\n")


def example_2_gif_generation():
    """Example 2: Generate animated GIF from episode.
    
    Captures frames during episode and creates GIF.
    Best for: Sharing results, documentation, presentations.
    
    Note: Requires 'kaleido' package for frame capture.
          Install with: pixi add kaleido
    """
    print("=" * 70)
    print("EXAMPLE 2: GIF Generation")
    print("=" * 70)
    
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("❌ This example requires 'imageio' package")
        print("   Install with: pixi add imageio")
        return
    
    # Setup
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
    
    renderer = NavigationRenderer(config=config, width=800, height=600)
    env = GridEnvironment(arena=arena, max_steps=100, seed=42, renderer=renderer)
    
    # Run episode and collect frames
    print("\nRunning episode and capturing frames...")
    frames = []
    obs, info = env.reset(seed=42)
    
    # Capture initial frame
    frame = env.render(mode='rgb_array')
    if frame is not None and frame.sum() > 0:  # Check if not placeholder
        frames.append(frame)
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Capture frame every N steps to reduce GIF size
        if step % 2 == 0:
            frame = env.render(mode='rgb_array')
            if frame is not None and frame.sum() > 0:
                frames.append(frame)
        
        if terminated or truncated:
            break
    
    # Save GIF
    if frames:
        os.makedirs('output/visualizations', exist_ok=True)
        gif_path = 'output/visualizations/episode.gif'
        
        print(f"\nSaving GIF with {len(frames)} frames...")
        imageio.mimsave(gif_path, frames, fps=5, loop=0)
        print(f"✅ GIF saved to: {gif_path}")
        print(f"   File size: {os.path.getsize(gif_path) / 1024:.1f} KB")
    else:
        print("⚠️  No frames captured. Install kaleido:")
        print("   pixi add kaleido")
    
    env.close()
    print("\n✅ Example 2 complete\n")


def example_3_video_generation():
    """Example 3: Generate MP4 video from episode.
    
    Higher quality than GIF, smaller file size for longer episodes.
    Best for: Long episodes, high-quality recordings, analysis.
    
    Note: Requires 'kaleido' and 'imageio-ffmpeg' packages.
    """
    print("=" * 70)
    print("EXAMPLE 3: Video (MP4) Generation")
    print("=" * 70)
    
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("❌ This example requires 'imageio' package")
        return
    
    # Setup (same as GIF example)
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
    
    renderer = NavigationRenderer(config=config, width=1024, height=768)
    env = GridEnvironment(arena=arena, max_steps=100, seed=42, renderer=renderer)
    
    # Run episode and collect frames
    print("\nRunning episode and capturing frames...")
    frames = []
    obs, info = env.reset(seed=42)
    
    frame = env.render(mode='rgb_array')
    if frame is not None and frame.sum() > 0:
        frames.append(frame)
    
    for step in range(100):  # Longer episode for video
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Capture every frame for smooth video
        frame = env.render(mode='rgb_array')
        if frame is not None and frame.sum() > 0:
            frames.append(frame)
        
        if terminated or truncated:
            break
    
    # Save video
    if frames:
        os.makedirs('output/visualizations', exist_ok=True)
        video_path = 'output/visualizations/episode.mp4'
        
        print(f"\nSaving video with {len(frames)} frames...")
        # Use mimwrite for video (requires ffmpeg)
        try:
            imageio.mimwrite(video_path, frames, fps=10, codec='libx264')
            print(f"✅ Video saved to: {video_path}")
            print(f"   File size: {os.path.getsize(video_path) / 1024:.1f} KB")
        except Exception as e:
            print(f"⚠️  Could not save video: {e}")
            print("   Install ffmpeg: pixi add imageio-ffmpeg")
    else:
        print("⚠️  No frames captured. Install kaleido:")
        print("   pixi add kaleido")
    
    env.close()
    print("\n✅ Example 3 complete\n")


def example_4_static_frames():
    """Example 4: Capture static frames at key timesteps.
    
    Saves individual frames at specific points (start, middle, end, target reached).
    Best for: Papers, reports, detailed analysis.
    """
    print("=" * 70)
    print("EXAMPLE 4: Static Frame Capture")
    print("=" * 70)
    
    # Setup
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
    
    renderer = NavigationRenderer(config=config, width=1200, height=900)
    env = GridEnvironment(arena=arena, max_steps=100, seed=42, renderer=renderer)
    
    # Run episode and save key frames
    print("\nRunning episode and capturing key frames...")
    os.makedirs('output/frames', exist_ok=True)
    
    obs, info = env.reset(seed=42)
    
    # Capture start
    frame = env.render(mode='rgb_array')
    if frame is not None and frame.sum() > 0:
        try:
            from PIL import Image
            Image.fromarray(frame).save('output/frames/frame_000_start.png')
            print("✅ Saved: frame_000_start.png")
        except:
            pass
    
    target_reached_frame_saved = False
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Save at key timesteps
        if step == 25:  # Quarter
            frame = env.render(mode='rgb_array')
            if frame is not None and frame.sum() > 0:
                try:
                    from PIL import Image
                    Image.fromarray(frame).save('output/frames/frame_025_quarter.png')
                    print("✅ Saved: frame_025_quarter.png")
                except:
                    pass
        
        elif step == 50:  # Halfway
            frame = env.render(mode='rgb_array')
            if frame is not None and frame.sum() > 0:
                try:
                    from PIL import Image
                    Image.fromarray(frame).save('output/frames/frame_050_halfway.png')
                    print("✅ Saved: frame_050_halfway.png")
                except:
                    pass
        
        # Save when target reached
        if info.get('target_reached', False) and not target_reached_frame_saved:
            frame = env.render(mode='rgb_array')
            if frame is not None and frame.sum() > 0:
                try:
                    from PIL import Image
                    Image.fromarray(frame).save(f'output/frames/frame_{step:03d}_target_reached.png')
                    print(f"✅ Saved: frame_{step:03d}_target_reached.png")
                    target_reached_frame_saved = True
                except:
                    pass
        
        if terminated or truncated:
            break
    
    # Save final frame
    frame = env.render(mode='rgb_array')
    if frame is not None and frame.sum() > 0:
        try:
            from PIL import Image
            Image.fromarray(frame).save(f'output/frames/frame_{step:03d}_final.png')
            print(f"✅ Saved: frame_{step:03d}_final.png")
        except:
            pass
    
    print("\n✅ All frames saved to: output/frames/")
    env.close()
    print("\n✅ Example 4 complete\n")


def example_5_html_export():
    """Example 5: Export interactive HTML file.
    
    Creates standalone HTML file with interactive 3D plot.
    Best for: Sharing results, embedding in websites, offline viewing.
    """
    print("=" * 70)
    print("EXAMPLE 5: HTML Export")
    print("=" * 70)
    
    # Setup
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
    
    renderer = NavigationRenderer(config=config, width=1200, height=900)
    env = GridEnvironment(arena=arena, max_steps=100, seed=42, renderer=renderer)
    
    # Run episode
    print("\nRunning episode...")
    obs, info = env.reset(seed=42)
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    # Export to HTML
    os.makedirs('output/visualizations', exist_ok=True)
    html_path = 'output/visualizations/episode_interactive.html'
    
    print(f"\nExporting to HTML...")
    # Get figure from renderer
    fig = renderer._create_figure()
    fig.write_html(html_path)
    
    print(f"✅ Interactive HTML saved to: {html_path}")
    print(f"   File size: {os.path.getsize(html_path) / 1024:.1f} KB")
    print(f"   Open in browser to view (fully interactive)")
    
    env.close()
    print("\n✅ Example 5 complete\n")


def main():
    """Run all visualization examples."""
    print("\n" + "=" * 70)
    print("NAVIGATION ARENA - VISUALIZATION EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates different ways to visualize episodes:\n")
    print("1. Live Interactive (browser)")
    print("2. GIF Generation")
    print("3. Video (MP4) Generation")
    print("4. Static Frame Capture")
    print("5. HTML Export")
    print("\n" + "=" * 70 + "\n")
    
    # Run examples
    example_1_live_interactive()
    time.sleep(1)
    
    example_2_gif_generation()
    time.sleep(1)
    
    example_3_video_generation()
    time.sleep(1)
    
    example_4_static_frames()
    time.sleep(1)
    
    example_5_html_export()
    
    print("=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nOutput files saved to:")
    print("  - output/visualizations/  (GIFs, videos, HTML)")
    print("  - output/frames/          (Static images)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

