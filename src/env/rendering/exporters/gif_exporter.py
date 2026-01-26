"""GIF export functionality for navigation renderer."""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..navigation_renderer import NavigationRenderer


def save_gif(
    renderer: 'NavigationRenderer',
    output_path: str,
    fps: int = 10,
    subsample: int = 1
) -> None:
    """Export episode as animated GIF.
    
    Args:
        renderer: NavigationRenderer instance with recorded states.
        output_path: Path to save GIF file.
        fps: Frames per second (default: 10).
        subsample: Use every Nth frame to reduce file size (default: 1 = all frames).
    
    Example:
        >>> renderer = NavigationRenderer(config)
        >>> # ... run episode ...
        >>> renderer.save_gif('episode.gif', fps=10)
    
    Note:
        Requires 'imageio' and 'kaleido' packages.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("ERROR: GIF export requires 'imageio' package")
        print(" Install with: pixi add imageio")
        return
    
    if not renderer.states:
        print("ERROR: No states recorded. Run episode first.")
        return
    
    print(f"Generating GIF with {len(renderer.states)} frames...")
    
    # Collect frames
    frames = []
    for i in range(0, len(renderer.states), subsample):
        # Temporarily set states to render each frame
        original_states = renderer.states
        renderer.states = original_states[:i+1]
        
        frame = renderer.render(mode='rgb_array')
        
        # Restore full state history
        renderer.states = original_states
        
        if frame is not None and frame.sum() > 0:
            frames.append(frame)
    
    if not frames:
        print("ERROR: No frames captured. Install kaleido:")
        print("   pixi add kaleido")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    
    print(f"GIF saved to: {output_path}")
    print(f"   Frames: {len(frames)} (subsampled by {subsample})")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")

