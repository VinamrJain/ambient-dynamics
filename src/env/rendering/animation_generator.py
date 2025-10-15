"""Animation generator for grid environment visualization."""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

from ..utils.types import GridConfig
from .visualization_recorder import VisualizationRecorder
from .grid_3d_renderer import Grid3DRenderer


class AnimationGenerator:
    """Generates animations from recorded episode data."""
    
    def __init__(self, config: GridConfig, figsize: tuple = (12, 10)):
        """
        Initialize animation generator.
        
        Args:
            config: Grid configuration
            figsize: Figure size for animation frames
        """
        self.config = config
        self.renderer = Grid3DRenderer(config, figsize=figsize)
        
    def generate_gif(self, recorder: VisualizationRecorder,
                    output_path: str,
                    fps: int = 5,
                    show_wind: bool = True,
                    show_trajectory: bool = True,
                    show_grid: bool = True,
                    step_interval: int = 1) -> None:
        """
        Generate animated GIF from recorded episode.
        
        Args:
            recorder: VisualizationRecorder with episode data
            output_path: Path to save GIF file
            fps: Frames per second
            show_wind: Whether to show wind field
            show_trajectory: Whether to show trajectory
            show_grid: Whether to show grid points
            step_interval: Only render every nth step (for long episodes)
        """
        if len(recorder) == 0:
            raise ValueError("Recorder has no data to animate")
        
        # Determine steps to render
        steps_to_render = range(0, len(recorder), step_interval)
        
        # Create figure and axis
        fig, ax = self.renderer.create_figure()
        
        def update_frame(frame_idx):
            """Update function for animation."""
            ax.clear()
            
            # Reset axis properties
            ax.set_xlabel('X (i)', fontsize=10)
            ax.set_ylabel('Y (j)', fontsize=10)
            ax.set_zlabel('Z (k)', fontsize=10)
            ax.set_xlim(0.5, self.config.n_x + 0.5)
            ax.set_ylim(0.5, self.config.n_y + 0.5)
            ax.set_zlim(0.5, self.config.n_z + 0.5)
            ax.grid(True, alpha=0.3)
            
            step_idx = steps_to_render[frame_idx]
            
            # Plot grid points
            if show_grid:
                self.renderer.plot_grid_points(ax)
            
            # Plot wind field (only once, not every frame for efficiency)
            if show_wind and recorder.get_wind_field() is not None and frame_idx == 0:
                self.renderer.plot_wind_field(ax, recorder.get_wind_field())
            elif show_wind and recorder.get_wind_field() is not None:
                # For subsequent frames, redraw wind field
                self.renderer.plot_wind_field(ax, recorder.get_wind_field())
            
            # Plot target
            if recorder.target_position is not None:
                self.renderer.plot_target(ax, recorder.target_position,
                                        vicinity_radius=recorder.target_vicinity_radius)
            
            # Plot trajectory up to current step
            if show_trajectory:
                traj = recorder.get_trajectory_array()[:step_idx+1]
                if len(traj) > 1:
                    self.renderer.plot_trajectory(ax, traj)
            
            # Plot actor at current position
            current_pos = recorder.trajectory[step_idx]
            self.renderer.plot_actor(ax, current_pos)
            
            # Set title
            reward = recorder.rewards[step_idx] if step_idx < len(recorder.rewards) else 0.0
            title = f"Step {step_idx}/{len(recorder)-1} | Reward: {reward:.2f}"
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            # Add legend
            ax.legend(loc='upper left', fontsize=9)
        
        # Create animation
        anim = FuncAnimation(fig, update_frame, frames=len(steps_to_render),
                           interval=1000//fps, blit=False)
        
        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        plt.close(fig)
        
        print(f"Animation saved to: {output_path}")
    
    def generate_frames(self, recorder: VisualizationRecorder,
                       output_dir: str,
                       show_wind: bool = True,
                       show_trajectory: bool = True,
                       show_grid: bool = True,
                       step_interval: int = 1,
                       dpi: int = 150) -> None:
        """
        Generate individual frame images from recorded episode.
        
        Args:
            recorder: VisualizationRecorder with episode data
            output_dir: Directory to save frame images
            show_wind: Whether to show wind field
            show_trajectory: Whether to show trajectory
            show_grid: Whether to show grid points
            step_interval: Only render every nth step
            dpi: Image resolution
        """
        if len(recorder) == 0:
            raise ValueError("Recorder has no data to render")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine steps to render
        steps_to_render = range(0, len(recorder), step_interval)
        
        for i, step_idx in enumerate(steps_to_render):
            fig = self.renderer.render_frame(recorder, step_idx,
                                            show_wind=show_wind,
                                            show_trajectory=show_trajectory,
                                            show_grid=show_grid)
            
            # Save frame
            frame_path = output_path / f"frame_{i:04d}.png"
            self.renderer.save_figure(fig, str(frame_path), dpi=dpi)
            
        print(f"Frames saved to: {output_dir}")
        print(f"Total frames: {len(steps_to_render)}")

