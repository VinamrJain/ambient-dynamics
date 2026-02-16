"""Shared helpers for interactive field visualization scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping


def add_project_root_to_path() -> Path:
    """Add repository root to sys.path for direct script execution."""
    project_root = Path(__file__).resolve().parents[2]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root


def get_output_dir() -> Path:
    """Create and return the viz output directory."""
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(fig, filename: str, dpi: int = 150, bbox_inches: str | None = None) -> Path:
    """Save a Matplotlib figure under tests/viz_field/output."""
    output_path = get_output_dir() / filename
    kwargs = {"dpi": dpi}
    if bbox_inches is not None:
        kwargs["bbox_inches"] = bbox_inches
    fig.savefig(output_path, **kwargs)
    return output_path


def format_params(params: Mapping[str, object]) -> str:
    """Format parameters for plot subtitles and figure text."""
    return ", ".join(f"{key}={value}" for key, value in params.items())


def make_grid_locations_2d(n_x: int, n_y: int):
    """Return flattened 2D grid coordinates with 1-indexed convention."""
    import numpy as np

    x_coords = np.arange(1, n_x + 1)
    y_coords = np.arange(1, n_y + 1)
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
    return np.column_stack([x_mesh.ravel(), y_mesh.ravel()])
