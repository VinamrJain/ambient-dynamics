"""Evaluation harness -- runners, configs, logging, and parallel launching."""

from .runner import run_episode, run_experiment, EpisodeResult, ExperimentResult
from .experiment_config import (
    load_config,
    build_env,
    build_agent,
    derive_seed,
    register_agent,
)
from .metrics import (
    WandbLogger,
    TensorBoardLogger,
    PrintLogger,
    CompositeLogger,
)
from .launcher import launch_suite

__all__ = [
    "run_episode",
    "run_experiment",
    "EpisodeResult",
    "ExperimentResult",
    "load_config",
    "build_env",
    "build_agent",
    "derive_seed",
    "register_agent",
    "WandbLogger",
    "TensorBoardLogger",
    "PrintLogger",
    "CompositeLogger",
    "launch_suite",
]
