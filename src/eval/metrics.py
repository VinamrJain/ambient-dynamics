"""Logger backends for experiment tracking.

Implements the ``Logger`` protocol from ``src.agents.agent`` for
Weights & Biases and TensorBoard, plus a ``CompositeLogger`` that
fans out to multiple backends.
"""

from __future__ import annotations

from typing import Any

from ..agents.agent import Logger


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------

class WandbLogger:
    """Thin wrapper around ``wandb``."""

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        **init_kwargs,
    ) -> None:
        import wandb
        self._run = wandb.init(
            project=project,
            name=name,
            config=config or {},
            group=group,
            tags=tags,
            **init_kwargs,
        )

    def log_scalar(self, key: str, value: float, step: int) -> None:
        import wandb
        wandb.log({key: value}, step=step)

    def log_dict(self, data: dict[str, Any], step: int) -> None:
        import wandb
        wandb.log(data, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        import wandb
        wandb.config.update(config, allow_val_change=True)

    def finish(self) -> None:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------------

class TensorBoardLogger:
    """Thin wrapper around ``torch.utils.tensorboard.SummaryWriter``."""

    def __init__(self, log_dir: str) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self._writer.add_scalar(key, value, step)

    def log_dict(self, data: dict[str, Any], step: int) -> None:
        for k, v in data.items():
            self._writer.add_scalar(k, v, step)

    def log_config(self, config: dict[str, Any]) -> None:
        text = "\n".join(f"**{k}**: {v}" for k, v in config.items())
        self._writer.add_text("config", text)

    def close(self) -> None:
        self._writer.close()


# ---------------------------------------------------------------------------
# PrintLogger (useful for debugging / CI)
# ---------------------------------------------------------------------------

class PrintLogger:
    """Prints metrics to stdout."""

    def log_scalar(self, key: str, value: float, step: int) -> None:
        print(f"[step {step}] {key} = {value:.4f}")

    def log_dict(self, data: dict[str, Any], step: int) -> None:
        items = "  ".join(f"{k}={v:.4f}" for k, v in data.items())
        print(f"[step {step}] {items}")

    def log_config(self, config: dict[str, Any]) -> None:
        print(f"[config] {config}")


# ---------------------------------------------------------------------------
# Composite (fan-out to multiple backends)
# ---------------------------------------------------------------------------

class CompositeLogger:
    """Delegates every call to a list of child loggers."""

    def __init__(self, loggers: list[Logger]) -> None:
        self._loggers = list(loggers)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        for lg in self._loggers:
            lg.log_scalar(key, value, step)

    def log_dict(self, data: dict[str, Any], step: int) -> None:
        for lg in self._loggers:
            lg.log_dict(data, step)

    def log_config(self, config: dict[str, Any]) -> None:
        for lg in self._loggers:
            lg.log_config(config)
