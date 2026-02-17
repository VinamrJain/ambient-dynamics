"""Parallel experiment launcher.

Runs multiple experiment configs (or the same config with different seeds)
in parallel using ``concurrent.futures.ProcessPoolExecutor``.

Usage::

    python -m src.eval.launcher --config experiments/configs/my_suite.yaml

Or programmatically::

    from src.eval.launcher import launch_suite
    results = launch_suite(suite_config, max_workers=4)
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .experiment_config import load_config
from .runner import run_experiment, ExperimentResult


# ---------------------------------------------------------------------------
# Single-run entry point (pickle-friendly for multiprocessing)
# ---------------------------------------------------------------------------

def _run_single(
    cfg: dict[str, Any],
    num_episodes: int,
    master_seed: int,
    train_episodes: int,
    run_id: str,
) -> dict[str, Any]:
    """Run one experiment and return the summary as a plain dict.

    This function is the target for ``ProcessPoolExecutor.submit``.
    It must be a module-level function (not a lambda/closure) so that
    it is picklable.
    """
    result = run_experiment(
        cfg,
        num_episodes=num_episodes,
        master_seed=master_seed,
        train_episodes=train_episodes,
    )
    summary = result.summary()
    summary["run_id"] = run_id
    summary["master_seed"] = master_seed
    return summary


# ---------------------------------------------------------------------------
# Suite launcher
# ---------------------------------------------------------------------------

def launch_suite(
    suite_cfg: dict[str, Any],
    *,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Launch a suite of experiments in parallel.

    Parameters
    ----------
    suite_cfg : dict
        Top-level YAML config with keys:

        - ``defaults``: shared experiment defaults (env, eval params)
        - ``runs``: list of per-run overrides (agent, seed, etc.)

        Each run inherits from ``defaults`` and overrides specific keys.

    max_workers : int
        Number of parallel worker processes.

    Returns
    -------
    list[dict]
        One summary dict per run.
    """
    defaults = suite_cfg.get("defaults", {})
    runs = suite_cfg.get("runs", [])

    if not runs:
        raise ValueError("Suite config has no 'runs' entries.")

    # Build per-run configs by merging defaults
    jobs: list[tuple[dict, int, int, int, str]] = []
    for i, run_override in enumerate(runs):
        cfg = _deep_merge(defaults, run_override)
        num_episodes = cfg.get("eval", {}).get("num_episodes", 10)
        master_seed = cfg.get("eval", {}).get("master_seed", 0)
        train_episodes = cfg.get("eval", {}).get("train_episodes", 0)
        run_id = run_override.get("id", f"run_{i}")
        jobs.append((cfg, num_episodes, master_seed, train_episodes, run_id))

    # Execute
    summaries: list[dict[str, Any]] = []

    if max_workers <= 1:
        for args in jobs:
            summaries.append(_run_single(*args))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run_single, *args): args[-1] for args in jobs
            }
            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    summaries.append(future.result())
                except Exception as exc:
                    print(f"[ERROR] Run '{run_id}' failed: {exc}")
                    summaries.append({"run_id": run_id, "error": str(exc)})

    return summaries


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a suite of RL experiments in parallel.",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML suite config file.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write JSON results (optional).",
    )
    args = parser.parse_args()

    suite_cfg = load_config(args.config)
    print(f"Launching suite with {len(suite_cfg.get('runs', []))} runs, "
          f"max_workers={args.workers}")

    t0 = time.time()
    results = launch_suite(suite_cfg, max_workers=args.workers)
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    for r in results:
        print(f"  {r.get('run_id', '?'):>12s}  |  {r.get('agent', '?'):>12s}  |  "
              f"reward={r.get('mean_reward', 0):.2f} ± {r.get('std_reward', 0):.2f}  |  "
              f"reach={r.get('reach_rate', 0):.0%}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
