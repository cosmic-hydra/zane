"""Hyperparameter Optimization for ZANE.
Grid/random/Bayesian search with early stopping.
Inspired by DeepChem dc.hyper and Optuna.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HPOConfig:
    method: str = "random"
    n_trials: int = 50
    metric: str = "val_loss"
    direction: str = "minimize"
    seed: int = 42
    early_stopping_rounds: int = 10
    save_results: bool = True
    results_dir: str = "./hpo_results"


@dataclass
class TrialResult:
    trial_id: int
    params: dict[str, Any]
    metric_value: float
    duration_seconds: float
    status: str = "completed"


class SearchSpace:
    """Define hyperparameter search spaces."""

    def __init__(self):
        self.params = {}

    def add_float(self, name, low, high, log=False):
        self.params[name] = {"type": "float", "low": low, "high": high, "log": log}
        return self

    def add_int(self, name, low, high):
        self.params[name] = {"type": "int", "low": low, "high": high}
        return self

    def add_categorical(self, name, choices):
        self.params[name] = {"type": "categorical", "choices": choices}
        return self

    def sample(self, rng=None):
        rng = rng or random.Random()
        params = {}
        for name, s in self.params.items():
            if s["type"] == "float":
                params[name] = (
                    np.exp(rng.uniform(np.log(s["low"]), np.log(s["high"])))
                    if s.get("log")
                    else rng.uniform(s["low"], s["high"])
                )
            elif s["type"] == "int":
                params[name] = rng.randint(s["low"], s["high"])
            elif s["type"] == "categorical":
                params[name] = rng.choice(s["choices"])
        return params

    def grid(self, resolution=5):
        axes = []
        for name, s in self.params.items():
            if s["type"] == "float":
                vals = (
                    np.exp(np.linspace(np.log(s["low"]), np.log(s["high"]), resolution))
                    if s.get("log")
                    else np.linspace(s["low"], s["high"], resolution)
                )
                axes.append([(name, v) for v in vals])
            elif s["type"] == "int":
                axes.append(
                    [(name, v) for v in range(s["low"], s["high"] + 1, max(1, (s["high"] - s["low"]) // resolution))]
                )
            elif s["type"] == "categorical":
                axes.append([(name, c) for c in s["choices"]])
        return [dict(combo) for combo in itertools.product(*axes)]


class HPOptimizer:
    """HPO for ZANE models. Example:
    space = SearchSpace().add_float("lr",1e-5,1e-2,log=True).add_int("layers",2,8)
    best = HPOptimizer(HPOConfig(n_trials=50)).optimize(space, train_fn)
    """

    def __init__(self, config: HPOConfig):
        self.config = config
        self.results: list[TrialResult] = []
        self.best_result = None

    def optimize(self, space, train_fn):
        rng = random.Random(self.config.seed)
        candidates = (
            space.grid()[: self.config.n_trials]
            if self.config.method == "grid"
            else [space.sample(rng) for _ in range(self.config.n_trials)]
        )
        no_improve = 0
        for i, params in enumerate(candidates):
            t0 = time.time()
            try:
                metric = train_fn(params)
                status = "completed"
            except Exception as e:
                logger.warning(f"Trial {i} failed: {e}")
                metric = float("inf") if self.config.direction == "minimize" else float("-inf")
                status = "failed"
            tr = TrialResult(i, params, metric, time.time() - t0, status)
            self.results.append(tr)
            if (
                self.best_result is None
                or (self.config.direction == "minimize" and metric < self.best_result.metric_value)
                or (self.config.direction == "maximize" and metric > self.best_result.metric_value)
            ):
                self.best_result = tr
                no_improve = 0
                logger.info(f"Trial {i}: {metric:.6f} NEW BEST")
            else:
                no_improve += 1
            if no_improve >= self.config.early_stopping_rounds:
                logger.info(f"Early stop at trial {i}")
                break
        if self.config.save_results:
            self._save()
        return self.best_result

    def _save(self):
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(self.config.results_dir) / "hpo_results.json", "w") as f:
            json.dump(
                {
                    "best": asdict(self.best_result) if self.best_result else None,
                    "trials": [asdict(r) for r in self.results],
                },
                f,
                indent=2,
                default=str,
            )

    def summary(self):
        vals = [r.metric_value for r in self.results if r.status == "completed"]
        return {
            "trials": len(self.results),
            "best": self.best_result.metric_value if self.best_result else None,
            "best_params": self.best_result.params if self.best_result else None,
            "mean": float(np.mean(vals)) if vals else None,
        }
