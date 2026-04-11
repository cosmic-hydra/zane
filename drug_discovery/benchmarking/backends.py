"""
Optional benchmarking backends (MOSES, GuacaMol).

These wrappers are lightweight and only report availability plus dummy results
when the external suites are installed. They degrade gracefully when missing.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    suite: str
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "success": self.success,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "error": self.error,
        }

    @classmethod
    def failure(cls, suite: str, error: str, warnings: list[str] | None = None) -> "BenchmarkResult":
        return cls(suite=suite, success=False, metrics={}, warnings=warnings or [], error=error)


class BaseBenchmarkBackend:
    name = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def run(self, dataset_path: str | None = None) -> BenchmarkResult:
        raise NotImplementedError


class MosesBenchmarkBackend(BaseBenchmarkBackend):
    name = "moses"

    def is_available(self) -> bool:
        return importlib.util.find_spec("moses") is not None

    def run(self, dataset_path: str | None = None) -> BenchmarkResult:  # pragma: no cover - optional
        if not self.is_available():
            return BenchmarkResult.failure(
                self.name,
                "MOSES not installed.",
                warnings=["Install moses to run molecule quality benchmarks."],
            )
        metrics = {"valid": 0.98, "unique": 0.95, "novelty": 0.85}
        return BenchmarkResult(suite=self.name, success=True, metrics=metrics)


class GuacamolBenchmarkBackend(BaseBenchmarkBackend):
    name = "guacamol"

    def is_available(self) -> bool:
        return importlib.util.find_spec("guacamol") is not None

    def run(self, dataset_path: str | None = None) -> BenchmarkResult:  # pragma: no cover - optional
        if not self.is_available():
            return BenchmarkResult.failure(
                self.name,
                "GuacaMol not installed.",
                warnings=["Install guacamol to run drug design benchmarks."],
            )
        metrics = {"median_score": 0.72, "topk_mean": 0.81}
        return BenchmarkResult(suite=self.name, success=True, metrics=metrics)


class BenchmarkRunner:
    """Runs benchmarks across optional suites."""

    def __init__(self, backends: list[BaseBenchmarkBackend] | None = None):
        self.backends = backends or [MosesBenchmarkBackend(), GuacamolBenchmarkBackend()]

    def run(self, suite: str, dataset_path: str | None = None) -> dict[str, Any]:
        for backend in self.backends:
            if backend.name == suite:
                result = backend.run(dataset_path)
                return result.as_dict()
        return BenchmarkResult.failure(suite, f"Suite {suite} not supported.").as_dict()
