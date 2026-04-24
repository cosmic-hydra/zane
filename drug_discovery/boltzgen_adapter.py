"""
Lightweight BoltzGen integration helpers.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BoltzGenResult:
    """Result of a BoltzGen execution."""

    command: list[str]
    output_dir: Path
    returncode: int
    stdout: str
    stderr: str
    metrics: list[dict[str, object]] | None
    metrics_file: Path | None

    @property
    def success(self) -> bool:
        """Whether the run completed successfully."""
        return self.returncode == 0


class BoltzGenRunner:
    """
    Minimal wrapper around the BoltzGen CLI.

    Supports command construction, invocation, and parsing of the ranking CSVs
    emitted by the official BoltzGen pipeline.
    """

    def __init__(
        self, executable: str = "boltzgen", cache_dir: str | Path | None = None, work_dir: str | Path | None = None
    ):
        self.executable = executable
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.work_dir = Path(work_dir) if work_dir else Path.cwd() / "outputs" / "boltzgen"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def build_run_command(
        self,
        design_spec: str | Path,
        output_dir: str | Path,
        protocol: str = "protein-anything",
        num_designs: int = 50,
        budget: int = 10,
        steps: Sequence[str] | None = None,
        devices: int | None = None,
        reuse: bool = True,
        config_overrides: Sequence[str] | None = None,
        extra_args: Sequence[str] | None = None,
    ) -> list[str]:
        """
        Build the BoltzGen CLI invocation.

        Args:
            design_spec: Path to a BoltzGen design YAML.
            output_dir: Where pipeline artifacts should be written.
            protocol: BoltzGen protocol identifier.
            num_designs: Intermediate designs to generate before filtering.
            budget: Final number of designs to keep.
            steps: Optional subset of steps to run (e.g. ["design", "analysis"]).
            devices: Number of devices to request.
            reuse: Whether to reuse existing intermediate files.
            config_overrides: Optional configuration overrides passed via --config.
            extra_args: Additional CLI flags passed verbatim.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.executable,
            "run",
            str(design_spec),
            "--output",
            str(output_path),
            "--protocol",
            protocol,
            "--num_designs",
            str(num_designs),
            "--budget",
            str(budget),
        ]

        if steps:
            cmd.extend(["--steps", *steps])
        if devices is not None:
            cmd.extend(["--devices", str(devices)])
        if reuse:
            cmd.append("--reuse")
        if self.cache_dir:
            cmd.extend(["--cache", str(self.cache_dir)])
        for override in config_overrides or []:
            cmd.extend(["--config", override])
        if extra_args:
            cmd.extend(list(extra_args))

        return cmd

    def run(
        self,
        design_spec: str | Path,
        output_dir: str | Path | None = None,
        protocol: str = "protein-anything",
        num_designs: int = 50,
        budget: int = 10,
        steps: Sequence[str] | None = None,
        devices: int | None = None,
        reuse: bool = True,
        config_overrides: Sequence[str] | None = None,
        extra_args: Sequence[str] | None = None,
        parse_results: bool = True,
        env: dict[str, str] | None = None,
    ) -> BoltzGenResult:
        """
        Execute the BoltzGen pipeline via subprocess.

        Returns:
            BoltzGenResult with command, outputs, and parsed metrics when available.
        """
        self._ensure_available()

        resolved_output = Path(output_dir) if output_dir else self.work_dir / "run"
        resolved_output.mkdir(parents=True, exist_ok=True)

        command = self.build_run_command(
            design_spec=design_spec,
            output_dir=resolved_output,
            protocol=protocol,
            num_designs=num_designs,
            budget=budget,
            steps=steps,
            devices=devices,
            reuse=reuse,
            config_overrides=config_overrides,
            extra_args=extra_args,
        )

        env_vars = os.environ.copy()
        if env:
            env_vars.update(env)
        if self.cache_dir and "HF_HOME" not in env_vars:
            env_vars["HF_HOME"] = str(self.cache_dir)

        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env_vars)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"BoltzGen executable '{self.executable}' not found. Install with `pip install boltzgen`."
            ) from exc

        metrics: list[dict[str, object]] | None = None
        metrics_file: Path | None = None
        if parse_results and completed.returncode == 0:
            metrics, metrics_file = self.parse_metrics(resolved_output, budget=budget)

        return BoltzGenResult(
            command=command,
            output_dir=resolved_output,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            metrics=metrics,
            metrics_file=metrics_file,
        )

    def parse_metrics(
        self, output_dir: str | Path, budget: int | None = None
    ) -> tuple[list[dict[str, object]] | None, Path | None]:
        """
        Parse BoltzGen ranking CSVs from an output directory.
        """
        metrics_path = self._locate_metrics_file(Path(output_dir), budget=budget)
        if metrics_path is None:
            return None, None

        with metrics_path.open() as f:
            reader = csv.DictReader(f)
            metrics = [self._convert_row(row) for row in reader]

        return metrics, metrics_path

    @staticmethod
    def summarize_metrics(
        metrics: Iterable[dict[str, object]] | None, top_k: int = 5, score_key: str | None = None
    ) -> list[dict[str, object]]:
        """
        Summarize metrics by returning the top_k entries optionally sorted by score_key.
        """
        if not metrics:
            return []

        metrics_list = list(metrics)
        if score_key and all(score_key in m for m in metrics_list):
            sorted_metrics = sorted(metrics_list, key=lambda row: row.get(score_key) or 0, reverse=False)
        else:
            sorted_metrics = metrics_list

        return sorted_metrics[:top_k]

    def _locate_metrics_file(self, output_dir: Path, budget: int | None = None) -> Path | None:
        candidates: list[Path] = []
        if budget:
            candidates.append(output_dir / "final_ranked_designs" / f"final_designs_metrics_{budget}.csv")
        candidates.append(output_dir / "final_ranked_designs" / "final_designs_metrics.csv")
        candidates.append(output_dir / "final_ranked_designs" / "all_designs_metrics.csv")

        for path in candidates:
            if path.exists():
                return path

        for path in output_dir.glob("**/*metrics*.csv"):
            return path

        return None

    @staticmethod
    def _convert_row(row: dict[str, str]) -> dict[str, object]:
        converted: dict[str, object] = {}
        for key, value in row.items():
            converted[key] = BoltzGenRunner._coerce_value(value)
        return converted

    @staticmethod
    def _coerce_value(value: str) -> object:
        stripped = value.strip()
        if stripped == "":
            return ""
        try:
            integer = int(stripped)
            return integer
        except ValueError:
            pass
        try:
            floating = float(stripped)
            return floating
        except ValueError:
            pass
        return stripped

    def _ensure_available(self) -> None:
        if shutil.which(self.executable) is None:
            raise FileNotFoundError(
                f"BoltzGen executable '{self.executable}' not found. Install with `pip install boltzgen`."
            )
