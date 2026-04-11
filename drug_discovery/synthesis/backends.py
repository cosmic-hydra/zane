"""
Optional retrosynthesis backends that wrap external toolkits.

The backends rely on third-party projects such as AiZynthFinder. They are
loaded lazily and degrade gracefully when the optional dependency or
configuration is not available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drug_discovery.integrations import get_integration_status


@dataclass
class RouteCandidate:
    """Structured description of a retrosynthesis route."""

    score: float | None = None
    steps: int | None = None
    precursors: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendResult:
    """Outcome of a backend run."""

    backend: str
    success: bool
    routes: list[RouteCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def failure(cls, backend: str, error: str, warnings: list[str] | None = None) -> "BackendResult":
        return cls(backend=backend, success=False, routes=[], warnings=warnings or [], info={}, error=error)

    def as_dict(self) -> dict[str, Any]:
        """Convert to plain dict for JSON serialization."""
        return {
            "backend": self.backend,
            "success": self.success,
            "routes": [route.__dict__ for route in self.routes],
            "warnings": self.warnings,
            "info": self.info,
            "error": self.error,
        }


class BaseRetrosynthesisBackend:
    """Interface for external retrosynthesis backends."""

    name = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def plan(self, smiles: str, max_depth: int = 5) -> BackendResult:
        raise NotImplementedError


class AiZynthFinderBackend(BaseRetrosynthesisBackend):
    """
    Thin wrapper around AiZynthFinder.

    Requires:
        - Package: `aizynthfinder`
        - Config: path to an AiZynthFinder config YAML.
    """

    name = "aizynthfinder"

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path) if config_path else None

    def is_available(self) -> bool:
        return bool(self.config_path and get_integration_status("aizynthfinder").importable)

    def plan(self, smiles: str, max_depth: int = 5) -> BackendResult:
        if not self.is_available():
            status = get_integration_status("aizynthfinder")
            return BackendResult.failure(
                self.name,
                "AiZynthFinder unavailable (missing dependency or config path).",
                warnings=[
                    "Install aizynthfinder and provide a config file via RetrosynthesisPlanner(aizynth_config=...).",
                    f"Submodule registered: {status.submodule_registered}; local checkout present: {status.local_checkout_present}.",
                ],
            )

        try:
            from aizynthfinder.aizynthfinder import AiZynthFinder

            finder = AiZynthFinder(configfile=str(self.config_path))
            finder.target_smiles = smiles
            if max_depth:
                finder.config.search_tree.max_depth = max_depth

            finder.tree_search()

            routes: list[RouteCandidate] = []
            display_routes = getattr(getattr(finder, "routes", None), "displayed_tree_routes", [])
            for route in display_routes or []:
                score = getattr(route, "score", None)
                steps = getattr(route, "number_of_steps", None)
                precursors = getattr(route, "precursors", None)
                precursor_smiles = [str(pc) for pc in precursors] if precursors else None
                routes.append(RouteCandidate(score=score, steps=steps, precursors=precursor_smiles))

            if not routes:
                return BackendResult.failure(self.name, "No routes found.")

            return BackendResult(
                backend=self.name,
                success=True,
                routes=routes,
                info={"used_config": str(self.config_path)},
            )
        except Exception as exc:  # pragma: no cover - depends on external lib
            return BackendResult.failure(self.name, f"AiZynthFinder error: {exc}")
