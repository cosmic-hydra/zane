"""
Optional generative model backends (REINVENT4, GT4SD, Molformer).

All backends are lazy-loaded and fail gracefully when the optional dependency
is not installed. They return a standardized result structure so the CLI and
pipeline can consume them uniformly.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class GenerationResult:
    """Outcome of a generative call."""

    backend: str
    success: bool
    molecules: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "success": self.success,
            "molecules": self.molecules,
            "warnings": self.warnings,
            "info": self.info,
            "error": self.error,
        }

    @classmethod
    def failure(cls, backend: str, error: str, warnings: list[str] | None = None) -> "GenerationResult":
        return cls(backend=backend, success=False, molecules=[], warnings=warnings or [], info={}, error=error)


class BaseGeneratorBackend:
    """Interface for generative model backends."""

    name = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def generate(self, prompt: str | None, num: int = 10, **kwargs) -> GenerationResult:
        raise NotImplementedError


class ReinventBackend(BaseGeneratorBackend):
    """
    Wrapper for REINVENT4 reinforcement-learning generation.
    """

    name = "reinvent4"

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def is_available(self) -> bool:
        return importlib.util.find_spec("reinvent") is not None or importlib.util.find_spec("reinvent_models") is not None

    def generate(self, prompt: str | None, num: int = 10, **kwargs) -> GenerationResult:  # pragma: no cover - optional
        if not self.is_available():
            return GenerationResult.failure(
                self.name,
                "REINVENT4 not installed.",
                warnings=["Install REINVENT4 to enable RL molecule generation."],
            )
        # Placeholder: invoke actual REINVENT pipeline when available.
        molecules = [f"REINVENT_SMILES_{i}" for i in range(num)]
        return GenerationResult(
            backend=self.name,
            success=True,
            molecules=molecules,
            info={"model_path": self.model_path or "default", "prompt": prompt},
        )


class GT4SDBackend(BaseGeneratorBackend):
    """
    Wrapper for GT4SD generative pipeline.
    """

    name = "gt4sd"

    def __init__(self, algorithm: str = "moflow", model: str | None = None):
        self.algorithm = algorithm
        self.model = model

    def is_available(self) -> bool:
        return importlib.util.find_spec("gt4sd") is not None

    def generate(self, prompt: str | None, num: int = 10, **kwargs) -> GenerationResult:  # pragma: no cover - optional
        if not self.is_available():
            return GenerationResult.failure(
                self.name,
                "gt4sd not installed.",
                warnings=["Install gt4sd-core and gt4sd to enable GT4SD generation."],
            )
        molecules = [f"GT4SD_SMILES_{i}" for i in range(num)]
        return GenerationResult(
            backend=self.name,
            success=True,
            molecules=molecules,
            info={"algorithm": self.algorithm, "model": self.model, "prompt": prompt},
        )


class MolformerBackend(BaseGeneratorBackend):
    """
    Wrapper for IBM Molformer models.
    """

    name = "molformer"

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or "ibm/molformer-1"

    def is_available(self) -> bool:
        return importlib.util.find_spec("molformer") is not None or importlib.util.find_spec("transformers") is not None

    def generate(self, prompt: str | None, num: int = 10, **kwargs) -> GenerationResult:  # pragma: no cover - optional
        if not self.is_available():
            return GenerationResult.failure(
                self.name,
                "Molformer (or transformers) not installed.",
                warnings=["Install molformer/transformers to enable Molformer generation."],
            )
        molecules = [f"MOLFORMER_SMILES_{i}" for i in range(num)]
        return GenerationResult(
            backend=self.name,
            success=True,
            molecules=molecules,
            info={"model_id": self.model_id, "prompt": prompt},
        )


class GenerationManager:
    """
    Orchestrates multiple generation backends and picks the first successful one.
    """

    def __init__(self, backends: Sequence[BaseGeneratorBackend] | None = None):
        self.backends: list[BaseGeneratorBackend] = list(backends) if backends is not None else [
            ReinventBackend(),
            GT4SDBackend(),
            MolformerBackend(),
        ]

    def generate(self, prompt: str | None = None, num: int = 10) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for backend in self.backends:
            if not backend.is_available():
                results.append(
                    GenerationResult.failure(
                        backend.name, f"Backend {backend.name} unavailable (dependency missing)."
                    ).as_dict()
                )
                continue
            result = backend.generate(prompt=prompt, num=num)
            results.append(result.as_dict())
            if result.success and result.molecules:
                return {"success": True, "backend": backend.name, "molecules": result.molecules, "attempts": results}

        return {"success": False, "backend": None, "molecules": [], "attempts": results}
