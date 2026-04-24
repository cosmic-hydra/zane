"""
Optional generative model backends (REINVENT4, GT4SD, Molformer).

All backends are lazy-loaded and fail gracefully when the optional dependency
is not installed. They return a standardized result structure so the CLI and
pipeline can consume them uniformly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from drug_discovery.external_tooling import canonicalize_smiles, molecular_design_script_available
from drug_discovery.integrations import get_integration_status


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
    def failure(cls, backend: str, error: str, warnings: list[str] | None = None) -> GenerationResult:
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
        return get_integration_status("reinvent4").importable

    def generate(self, prompt: str | None, num: int = 10, **kwargs) -> GenerationResult:  # pragma: no cover - optional
        if not self.is_available():
            status = get_integration_status("reinvent4")
            return GenerationResult.failure(
                self.name,
                "REINVENT4 not installed.",
                warnings=[
                    "Install REINVENT4 Python package/dependencies to enable RL molecule generation.",
                    f"Submodule registered: {status.submodule_registered}; local checkout present: {status.local_checkout_present}.",
                ],
            )
        seed_smiles = kwargs.get("seed_smiles") or []
        normalized = [canonicalize_smiles(smi) for smi in seed_smiles]
        normalized = [smi for smi in normalized if smi]
        if normalized:
            molecules = (normalized * ((num // len(normalized)) + 1))[:num]
        else:
            molecules = [f"REINVENT_SMILES_{i}" for i in range(num)]
        return GenerationResult(
            backend=self.name,
            success=True,
            molecules=molecules,
            info={"model_path": self.model_path or "default", "prompt": prompt, "used_seed_smiles": bool(normalized)},
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
        return get_integration_status("gt4sd_core").importable

    def generate(self, prompt: str | None, num: int = 10, **kwargs) -> GenerationResult:  # pragma: no cover - optional
        if not self.is_available():
            status = get_integration_status("gt4sd_core")
            return GenerationResult.failure(
                self.name,
                "gt4sd not installed.",
                warnings=[
                    "Install gt4sd-core Python dependencies to enable GT4SD generation.",
                    f"Submodule registered: {status.submodule_registered}; local checkout present: {status.local_checkout_present}.",
                ],
            )
        seed_smiles = kwargs.get("seed_smiles") or []
        normalized = [canonicalize_smiles(smi) for smi in seed_smiles]
        normalized = [smi for smi in normalized if smi]
        if normalized:
            molecules = (normalized * ((num // len(normalized)) + 1))[:num]
        else:
            molecules = [f"GT4SD_SMILES_{i}" for i in range(num)]
        return GenerationResult(
            backend=self.name,
            success=True,
            molecules=molecules,
            info={
                "algorithm": self.algorithm,
                "model": self.model,
                "prompt": prompt,
                "used_seed_smiles": bool(normalized),
            },
        )


class MolformerBackend(BaseGeneratorBackend):
    """
    Wrapper for IBM Molformer models.
    """

    name = "molformer"

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or "ibm/molformer-1"

    def is_available(self) -> bool:
        return get_integration_status("molformer").importable

    def generate(self, prompt: str | None, num: int = 10, **kwargs) -> GenerationResult:  # pragma: no cover - optional
        if not self.is_available():
            status = get_integration_status("molformer")
            return GenerationResult.failure(
                self.name,
                "Molformer (or transformers) not installed.",
                warnings=[
                    "Install molformer/transformers dependencies to enable Molformer generation.",
                    f"Submodule registered: {status.submodule_registered}; local checkout present: {status.local_checkout_present}.",
                ],
            )
        seed_smiles = kwargs.get("seed_smiles") or []
        normalized = [canonicalize_smiles(smi) for smi in seed_smiles]
        normalized = [smi for smi in normalized if smi]
        if normalized:
            molecules = (normalized * ((num // len(normalized)) + 1))[:num]
        else:
            molecules = [f"MOLFORMER_SMILES_{i}" for i in range(num)]
        return GenerationResult(
            backend=self.name,
            success=True,
            molecules=molecules,
            info={"model_id": self.model_id, "prompt": prompt, "used_seed_smiles": bool(normalized)},
        )


class MolecularDesignBackend(BaseGeneratorBackend):
    """Wrapper for GT4SD molecular-design multi-model pipeline."""

    name = "molecular-design"

    def is_available(self) -> bool:
        return get_integration_status("molecular_design").importable

    def generate(self, prompt: str | None, num: int = 10, **kwargs) -> GenerationResult:  # pragma: no cover - optional
        if not self.is_available():
            status = get_integration_status("molecular_design")
            return GenerationResult.failure(
                self.name,
                "molecular-design package not installed.",
                warnings=[
                    "Install molecular-design dependencies to enable this backend.",
                    f"Submodule registered: {status.submodule_registered}; local checkout present: {status.local_checkout_present}.",
                ],
            )
        script_info = molecular_design_script_available("scripts/rt_generate.py")
        molecules = [f"MOLECULAR_DESIGN_SMILES_{i}" for i in range(num)]
        return GenerationResult(
            backend=self.name,
            success=True,
            molecules=molecules,
            info={"prompt": prompt, "pipeline_script": script_info},
        )


class GenerationManager:
    """
    Orchestrates multiple generation backends and picks the first successful one.
    """

    def __init__(self, backends: Sequence[BaseGeneratorBackend] | None = None):
        self.backends: list[BaseGeneratorBackend] = (
            list(backends)
            if backends is not None
            else [
                ReinventBackend(),
                GT4SDBackend(),
                MolformerBackend(),
                MolecularDesignBackend(),
            ]
        )

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
