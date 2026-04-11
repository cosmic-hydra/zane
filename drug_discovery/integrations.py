"""Integration registry for optional external drug-discovery ecosystems.

This module centralizes:
- upstream repository metadata
- local submodule checkout detection
- python module availability checks
- optional local checkout path injection for imports
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IntegrationSpec:
    key: str
    name: str
    purpose: str
    url: str
    python_modules: tuple[str, ...] = ()
    submodule_path: str | None = None
    is_collection: bool = False


@dataclass(frozen=True)
class IntegrationStatus:
    key: str
    name: str
    purpose: str
    url: str
    submodule_path: str | None
    submodule_registered: bool
    local_checkout_present: bool
    python_module: str | None
    importable: bool
    is_collection: bool

    @property
    def available(self) -> bool:
        if self.is_collection:
            return True
        return self.importable or self.local_checkout_present

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "name": self.name,
            "purpose": self.purpose,
            "url": self.url,
            "submodule_path": self.submodule_path,
            "submodule_registered": self.submodule_registered,
            "local_checkout_present": self.local_checkout_present,
            "python_module": self.python_module,
            "importable": self.importable,
            "available": self.available,
            "is_collection": self.is_collection,
        }


INTEGRATIONS: dict[str, IntegrationSpec] = {
    "aizynthfinder": IntegrationSpec(
        key="aizynthfinder",
        name="AiZynthFinder",
        purpose="Retrosynthesis (core)",
        url="https://github.com/MolecularAI/aizynthfinder",
        python_modules=("aizynthfinder",),
        submodule_path="external/aizynthfinder",
    ),
    "reinvent4": IntegrationSpec(
        key="reinvent4",
        name="REINVENT4",
        purpose="RL molecule generation",
        url="https://github.com/MolecularAI/REINVENT4",
        python_modules=("reinvent", "reinvent_models"),
        submodule_path="external/REINVENT4",
    ),
    "molecular_design": IntegrationSpec(
        key="molecular_design",
        name="molecular-design",
        purpose="Generative pipeline (multi-model)",
        url="https://github.com/GT4SD/molecular-design",
        python_modules=("molecular_design",),
        submodule_path="external/molecular-design",
    ),
    "gt4sd_core": IntegrationSpec(
        key="gt4sd_core",
        name="gt4sd-core",
        purpose="Generative models framework",
        url="https://github.com/GT4SD/gt4sd-core",
        python_modules=("gt4sd",),
        submodule_path="external/gt4sd-core",
    ),
    "rdkit": IntegrationSpec(
        key="rdkit",
        name="RDKit",
        purpose="Cheminformatics toolkit",
        url="https://github.com/rdkit/rdkit",
        python_modules=("rdkit",),
        submodule_path="external/rdkit",
    ),
    "molformer": IntegrationSpec(
        key="molformer",
        name="Molformer",
        purpose="Molecular transformer (advanced)",
        url="https://github.com/IBM/molformer",
        python_modules=("molformer", "transformers"),
        submodule_path="external/molformer",
    ),
    "moses": IntegrationSpec(
        key="moses",
        name="MOSES",
        purpose="Benchmarking (molecule quality)",
        url="https://github.com/molecularsets/moses",
        python_modules=("moses",),
        submodule_path="external/moses",
    ),
    "guacamol": IntegrationSpec(
        key="guacamol",
        name="GuacaMol",
        purpose="Benchmark tasks (drug design)",
        url="https://github.com/BenevolentAI/guacamol",
        python_modules=("guacamol",),
        submodule_path="external/guacamol",
    ),
    "molecularai_org": IntegrationSpec(
        key="molecularai_org",
        name="MolecularAI organization",
        purpose="AstraZeneca AI chemistry org (collection)",
        url="https://github.com/MolecularAI",
        is_collection=True,
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _gitmodules_paths() -> set[str]:
    gitmodules = _repo_root() / ".gitmodules"
    if not gitmodules.exists():
        return set()
    paths: set[str] = set()
    for line in gitmodules.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("path = "):
            paths.add(stripped.split("=", 1)[1].strip())
    return paths


def local_checkout_present(submodule_path: str | None) -> bool:
    if not submodule_path:
        return False
    full = _repo_root() / submodule_path
    if not full.exists() or not full.is_dir():
        return False
    try:
        next(full.iterdir())
    except StopIteration:
        return False
    return True


def ensure_local_checkout_on_path(integration_key: str) -> None:
    spec = INTEGRATIONS[integration_key]
    if not spec.submodule_path:
        return
    checkout = _repo_root() / spec.submodule_path
    if checkout.exists() and checkout.is_dir():
        checkout_str = str(checkout)
        if checkout_str not in sys.path:
            sys.path.insert(0, checkout_str)


def resolve_importable_module(integration_key: str) -> str | None:
    spec = INTEGRATIONS[integration_key]
    ensure_local_checkout_on_path(integration_key)
    for module_name in spec.python_modules:
        if importlib.util.find_spec(module_name) is not None:
            return module_name
    return None


def get_integration_status(integration_key: str) -> IntegrationStatus:
    spec = INTEGRATIONS[integration_key]
    module_name = resolve_importable_module(integration_key)
    registered_paths = _gitmodules_paths()
    return IntegrationStatus(
        key=spec.key,
        name=spec.name,
        purpose=spec.purpose,
        url=spec.url,
        submodule_path=spec.submodule_path,
        submodule_registered=(spec.submodule_path in registered_paths if spec.submodule_path else False),
        local_checkout_present=local_checkout_present(spec.submodule_path),
        python_module=module_name,
        importable=module_name is not None,
        is_collection=spec.is_collection,
    )


def get_all_integration_statuses() -> list[IntegrationStatus]:
    return [get_integration_status(key) for key in INTEGRATIONS]
