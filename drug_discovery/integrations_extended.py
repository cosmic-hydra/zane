"""Extended External Tool Integrations for ZANE — 10 cutting-edge tools."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IntegrationStatus:
    name: str
    available: bool
    version: str = ""
    install_cmd: str = ""
    description: str = ""
    category: str = ""


EXTENDED_INTEGRATIONS = {
    "unimol": {
        "import": "unimol_tools",
        "install": "pip install unimol_tools",
        "description": "Unified molecular representation learning",
        "category": "representation",
    },
    "esm": {
        "import": "esm",
        "install": "pip install fair-esm",
        "description": "Meta ESM-2 protein language model",
        "category": "protein",
    },
    "esmfold": {
        "import": "esm.esmfold.v1",
        "install": "pip install fair-esm[esmfold]",
        "description": "ESMFold structure prediction",
        "category": "protein",
    },
    "mace": {
        "import": "mace",
        "install": "pip install mace-torch",
        "description": "MACE equivariant interatomic potentials",
        "category": "simulation",
    },
    "boltz": {
        "import": "boltz",
        "install": "pip install boltz",
        "description": "Boltz-1 protein structure prediction",
        "category": "protein",
    },
    "py3dmol": {
        "import": "py3Dmol",
        "install": "pip install py3Dmol",
        "description": "3D molecular visualization",
        "category": "visualization",
    },
    "torchani": {
        "import": "torchani",
        "install": "pip install torchani",
        "description": "ANI neural network potentials",
        "category": "simulation",
    },
    "deepchem": {
        "import": "deepchem",
        "install": "pip install deepchem",
        "description": "Deep learning for drug discovery",
        "category": "framework",
    },
    "chemprop": {
        "import": "chemprop",
        "install": "pip install chemprop",
        "description": "D-MPNN property prediction",
        "category": "model",
    },
    "torchdrug": {
        "import": "torchdrug",
        "install": "pip install torchdrug",
        "description": "GNN-based drug discovery platform",
        "category": "framework",
    },
}


def check_integration(name):
    info = EXTENDED_INTEGRATIONS.get(name, {})
    if not info:
        return IntegrationStatus(name=name, available=False)
    try:
        mod = importlib.import_module(info["import"])
        return IntegrationStatus(
            name=name,
            available=True,
            version=getattr(mod, "__version__", "?"),
            install_cmd=info["install"],
            description=info["description"],
            category=info.get("category", ""),
        )
    except ImportError:
        return IntegrationStatus(
            name=name,
            available=False,
            install_cmd=info["install"],
            description=info["description"],
            category=info.get("category", ""),
        )


def check_all_integrations():
    return {n: check_integration(n) for n in EXTENDED_INTEGRATIONS}


def integration_report():
    statuses = check_all_integrations()
    lines = ["ZANE Extended Integration Status", "=" * 50]
    for cat in sorted(set(s.category for s in statuses.values())):
        lines.append(f"\n[{cat.upper()}]")
        for name, s in sorted(statuses.items()):
            if s.category == cat:
                icon = "Y" if s.available else "N"
                lines.append(f"  [{icon}] {name} — {s.description}")
                if not s.available:
                    lines.append(f"      Install: {s.install_cmd}")
    return "\n".join(lines)
