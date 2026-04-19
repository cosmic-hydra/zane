"""ZANE Models — Unified model registry with 2026 Q2 SOTA architectures."""

from drug_discovery.models.equivariant_gnn import (
    EquivariantGNN, EquivariantGNNConfig, GaussianRBF, CosineCutoff,
    EGNNLayer, SchNetLayer, build_radius_graph)
from drug_discovery.models.diffusion_generator import (
    MolecularDiffusionModel, DiffusionMoleculeGenerator, DiffusionConfig)

MODEL_REGISTRY = {
    "egnn": {"class": EquivariantGNN, "config": EquivariantGNNConfig, "variant": "egnn"},
    "schnet": {"class": EquivariantGNN, "config": EquivariantGNNConfig, "variant": "schnet"},
    "diffusion": {"class": MolecularDiffusionModel, "config": DiffusionConfig, "variant": None},
}

try:
    from drug_discovery.models.gnn import GNNModel
    MODEL_REGISTRY["gnn"] = {"class": GNNModel, "config": None, "variant": None}
except ImportError: pass
try:
    from drug_discovery.models.transformer import TransformerModel
    MODEL_REGISTRY["transformer"] = {"class": TransformerModel, "config": None, "variant": None}
except ImportError: pass

def get_model(name, **kwargs):
    """Instantiate a model by name: egnn, schnet, diffusion, gnn, transformer."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(sorted(MODEL_REGISTRY))}")
    entry = MODEL_REGISTRY[name]
    if entry["config"]:
        kw = kwargs.copy()
        if entry["variant"]: kw.setdefault("variant", entry["variant"])
        cfg = entry["config"](**{k:v for k,v in kw.items() if hasattr(entry["config"], k)})
        return entry["class"](cfg)
    return entry["class"](**kwargs)

def list_models(): return sorted(MODEL_REGISTRY.keys())
