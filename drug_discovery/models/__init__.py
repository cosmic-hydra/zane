"""ZANE Models — Unified registry with guarded imports and legacy compat."""

import logging
logger = logging.getLogger(__name__)
MODEL_REGISTRY = {}

try:
    from drug_discovery.models.equivariant_gnn import (
        EquivariantGNN, EquivariantGNNConfig, GaussianRBF, CosineCutoff,
        EGNNLayer, SchNetLayer, build_radius_graph)
    MODEL_REGISTRY["egnn"] = {"class": EquivariantGNN, "config": EquivariantGNNConfig, "variant": "egnn"}
    MODEL_REGISTRY["schnet"] = {"class": EquivariantGNN, "config": EquivariantGNNConfig, "variant": "schnet"}
except ImportError as e:
    logger.debug(f"Equivariant GNN not available: {e}")

try:
    from drug_discovery.models.diffusion_generator import (
        MolecularDiffusionModel, DiffusionMoleculeGenerator, DiffusionConfig)
    MODEL_REGISTRY["diffusion"] = {"class": MolecularDiffusionModel, "config": DiffusionConfig, "variant": None}
except ImportError as e:
    logger.debug(f"Diffusion generator not available: {e}")

try:
    from drug_discovery.models.gflownet import GFlowNetPolicy, GFlowNetTrainer, GFlowNetConfig
    MODEL_REGISTRY["gflownet"] = {"class": GFlowNetPolicy, "config": GFlowNetConfig, "variant": None}
except ImportError as e:
    logger.debug(f"GFlowNet not available: {e}")

try:
    from drug_discovery.models.gnn import MolecularGNN, MolecularMPNN
    MODEL_REGISTRY["gnn"] = {"class": MolecularGNN, "config": None, "variant": None}
except ImportError:
    pass
try:
    from drug_discovery.models.transformer import MolecularTransformer
    MODEL_REGISTRY["transformer"] = {"class": MolecularTransformer, "config": None, "variant": None}
except ImportError:
    pass
try:
    from drug_discovery.models.ensemble import EnsembleModel, MultiTaskModel, HybridModel
except ImportError:
    pass
try:
    from drug_discovery.models.gnn import GNNModel
    MODEL_REGISTRY["legacy_gnn"] = {"class": GNNModel, "config": None, "variant": None}
except ImportError:
    pass
try:
    from drug_discovery.models.transformer import TransformerModel
    MODEL_REGISTRY["legacy_transformer"] = {"class": TransformerModel, "config": None, "variant": None}
except ImportError:
    pass


__all__ = ["get_model", "list_models", "MODEL_REGISTRY", "MolecularGNN", "MolecularMPNN", "MolecularTransformer", "SMILESTransformer", "EnsembleModel", "MultiTaskModel", "HybridModel"]

try:
    from drug_discovery.models.gnn import MolecularGNN, MolecularMPNN
except ImportError:
    pass
try:
    from drug_discovery.models.transformer import MolecularTransformer, SMILESTransformer
except ImportError:
    pass
try:
    from drug_discovery.models.ensemble import EnsembleModel, MultiTaskModel, HybridModel
except ImportError:
    pass

def get_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(sorted(MODEL_REGISTRY.keys()))}")
    entry = MODEL_REGISTRY[name]
    if entry["config"]:
        kw = kwargs.copy()
        if entry.get("variant"): kw.setdefault("variant", entry["variant"])
        valid = set(entry["config"].__dataclass_fields__) if hasattr(entry["config"], "__dataclass_fields__") else set()
        filtered = {k: v for k, v in kw.items() if k in valid} if valid else kw
        return entry["class"](entry["config"](**filtered))
    return entry["class"](**kwargs)

def list_models(): return sorted(MODEL_REGISTRY.keys())

__all__ = ["get_model", "list_models", "MODEL_REGISTRY"]
