"""ZANE Models — Molecular Graph Neural Networks and Transformers."""

MODEL_REGISTRY = {}

try:
    from drug_discovery.models.equivariant_gnn import (
        CosineCutoff as CosineCutoff,
    )
    from drug_discovery.models.equivariant_gnn import (
        EGNNLayer as EGNNLayer,
    )
    from drug_discovery.models.equivariant_gnn import (
        EquivariantGNN as EquivariantGNN,
    )
    from drug_discovery.models.equivariant_gnn import (
        EquivariantGNNConfig as EquivariantGNNConfig,
    )
    from drug_discovery.models.equivariant_gnn import (
        GaussianRBF as GaussianRBF,
    )
    from drug_discovery.models.equivariant_gnn import (
        SchNetLayer as SchNetLayer,
    )
    from drug_discovery.models.equivariant_gnn import (
        build_radius_graph as build_radius_graph,
    )

    MODEL_REGISTRY["equivariant_gnn"] = {
        "class": EquivariantGNN,
        "config": EquivariantGNNConfig,
        "variant": None,
    }
except ImportError:
    pass

try:
    from drug_discovery.models.diffusion_generator import (
        DiffusionConfig as DiffusionConfig,
    )
    from drug_discovery.models.diffusion_generator import (
        DiffusionMoleculeGenerator as DiffusionMoleculeGenerator,
    )
    from drug_discovery.models.diffusion_generator import (
        MolecularDiffusionModel as MolecularDiffusionModel,
    )

    MODEL_REGISTRY["diffusion"] = {
        "class": MolecularDiffusionModel,
        "config": DiffusionConfig,
        "variant": None,
    }
except ImportError:
    pass

try:
    from drug_discovery.models.gflownet import (
        GFlowNetConfig as GFlowNetConfig,
    )
    from drug_discovery.models.gflownet import (
        GFlowNetPolicy as GFlowNetPolicy,
    )
    from drug_discovery.models.gflownet import (
        GFlowNetTrainer as GFlowNetTrainer,
    )
    from drug_discovery.models.gflownet import (
        PhysicsRewardConfig as PhysicsRewardConfig,
    )
    from drug_discovery.models.gflownet import (
        PhysicsRewardFunction as PhysicsRewardFunction,
    )

    MODEL_REGISTRY["gflownet"] = {
        "class": GFlowNetPolicy,
        "config": GFlowNetConfig,
        "variant": None,
    }
except ImportError:
    pass

try:
    from drug_discovery.models.gnn import MolecularGNN as MolecularGNN
    from drug_discovery.models.gnn import MolecularMPNN as MolecularMPNN
    from drug_discovery.models.gnn import MolecularGIN as MolecularGIN

    MODEL_REGISTRY["gnn"] = {"class": MolecularGNN, "config": None, "variant": None}
    MODEL_REGISTRY["gin"] = {"class": MolecularGIN, "config": None, "variant": None}
except ImportError:
    pass

try:
    from drug_discovery.models.transformer import MolecularTransformer as MolecularTransformer
    from drug_discovery.models.transformer import ModernMolecularTransformer as ModernMolecularTransformer

    MODEL_REGISTRY["transformer"] = {
        "class": MolecularTransformer,
        "config": None,
        "variant": None,
    }
    MODEL_REGISTRY["modern_transformer"] = {
        "class": ModernMolecularTransformer,
        "config": None,
        "variant": None,
    }
except ImportError:
    pass

try:
    from drug_discovery.models.ensemble import (
        EnsembleModel as EnsembleModel,
    )
    from drug_discovery.models.ensemble import (
        HybridModel as HybridModel,
    )
    from drug_discovery.models.ensemble import (
        MultiTaskModel as MultiTaskModel,
    )
except ImportError:
    pass

__all__ = [
    "MODEL_REGISTRY",
    "MolecularGNN",
    "MolecularMPNN",
    "MolecularGIN",
    "MolecularTransformer",
    "EnsembleModel",
    "HybridModel",
    "MultiTaskModel",
    "EquivariantGNN",
    "EquivariantGNNConfig",
    "EGNNLayer",
    "SchNetLayer",
    "GaussianRBF",
    "CosineCutoff",
    "build_radius_graph",
    "MolecularDiffusionModel",
    "DiffusionConfig",
    "DiffusionMoleculeGenerator",
    "GFlowNetPolicy",
    "GFlowNetConfig",
    "GFlowNetTrainer",
    "PhysicsRewardFunction",
    "PhysicsRewardConfig",
]
