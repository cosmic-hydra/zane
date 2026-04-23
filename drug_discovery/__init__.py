"""ZANE — AI-native Drug Discovery Platform."""

__version__ = "2026.4.1"
__all__ = ["__version__"]

try:
    from drug_discovery.pipeline import DrugDiscoveryPipeline

    __all__.append("DrugDiscoveryPipeline")
except Exception:
    # Keep imports lazy when optional dependencies (e.g., torch-geometric) are unavailable.
    pass

try:
    from drug_discovery.drugmaking import (
        CustomDrugmakingModule,
        CompoundTestResult,
        CandidateResult,
        OptimizationConfig,
        CounterSubstanceFinder,
        CounterSubstanceResult,
    )

    __all__.extend([
        "CustomDrugmakingModule",
        "CompoundTestResult",
        "CandidateResult",
        "OptimizationConfig",
        "CounterSubstanceFinder",
        "CounterSubstanceResult",
    ])
except Exception:
    # Keep drugmaking module lazy when dependencies are unavailable.
    pass

# ── Module 2: Quantum Machine Learning Engine ────────────────────────────────
try:
    from drug_discovery.qml import (
        ActiveSpaceApproximator,
        ActiveSpaceResult,
        VQECircuit,
        VQEResult,
        ZeroNoiseExtrapolation,
        ZNEResult,
        ErrorMitigationConfig,
        QuantumDriver,
        LocalSimulator,
        AWSBraketDriver,
    )

    __all__.extend([
        "ActiveSpaceApproximator",
        "ActiveSpaceResult",
        "VQECircuit",
        "VQEResult",
        "ZeroNoiseExtrapolation",
        "ZNEResult",
        "ErrorMitigationConfig",
        "QuantumDriver",
        "LocalSimulator",
        "AWSBraketDriver",
    ])
except Exception:
    pass

# ── Module 3: Multi-Omics Digital Twin & ADMET Predictor ────────────────────
try:
    from drug_discovery.multi_omics import (
        SingleCellLoader,
        SpatialTranscriptomicsLoader,
        CellData,
        HeterogeneousGraph,
        GraphNode,
        GraphEdge,
        DrugTargetInteraction,
        ADMETPredictor,
        ADMETProfile,
        ADMETConfig,
    )

    __all__.extend([
        "SingleCellLoader",
        "SpatialTranscriptomicsLoader",
        "CellData",
        "HeterogeneousGraph",
        "GraphNode",
        "GraphEdge",
        "DrugTargetInteraction",
        "ADMETPredictor",
        "ADMETProfile",
        "ADMETConfig",
    ])
except Exception:
    pass

# ── Module 4: 4D Geometric Deep Learning & FEP ────────────────────────────────
try:
    from drug_discovery.geometric_dl import (
        SE3Transformer,
        SE3EquivariantBlock,
        BindingFreeEnergyCalculator,
        FEPConfig,
        FEPResult,
        OpenMMDriver,
        TransientPocketPredictor,
        PocketPrediction,
    )

    __all__.extend([
        "SE3Transformer",
        "SE3EquivariantBlock",
        "BindingFreeEnergyCalculator",
        "FEPConfig",
        "FEPResult",
        "OpenMMDriver",
        "TransientPocketPredictor",
        "PocketPrediction",
    ])
except Exception:
    pass

# ── Module 5: Target-Aware 3D Molecular Diffusion ────────────────────────────
try:
    from drug_discovery.diffusion import (
        EquivariantDiffusionModel,
        DiffusionConfig,
        DiffusionResult,
        PocketAwareGenerator,
        PocketContext,
        GeneratedMolecule,
    )

    __all__.extend([
        "EquivariantDiffusionModel",
        "DiffusionConfig",
        "DiffusionResult",
        "PocketAwareGenerator",
        "PocketContext",
        "GeneratedMolecule",
    ])
except Exception:
    pass

# ── Module 6: Active Learning Brain & Bayesian Optimization ─────────────────
try:
    from drug_discovery.active_learning import (
        GaussianProcessSurrogate,
        SurrogateConfig,
        ExpectedImprovement,
        UpperConfidenceBound,
        ThompsonSampling,
        BayesianOptimizer,
        MultiFidelityOptimizer,
        ResourceAllocator,
        OptimizationResult,
        ResourceBudget,
    )

    __all__.extend([
        "GaussianProcessSurrogate",
        "SurrogateConfig",
        "ExpectedImprovement",
        "UpperConfidenceBound",
        "ThompsonSampling",
        "BayesianOptimizer",
        "MultiFidelityOptimizer",
        "ResourceAllocator",
        "OptimizationResult",
        "ResourceBudget",
    ])
except Exception:
    pass
