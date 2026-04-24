"""ZANE — AI-native Drug Discovery Platform."""

__version__ = "2026.4.1"
__all__ = ["__version__"]

try:
    from drug_discovery.pipeline import DrugDiscoveryPipeline as DrugDiscoveryPipeline

    __all__.append("DrugDiscoveryPipeline")
except Exception:
    # Keep imports lazy when optional dependencies (e.g., torch-geometric) are unavailable.
    pass

try:
    from drug_discovery.drugmaking import (
        CandidateResult as CandidateResult,
    )
    from drug_discovery.drugmaking import (
        CompoundTestResult as CompoundTestResult,
    )
    from drug_discovery.drugmaking import (
        CounterSubstanceFinder as CounterSubstanceFinder,
    )
    from drug_discovery.drugmaking import (
        CounterSubstanceResult as CounterSubstanceResult,
    )
    from drug_discovery.drugmaking import (
        CustomDrugmakingModule as CustomDrugmakingModule,
    )
    from drug_discovery.drugmaking import (
        OptimizationConfig as OptimizationConfig,
    )

    __all__.extend(
        [
            "CustomDrugmakingModule",
            "CompoundTestResult",
            "CandidateResult",
            "OptimizationConfig",
            "CounterSubstanceFinder",
            "CounterSubstanceResult",
        ]
    )
except Exception:
    # Keep drugmaking module lazy when dependencies are unavailable.
    pass

# ── Module 2: Quantum Machine Learning Engine ────────────────────────────────
try:
    from drug_discovery.qml import (
        ActiveSpaceApproximator as ActiveSpaceApproximator,
    )
    from drug_discovery.qml import (
        ActiveSpaceResult as ActiveSpaceResult,
    )
    from drug_discovery.qml import (
        AWSBraketDriver as AWSBraketDriver,
    )
    from drug_discovery.qml import (
        ErrorMitigationConfig as ErrorMitigationConfig,
    )
    from drug_discovery.qml import (
        LocalSimulator as LocalSimulator,
    )
    from drug_discovery.qml import (
        QuantumDriver as QuantumDriver,
    )
    from drug_discovery.qml import (
        VQECircuit as VQECircuit,
    )
    from drug_discovery.qml import (
        VQEResult as VQEResult,
    )
    from drug_discovery.qml import (
        ZeroNoiseExtrapolation as ZeroNoiseExtrapolation,
    )
    from drug_discovery.qml import (
        ZNEResult as ZNEResult,
    )

    __all__.extend(
        [
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
        ]
    )
except Exception:
    pass

# ── Module 3: Multi-Omics Digital Twin & ADMET Predictor ────────────────────
try:
    from drug_discovery.multi_omics import (
        ADMETConfig as ADMETConfig,
    )
    from drug_discovery.multi_omics import (
        ADMETPredictor as ADMETPredictor,
    )
    from drug_discovery.multi_omics import (
        ADMETProfile as ADMETProfile,
    )
    from drug_discovery.multi_omics import (
        CellData as CellData,
    )
    from drug_discovery.multi_omics import (
        DrugTargetInteraction as DrugTargetInteraction,
    )
    from drug_discovery.multi_omics import (
        GraphEdge as GraphEdge,
    )
    from drug_discovery.multi_omics import (
        GraphNode as GraphNode,
    )
    from drug_discovery.multi_omics import (
        HeterogeneousGraph as HeterogeneousGraph,
    )
    from drug_discovery.multi_omics import (
        SingleCellLoader as SingleCellLoader,
    )
    from drug_discovery.multi_omics import (
        SpatialTranscriptomicsLoader as SpatialTranscriptomicsLoader,
    )

    __all__.extend(
        [
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
        ]
    )
except Exception:
    pass

# ── Module 4: 4D Geometric Deep Learning & FEP ────────────────────────────────
try:
    from drug_discovery.geometric_dl import (
        BindingFreeEnergyCalculator as BindingFreeEnergyCalculator,
    )
    from drug_discovery.geometric_dl import (
        FEPConfig as FEPConfig,
    )
    from drug_discovery.geometric_dl import (
        FEPResult as FEPResult,
    )
    from drug_discovery.geometric_dl import (
        OpenMMDriver as OpenMMDriver,
    )
    from drug_discovery.geometric_dl import (
        PocketPrediction as PocketPrediction,
    )
    from drug_discovery.geometric_dl import (
        SE3EquivariantBlock as SE3EquivariantBlock,
    )
    from drug_discovery.geometric_dl import (
        SE3Transformer as SE3Transformer,
    )
    from drug_discovery.geometric_dl import (
        TransientPocketPredictor as TransientPocketPredictor,
    )

    __all__.extend(
        [
            "SE3Transformer",
            "SE3EquivariantBlock",
            "BindingFreeEnergyCalculator",
            "FEPConfig",
            "FEPResult",
            "OpenMMDriver",
            "TransientPocketPredictor",
            "PocketPrediction",
        ]
    )
except Exception:
    pass

# ── Module 5: Target-Aware 3D Molecular Diffusion ────────────────────────────
try:
    from drug_discovery.diffusion import (
        DiffusionConfig as DiffusionConfig,
    )
    from drug_discovery.diffusion import (
        DiffusionResult as DiffusionResult,
    )
    from drug_discovery.diffusion import (
        EquivariantDiffusionModel as EquivariantDiffusionModel,
    )
    from drug_discovery.diffusion import (
        GeneratedMolecule as GeneratedMolecule,
    )
    from drug_discovery.diffusion import (
        PocketAwareGenerator as PocketAwareGenerator,
    )
    from drug_discovery.diffusion import (
        PocketContext as PocketContext,
    )

    __all__.extend(
        [
            "EquivariantDiffusionModel",
            "DiffusionConfig",
            "DiffusionResult",
            "PocketAwareGenerator",
            "PocketContext",
            "GeneratedMolecule",
        ]
    )
except Exception:
    pass

# ── Module 6: Active Learning Brain & Bayesian Optimization ─────────────────
try:
    from drug_discovery.active_learning import (
        BayesianOptimizer as BayesianOptimizer,
    )
    from drug_discovery.active_learning import (
        ExpectedImprovement as ExpectedImprovement,
    )
    from drug_discovery.active_learning import (
        GaussianProcessSurrogate as GaussianProcessSurrogate,
    )
    from drug_discovery.active_learning import (
        MultiFidelityOptimizer as MultiFidelityOptimizer,
    )
    from drug_discovery.active_learning import (
        OptimizationResult as OptimizationResult,
    )
    from drug_discovery.active_learning import (
        ResourceAllocator as ResourceAllocator,
    )
    from drug_discovery.active_learning import (
        ResourceBudget as ResourceBudget,
    )
    from drug_discovery.active_learning import (
        SurrogateConfig as SurrogateConfig,
    )
    from drug_discovery.active_learning import (
        ThompsonSampling as ThompsonSampling,
    )
    from drug_discovery.active_learning import (
        UpperConfidenceBound as UpperConfidenceBound,
    )

    __all__.extend(
        [
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
        ]
    )
except Exception:
    pass

# ── Module 11-14: Apex Orchestrator & Advanced Modules ────────────────────────
try:
    from drug_discovery.agentic import (
        AgenticSwarm as AgenticSwarm,
    )
    from drug_discovery.agentic import (
        INDGenerator as INDGenerator,
    )
    from drug_discovery.apex_orchestrator import ApexOrchestrator as ApexOrchestrator
    from drug_discovery.neuromorphic import (
        NeuromorphicInferenceEngine as NeuromorphicInferenceEngine,
    )
    from drug_discovery.neuromorphic import (
        SNNCompiler as SNNCompiler,
    )
    from drug_discovery.quantum_chemistry import (
        FermiNetSolver as FermiNetSolver,
    )
    from drug_discovery.quantum_chemistry import (
        QEDSandbox as QEDSandbox,
    )

    __all__.extend(
        [
            "ApexOrchestrator",
            "SNNCompiler",
            "NeuromorphicInferenceEngine",
            "FermiNetSolver",
            "QEDSandbox",
            "AgenticSwarm",
            "INDGenerator",
        ]
    )
except Exception:
    pass
