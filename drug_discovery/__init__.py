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
