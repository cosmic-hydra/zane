"""
Custom Drugmaking Process Module for ZANE.

This module provides end-to-end drug design capabilities:
- Generation of novel compounds from scratch
- Effectiveness and toxicity testing
- Multi-objective optimization to balance success and safety
- Counter-substance finder for risk mitigation

Example usage::

    from drug_discovery.drugmaking import CustomDrugmakingModule, CounterSubstanceFinder

    # Create the drugmaking module
    module = CustomDrugmakingModule()

    # Run end-to-end optimization
    result = module.run_end_to_end(target_objectives=["potency", "safety"])

    # Find counter-substances for risk mitigation
    finder = CounterSubstanceFinder()
    counter_substances = finder.find_counter_substances(drug_smiles="CCO")
"""

from __future__ import annotations

from drug_discovery.drugmaking.process import (
    CandidateResult,
    CompoundTestResult,
    CustomDrugmakingModule,
    OptimizationConfig,
)
from drug_discovery.drugmaking.risk_mitigation import (
    CounterSubstanceFinder,
    CounterSubstanceResult,
)

__all__ = [
    "CustomDrugmakingModule",
    "CompoundTestResult",
    "CandidateResult",
    "OptimizationConfig",
    "CounterSubstanceFinder",
    "CounterSubstanceResult",
]
