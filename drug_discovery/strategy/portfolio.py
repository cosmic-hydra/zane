"""Program-level strategy engine that combines discovery and manufacturing views."""

from __future__ import annotations

from typing import Any

from .manufacturing import ManufacturingStrategyPlanner
from .tpp import TPPScorer, TargetProductProfile


class ProgramStrategyEngine:
    """Prioritizes candidates for preclinical progression and manufacturing readiness."""

    def __init__(self, tpp: TargetProductProfile | None = None):
        self.tpp_scorer = TPPScorer(tpp=tpp)
        self.mfg = ManufacturingStrategyPlanner()

    def evaluate_candidates(self, smiles_list: list[str], top_k: int = 5) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for smiles in smiles_list:
            profile = self.tpp_scorer.build_profile(smiles)
            tpp = self.tpp_scorer.score(profile)
            mfg = self.mfg.plan(smiles).as_dict()

            readiness = (
                0.45 * tpp["tpp_score"]
                + 0.2 * (1.0 - mfg["process_risk"])
                + 0.2 * (1.0 - mfg["cogs_index"])
                + 0.15 * mfg["green_chemistry_score"]
            )

            rows.append(
                {
                    "smiles": smiles,
                    "tpp": tpp,
                    "manufacturing": mfg,
                    "program_readiness": round(max(0.0, min(1.0, readiness)), 4),
                    "go_for_scale_up": bool(readiness >= 0.62 and mfg["route_steps"] <= 7),
                }
            )

        rows.sort(key=lambda item: item["program_readiness"], reverse=True)
        top = rows[: max(1, top_k)]

        return {
            "success": True,
            "strategy": "discovery_to_manufacturing",
            "num_input_candidates": len(smiles_list),
            "selected": top,
            "selection_policy": {
                "primary_metric": "program_readiness",
                "go_threshold": 0.62,
                "max_route_steps": 7,
            },
        }


# Simple compatibility facade expected by tests
class PortfolioOptimizer:
    def __init__(self):
        self.engine = ProgramStrategyEngine()

    def optimize(self, smiles_list: list[str], top_k: int = 5):
        return self.engine.evaluate_candidates(smiles_list, top_k=top_k)
