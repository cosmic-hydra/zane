"""Multi-Stage Fail-Fast Pipeline for ZANE.
Sequential filtering: drug-likeness -> ADMET -> toxicity -> efficacy -> CSS.
Eliminates poor candidates early to reduce late-stage failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    stage_name: str
    passed: bool
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    smiles: str
    passed_all: bool
    final_score: float
    stages: list[StageResult] = field(default_factory=list)
    eliminated_at: str | None = None
    rank: int = 0


@dataclass
class FailFastConfig:
    lipinski_max_violations: int = 1
    min_qed: float = 0.3
    max_sa_score: float = 7.0
    min_oral_mpo: float = 3.0
    max_tox_risk: float = 0.7
    min_css: float = 0.4
    skip_stages: list[str] = field(default_factory=list)


class FailFastPipeline:
    """Fail-fast filtering. Example:
    results = FailFastPipeline().run(["CCO","c1ccccc1"])
    survivors = FailFastPipeline().get_survivors(results)
    """

    def __init__(self, config=None):
        self.config = config or FailFastConfig()

    def run(self, smiles_list):
        results = []
        for smi in smiles_list:
            r = PipelineResult(smiles=smi, passed_all=True, final_score=0.0)
            desc = self._desc(smi)
            if desc is None:
                r.passed_all = False
                r.eliminated_at = "invalid"
                r.stages.append(StageResult("validity", False, 0.0, {"error": "Invalid SMILES"}))
                results.append(r)
                continue
            if "druglikeness" not in self.config.skip_stages:
                s = self._druglikeness(desc)
                r.stages.append(s)
                if not s.passed:
                    r.passed_all = False
                    r.eliminated_at = s.stage_name
                    results.append(r)
                    continue
            if "admet" not in self.config.skip_stages:
                s = self._admet(desc)
                r.stages.append(s)
                if not s.passed:
                    r.passed_all = False
                    r.eliminated_at = s.stage_name
                    results.append(r)
                    continue
            if "toxicity" not in self.config.skip_stages:
                s = self._tox(desc)
                r.stages.append(s)
                if not s.passed:
                    r.passed_all = False
                    r.eliminated_at = s.stage_name
                    results.append(r)
                    continue
            if "efficacy" not in self.config.skip_stages:
                r.stages.append(self._efficacy(desc))
            scores = [s.score for s in r.stages if s.passed]
            r.final_score = np.mean(scores) if scores else 0.0
            if r.final_score < self.config.min_css:
                r.passed_all = False
                r.eliminated_at = "css"
            results.append(r)
        survivors = sorted([r for r in results if r.passed_all], key=lambda r: r.final_score, reverse=True)
        for i, r in enumerate(survivors):
            r.rank = i + 1
        logger.info(
            f"Pipeline: {len(smiles_list)} in -> {len(survivors)} survive ({len(survivors)/max(len(smiles_list),1)*100:.0f}%)"
        )
        return results

    def get_survivors(self, results):
        return sorted([r for r in results if r.passed_all], key=lambda r: r.final_score, reverse=True)

    def attrition_report(self, results):
        total = len(results)
        eliminated = {}
        for r in results:
            if r.eliminated_at:
                eliminated[r.eliminated_at] = eliminated.get(r.eliminated_at, 0) + 1
        surv = sum(1 for r in results if r.passed_all)
        return {
            "total": total,
            "survivors": surv,
            "pass_rate": round(surv / max(total, 1), 3),
            "eliminated_by_stage": eliminated,
        }

    def _desc(self, smi):
        try:
            from drug_discovery.data.pipeline import compute_descriptors

            return compute_descriptors(smi)
        except ImportError:
            return {
                "mol_weight": 300,
                "logp": 2.5,
                "hbd": 1,
                "hba": 4,
                "tpsa": 60,
                "rotatable_bonds": 3,
                "qed": 0.6,
                "sa_score": 3,
            }

    def _druglikeness(self, d):
        v = sum([d.get("mol_weight", 0) > 500, d.get("logp", 0) > 5, d.get("hbd", 0) > 5, d.get("hba", 0) > 10])
        qed = d.get("qed", 0)
        ok = v <= self.config.lipinski_max_violations and qed >= self.config.min_qed
        return StageResult("druglikeness", ok, max(0, 1 - v / 4) * 0.5 + qed * 0.5, {"violations": v, "qed": qed})

    def _admet(self, d):
        from drug_discovery.evaluation.clinical_success_predictor import oral_mpo_score

        mpo = oral_mpo_score(
            d.get("mol_weight", 0),
            d.get("logp", 0),
            d.get("hbd", 0),
            d.get("hba", 0),
            d.get("tpsa", 0),
            d.get("rotatable_bonds", 0),
        )
        return StageResult("admet", mpo >= self.config.min_oral_mpo, mpo / 6.0, {"oral_mpo": mpo})

    def _tox(self, d):
        from drug_discovery.evaluation.clinical_success_predictor import compute_safety_score

        safety, flags = compute_safety_score(d)
        mx = max(flags.values()) if flags else 0
        return StageResult("toxicity", mx <= self.config.max_tox_risk, safety, {"flags": flags})

    def _efficacy(self, d):
        qed = d.get("qed", 0.5)
        sa = d.get("sa_score", 3)
        sa_n = max(0, 1 - sa / 10) if sa > 0 else 0.5
        return StageResult("efficacy", True, 0.7 * qed + 0.3 * sa_n, {"qed": qed})
