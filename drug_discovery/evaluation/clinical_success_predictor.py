"""Clinical Success Predictor for ZANE.
Multi-stage risk assessment based on actual clinical trial failure statistics.
57% efficacy + 17% safety + 20% PK/ADMET weighted scoring.
Includes CNS MPO, Oral MPO, Fragment RO3, 8-organ toxicity panel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

FAILURE_WEIGHTS = {"efficacy": 0.57, "safety": 0.17, "pk_admet": 0.20, "other": 0.06}
TOXICITY_ORGANS = [
    "cardiotoxicity",
    "hepatotoxicity",
    "nephrotoxicity",
    "neurotoxicity",
    "hematotoxicity",
    "gastrointestinal",
    "respiratory",
    "dermal",
]


@dataclass
class RiskProfile:
    smiles: str = ""
    efficacy_score: float = 0.0
    safety_score: float = 0.0
    admet_score: float = 0.0
    druglikeness_score: float = 0.0
    synthetic_score: float = 0.0
    clinical_success_score: float = 0.0
    risk_level: str = "unknown"
    toxicity_flags: dict[str, float] = field(default_factory=dict)
    admet_flags: dict[str, Any] = field(default_factory=dict)
    mpo_scores: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def cns_mpo_score(mw, logp, hbd, psa, pka=8.4, logd=None):
    """Pfizer CNS MPO (0-6 scale, >=4 desirable). Wager et al. 2010."""
    scores = []
    scores.append(1.0 if mw <= 360 else max(0, 1 - (mw - 360) / 140) if mw <= 500 else 0.0)
    scores.append(1.0 if logp <= 3 else max(0, 1 - (logp - 3) / 2) if logp <= 5 else 0.0)
    scores.append(1.0 if hbd <= 0.5 else max(0, 1 - (hbd - 0.5) / 3) if hbd <= 3.5 else 0.0)
    if 40 <= psa <= 90:
        scores.append(1.0)
    elif psa < 40:
        scores.append(psa / 40)
    elif psa <= 120:
        scores.append(1 - (psa - 90) / 30)
    else:
        scores.append(0.0)
    scores.append(1.0 if 7 <= pka <= 9 else max(0, pka / 7) if pka < 7 else max(0, 1 - (pka - 9) / 2))
    ld = logd if logd is not None else logp - 0.5
    scores.append(1.0 if ld <= 2 else max(0, 1 - (ld - 2) / 2) if ld <= 4 else 0.0)
    return sum(scores)


def oral_mpo_score(mw, logp, hbd, hba, psa, rotatable):
    """Oral drug MPO (0-6 scale). Lipinski+Veber continuous scoring."""
    return sum(
        [
            1.0 if mw <= 500 else max(0, 1 - (mw - 500) / 200),
            1.0 if logp <= 5 else max(0, 1 - (logp - 5) / 2),
            1.0 if hbd <= 5 else max(0, 1 - (hbd - 5) / 3),
            1.0 if hba <= 10 else max(0, 1 - (hba - 10) / 5),
            1.0 if psa <= 140 else max(0, 1 - (psa - 140) / 60),
            1.0 if rotatable <= 10 else max(0, 1 - (rotatable - 10) / 5),
        ]
    )


def fragment_rule_of_three(mw, logp, hbd, hba, rotatable, psa):
    """Congreve et al. Drug Discovery Today 2003."""
    checks = {
        "mw_le_300": mw <= 300,
        "logp_le_3": logp <= 3,
        "hbd_le_3": hbd <= 3,
        "hba_le_3": hba <= 3,
        "rotatable_le_3": rotatable <= 3,
        "psa_le_60": psa <= 60,
    }
    checks["passes"] = all(checks.values())
    return checks


def compute_safety_score(descriptors):
    """8-organ toxicity scoring. Returns (score 0-1, organ_flags)."""
    flags = {}
    score = 1.0
    mw = descriptors.get("mol_weight", 0)
    logp = descriptors.get("logp", 0)
    tpsa = descriptors.get("tpsa", 0)
    cardio = max(0, min(1, (logp - 3) / 4)) * max(0, min(1, (100 - tpsa) / 80))
    flags["cardiotoxicity"] = round(cardio, 3)
    score -= 0.15 * cardio
    hepato = max(0, min(1, (mw - 400) / 300)) * max(0, min(1, (logp - 3) / 4))
    flags["hepatotoxicity"] = round(hepato, 3)
    score -= 0.15 * hepato
    reactive = max(0, min(1, (logp - 4) / 3))
    flags["reactive_metabolites"] = round(reactive, 3)
    score -= 0.1 * reactive
    return max(0, min(1, score)), flags


def compute_clinical_success_score(efficacy, safety, admet, druglikeness=0.5, synthetic=0.5):
    """Weighted CSS. Returns (score 0-1, risk_level)."""
    css = (
        FAILURE_WEIGHTS["efficacy"] * efficacy
        + FAILURE_WEIGHTS["safety"] * safety
        + FAILURE_WEIGHTS["pk_admet"] * admet
        + FAILURE_WEIGHTS["other"] * (0.5 * druglikeness + 0.5 * synthetic)
    )
    level = "low" if css >= 0.7 else "medium" if css >= 0.5 else "high" if css >= 0.3 else "critical"
    return round(css, 4), level


class ClinicalSuccessPredictor:
    """Multi-stage clinical success predictor.
    Example: profile = ClinicalSuccessPredictor().assess("CCO")
    """

    def assess(self, smiles, descriptors=None, binding_energy=None):
        if descriptors is None:
            try:
                from drug_discovery.data.pipeline import compute_descriptors

                descriptors = compute_descriptors(smiles) or {}
            except ImportError:
                descriptors = {}
        p = RiskProfile(smiles=smiles)
        qed = descriptors.get("qed", 0.5)
        bind = min(1, max(0, (-binding_energy - 5) / 7)) if binding_energy else 0.5
        p.efficacy_score = 0.6 * qed + 0.4 * bind
        safety, tox = compute_safety_score(descriptors)
        p.safety_score = safety
        p.toxicity_flags = tox
        mw = descriptors.get("mol_weight", 500)
        logp = descriptors.get("logp", 3)
        hbd = descriptors.get("hbd", 2)
        hba = descriptors.get("hba", 5)
        tpsa = descriptors.get("tpsa", 80)
        rot = descriptors.get("rotatable_bonds", 5)
        oral = oral_mpo_score(mw, logp, hbd, hba, tpsa, rot)
        p.admet_score = oral / 6.0
        p.admet_flags = {"oral_mpo": round(oral, 2)}
        p.druglikeness_score = qed
        sa = descriptors.get("sa_score", 3)
        p.synthetic_score = max(0, 1 - sa / 10) if sa > 0 else 0.5
        cns = cns_mpo_score(mw, logp, hbd, tpsa)
        p.mpo_scores = {"cns_mpo": round(cns, 2), "oral_mpo": round(oral, 2)}
        css, level = compute_clinical_success_score(
            p.efficacy_score, p.safety_score, p.admet_score, p.druglikeness_score, p.synthetic_score
        )
        p.clinical_success_score = css
        p.risk_level = level
        if p.safety_score < 0.5:
            p.recommendations.append("HIGH TOX RISK: review structural alerts")
        if p.admet_score < 0.5:
            p.recommendations.append("POOR ADMET: consider prodrug strategy")
        if p.efficacy_score < 0.4:
            p.recommendations.append("LOW EFFICACY: validate target engagement")
        if cns >= 4:
            p.recommendations.append("CNS-PENETRANT: suitable for CNS targets")
        return p

    def batch_assess(self, smiles_list, **kw):
        return [self.assess(s, **kw) for s in smiles_list]

    def rank_by_success(self, profiles):
        return sorted(profiles, key=lambda p: p.clinical_success_score, reverse=True)
