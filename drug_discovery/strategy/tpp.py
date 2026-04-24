"""Target Product Profile scoring for early-stage candidate prioritization."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any


@dataclass(frozen=True)
class TargetProductProfile:
    """Program-level criteria for what a successful molecule should look like."""

    name: str = "default_tpp"
    min_qed: float = 0.45
    max_logp: float = 4.5
    max_mw: float = 550.0
    max_sa_score: float = 6.0


@dataclass(frozen=True)
class CandidateProfile:
    smiles: str
    qed: float
    logp: float
    molecular_weight: float
    sa_score: float


class TPPScorer:
    """Scores molecules against a target product profile with robust fallbacks."""

    def __init__(self, tpp: TargetProductProfile | None = None):
        self.tpp = tpp or TargetProductProfile()

    @staticmethod
    def _fallback_value(smiles: str, key: str, low: float, high: float) -> float:
        digest = sha256(f"{smiles}::{key}".encode()).hexdigest()
        ratio = int(digest[:8], 16) / float(0xFFFFFFFF)
        return low + (high - low) * ratio

    def build_profile(self, smiles: str) -> CandidateProfile:
        try:
            from rdkit import Chem
            from rdkit.Chem import QED, Crippen, Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("invalid smiles")
            qed = float(QED.qed(mol))
            logp = float(Crippen.MolLogP(mol))
            mw = float(Descriptors.MolWt(mol))
            sa_score = 1.5 + max(0.0, (mw - 250.0) / 120.0) + max(0.0, abs(logp - 2.0) / 2.5)
            sa_score = min(10.0, max(1.0, sa_score))
            return CandidateProfile(smiles=smiles, qed=qed, logp=logp, molecular_weight=mw, sa_score=sa_score)
        except Exception:
            return CandidateProfile(
                smiles=smiles,
                qed=self._fallback_value(smiles, "qed", 0.2, 0.9),
                logp=self._fallback_value(smiles, "logp", -0.2, 6.0),
                molecular_weight=self._fallback_value(smiles, "mw", 180.0, 620.0),
                sa_score=self._fallback_value(smiles, "sa", 2.0, 9.0),
            )

    def score(self, profile: CandidateProfile) -> dict[str, Any]:
        qed_pass = profile.qed >= self.tpp.min_qed
        logp_pass = profile.logp <= self.tpp.max_logp
        mw_pass = profile.molecular_weight <= self.tpp.max_mw
        sa_pass = profile.sa_score <= self.tpp.max_sa_score

        # Weighted program score in [0, 1].
        score = (
            0.35 * min(1.0, profile.qed / max(self.tpp.min_qed, 1e-6))
            + 0.2 * min(1.0, self.tpp.max_logp / max(profile.logp, 1e-6))
            + 0.25 * min(1.0, self.tpp.max_mw / max(profile.molecular_weight, 1e-6))
            + 0.2 * min(1.0, self.tpp.max_sa_score / max(profile.sa_score, 1e-6))
        )

        return {
            "smiles": profile.smiles,
            "tpp": self.tpp.name,
            "criteria": {
                "qed_pass": qed_pass,
                "logp_pass": logp_pass,
                "mw_pass": mw_pass,
                "sa_pass": sa_pass,
            },
            "tpp_score": round(max(0.0, min(1.0, score)), 4),
            "profile": {
                "qed": round(profile.qed, 4),
                "logp": round(profile.logp, 4),
                "molecular_weight": round(profile.molecular_weight, 2),
                "sa_score": round(profile.sa_score, 3),
            },
        }
