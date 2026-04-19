"""Drug modeling utilities for candidate scoring and prioritization."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DrugModelingResult:
    """Container for a single candidate's modeling output."""

    smiles: str
    molecular_weight: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    qed: float
    synthetic_accessibility: float
    lipinski_violations: int
    developability_score: float
    recommendation: str

    def to_dict(self) -> dict:
        """Convert to a JSON-safe dictionary."""
        return asdict(self)


class DrugModeler:
    """Descriptor-driven drug modeling and ranking helper."""

    def _build_molecule(self, smiles: str):
        try:
            from rdkit import Chem
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("RDKit is required for drug modeling.") from exc

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return Chem.AddHs(mol)

    @staticmethod
    def _safe_sa_score(mol) -> float:
        """Estimate synthetic accessibility from simple structural heuristics.

        This keeps the module lightweight and deterministic without third-party SA dependencies.
        Lower values are better on a scale near [1, 10].
        """
        from rdkit.Chem import Descriptors, rdMolDescriptors

        rings = rdMolDescriptors.CalcNumRings(mol)
        heavy = mol.GetNumHeavyAtoms()
        rot = Descriptors.NumRotatableBonds(mol)
        aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)

        # Penalize high flexibility/size, reward aromatic scaffolds moderately.
        score = 2.0 + (heavy / 20.0) + (rot / 8.0) + (rings / 10.0) - (aromatic / 12.0)
        return float(max(1.0, min(10.0, score)))

    @staticmethod
    def _lipinski_violations(mw: float, logp: float, hbd: int, hba: int) -> int:
        violations = 0
        violations += int(mw > 500.0)
        violations += int(logp > 5.0)
        violations += int(hbd > 5)
        violations += int(hba > 10)
        return violations

    @staticmethod
    def _recommendation(score: float, violations: int, qed: float) -> str:
        if score >= 0.75 and violations <= 1 and qed >= 0.7:
            return "promote"
        if score >= 0.55 and violations <= 2 and qed >= 0.5:
            return "review"
        return "hold"

    def model_candidate(self, smiles: str) -> DrugModelingResult:
        """Model a candidate and return descriptor + decision outputs."""
        from rdkit.Chem import QED, Crippen, Descriptors, rdMolDescriptors

        mol = self._build_molecule(smiles)

        mw = float(Descriptors.MolWt(mol))
        logp = float(Crippen.MolLogP(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        hbd = int(rdMolDescriptors.CalcNumHBD(mol))
        hba = int(rdMolDescriptors.CalcNumHBA(mol))
        rot = int(Descriptors.NumRotatableBonds(mol))
        qed = float(QED.qed(mol))
        sa = self._safe_sa_score(mol)
        lipinski = self._lipinski_violations(mw=mw, logp=logp, hbd=hbd, hba=hba)

        # Weighted developability score in [0,1]
        sa_component = 1.0 - ((sa - 1.0) / 9.0)
        lip_component = max(0.0, 1.0 - (lipinski / 4.0))
        tpsa_component = 1.0 if 20.0 <= tpsa <= 140.0 else 0.6
        score = float(max(0.0, min(1.0, 0.45 * qed + 0.25 * sa_component + 0.2 * lip_component + 0.1 * tpsa_component)))

        return DrugModelingResult(
            smiles=smiles,
            molecular_weight=mw,
            logp=logp,
            tpsa=tpsa,
            hbd=hbd,
            hba=hba,
            rotatable_bonds=rot,
            qed=qed,
            synthetic_accessibility=sa,
            lipinski_violations=lipinski,
            developability_score=score,
            recommendation=self._recommendation(score=score, violations=lipinski, qed=qed),
        )

    def rank_candidates(self, smiles_list: list[str]) -> list[DrugModelingResult]:
        """Model and rank candidates by developability score (descending)."""
        results = [self.model_candidate(smiles) for smiles in smiles_list]
        results.sort(key=lambda x: x.developability_score, reverse=True)
        return results

    def model_candidates(self, smiles_list: list[str]) -> list[dict]:
        """Convenience method returning dictionaries for CLI/JSON workflows."""
        return [result.to_dict() for result in self.rank_candidates(smiles_list)]


# Compatibility alias expected by tests
class DrugModel(DrugModeler):
    """Alias for DrugModeler."""

    pass
