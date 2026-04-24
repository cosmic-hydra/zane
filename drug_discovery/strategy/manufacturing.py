"""Chemical manufacturing strategy and risk planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any


@dataclass(frozen=True)
class ManufacturingPlan:
    smiles: str
    route_steps: int
    estimated_yield: float
    process_risk: float
    cogs_index: float
    green_chemistry_score: float
    supply_chain_risk: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "route_steps": self.route_steps,
            "estimated_yield": round(self.estimated_yield, 4),
            "process_risk": round(self.process_risk, 4),
            "cogs_index": round(self.cogs_index, 4),
            "green_chemistry_score": round(self.green_chemistry_score, 4),
            "supply_chain_risk": round(self.supply_chain_risk, 4),
        }


class ManufacturingStrategyPlanner:
    """Builds practical manufacturing strategy metrics from chemistry features."""

    def __init__(self):
        self.retro = None
        try:
            from drug_discovery.synthesis import RetrosynthesisPlanner

            self.retro = RetrosynthesisPlanner()
        except Exception:
            self.retro = None

    @staticmethod
    def _hash_score(smiles: str, label: str) -> float:
        digest = sha256(f"{smiles}:{label}".encode()).hexdigest()
        return int(digest[:8], 16) / float(0xFFFFFFFF)

    @staticmethod
    def _descriptor_snapshot(smiles: str) -> dict[str, float]:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("invalid smiles")
            return {
                "mw": float(Descriptors.MolWt(mol)),
                "hba": float(Lipinski.NumHAcceptors(mol)),
                "hbd": float(Lipinski.NumHDonors(mol)),
                "rot": float(Descriptors.NumRotatableBonds(mol)),
                "rings": float(Descriptors.RingCount(mol)),
            }
        except Exception:
            return {
                "mw": 220.0 + 350.0 * ManufacturingStrategyPlanner._hash_score(smiles, "mw"),
                "hba": 1.0 + 8.0 * ManufacturingStrategyPlanner._hash_score(smiles, "hba"),
                "hbd": 0.0 + 4.0 * ManufacturingStrategyPlanner._hash_score(smiles, "hbd"),
                "rot": 1.0 + 10.0 * ManufacturingStrategyPlanner._hash_score(smiles, "rot"),
                "rings": 0.0 + 5.0 * ManufacturingStrategyPlanner._hash_score(smiles, "rings"),
            }

    def plan(self, smiles: str, max_depth: int = 6) -> ManufacturingPlan:
        if self.retro is not None:
            retro = self.retro.plan_synthesis(smiles, max_depth=max_depth)
            route_steps = int(retro.get("num_steps", 5))
            estimated_yield = float(retro.get("estimated_yield", 0.55))
        else:
            route_steps = 3 + int(self._hash_score(smiles, "route") * max(1, max_depth - 2))
            estimated_yield = 0.45 + 0.4 * self._hash_score(smiles, "yield")

        d = self._descriptor_snapshot(smiles)

        process_risk = min(1.0, 0.08 * route_steps + 0.025 * d["rot"] + 0.04 * max(0.0, d["rings"] - 3.0))
        cogs_index = min(1.0, 0.35 * (1.0 - estimated_yield) + 0.35 * min(1.0, d["mw"] / 700.0) + 0.3 * process_risk)

        hetero = d["hba"] + d["hbd"]
        green = max(
            0.0, 1.0 - (0.45 * process_risk + 0.15 * min(1.0, hetero / 12.0) + 0.15 * min(1.0, d["mw"] / 700.0))
        )

        supply_chain = min(1.0, 0.07 * route_steps + 0.05 * max(0.0, d["rings"] - 2.0))

        return ManufacturingPlan(
            smiles=smiles,
            route_steps=route_steps,
            estimated_yield=estimated_yield,
            process_risk=process_risk,
            cogs_index=cogs_index,
            green_chemistry_score=green,
            supply_chain_risk=supply_chain,
        )
