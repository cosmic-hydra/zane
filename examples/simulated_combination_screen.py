"""Simulation-only drug combination screening example.

This script validates ZANE's in-silico ADMET stack by scoring predefined
molecules and ranking pairwise combinations using proxy metrics:

- efficacy_proxy: mean QED score (higher is better)
- side_effect_risk_proxy: structural toxicity flags + Lipinski violations + SA

The output is a ranked table intended for simulation/testing only.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import sys

import pandas as pd

# Allow direct execution: python examples/simulated_combination_screen.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drug_discovery.evaluation import ADMETPredictor


@dataclass(frozen=True)
class MoleculeSpec:
    name: str
    smiles: str


SIMULATION_LIBRARY = [
    MoleculeSpec("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    MoleculeSpec("Naproxen", "COC1=CC=CC2=C1C=C(C=C2)C(C)C(=O)O"),
    MoleculeSpec("Dextromethorphan", "CN1CCC23CCCCC2C1CC4=C3C=CC(=C4)OC"),
    MoleculeSpec("Guaifenesin", "COC1=CC=C(C=C1)OCC(O)CO"),
    MoleculeSpec("Pseudoephedrine", "CC(C)NCC(C1=CC=CC=C1)O"),
    MoleculeSpec("Loratadine", "CCOC(=O)N1CCC(=C2C3=CC=CC=C3CCC4=CC=CC=C24)CC1"),
    MoleculeSpec("Cetirizine", "CN1CCN(CC1)CCOCCOCC(=O)O"),
]


def molecule_score(admet: ADMETPredictor, spec: MoleculeSpec) -> dict:
    lipinski = admet.check_lipinski_rule(spec.smiles) or {}
    toxicity = admet.predict_toxicity_flags(spec.smiles) or {}
    qed = admet.calculate_qed(spec.smiles)
    sa = admet.calculate_synthetic_accessibility(spec.smiles)

    toxicity_hits = sum(1 for v in toxicity.values() if bool(v))
    lipinski_violations = int(lipinski.get("num_violations", 4))
    qed_val = float(qed if qed is not None else 0.0)
    sa_val = float(sa if sa is not None else 10.0)

    # Lower is better. SA is 1-10, so lightly scaled here.
    side_effect_risk_proxy = toxicity_hits + lipinski_violations + (sa_val / 10.0)

    return {
        "name": spec.name,
        "smiles": spec.smiles,
        "qed": qed_val,
        "sa": sa_val,
        "toxicity_hits": toxicity_hits,
        "lipinski_violations": lipinski_violations,
        "side_effect_risk_proxy": side_effect_risk_proxy,
    }


def rank_combinations(molecule_rows: list[dict]) -> pd.DataFrame:
    rows = []
    for a, b in combinations(molecule_rows, 2):
        efficacy_proxy = (a["qed"] + b["qed"]) / 2.0
        risk_proxy = (a["side_effect_risk_proxy"] + b["side_effect_risk_proxy"]) / 2.0

        # Higher is better.
        combo_score = efficacy_proxy - (0.35 * risk_proxy)

        rows.append(
            {
                "drug_a": a["name"],
                "drug_b": b["name"],
                "efficacy_proxy": round(efficacy_proxy, 4),
                "risk_proxy": round(risk_proxy, 4),
                "combo_score": round(combo_score, 4),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(by="combo_score", ascending=False).reset_index(drop=True)


def main() -> None:
    admet = ADMETPredictor()

    molecule_rows = [molecule_score(admet, spec) for spec in SIMULATION_LIBRARY]
    molecule_df = pd.DataFrame(molecule_rows).sort_values(by="qed", ascending=False).reset_index(drop=True)

    combo_df = rank_combinations(molecule_rows)

    print("\n=== Simulation Molecule Scores ===")
    print(molecule_df[["name", "qed", "sa", "toxicity_hits", "lipinski_violations", "side_effect_risk_proxy"]])

    print("\n=== Top 10 Simulated Combinations ===")
    print(combo_df.head(10))

    out_path = "./outputs/reports/simulated_combination_ranking.csv"
    combo_df.to_csv(out_path, index=False)
    print(f"\nSaved ranking to: {out_path}")


if __name__ == "__main__":
    main()
