"""Structural Alerts & PAINS Filter for ZANE.
16 PAINS SMARTS + 15 Brenk alerts + 8 reactive metabolite motifs.
Ref: Baell & Holloway (J Med Chem 2010), Brenk et al. (ChemMedChem 2008)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PAINS_SMARTS = {
    "quinone": "[#6]1(=[O,N])[#6]=,:[#6][#6](=[O,N])[#6]=,:[#6]1",
    "catechol": "[OH]c1cc([OH])ccc1",
    "rhodanine": "O=C1CSC(=S)N1",
    "hydroxyphenyl_hydrazone": "[OH]c1ccc(/N=N/)cc1",
    "2_amino_phenol": "[NH2]c1ccccc1[OH]",
    "imine_one": "[CX3](=[OX1])[CX3]=[NX2]",
    "mannich": "[NX3][CX4][OH]",
    "azo": "[#7]=[#7]",
    "diazo": "[N]=[N]=[C]",
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
    "acyl_hydrazine": "[CX3](=[OX1])[NX3][NX3]",
    "sulfonyl_halide": "[SX4](=[OX1])(=[OX1])[FX1,ClX1,BrX1]",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "anil_di_alk_A": "c1cc([#7])ccc1[CX4]",
    "triflate": "[CX4][O]S(=O)(=O)C(F)(F)F",
    "alkylidene_barbiturate": "O=C1CC(=C)C(=O)N1",
}
BRENK_SMARTS = {
    "nitro_aromatic": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1[NX3+](=O)[O-]",
    "aldehyde": "[CX3H1](=O)",
    "epoxide": "C1OC1",
    "thiol": "[SX2H]",
    "acyl_halide": "[CX3](=[OX1])[FX1,ClX1,BrX1,IX1]",
    "sulfonate_ester": "[SX4](=O)(=O)[OX2]",
    "peroxide": "[OX2][OX2]",
    "azide": "[NX1]=[NX2]=[NX1]",
    "beta_lactam": "C1C(=O)NC1",
    "vinyl_halide": "[CX3]=[CX3][FX1,ClX1,BrX1]",
    "aliphatic_nitro": "[CX4][NX3+](=O)[O-]",
    "thiocarbonyl": "[CX3]=[SX1]",
    "acyl_cyanide": "[CX3](=[OX1])C#N",
    "phosphoramide": "[PX4](=O)([NX3])([NX3])",
    "polycyclic_aromatic": "c1ccc2c(c1)ccc1ccc3ccccc3c1c2",
}
REACTIVE_METAB_SMARTS = {
    "aniline": "[NH2]c1ccccc1",
    "nitroaromatic": "[cR1][NX3+](=O)[O-]",
    "thiophene": "c1ccsc1",
    "furan": "c1ccoc1",
    "hydroquinone": "[OH]c1ccc([OH])cc1",
    "terminal_alkyne": "C#[CH]",
    "hydroxamic_acid": "[CX3](=O)[NX3][OH]",
    "epoxide_precursor": "C=C[CX4]",
}


@dataclass
class AlertResult:
    alert_name: str
    alert_class: str
    pattern: str
    severity: str


@dataclass
class StructuralAlertReport:
    smiles: str
    total_alerts: int = 0
    pains_alerts: list[AlertResult] = field(default_factory=list)
    brenk_alerts: list[AlertResult] = field(default_factory=list)
    reactive_alerts: list[AlertResult] = field(default_factory=list)
    is_clean: bool = True
    risk_score: float = 0.0

    def to_dict(self):
        return {
            "smiles": self.smiles,
            "total": self.total_alerts,
            "clean": self.is_clean,
            "risk": round(self.risk_score, 3),
            "pains": [a.alert_name for a in self.pains_alerts],
            "brenk": [a.alert_name for a in self.brenk_alerts],
            "reactive": [a.alert_name for a in self.reactive_alerts],
        }


def _check_smarts(smiles, smarts_dict, cls, severity):
    alerts = []
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return alerts
        for name, pat in smarts_dict.items():
            try:
                q = Chem.MolFromSmarts(pat)
                if q and mol.HasSubstructMatch(q):
                    alerts.append(AlertResult(name, cls, pat, severity))
            except:
                continue
    except ImportError:
        smi = smiles.lower()
        for name, hit in {"nitro": "[n+](=o)[o-]" in smi, "azo": "n=n" in smi, "thiol": "sh" in smi}.items():
            if hit:
                alerts.append(AlertResult(name, cls, "heuristic", severity))
    return alerts


class StructuralAlertScreener:
    """PAINS + Brenk + reactive metabolite screening.
    Example: report = StructuralAlertScreener().screen("O=C1CSC(=S)N1")
    """

    def screen(self, smiles):
        r = StructuralAlertReport(smiles=smiles)
        r.pains_alerts = _check_smarts(smiles, PAINS_SMARTS, "pains", "danger")
        r.brenk_alerts = _check_smarts(smiles, BRENK_SMARTS, "brenk", "critical")
        r.reactive_alerts = _check_smarts(smiles, REACTIVE_METAB_SMARTS, "reactive_metabolite", "warning")
        r.total_alerts = len(r.pains_alerts) + len(r.brenk_alerts) + len(r.reactive_alerts)
        r.is_clean = r.total_alerts == 0
        r.risk_score = min(1.0, len(r.brenk_alerts) * 0.4 + len(r.pains_alerts) * 0.25 + len(r.reactive_alerts) * 0.15)
        return r

    def batch_screen(self, smiles_list):
        reports = [self.screen(s) for s in smiles_list]
        clean = sum(1 for r in reports if r.is_clean)
        logger.info(f"Alerts: {len(smiles_list)} screened, {clean} clean ({clean/max(len(smiles_list),1)*100:.0f}%)")
        return reports

    def get_clean_molecules(self, reports):
        return [r.smiles for r in reports if r.is_clean]

    def filter_and_rank(self, smiles_list):
        reports = self.batch_screen(smiles_list)
        return [(r.smiles, r.risk_score) for r in sorted(reports, key=lambda r: r.risk_score)]
