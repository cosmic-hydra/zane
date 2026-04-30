import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Dict, Any

class SevereToxicityVeto(Exception):
    """Exception raised when a molecule fails critical human organ safety checks."""
    pass

class IdiosyncraticToxScreener:
    """
    Identifies rare but fatal human toxicities, focusing on:
    - Drug-Induced Liver Injury (DILI) via BSEP inhibition heuristics.
    - Mitochondrial toxicity (oxidative phosphorylation decoupling).
    """

    def __init__(self, pili_threshold: float = 3.0, tpsa_threshold: float = 75.0):
        self.pili_threshold = pili_threshold
        self.tpsa_threshold = tpsa_threshold

    def predict_dili_risk(self, smiles: str) -> float:
        """
        Calculates DILI risk based on the 'Rule of 2' (LogP >= 3 and TPSA <= 75).
        Generates a heuristic risk score in range [0, 1].
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 1.0

        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        # Basic Rule of 2 DILI risk assessment
        risk_score = 0.0
        if logp >= self.pili_threshold:
            risk_score += 0.5
        if tpsa <= self.tpsa_threshold:
            risk_score += 0.5
            
        return risk_score

    def flag_mitochondrial_toxicity(self, smiles: str) -> bool:
        """
        Checks for structural alerts that decouple mitochondrial oxidative phosphorylation.
        Structural alerts include: phenols with multiple halogen/nitro groups, 
        lipophilic weak acids, and certain quinones.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # Known mitochondrial toxins alerts (Uncouplers)
        alerts = {
            "halogenated_phenol": "Oc1c([F,Cl,Br,I])cc([F,Cl,Br,I])cc1",
            "nitro_phenol": "Oc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1",
            "lipophilic_weak_acid": "c1ccccc1-[C,N,S](=O)=O", # Generic benzoic/sulfonic acid
            "quinone": "C1(=O)C=CC(=O)C=C1"
        }

        found_alerts = []
        for name, smarts in alerts.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                found_alerts.append(name)
        
        if found_alerts:
            raise SevereToxicityVeto(
                f"Mitochondrial toxicity alert: Found {', '.join(found_alerts)}. "
                "Potential oxidative phosphorylation decoupling detected."
            )
            
        return False

    def screen_molecule(self, smiles: str) -> Dict[str, Any]:
        """Performs safety screen and raises Veto if fatal risks are detected."""
        dili_risk = self.predict_dili_risk(smiles)
        
        # We allow high DILI risk molecules to be flagged but not vetoed alone
        # unless combined with mitochondrial risk.
        try:
            self.flag_mitochondrial_toxicity(smiles)
            mito_risk = False
        except SevereToxicityVeto:
            mito_risk = True
            raise

        return {
            "smiles": smiles,
            "dili_risk_score": dili_risk,
            "mitochondrial_safe": not mito_risk,
            "organ_safety_pass": dili_risk < 1.0 and not mito_risk
        }
