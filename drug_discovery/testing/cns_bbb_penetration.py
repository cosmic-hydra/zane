import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from typing import Dict, Any

class BioavailabilityScreener:
    """
    Assesses Central Nervous System (CNS) penetration and active efflux risk.
    Computes CNS MPO scores and predicts P-glycoprotein (P-gp) substrate status.
    """

    def __init__(self):
        pass

    def calculate_cns_mpo(self, smiles: str) -> float:
        """
        Computes the CNS Multiparameter Optimization (MPO) score (0-6).
        Parameters: ClogP, MW, TPSA, HBD, pKa.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        # Calculate raw parameters
        clogp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        
        # pKa estimation (very heuristic for this scaffolding)
        # Bases often have pKa ~9, Acids ~4.
        pka = 8.0 # Default fallback
        if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1,H0;c,C]")): # Amine
            pka = 9.5
        elif mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]")): # Carboxylic acid
            pka = 4.5

        # Desirability functions (normalized to [0, 1])
        def f_clogp(x): return 1.0 if x <= 3.0 else np.exp(-(x-3.0)**2 / 2.0)
        def f_mw(x): return 1.0 if x <= 360.0 else np.exp(-(x-360.0)**2 / 5000.0)
        def f_tpsa(x): 
            if 40 <= x <= 90: return 1.0
            elif x < 40: return x/40.0
            else: return np.exp(-(x-90.0)**2 / 1000.0)
        def f_hbd(x): return 1.0 if x <= 0 else 0.8 if x == 1 else 0.0
        def f_pka(x): return np.exp(-(x-8.0)**2 / 10.0)

        scores = [f_clogp(clogp), f_mw(mw), f_tpsa(tpsa), f_hbd(hbd), f_pka(pka)]
        return float(np.sum(scores))

    def predict_p_glycoprotein_efflux(self, smiles: str) -> bool:
        """
        Predicts if the drug is a P-gp substrate (active efflux risk).
        P-gp substrates are typically large (MW > 400), lipophilic, 
        and have many H-bond acceptors.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return True

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hba = Descriptors.NumHAcceptors(mol)

        # Heuristic Rule: P-gp Substrate if MW > 400 and (LogP > 3 or HBA > 8)
        if mw > 400 and (logp > 3.0 or hba > 8):
            return True
            
        return False

    def get_cns_profile(self, smiles: str) -> Dict[str, Any]:
        """Returns the CNS penetration and efflux profile."""
        mpo_score = self.calculate_cns_mpo(smiles)
        pgp_efflux = self.predict_p_glycoprotein_efflux(smiles)
        
        return {
            "smiles": smiles,
            "cns_mpo_score": mpo_score,
            "pgp_efflux_risk": pgp_efflux,
            "likely_cns_penetrant": mpo_score >= 4.0 and not pgp_efflux
        }
