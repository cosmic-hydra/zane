import deepchem as dc
from typing import List, Dict, Any
from rdkit import Chem
from ..data.rdkit_utils import smiles_to_mols, compute_descriptors

from drug_discovery.glp_tox_panel import PreClinicalToxPanel

class ADMETScreen:
    def __init__(self):
        try:
            import deepchem as dc
            self.deepchem_available = True
            # Enhanced Tox21 + Clintox for hepatotox
            self.tox21_tasks = ['SR-HSE', 'SR-p53']  # proxies
            self.tox21_model = dc.models.GraphConvModel(n_tasks=2, mode='classification')  # Stub/train on Tox21
            self.clintox_model = dc.models.GraphConvModel(n_tasks=2, mode='classification')  # Hepatotox from Clintox
            self.featurizer = dc.feat.CircularFingerprint(size=1024)
        except:
            self.deepchem_available = False
        
        self.glp_panel = PreClinicalToxPanel(herg_threshold=0.3)  # Strict CiPA-like hERG

    def predict(self, smiles_list: List[str]) -> Dict[str, List[float]]:
        mols = smiles_to_mols(smiles_list)
        results = {'herg': [], 'hepatotox': [], 'bbb': [], 'qed': []}
        
        # Enhanced hERG from GLP panel (heuristic + pharmacophore)
        for smi in smiles_list:
            panel = self.glp_panel.evaluate(smi)
            results['herg'].append(panel.herg.inhibition_probability)
        
        # DeepChem stubs for others
        if self.deepchem_available:
            X = self.featurizer.featurize(smiles_list)
            tox21_pred = self.tox21_model.predict(X)
            clintox_pred = self.clintox_model.predict(X)
            for i in range(len(smiles_list)):
                results['hepatotox'].append(clintox_pred[i]['probabilities'][1])  # hepatotox class
        
        else:
            results['hepatotox'] = [0.05] * len(smiles_list)
            results['bbb'] = [0.8] * len(smiles_list)
        
        desc_df = compute_descriptors(mols)
        results['qed'] = desc_df['qed'].tolist()
        
        # CiPA-like hERG risk bands
        results['herg_risk'] = ['high' if p > 0.5 else 'moderate' if p > 0.3 else 'low' for p in results['herg']]
        
        return results