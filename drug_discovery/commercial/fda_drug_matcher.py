import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class CommercialDrugMapper:
    def __init__(self):
        self.fda_db: Optional[pd.DataFrame] = None
        self.fingerprints = []

    def load_fda_orange_book(self, database_path: str):
        """
        Loads a database of currently approved, commercially available medicines.
        Expected columns: 'drug_name', 'smiles', 'commercial_dose'
        """
        try:
            self.fda_db = pd.read_csv(database_path)
            self._precompute_fingerprints()
            logger.info(f"Loaded {len(self.fda_db)} drugs from {database_path}")
        except Exception as e:
            logger.error(f"Error loading FDA Orange Book: {e}")
            # Fallback for demonstration if file doesn't exist
            self.fda_db = pd.DataFrame([
                {"drug_name": "Imatinib", "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", "commercial_dose": "400mg daily"},
                {"drug_name": "Metformin", "smiles": "CN(C)C(=N)N=C(N)N", "commercial_dose": "500mg twice daily"},
                {"drug_name": "Atorvastatin", "smiles": "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4", "commercial_dose": "20mg daily"}
            ])
            self._precompute_fingerprints()

    def _precompute_fingerprints(self):
        self.fingerprints = []
        for smiles in self.fda_db['smiles']:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                self.fingerprints.append(fp)
            else:
                self.fingerprints.append(None)

    def find_closest_commercial_match(self, generated_smiles: str) -> dict:
        """
        Calculates Morgan Fingerprint Tanimoto similarity between ZANE's de novo molecule 
        and all FDA-approved drugs.
        """
        if self.fda_db is None or not self.fingerprints:
            return {"closest_drug": "Unknown", "similarity": 0.0, "commercial_dose": "N/A", "smiles": ""}

        query_mol = Chem.MolFromSmiles(generated_smiles)
        if not query_mol:
            return {"closest_drug": "Invalid SMILES", "similarity": 0.0, "commercial_dose": "N/A", "smiles": ""}

        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
        
        max_sim = -1.0
        best_match_idx = -1
        
        for i, fp in enumerate(self.fingerprints):
            if fp:
                sim = DataStructs.TanimotoSimilarity(query_fp, fp)
                if sim > max_sim:
                    max_sim = sim
                    best_match_idx = i
        
        if best_match_idx != -1:
            match = self.fda_db.iloc[best_match_idx]
            return {
                "closest_drug": match['drug_name'],
                "similarity": round(max_sim, 4),
                "commercial_dose": match['commercial_dose'],
                "smiles": match['smiles']
            }
        
        return {"closest_drug": "No Match", "similarity": 0.0, "commercial_dose": "N/A", "smiles": ""}

    def compare_compounds(self, zane_compounds: List[dict], commercial_match: dict) -> dict:
        """
        Compares ZANE's multi-compound drug with the commercial equivalent.
        """
        zane_smiles_set = {c['smiles'] for c in zane_compounds}
        comm_smiles = commercial_match.get('smiles', '')
        
        # Extra: In ZANE but not the main commercial ingredient
        extra = [c['smiles'] for c in zane_compounds if c['smiles'] != comm_smiles]
        
        # Missing: In commercial but not in ZANE (Mocked for demonstration)
        missing = ["Magnesium Stearate (Excipient)", "Hypromellose (Coating)"] if comm_smiles else []
        
        return {
            "extra_compounds": extra[:5], # Limit display
            "missing_compounds": missing
        }
