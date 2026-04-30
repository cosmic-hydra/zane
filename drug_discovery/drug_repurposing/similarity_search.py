import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from typing import List, Tuple, Dict

class DrugSimilaritySearch:
    """
    Find drugs similar to a query molecule for potential repurposing.
    """
    
    def __init__(self, drug_library_smiles: List[str], drug_names: List[str] = None):
        self.library_smiles = drug_library_smiles
        self.library_mols = [Chem.MolFromSmiles(s) for s in drug_library_smiles]
        self.library_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in self.library_mols if m]
        self.drug_names = drug_names or [f"Drug_{i}" for i in range(len(drug_library_smiles))]

    def find_similar(self, query_smiles: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        query_mol = Chem.MolFromSmiles(query_smiles)
        if not query_mol:
            return []
            
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
        
        results = []
        for i, fp in enumerate(self.library_fps):
            similarity = DataStructs.TanimotoSimilarity(query_fp, fp)
            if similarity >= threshold:
                results.append({
                    "name": self.drug_names[i],
                    "smiles": self.library_smiles[i],
                    "similarity": similarity
                })
                
        return sorted(results, key=lambda x: x["similarity"], reverse=True)
