from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class DeNovoStrictEnforcer:
    """
    Ensures that generated molecules are novel and not regurgitations of known drugs.
    """
    def __init__(self, threshold: float = 0.45):
        self.threshold = threshold
        self.known_fingerprints = []
        self.known_smiles = set()

    def load_public_archives(self, chembl_db_path: str):
        """
        Loads a database of known SMILES strings and pre-calculates fingerprints.
        In a production environment, this would use a memory-mapped database or a vector store.
        """
        if not os.path.exists(chembl_db_path):
            logger.warning(f"ChEMBL database path {chembl_db_path} not found. Running with empty novelty database.")
            return

        logger.info(f"Loading public drug archives from {chembl_db_path}...")
        try:
            with open(chembl_db_path, 'r') as f:
                for line in f:
                    smiles = line.strip().split()[0]
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        self.known_fingerprints.append(fp)
                        self.known_smiles.add(smiles)
            logger.info(f"Loaded {len(self.known_fingerprints)} molecules into novelty enforcer.")
        except Exception as e:
            logger.error(f"Failed to load archives: {str(e)}")

    def calculate_novelty_penalty(self, generated_smiles: str) -> float:
        """
        Calculates the Morgan Fingerprint Tanimoto similarity against the known database.
        Applies a catastrophic negative reward if similarity exceeds the threshold.
        """
        mol = Chem.MolFromSmiles(generated_smiles)
        if not mol:
            return -10.0 # Invalid molecule penalty

        # Check for exact match first
        if generated_smiles in self.known_smiles:
            logger.info(f"Catastrophic Penalty: Exact match for known drug {generated_smiles}")
            return -1000.0

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        
        if not self.known_fingerprints:
            return 0.0

        # Calculate max Tanimoto similarity
        similarities = DataStructs.BulkTanimotoSimilarity(fp, self.known_fingerprints)
        max_sim = max(similarities)

        if max_sim > self.threshold:
            # Catastrophic negative reward for minor tweaks of existing drugs
            penalty = -100.0 * (max_sim / self.threshold)**2
            logger.info(f"Novelty Veto: Max similarity {max_sim:.4f} exceeds threshold {self.threshold}. Penalty: {penalty:.2f}")
            return penalty

        return 0.0
