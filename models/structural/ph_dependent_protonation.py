from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from typing import List, Optional
try:
    from dimorphite_dl import DimorphiteDL
except ImportError:
    DimorphiteDL = None
import logging

logger = logging.getLogger(__name__)

class MicroenvironmentIonizationEngine:
    """
    Handles molecular ionization states based on localized pH environments.
    """
    def __init__(self):
        if DimorphiteDL:
            self.engine = DimorphiteDL(min_ph=0.0, max_ph=14.0, silent=True)
        else:
            logger.warning("dimorphite_dl not found. pH-dependent protonation will be disabled.")
            self.engine = None

    def predict_ionization_state(self, smiles: str, target_ph: float) -> str:
        """
        Calculates the molecule's dominant protonation state at a specific pH.
        """
        if not self.engine:
            return smiles

        try:
            protonated_smiles_list = self.engine.protonate(smiles, target_ph)
            if protonated_smiles_list:
                # Dimorphite returns a list of possible states; we take the most dominant
                return protonated_smiles_list[0]
        except Exception as e:
            logger.error(f"Error predicting ionization for {smiles} at pH {target_ph}: {str(e)}")
        
        return smiles

    def calculate_ph_dependent_solubility(self, smiles: str, target_ph: float) -> float:
        """
        Estimates solubility based on the ionization state at a specific pH.
        Penalizes if the drug is likely to precipitate.
        """
        protonated_smiles = self.predict_ionization_state(smiles, target_ph)
        mol = Chem.MolFromSmiles(protonated_smiles)
        
        if not mol:
            return 0.0

        # LogP as a proxy for lipophilicity/solubility
        logp = Descriptors.MolLogP(mol)
        
        # Basic Henderson-Hasselbalch inspired solubility heuristic:
        # Ionized forms (charged) are generally more soluble.
        # We check for charges in the molecule.
        num_charges = sum(abs(atom.GetFormalCharge()) for atom in mol.GetAtoms())
        
        # If the molecule is neutral and has high LogP, it might precipitate in aqueous environment
        # Solubility score: higher is better
        solubility_score = 1.0 / (1.0 + 10**(logp - 2.0))
        
        # Increase solubility score if ionized
        if num_charges > 0:
            solubility_score *= (1.5 * num_charges)

        return min(solubility_score, 1.0)

    def get_pka_values(self, smiles: str) -> List[float]:
        """
        Placeholder for pKa prediction. In a full implementation, this would 
        interface with a dedicated pKa model.
        """
        # Dimorphite-DL handles pH distribution, but we can't easily extract exact pKa 
        # without running a range. This is a simplified proxy.
        return [7.0] # Mock pKa
