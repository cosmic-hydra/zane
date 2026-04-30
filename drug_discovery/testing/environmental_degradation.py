import numpy as np
import scipy.constants as const
from rdkit import Chem
from typing import Dict, Any

class EnvironmentalStressSimulator:
    """
    Simulates drug degradation under extreme environmental stress conditions:
    Heat (Arrhenius decay), humidity, and light (photostability).
    """

    def __init__(self, pre_exponential_factor: float = 1e13):
        """
        Args:
            pre_exponential_factor (A): Frequency of collisions/orientation factor.
                                        Default is a standard value for first-order reactions.
        """
        self.A = pre_exponential_factor
        self.R = const.R  # Ideal gas constant in J/(mol*K)

    def calculate_arrhenius_decay(
        self, 
        activation_energy_kj: float, 
        temp_celsius: float, 
        days: int
    ) -> float:
        """
        Calculates the percentage of API remaining using the Arrhenius equation.
        
        k = A * exp(-Ea / RT)
        """
        # Convert T to Kelvin
        T = temp_celsius + 273.15
        # Convert Ea to Joules
        Ea = activation_energy_kj * 1000.0
        
        # Calculate rate constant k (1/sec)
        k = self.A * np.exp(-Ea / (self.R * T))
        
        # Total seconds of exposure
        seconds = days * 24 * 3600
        
        # Concentration remaining (First order: C = C0 * exp(-kt))
        percent_remaining = np.exp(-k * seconds) * 100.0
        
        return float(np.clip(percent_remaining, 0.0, 100.0))

    def check_photostability(self, smiles: str) -> bool:
        """
        Checks for UV-absorbing chromophores using RDKit SMARTS matching.
        Flagged if highly susceptible to photo-degradation.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # Structural alerts for photo-instability
        phototoxicity_alerts = {
            "conjugated_polyenes": "C=C-C=C-C=C",
            "aromatic_ketones": "c1ccccc1C(=O)C",
            "nitro_aromatics": "c1ccccc1[N+](=O)[O-]",
            "phenothiazines": "c1ccc2c(c1)Sc3ccccc3N2",
            "quinolones": "c1ccc2c(c1)nc(cc2=O)C(=O)O",
            "extended_aromatic_systems": "c1ccc2c(c1)ccc3ccccc32" # Pyrene-like
        }

        for name, smarts in phototoxicity_alerts.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True
        
        return False

    def simulate_stress_report(self, smiles: str, Ea: float, temp: float, days: int) -> Dict[str, Any]:
        """Comprehensive environmental validation report."""
        remaining = self.calculate_arrhenius_decay(Ea, temp, days)
        photo_risk = self.check_photostability(smiles)
        
        return {
            "smiles": smiles,
            "api_remaining_percent": remaining,
            "photostability_risk": photo_risk,
            "stable_under_conditions": remaining > 90.0 and not photo_risk
        }
