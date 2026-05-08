import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class EpigeneticAgeCalculator:
    """
    Calculates biological age using DNA methylation data and adjusts therapeutic 
    dosing to prevent overdose in senescent biological systems.
    """
    def __init__(self):
        # Pre-trained Horvath Clock coefficients (simplified mock subset)
        # Real Horvath clock uses 353 specific CpG sites
        self.horvath_coefficients = {
            "cg00075967": 0.05,
            "cg00374713": 0.12,
            "cg00864867": -0.08,
            "cg01234567": 0.22,
            "intercept": 0.65
        }

    def calculate_horvath_clock(self, methylation_array_path: str) -> float:
        """
        Ingests DNA methylation data (CpG island status) to calculate biological age.
        """
        try:
            # Expecting a CSV with 'cpg_id' and 'beta_value' (0 to 1)
            df = pd.read_csv(methylation_array_path)
            
            # Linear combination of methylation levels
            log_age = self.horvath_coefficients["intercept"]
            for cpg, coeff in self.horvath_coefficients.items():
                if cpg == "intercept": continue
                
                beta_value = df.loc[df['cpg_id'] == cpg, 'beta_value']
                if not beta_value.empty:
                    log_age += coeff * beta_value.values[0]
            
            # Inverse of the age transformation used by Horvath
            # (Simplified: assuming biological_age = exp(log_age))
            biological_age = np.exp(log_age)
            
            logger.info(f"Epigenetic analysis complete. Biological age: {biological_age:.1f} years.")
            return float(biological_age)
            
        except Exception as e:
            logger.error(f"Failed to calculate epigenetic age: {str(e)}")
            return 45.0 # Default adult age

    def adjust_dosing_for_senescence(self, base_dose: float, biological_age: float, chronological_age: float = 50.0) -> float:
        """
        Dynamically scales the therapeutic dose if the patient's epigenetic age 
        indicates severe cellular senescence.
        """
        # Calculate the aging acceleration factor
        age_acceleration = biological_age - chronological_age
        
        # If the patient is "biologically older" than their years, reduce dose
        # due to expected decline in metabolic capacity and cellular repair.
        reduction_factor = 1.0
        
        if age_acceleration > 5.0:
            # Reduce dose by 1% for every year of acceleration beyond 5 years
            excess_aging = age_acceleration - 5.0
            reduction_factor = max(0.5, 1.0 - (excess_aging * 0.02))
            
        adjusted_dose = base_dose * reduction_factor
        
        if reduction_factor < 1.0:
            logger.warning(f"Dose adjusted for senescence. Factor: {reduction_factor:.2f}. "
                           f"New dose: {adjusted_dose:.2f}")
            
        return float(adjusted_dose)
