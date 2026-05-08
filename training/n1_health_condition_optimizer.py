import torch
import numpy as np
import scipy.optimize as opt
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ConditionAdaptiveRewardFunction:
    """
    Dynamically adjusts the reinforcement learning reward landscape based on 
    individual patient physiological constraints.
    """
    def __init__(self, renal_threshold: float = 30.0, hepatic_threshold: float = 120.0):
        self.renal_threshold = renal_threshold
        self.hepatic_threshold = hepatic_threshold

    def dynamic_clearance_penalty(self, patient_state: Any, predicted_properties: Dict[str, float]) -> float:
        """
        Injects a penalty if a molecule's predicted clearance route conflicts 
        with the patient's organ impairment.
        
        If patient_state.eGFR < 30, molecules with high renal clearance are heavily penalized.
        """
        penalty = 0.0
        egfr = getattr(patient_state, 'egfr', 90.0)
        
        # Predicted renal clearance (normalized 0-1)
        renal_clearance = predicted_properties.get('renal_clearance', 0.5)
        
        if egfr < self.renal_threshold:
            # Severe renal impairment: patient cannot clear drugs via kidneys
            if renal_clearance > 0.2:
                # Calculate penalty based on severity of impairment and predicted clearance
                impairment_factor = (self.renal_threshold - egfr) / self.renal_threshold
                penalty -= 100.0 * impairment_factor * renal_clearance
                logger.info(f"Injecting renal clearance penalty: {penalty:.2f} for eGFR: {egfr}")
        
        return penalty

    def hepatic_safety_adjustment(self, patient_state: Any, predicted_properties: Dict[str, float]) -> float:
        """
        Adjusts reward for hepatic clearance if the liver is impaired.
        """
        ast = getattr(patient_state, 'ast', 25.0)
        alt = getattr(patient_state, 'alt', 25.0)
        
        hepatic_toxicity = predicted_properties.get('hepatic_toxicity', 0.1)
        
        if ast > self.hepatic_threshold or alt > self.hepatic_threshold:
            # Liver is already under stress; any hepatic toxicity is amplified
            return -50.0 * hepatic_toxicity
        
        return 0.0

    def compute_n1_optimized_reward(self, 
                                    base_reward: float, 
                                    patient_state: Any, 
                                    predicted_properties: Dict[str, float]) -> float:
        """
        Combines the base drug-likeness reward with patient-specific physiological penalties.
        """
        clearance_penalty = self.dynamic_clearance_penalty(patient_state, predicted_properties)
        hepatic_adjustment = self.hepatic_safety_adjustment(patient_state, predicted_properties)
        
        # Total optimized reward
        total_reward = base_reward + clearance_penalty + hepatic_adjustment
        
        return float(total_reward)
