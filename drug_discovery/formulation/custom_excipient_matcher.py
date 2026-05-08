import pandas as pd
import networkx as nx
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PersonalizedFormulationEngine:
    """
    Matches drug candidates with biologically safe excipients tailored to 
    specific patient health profiles and conditions.
    """
    def __init__(self):
        # Database of excipients and their contraindications
        # In a production system, this would be loaded from a validated medical database
        self.excipient_knowledge_base = pd.DataFrame([
            {"name": "Sodium Chloride", "category": "Salt", "contraindications": ["hypernatremia", "hypertension"]},
            {"name": "Sucrose", "category": "Sweetener", "contraindications": ["diabetes", "fructose_intolerance"]},
            {"name": "Lactose", "category": "Filler", "contraindications": ["lactose_intolerance"]},
            {"name": "Ethanol", "category": "Solvent", "contraindications": ["liver_failure", "alcoholism", "pregnancy"]},
            {"name": "Propylene Glycol", "category": "Solvent", "contraindications": ["renal_failure"]},
            {"name": "Mannitol", "category": "Diuretic/Sweetener", "contraindications": ["anuria", "severe_dehydration"]},
            {"name": "Aspartame", "category": "Sweetener", "contraindications": ["phenylketonuria"]}
        ])
        
        # Build a relationship graph for cross-reactivity (optional complexity)
        self.reactivity_graph = nx.Graph()

    def veto_contraindicated_vehicles(self, 
                                     patient_state: Any, 
                                     proposed_formulations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters out formulations that contain excipients incompatible with the patient's current health state.
        """
        patient_conditions = set(getattr(patient_state, 'conditions', []))
        
        # Derived conditions from biomarkers
        sodium_level = getattr(patient_state, 'sodium_level', 140.0)
        egfr = getattr(patient_state, 'egfr', 90.0)
        ast = getattr(patient_state, 'ast', 25.0)
        
        if sodium_level > 145:
            patient_conditions.add("hypernatremia")
        if egfr < 30:
            patient_conditions.add("renal_failure")
        if ast > 120:
            patient_conditions.add("liver_failure")
            
        safe_formulations = []
        
        for formulation in proposed_formulations:
            excipients = formulation.get("excipients", [])
            is_vetoed = False
            veto_reason = ""
            
            for excipient_name in excipients:
                # Lookup excipient in knowledge base
                match = self.excipient_knowledge_base[
                    self.excipient_knowledge_base['name'].str.lower() == excipient_name.lower()
                ]
                
                if not match.empty:
                    contraindications = match.iloc[0]['contraindications']
                    # Check for intersection between patient conditions and contraindications
                    overlap = patient_conditions.intersection(set(contraindications))
                    if overlap:
                        is_vetoed = True
                        veto_reason = f"Excipient {excipient_name} is contraindicated for: {list(overlap)}"
                        break
            
            if not is_vetoed:
                safe_formulations.append(formulation)
            else:
                logger.info(f"Vetoing formulation {formulation.get('id', 'unknown')}: {veto_reason}")
                
        return safe_formulations

    def suggest_safe_alternatives(self, category: str, patient_state: Any) -> List[str]:
        """
        Suggests safe excipients within a specific category for a given patient.
        """
        patient_conditions = set(getattr(patient_state, 'conditions', []))
        # (Biomarker derived conditions logic here...)
        
        category_matches = self.excipient_knowledge_base[
            self.excipient_knowledge_base['category'].str.lower() == category.lower()
        ]
        
        alternatives = []
        for _, row in category_matches.iterrows():
            if not set(row['contraindications']).intersection(patient_conditions):
                alternatives.append(row['name'])
        
        return alternatives
