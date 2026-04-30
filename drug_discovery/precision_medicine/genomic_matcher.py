from typing import List, Dict, Any

class GenomicDrugMatcher:
    """
    Match drugs to specific genomic variants.
    """
    
    def __init__(self, variant_drug_database: Dict[str, List[str]]):
        """
        Args:
            variant_drug_database: Mapping from variant (e.g., 'EGFR T790M') to list of effective drugs.
        """
        self.db = variant_drug_database
        
    def match_drugs_to_patient(self, patient_variants: List[str]) -> List[str]:
        """
        Find drugs that match the patient's genetic profile.
        """
        recommended_drugs = set()
        for variant in patient_variants:
            if variant in self.db:
                recommended_drugs.update(self.db[variant])
        return list(recommended_drugs)
