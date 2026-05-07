from __future__ import annotations

from typing import Any


class GenomicDrugMatcher:
    """
    Match drugs to specific genomic variants.
    """

    def __init__(
        self,
        variant_drug_database: dict[str, list[str]],
        clinical_profile_drug_database: dict[str, list[str]] | None = None,
        virus_drug_database: dict[str, list[str]] | None = None,
    ):
        """
        Args:
            variant_drug_database: Mapping from variant (e.g., 'EGFR T790M') to list of effective drugs.
            clinical_profile_drug_database: Optional mapping from clinical profile markers/conditions
                (e.g., 'hypertension', 'renal_impairment') to treatment options.
            virus_drug_database: Optional mapping from virus/pathogen names
                (e.g., 'SARS-CoV-2', 'influenza') to antiviral options.
        """
        self.db = variant_drug_database
        self.clinical_profile_db = clinical_profile_drug_database or {}
        self.virus_db = virus_drug_database or {}

    @staticmethod
    def _to_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []

    def match_drugs_to_patient(self, patient_variants: list[str]) -> list[str]:
        """
        Find drugs that match the patient's genetic profile.
        """
        recommended_drugs = set()
        for variant in patient_variants:
            if variant in self.db:
                recommended_drugs.update(self.db[variant])
        return list(recommended_drugs)

    def match_drugs_to_clinical_profile(self, profile: dict[str, Any]) -> dict[str, list[str]]:
        """
        Find personalized treatment options from complete clinical profile data.
        """
        variants = self._to_list(profile.get("variants"))
        clinical_markers = self._to_list(profile.get("clinical_profiles"))
        clinical_markers.extend(self._to_list(profile.get("conditions")))
        clinical_markers.extend(self._to_list(profile.get("diagnoses")))
        viruses = self._to_list(profile.get("viral_infections"))
        viruses.extend(self._to_list(profile.get("viruses")))

        genomic = set(self.match_drugs_to_patient(variants))
        clinical = set()
        antiviral = set()

        for marker in clinical_markers:
            if marker in self.clinical_profile_db:
                clinical.update(self.clinical_profile_db[marker])

        for virus in viruses:
            if virus in self.virus_db:
                antiviral.update(self.virus_db[virus])

        combined = genomic.union(clinical).union(antiviral)
        return {
            "genomic_recommendations": sorted(genomic),
            "clinical_profile_recommendations": sorted(clinical),
            "antiviral_support": sorted(antiviral),
            "combined_recommendations": sorted(combined),
        }
