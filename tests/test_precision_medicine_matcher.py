import importlib.util
from pathlib import Path

_module_path = (
    Path(__file__).resolve().parent / ".." / "drug_discovery" / "precision_medicine" / "genomic_matcher.py"
).resolve()
_spec = importlib.util.spec_from_file_location("genomic_matcher", str(_module_path))
_mod = importlib.util.module_from_spec(_spec)
assert _spec is not None
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
GenomicDrugMatcher = _mod.GenomicDrugMatcher


def test_match_drugs_to_patient_variants():
    matcher = GenomicDrugMatcher({"EGFR T790M": ["osimertinib"], "BRAF V600E": ["vemurafenib"]})
    matched = matcher.match_drugs_to_patient(["EGFR T790M", "UNKNOWN"])
    assert "osimertinib" in matched
    assert "vemurafenib" not in matched


def test_match_drugs_to_clinical_profile_including_virus_support():
    matcher = GenomicDrugMatcher(
        variant_drug_database={"EGFR T790M": ["osimertinib"]},
        clinical_profile_drug_database={"hypertension": ["amlodipine"]},
        virus_drug_database={"SARS-CoV-2": ["nirmatrelvir/ritonavir"]},
    )
    profile = {
        "variants": ["EGFR T790M"],
        "conditions": "hypertension",
        "viral_infections": ["SARS-CoV-2"],
    }
    result = matcher.match_drugs_to_clinical_profile(profile)

    assert result["genomic_recommendations"] == ["osimertinib"]
    assert result["clinical_profile_recommendations"] == ["amlodipine"]
    assert result["antiviral_support"] == ["nirmatrelvir/ritonavir"]
    assert result["combined_recommendations"] == ["amlodipine", "nirmatrelvir/ritonavir", "osimertinib"]
