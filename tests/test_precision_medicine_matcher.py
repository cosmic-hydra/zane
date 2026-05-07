from drug_discovery.precision_medicine.genomic_matcher import GenomicDrugMatcher


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
