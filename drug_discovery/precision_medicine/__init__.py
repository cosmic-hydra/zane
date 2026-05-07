"""
Precision Medicine Module
Tailor drug treatments to individual patient profiles.
"""

from .genomic_matcher import GenomicDrugMatcher

try:
    from .patient_stratification import PatientStratifier
except Exception:  # pragma: no cover - optional dependency fallback
    PatientStratifier = None

__all__ = ["GenomicDrugMatcher", "PatientStratifier"]
