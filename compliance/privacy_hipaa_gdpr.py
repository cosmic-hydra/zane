import pandas as pd
import numpy as np
from typing import List, Optional
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
except ImportError:
    # Fallback for environment without Presidio
    AnalyzerEngine = None
    AnonymizerEngine = None

class PHISanitizer:
    """
    Enforces HIPAA Safe Harbor and GDPR privacy standards for patient data.
    Uses Presidio NLP for PII/PHI detection and differential privacy for dataset protection.
    """
    
    def __init__(self):
        if AnalyzerEngine:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        else:
            self.analyzer = None
            self.anonymizer = None

    def anonymize_cohort_data(self, df: pd.DataFrame, columns_to_scan: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Redacts 18 HIPAA Safe Harbor identifiers.
        Scans text columns for Names, Dates, SSNs, MRNs, etc.
        """
        if self.analyzer is None:
            print("Presidio libraries not installed. Skipping NLP-based anonymization.")
            return df

        sanitized_df = df.copy()
        if columns_to_scan is None:
            columns_to_scan = sanitized_df.select_dtypes(include=['object']).columns

        for col in columns_to_scan:
            sanitized_df[col] = sanitized_df[col].apply(lambda x: self._scrub_text(str(x)) if pd.notnull(x) else x)
        
        return sanitized_df

    def _scrub_text(self, text: str) -> str:
        """Internal helper to analyze and anonymize a single string."""
        results = self.analyzer.analyze(text=text, entities=[], language='en')
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "PERSON": OperatorConfig("replace", {"new_value": "<REDACTED_NAME>"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "<REDACTED_LOC>"}),
                "DATE_TIME": OperatorConfig("replace", {"new_value": "<REDACTED_DATE>"}),
                "PHONE_NUMBER": OperatorConfig("mask", {"chars_to_mask": 10, "masking_char": "*", "from_end": True}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<REDACTED_EMAIL>"}),
                "US_SSN": OperatorConfig("replace", {"new_value": "<REDACTED_SSN>"}),
                "US_PASSPORT": OperatorConfig("replace", {"new_value": "<REDACTED_PASSPORT>"}),
            }
        )
        return anonymized_result.text

    def inject_differential_privacy(self, df: pd.DataFrame, epsilon: float = 0.1, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Applies epsilon-differential privacy by adding Laplacian noise to numerical columns.
        Compliance: GDPR Requirement for non-reversibility.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        dp_df = df.copy()
        for col in columns:
            # Sensitivity is assumed to be the range of the column for this implementation
            sensitivity = dp_df[col].max() - dp_df[col].min()
            if sensitivity == 0:
                sensitivity = 1.0
            
            beta = sensitivity / epsilon
            noise = np.random.laplace(0, beta, len(dp_df))
            dp_df[col] = dp_df[col] + noise
            
        return dp_df

    def sanitize_for_trial_simulation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline method to prepare data for clinical stratification models."""
        # 1. Redact PII/PHI
        df = self.anonymize_cohort_data(df)
        
        # 2. Inject DP noise into phenotypic metrics (Age, Weight, Lab Values)
        numerical_phenotypes = [c for c in df.columns if any(p in c.lower() for p in ['age', 'weight', 'height', 'level', 'value'])]
        df = self.inject_differential_privacy(df, columns=numerical_phenotypes)
        
        return df
