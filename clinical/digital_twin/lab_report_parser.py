import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import pdfplumber
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatientHealthState(BaseModel):
    """
    Represents the mathematical constraints of a patient's health state.
    """
    patient_id: str
    egfr: float = Field(..., description="Estimated Glomerular Filtration Rate (mL/min/1.73m2)")
    ast: float = Field(..., description="Aspartate Aminotransferase (U/L)")
    alt: float = Field(..., description="Alanine Aminotransferase (U/L)")
    blood_ph: float = Field(default=7.4, description="Systemic blood pH")
    sodium_level: float = Field(..., description="Serum sodium level (mmol/L)")
    allergies: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    organ_viability: Dict[str, str] = Field(default_factory=dict)

class LabReportIngestor:
    """
    Ingests and parses unstructured health reports to extract biological constraints.
    """
    def __init__(self):
        self.constraints: Dict[str, Any] = {}

    def parse_metabolic_panel(self, pdf_path: str) -> PatientHealthState:
        """
        Parses a metabolic panel from a PDF and returns a PatientHealthState.
        """
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            logger.error(f"Failed to parse PDF {pdf_path}: {str(e)}")
            raise ValueError(f"Could not read PDF: {str(e)}")

        # Extract biomarkers using regex
        egfr = self._extract_value(text, r"eGFR[:\s]+(\d+\.?\d*)")
        ast = self._extract_value(text, r"AST[:\s]+(\d+\.?\d*)")
        alt = self._extract_value(text, r"ALT[:\s]+(\d+\.?\d*)")
        ph = self._extract_value(text, r"pH[:\s]+(\d+\.?\d*)") or 7.4
        sodium = self._extract_value(text, r"Sodium[:\s]+(\d+\.?\d*)")

        if egfr is None or ast is None or alt is None or sodium is None:
            logger.warning("Some critical biomarkers were not found in the report. Using defaults where appropriate.")

        return PatientHealthState(
            patient_id="PID-" + re.search(r"Patient ID[:\s]+(\w+)", text, re.I).group(1) if re.search(r"Patient ID[:\s]+(\w+)", text, re.I) else "UNKNOWN",
            egfr=egfr if egfr is not None else 90.0,
            ast=ast if ast is not None else 25.0,
            alt=alt if alt is not None else 25.0,
            blood_ph=ph,
            sodium_level=sodium if sodium is not None else 140.0,
            allergies=re.findall(r"Allergy[:\s]+(\w+)", text, re.I),
            conditions=re.findall(r"Condition[:\s]+(\w+)", text, re.I)
        )

    def _extract_value(self, text: str, pattern: str) -> Optional[float]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def evaluate_organ_viability(self, state: PatientHealthState) -> Dict[str, Any]:
        """
        Flags severe organ impairment and locks parameters into a constraint dictionary.
        """
        constraints = {}
        viability = {}

        # Renal Function Evaluation
        if state.egfr < 15:
            viability["kidney"] = "ACUTE_RENAL_FAILURE"
            constraints["renal_clearance_blocked"] = True
            constraints["max_renal_clearance"] = 0.05
        elif state.egfr < 30:
            viability["kidney"] = "SEVERE_IMPAIRMENT"
            constraints["renal_clearance_blocked"] = False
            constraints["max_renal_clearance"] = 0.2
        else:
            viability["kidney"] = "NORMAL"
            constraints["renal_clearance_blocked"] = False

        # Hepatic Function Evaluation
        # Simplistic check: AST or ALT > 3x Upper Limit of Normal (approx 40 U/L)
        if state.ast > 120 or state.alt > 120:
            viability["liver"] = "HEPATIC_IMPAIRMENT"
            constraints["hepatic_metabolism_restricted"] = True
        else:
            viability["liver"] = "NORMAL"
            constraints["hepatic_metabolism_restricted"] = False

        # Systemic pH Acidosis/Alkalosis
        if state.blood_ph < 7.35:
            viability["systemic_ph"] = "ACIDOSIS"
        elif state.blood_ph > 7.45:
            viability["systemic_ph"] = "ALKALOSIS"
        else:
            viability["systemic_ph"] = "NORMAL"

        state.organ_viability = viability
        self.constraints = constraints
        return constraints
