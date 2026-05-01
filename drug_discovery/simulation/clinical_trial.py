"""
In Silico Clinical Trial Simulator

Orchestrates Phase 3 clinical trial simulations by combining synthetic
patient cohorts with Bayesian PK/PD models.
"""

import logging

import numpy as np

from .bayesian_pkpd import BayesianPKPD
from .patient_generator import PatientGenerator

logger = logging.getLogger(__name__)


class ClinicalTrialSimulator:
    """Simulates clinical trials across synthetic populations."""

    def __init__(self):
        self.patient_gen = PatientGenerator()
        self.pkpd_model = BayesianPKPD()

    @staticmethod
    def _standard_normal_cdf(x: float) -> float:
        """
        Approximation of standard normal CDF using error function.
        This is a stdlib-only implementation without scipy.
        """
        # Abramowitz and Stegun formula 7.1.26
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x) / np.sqrt(2)

        t = 1.0 / (1.0 + p * x)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t

        erf = 1.0 - (((((a5 * t5 + a4 * t4) + a3 * t3) + a2 * t2) + a1 * t) * t) * np.exp(-x * x)

        return 0.5 * (1.0 + sign * erf)

    def simulate_phase3(
        self, drug_name: str, num_patients: int = 1000, dose_regimen: float = 10.0, control_efficacy: float = 0.2
    ) -> dict[str, any]:
        """
        Run a simulated Phase 3 clinical trial.

        Args:
            drug_name: Name of the drug candidate
            num_patients: Total number of patients (half in treatment, half in control)
            dose_regimen: Dose administered to treatment group
            control_efficacy: Expected efficacy in the placebo/control group
        """
        logger.info(f"Simulating Phase 3 trial for {drug_name} with {num_patients} patients.")

        # 1. Generate synthetic cohort
        cohort = self.patient_gen.generate_cohort(num_patients)

        # 2. Randomize into Treatment and Control
        cohort["group"] = np.random.choice(["treatment", "control"], size=num_patients)

        # 3. Simulate outcomes
        outcomes = []
        for _, patient in cohort.iterrows():
            if patient["group"] == "treatment":
                # Use PK/PD model for treatment effect
                efficacy = self.pkpd_model.predict_outcome(patient.to_dict(), dose_regimen)
            else:
                # Placebo effect or standard of care
                efficacy = np.random.normal(control_efficacy, 0.05)

            # Binary clinical outcome (e.g., responder vs non-responder)
            outcome = 1 if np.random.random() < efficacy else 0
            outcomes.append(outcome)

        cohort["outcome"] = outcomes

        # 4. Analyze Results
        treatment_results = cohort[cohort["group"] == "treatment"]["outcome"]
        control_results = cohort[cohort["group"] == "control"]["outcome"]

        treatment_rate = treatment_results.mean()
        control_rate = control_results.mean()
        relative_risk = treatment_rate / (control_rate + 1e-5)

        # Calculate p-value using two-sample z-test
        n_treatment = len(treatment_results)
        n_control = len(control_results)
        
        # Pooled standard error for proportion difference
        p_pooled = (treatment_results.sum() + control_results.sum()) / (n_treatment + n_control)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_treatment + 1/n_control))
        
        # Z-statistic
        z_stat = (treatment_rate - control_rate) / (se + 1e-10)  # Add small epsilon to avoid division by zero
        
        # Two-tailed p-value from standard normal distribution
        # P(|Z| > |z_stat|) = 2 * P(Z > |z_stat|)
        p_value = 2 * (1 - self._standard_normal_cdf(abs(z_stat)))

        report = {
            "drug_name": drug_name,
            "sample_size": num_patients,
            "treatment_response_rate": float(treatment_rate),
            "control_response_rate": float(control_rate),
            "relative_risk": float(relative_risk),
            "p_value": float(p_value),
            "z_statistic": float(z_stat),
            "status": "Success" if p_value < 0.05 else "Failed",
        }

        logger.info(f"Trial Simulation Completed: {report['status']}")
        return report
