"""
Biological Response Simulation - In Silico Testing of Drug Effects

Simulates biological responses to drug candidates including:
- Cellular response modeling (gene expression, signaling pathways)
- Pharmacokinetics (ADME: Absorption, Distribution, Metabolism, Excretion)
- Pharmacodynamics (dose-response relationships)
- Multi-scale modeling (molecular → cellular → tissue → organism)
- Systems biology simulations
- Off-target effect prediction
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from drug_discovery.external_tooling import canonicalize_smiles, gt4sd_properties
from drug_discovery.utils.rdkit_fallback import heuristic_props, is_smiles_plausible

try:  # pragma: no cover - optional dependency
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import Crippen, Descriptors  # type: ignore
except Exception:  # pragma: no cover - default path when RDKit unavailable
    Chem = None  # type: ignore
    Crippen = None  # type: ignore
    Descriptors = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ADMEProperties:
    """ADME (Absorption, Distribution, Metabolism, Excretion) properties."""

    absorption: float  # Oral bioavailability (0-1)
    distribution: float  # Volume of distribution (L/kg)
    metabolism: float  # Metabolic stability (0-1)
    excretion: float  # Clearance rate (mL/min/kg)
    half_life: float  # Elimination half-life (hours)
    bioavailability: float  # Overall bioavailability (0-1)


@dataclass
class DoseResponse:
    """Dose-response relationship parameters."""

    ec50: float  # Half-maximal effective concentration
    emax: float  # Maximum effect
    hill_coefficient: float  # Hill coefficient (slope)
    doses: np.ndarray  # Dose range
    responses: np.ndarray  # Predicted responses


@dataclass
class CellularResponse:
    """Cellular-level response to drug."""

    cell_viability: float  # 0-1
    proliferation_rate: float  # Relative to control
    apoptosis_rate: float  # 0-1
    gene_expression_changes: dict[str, float]  # Gene -> fold-change
    pathway_activation: dict[str, float]  # Pathway -> activation score


class ADMEPredictor:
    """Predict ADME properties of molecules."""

    def __init__(self):
        """Initialize ADME predictor."""
        pass

    def predict_adme(self, smiles: str) -> ADMEProperties | None:
        """
        Predict ADME properties from SMILES.

        Args:
            smiles: SMILES string

        Returns:
            ADME properties or None
        """
        try:
            if not is_smiles_plausible(smiles):
                return None

            smiles = canonicalize_smiles(smiles) or smiles

            if Chem is not None:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None

                ext_props = gt4sd_properties(smiles)
                mol_weight = ext_props.get("molecular_weight", Descriptors.MolWt(mol))
                logp = ext_props.get("logp", Crippen.MolLogP(mol))
                tpsa = ext_props.get("tpsa", Descriptors.TPSA(mol))
                num_hbd = Descriptors.NumHDonors(mol)
                num_hba = Descriptors.NumHAcceptors(mol)
                num_rotatable = Descriptors.NumRotatableBonds(mol)
            else:
                props = heuristic_props(smiles)
                mol_weight = props.molecular_weight
                logp = props.logp
                tpsa = props.tpsa
                num_hbd = props.h_donors
                num_hba = props.h_acceptors
                num_rotatable = props.rotatable_bonds
                ext_props = {}

            num_rotatable = ext_props.get("num_rotatable_bonds", locals().get("num_rotatable", 0.0))

            absorption = 1.0
            if mol_weight > 500:
                absorption *= 0.7
            if logp > 5:
                absorption *= 0.6
            if num_hbd > 5:
                absorption *= 0.7
            if num_hba > 10:
                absorption *= 0.7
            if tpsa > 140:
                absorption *= 0.5

            vd = 0.5 + logp * 0.3  # L/kg
            vd = max(0.1, min(vd, 10.0))

            metabolism = 0.8 - (logp - 2) * 0.1
            metabolism = max(0.1, min(metabolism, 1.0))

            clearance = 15.0 * (1.0 - logp / 10.0) * (500.0 / max(mol_weight, 1.0))
            clearance = max(1.0, min(clearance, 100.0))

            half_life = (0.693 * vd * 70) / clearance  # hours (70kg person)
            bioavailability = absorption * metabolism

            return ADMEProperties(
                absorption=float(absorption),
                distribution=float(vd),
                metabolism=float(metabolism),
                excretion=float(clearance),
                half_life=float(half_life),
                bioavailability=float(bioavailability),
            )

        except Exception as e:
            logger.error(f"ADME prediction failed for {smiles}: {e}")
            return None

    def check_drug_likeness(self, smiles: str) -> dict[str, Any]:
        """
        Check drug-likeness rules.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary with drug-likeness assessment
        """
        try:
            if not is_smiles_plausible(smiles):
                return {"drug_like": False, "violations": ["Invalid SMILES"]}

            smiles = canonicalize_smiles(smiles) or smiles

            if Chem is not None:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return {"drug_like": False, "violations": ["Invalid SMILES"]}
                mol_weight = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                num_hbd = Descriptors.NumHDonors(mol)
                num_hba = Descriptors.NumHAcceptors(mol)
                num_rotatable = Descriptors.NumRotatableBonds(mol)
                tpsa = Descriptors.TPSA(mol)
            else:
                props = heuristic_props(smiles)
                mol_weight = props.molecular_weight
                logp = props.logp
                num_hbd = props.h_donors
                num_hba = props.h_acceptors
                num_rotatable = props.rotatable_bonds
                tpsa = props.tpsa

            violations = []
            if mol_weight > 500:
                violations.append("Molecular weight > 500")
            if logp > 5:
                violations.append("LogP > 5")
            if num_hbd > 5:
                violations.append("H-bond donors > 5")
            if num_hba > 10:
                violations.append("H-bond acceptors > 10")

            if num_rotatable > 10:
                violations.append("Rotatable bonds > 10 (Veber)")
            if tpsa > 140:
                violations.append("TPSA > 140 (Veber)")

            drug_like = len(violations) <= 1

            return {
                "drug_like": drug_like,
                "violations": violations,
                "lipinski_violations": sum(1 for v in violations if "Veber" not in v),
                "properties": {
                    "mol_weight": mol_weight,
                    "logp": logp,
                    "hbd": num_hbd,
                    "hba": num_hba,
                    "rotatable_bonds": num_rotatable,
                    "tpsa": tpsa,
                },
            }

        except Exception as e:
            logger.error(f"Drug-likeness check failed: {e}")
            return {"drug_like": False, "violations": [str(e)]}


class DoseResponseSimulator:
    """Simulate dose-response relationships."""

    def __init__(self):
        """Initialize dose-response simulator."""
        pass

    def simulate_dose_response(
        self,
        ec50: float,
        emax: float = 1.0,
        hill_coefficient: float = 1.0,
        dose_range: tuple[float, float] | None = None,
        n_points: int = 50,
    ) -> DoseResponse:
        """
        Simulate dose-response curve using Hill equation.

        Args:
            ec50: Half-maximal effective concentration
            emax: Maximum effect
            hill_coefficient: Hill coefficient
            dose_range: Optional (min_dose, max_dose) tuple
            n_points: Number of dose points

        Returns:
            DoseResponse object
        """
        if dose_range is None:
            # Auto-range around EC50
            dose_range = (ec50 / 100, ec50 * 100)

        # Generate dose points (log scale)
        doses = np.logspace(
            np.log10(dose_range[0]),
            np.log10(dose_range[1]),
            n_points,
        )

        # Hill equation: E = Emax * [D]^n / (EC50^n + [D]^n)
        responses = emax * (doses**hill_coefficient) / (ec50**hill_coefficient + doses**hill_coefficient)

        return DoseResponse(
            ec50=ec50,
            emax=emax,
            hill_coefficient=hill_coefficient,
            doses=doses,
            responses=responses,
        )

    def estimate_effective_dose(
        self,
        dose_response: DoseResponse,
        target_effect: float = 0.9,
    ) -> float:
        """
        Estimate dose for target effect.

        Args:
            dose_response: DoseResponse object
            target_effect: Target effect level (0-1)

        Returns:
            Estimated dose
        """
        # Find dose closest to target effect
        idx = np.argmin(np.abs(dose_response.responses - target_effect))
        return float(dose_response.doses[idx])

    def compute_therapeutic_window(
        self,
        efficacy_ec50: float,
        toxicity_ec50: float,
    ) -> dict[str, float]:
        """
        Compute therapeutic window (safety margin).

        Args:
            efficacy_ec50: EC50 for efficacy
            toxicity_ec50: EC50 for toxicity

        Returns:
            Dictionary with therapeutic window metrics
        """
        therapeutic_index = toxicity_ec50 / efficacy_ec50

        return {
            "therapeutic_index": float(therapeutic_index),
            "efficacy_ec50": float(efficacy_ec50),
            "toxicity_ec50": float(toxicity_ec50),
            "safety_margin": "wide" if therapeutic_index > 10 else "moderate" if therapeutic_index > 3 else "narrow",
        }


class CellularResponseSimulator:
    """Simulate cellular responses to drug treatment."""

    def __init__(self):
        """Initialize cellular response simulator."""
        # Placeholder gene expression baseline
        self.baseline_genes = {
            "EGFR": 1.0,
            "TP53": 1.0,
            "MYC": 1.0,
            "VEGF": 1.0,
            "BCL2": 1.0,
        }

        # Placeholder pathways
        self.pathways = {
            "MAPK": 0.5,
            "PI3K/AKT": 0.5,
            "WNT": 0.5,
            "p53": 0.5,
        }

    def simulate_cellular_response(
        self,
        smiles: str,
        dose: float,
        treatment_time: float = 24.0,
    ) -> CellularResponse | None:
        """
        Simulate cellular response to drug treatment.

        Args:
            smiles: Drug SMILES
            dose: Dose concentration (μM)
            treatment_time: Treatment duration (hours)

        Returns:
            CellularResponse or None
        """
        try:
            smiles = canonicalize_smiles(smiles) or smiles
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Simplified simulation based on molecular properties
            logp = Crippen.MolLogP(mol)
            mol_weight = Descriptors.MolWt(mol)

            # Cell viability (decreases with dose and lipophilicity)
            toxicity_factor = max(0.0, (logp / 5.0) * (dose / 10.0))
            cell_viability = max(0.0, 1.0 - toxicity_factor)

            # Proliferation rate (inversely correlated with viability)
            proliferation_rate = cell_viability * 0.8

            # Apoptosis rate
            apoptosis_rate = max(0.0, min(1.0, toxicity_factor * 1.5))

            # Gene expression changes (placeholder)
            gene_expression_changes = {}
            for gene in self.baseline_genes:
                # Simplified: random perturbation based on dose
                fold_change = 1.0 + np.random.normal(0, dose / 10.0)
                fold_change = max(0.1, fold_change)
                gene_expression_changes[gene] = float(fold_change)

            # Pathway activation (placeholder)
            pathway_activation = {}
            for pathway in self.pathways:
                activation = 0.5 + np.random.normal(0, 0.2)
                activation = max(0.0, min(1.0, activation))
                pathway_activation[pathway] = float(activation)

            return CellularResponse(
                cell_viability=float(cell_viability),
                proliferation_rate=float(proliferation_rate),
                apoptosis_rate=float(apoptosis_rate),
                gene_expression_changes=gene_expression_changes,
                pathway_activation=pathway_activation,
            )

        except Exception as e:
            logger.error(f"Cellular response simulation failed: {e}")
            return None


class BiologicalResponseSimulator:
    """Comprehensive biological response simulator."""

    def __init__(self):
        """Initialize biological response simulator."""
        self.adme_predictor = ADMEPredictor()
        self.dose_response_simulator = DoseResponseSimulator()
        self.cellular_simulator = CellularResponseSimulator()

    def simulate_full_response(
        self,
        smiles: str,
        initial_dose: float = 10.0,
        treatment_duration: float = 24.0,
    ) -> dict[str, Any]:
        """
        Simulate complete biological response.

        Args:
            smiles: Drug SMILES
            initial_dose: Initial dose (mg/kg for oral)
            treatment_duration: Treatment duration (hours)

        Returns:
            Dictionary with all simulation results
        """
        results = {
            "smiles": smiles,
            "initial_dose": initial_dose,
            "treatment_duration": treatment_duration,
        }

        # ADME properties
        adme = self.adme_predictor.predict_adme(smiles)
        if adme:
            results["adme"] = {
                "absorption": adme.absorption,
                "distribution": adme.distribution,
                "metabolism": adme.metabolism,
                "excretion": adme.excretion,
                "half_life": adme.half_life,
                "bioavailability": adme.bioavailability,
            }

            # Effective dose after ADME
            effective_dose = initial_dose * adme.bioavailability
        else:
            effective_dose = initial_dose * 0.5  # Default

        # Drug-likeness
        drug_likeness = self.adme_predictor.check_drug_likeness(smiles)
        results["drug_likeness"] = drug_likeness

        # Dose-response (placeholder EC50)
        ec50 = effective_dose  # Simplified
        dose_response = self.dose_response_simulator.simulate_dose_response(
            ec50=ec50,
            emax=1.0,
            hill_coefficient=1.5,
        )

        results["dose_response"] = {
            "ec50": dose_response.ec50,
            "emax": dose_response.emax,
            "hill_coefficient": dose_response.hill_coefficient,
        }

        # Cellular response
        cellular_response = self.cellular_simulator.simulate_cellular_response(
            smiles,
            dose=effective_dose,
            treatment_time=treatment_duration,
        )

        if cellular_response:
            results["cellular_response"] = {
                "cell_viability": cellular_response.cell_viability,
                "proliferation_rate": cellular_response.proliferation_rate,
                "apoptosis_rate": cellular_response.apoptosis_rate,
                "affected_genes": len(cellular_response.gene_expression_changes),
                "affected_pathways": len(cellular_response.pathway_activation),
            }

        logger.info(f"Biological response simulation complete for {smiles}")

        return results

    def batch_simulate(
        self,
        smiles_list: list[str],
        dose: float = 10.0,
    ) -> pd.DataFrame:
        """
        Simulate biological responses for multiple molecules.

        Args:
            smiles_list: List of SMILES
            dose: Dose for all molecules

        Returns:
            DataFrame with simulation results
        """
        results = []

        for smiles in smiles_list:
            sim_result = self.simulate_full_response(smiles, initial_dose=dose)

            row = {
                "smiles": smiles,
                "drug_like": sim_result.get("drug_likeness", {}).get("drug_like", False),
                "bioavailability": sim_result.get("adme", {}).get("bioavailability", 0.0),
                "half_life": sim_result.get("adme", {}).get("half_life", 0.0),
                "ec50": sim_result.get("dose_response", {}).get("ec50", 0.0),
                "cell_viability": sim_result.get("cellular_response", {}).get("cell_viability", 0.0),
            }

            results.append(row)

        df = pd.DataFrame(results)
        logger.info(f"Batch simulation complete: {len(df)} molecules")

        return df
