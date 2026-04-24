"""ZANE Simulation — Physics-based simulation and ML-accelerated FEP."""

from .bayesian_pkpd import BayesianPKPD as BayesianPKPD
from .clinical_trial import ClinicalTrialSimulator as ClinicalTrialSimulator
from .coarse_grained_md import CGSimulator as CGSimulator
from .microgravity import MicrogravitySimulator as MicrogravitySimulator
from .orbital_logistics import OrbitalLogisticsOptimizer as OrbitalLogisticsOptimizer
from .patient_generator import PatientGenerator as PatientGenerator

__all__ = [
    "CGSimulator",
    "PatientGenerator",
    "BayesianPKPD",
    "ClinicalTrialSimulator",
    "MicrogravitySimulator",
    "OrbitalLogisticsOptimizer",
]

try:
    from drug_discovery.simulation.free_energy import (
        FEPConfig as FEPConfig,
    )
    from drug_discovery.simulation.free_energy import (
        FEPPipeline as FEPPipeline,
    )
    from drug_discovery.simulation.free_energy import (
        FEPSurrogateNetwork as FEPSurrogateNetwork,
    )
    from drug_discovery.simulation.free_energy import (
        generate_lambda_schedule as generate_lambda_schedule,
    )

    __all__.extend(["FEPPipeline", "FEPConfig", "FEPSurrogateNetwork", "generate_lambda_schedule"])
except ImportError:
    pass
