"""ZANE — AI-native Drug Discovery Platform."""

__version__ = "2026.4.1"
__all__ = ["__version__"]

# Core pipeline
try:
    from drug_discovery.pipeline.autonomous_pipeline import StreamingDataPipeline as DrugDiscoveryPipeline
    __all__.append("DrugDiscoveryPipeline")
except Exception:
    pass

# Utility modules
try:
    from drug_discovery.data.rdkit_utils import smiles_to_sdf
    from drug_discovery.generation.torchdrug_generator import TorchDrugGenerator
    from drug_discovery.screening.admet_models import ADMETScreen
    from drug_discovery.screening.filtering import filter_admet
    from drug_discovery.docking.vina_wrapper import VinaDocker
    from drug_discovery.docking.diffdock_placeholder import run_diffdock
    from drug_discovery.structure_analysis.cif_parser import parse_cif_to_mol
    from drug_discovery.structure_analysis.xrpd_analysis import analyze_xrpd

    __all__.extend([
        "TorchDrugGenerator",
        "ADMETScreen",
        "VinaDocker",
        "run_diffdock",
        "parse_cif_to_mol",
        "analyze_xrpd",
        "smiles_to_sdf",
        "filter_admet"
    ])
except Exception:
    pass
