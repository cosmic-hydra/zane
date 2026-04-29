"""ZANE — AI-native Drug Discovery Platform."""

__version__ = &quot;2026.4.1&quot;
__all__ = [&quot;__version__&quot;]

try:
    from drug_discovery.pipeline import DrugDiscoveryPipeline as DrugDiscoveryPipeline

    __all__.append(&quot;DrugDiscoveryPipeline&quot;)
except Exception:
    # Keep imports lazy when optional dependencies are unavailable.
    pass

# ... all existing try blocks ...

# New drug pipeline modules
try:
    from .data.rdkit_utils import smiles_to_sdf
    from .generation.torchdrug_generator import TorchDrugGenerator
    from .screening.admet_models import ADMETScreen
    from .screening.filtering import filter_admet
    from .docking.vina_wrapper import VinaDocker
    from .docking.diffdock_placeholder import run_diffdock
    from .structure_analysis.cif_parser import parse_cif_to_mol
    from .structure_analysis.xrpd_analysis import analyze_xrpd

    __all__.extend([
        &quot;TorchDrugGenerator&quot;,
        &quot;ADMETScreen&quot;,
        &quot;VinaDocker&quot;,
        &quot;run_diffdock&quot;,
        &quot;parse_cif_to_mol&quot;,
        &quot;analyze_xrpd&quot;,
        &quot;smiles_to_sdf&quot;,
        &quot;filter_admet&quot;
    ])
except Exception:
    pass