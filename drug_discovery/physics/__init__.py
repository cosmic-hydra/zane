"""ZANE Physics — Docking, molecular dynamics, physics-based scoring."""

__all__ = []
try:
    from drug_discovery.physics.docking import (
        DockingPipeline, DockingConfig, DockingResult, VinaBackend)
    __all__.extend(["DockingPipeline", "DockingConfig", "DockingResult", "VinaBackend"])
except ImportError:
    pass

try:
    from drug_discovery.physics.diffdock_adapter import DiffDockAdapter

    __all__.append("DiffDockAdapter")
except Exception:
    pass

try:
    from drug_discovery.physics.openmm_adapter import OpenMMAdapter

    __all__.append("OpenMMAdapter")
except Exception:
    pass

try:
    from drug_discovery.physics.protein_structure import OpenFoldAdapter

    __all__.append("OpenFoldAdapter")
except Exception:
    pass
