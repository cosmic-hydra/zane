"""
AlphaFold3 proxy module for ligand binding prediction.
Uses DiffDock + OpenFold for structure/pocket, with Ray-distributed batching.
"""

from .alphafold3_docking import AlphaFold3Docking, AF3Result

__all__ = ["AlphaFold3Docking", "AF3Result"]