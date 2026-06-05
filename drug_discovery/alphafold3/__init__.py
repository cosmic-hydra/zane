"""
AlphaFold3 proxy module for ligand binding prediction.
Uses DiffDock + OpenFold for structure/pocket, with Ray-distributed batching.
"""

from .alphafold3_docking import AF3Result, AlphaFold3Docking

__all__ = ["AF3Result", "AlphaFold3Docking"]
