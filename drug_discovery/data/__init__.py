"""
Data Layer - Foundation for Ultra-SOTA Drug Discovery Platform

This module provides comprehensive data ingestion, normalization, and management
for multiple biomedical databases including ChEMBL, PubChem, PDB, DrugBank,
and ClinicalTrials.gov.
"""

from drug_discovery.data.collector import DataCollector
from drug_discovery.data.dataset import MolecularDataset
from drug_discovery.data.feature_store import FeatureStore
from drug_discovery.data.normalizer import DataNormalizer
from drug_discovery.data.versioning import DatasetVersioning

__all__ = [
    "DataCollector",
    "MolecularDataset",
    "FeatureStore",
    "DataNormalizer",
    "DatasetVersioning",
]
