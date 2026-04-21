"""ZANE Data — Molecular data collection, featurization, pipelines."""

__all__ = []

try:
    from drug_discovery.data.collector import DataCollector

    __all__.append("DataCollector")
except Exception:
    pass

try:
    from drug_discovery.data.dataset import (
        MolecularDataset,
        MolecularFeaturizer,
        murcko_scaffold_split_molecular,
        murcko_scaffold_kfold_split_molecular,
        train_test_split_molecular,
    )

    __all__.extend(
        [
            "MolecularDataset",
            "MolecularFeaturizer",
            "murcko_scaffold_split_molecular",
            "murcko_scaffold_kfold_split_molecular",
            "train_test_split_molecular",
        ]
    )
except Exception:
    pass

try:
    from drug_discovery.data.pipeline import (
        MolecularDataset as PipelineDataset,
        validate_smiles,
        validate_batch,
        compute_descriptors,
        compute_morgan_fingerprint,
        smiles_to_graph,
        lipinski_filter,
        tanimoto_similarity,
        is_valid_smiles_fast,
    )

    __all__.extend(
        [
            "PipelineDataset",
            "validate_smiles",
            "validate_batch",
            "compute_descriptors",
            "compute_morgan_fingerprint",
            "smiles_to_graph",
            "lipinski_filter",
            "tanimoto_similarity",
            "is_valid_smiles_fast",
        ]
    )
except Exception:
    pass

try:
    from drug_discovery.data.feature_store import FeatureStore

    __all__.append("FeatureStore")
except Exception:
    pass

try:
    from drug_discovery.data.normalizer import DataNormalizer

    __all__.append("DataNormalizer")
except Exception:
    pass

try:
    from drug_discovery.data.versioning import DatasetVersioning

    __all__.append("DatasetVersioning")
except Exception:
    pass
