"""ZANE Data — Molecular data collection, featurization, pipelines."""

__all__ = ["DataCollector", "fetch_page"]
try:
    from drug_discovery.data.collector import DataCollector, fetch_page
    from drug_discovery.data.dataset import (
        MolecularDataset, MolecularFeaturizer,
        train_test_split_molecular, murcko_scaffold_split_molecular,
        murcko_scaffold_kfold_split_molecular)
    from drug_discovery.data.pipeline import (
        validate_smiles, validate_batch,
        compute_descriptors, compute_morgan_fingerprint,
        smiles_to_graph, lipinski_filter, tanimoto_similarity, is_valid_smiles_fast)
    __all__.extend(["MolecularDataset", "MolecularFeaturizer",
        "train_test_split_molecular", "murcko_scaffold_split_molecular",
        "murcko_scaffold_kfold_split_molecular",
        "validate_smiles", "validate_batch",
        "compute_descriptors", "compute_morgan_fingerprint", "smiles_to_graph",
        "lipinski_filter", "tanimoto_similarity", "is_valid_smiles_fast"])
except ImportError:
    pass
