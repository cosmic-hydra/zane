"""ZANE Data — Molecular collection, featurization and datasets."""

try:
    from drug_discovery.data.collector import DataCollector as DataCollector
    from drug_discovery.data.dataset import (
        MolecularDataset as MolecularDataset,
    )
    from drug_discovery.data.dataset import (
        MolecularFeaturizer as MolecularFeaturizer,
    )
    from drug_discovery.data.dataset import (
        murcko_scaffold_kfold_split_molecular as murcko_scaffold_kfold_split_molecular,
    )
    from drug_discovery.data.dataset import (
        murcko_scaffold_split_molecular as murcko_scaffold_split_molecular,
    )
    from drug_discovery.data.dataset import (
        train_test_split_molecular as train_test_split_molecular,
    )

    __all__ = [
        "DataCollector",
        "MolecularDataset",
        "MolecularFeaturizer",
        "murcko_scaffold_split_molecular",
        "train_test_split_molecular",
    ]
except ImportError:
    # Keep data module lazy when dependencies are unavailable.
    pass
