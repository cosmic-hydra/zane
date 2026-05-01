from __future__ import annotations

import pytest

pytest.importorskip("pandas")

import pandas as pd

from drug_discovery.training.nvidia_llm_finetune import (
    build_training_frame,
    build_training_texts,
    collect_public_molecule_data,
    format_molecule_record,
    normalize_molecule_sources,
)


def test_normalize_molecule_sources_filters_and_deduplicates():
    assert normalize_molecule_sources(["PubChem", "chembl", "chembl", "unknown", "approved_drugs"]) == [
        "pubchem",
        "chembl",
        "approved_drugs",
    ]


def test_format_molecule_record_includes_metadata():
    text = format_molecule_record(pd.Series({"source": "pubchem", "name": "Aspirin", "smiles": "CC(=O)O"}))

    assert "Source: pubchem" in text
    assert "Name: Aspirin" in text
    assert "SMILES: CC(=O)O" in text


def test_build_training_texts_and_frame():
    df = pd.DataFrame(
        [
            {"source": "pubchem", "name": "Aspirin", "smiles": "CC(=O)O"},
            {"source": "chembl", "name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
        ]
    )

    texts = build_training_texts(df)
    frame = build_training_frame(df)

    assert len(texts) == 2
    assert list(frame.columns) == ["text"]
    assert len(frame) == 2


class _MockCollector:
    def __init__(self):
        self.pubchem_called = False
        self.chembl_called = False
        self.approved_called = False
        self.drugbank_called = False

    def collect_from_pubchem(self, query: str, limit: int):
        self.pubchem_called = True
        return pd.DataFrame([{"source": "pubchem", "name": query, "smiles": "CC(=O)O"}])

    def collect_from_chembl(self, target=None, limit: int = 0, activity_type=None):
        self.chembl_called = True
        return pd.DataFrame([{"source": "chembl", "name": target or "chembl-hit", "smiles": "CCO"}])

    def collect_approved_drugs(self):
        self.approved_called = True
        return pd.DataFrame([{"source": "builtin", "name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}])

    def collect_from_drugbank(self, file_path=None, limit: int = 0):
        self.drugbank_called = True
        return pd.DataFrame([{"source": "drugbank", "name": "Caffeine", "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C"}])

    def merge_datasets(self, datasets):
        valid = [df for df in datasets if df is not None and not df.empty]
        if not valid:
            return pd.DataFrame(columns=["source", "name", "smiles"])
        return pd.concat(valid, ignore_index=True)


def test_collect_public_molecule_data_uses_requested_sources():
    collector = _MockCollector()

    df = collect_public_molecule_data(
        collector=collector,
        sources=["pubchem", "chembl", "approved_drugs", "drugbank"],
        pubchem_query="aspirin",
        chembl_target="kinase",
        limit_per_source=2,
    )

    assert not df.empty
    assert collector.pubchem_called is True
    assert collector.chembl_called is True
    assert collector.approved_called is True
    assert collector.drugbank_called is True


def test_collect_public_molecule_data_falls_back_to_approved_drugs():
    collector = _MockCollector()

    def empty_merge(datasets):
        return pd.DataFrame(columns=["source", "name", "smiles"])

    collector.merge_datasets = empty_merge

    df = collect_public_molecule_data(collector=collector, sources=["pubchem"], fallback_to_approved_drugs=True)

    assert not df.empty
    assert collector.approved_called is True