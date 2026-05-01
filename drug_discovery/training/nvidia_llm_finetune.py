"""Helpers for local NVIDIA LLM fine-tuning on public molecule databases."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import pandas as pd

from drug_discovery.data.collector import DataCollector

SUPPORTED_MOLECULE_SOURCES = ("pubchem", "chembl", "approved_drugs", "drugbank")


def normalize_molecule_sources(sources: Sequence[str] | None) -> list[str]:
    """Return a deduplicated, validated source list."""
    if sources is None:
        return list(SUPPORTED_MOLECULE_SOURCES)

    normalized: list[str] = []
    seen: set[str] = set()
    for source in sources:
        cleaned = str(source).strip().lower()
        if cleaned not in SUPPORTED_MOLECULE_SOURCES or cleaned in seen:
            continue
        normalized.append(cleaned)
        seen.add(cleaned)
    return normalized


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def collect_public_molecule_data(
    collector: DataCollector,
    sources: Sequence[str] | None = None,
    pubchem_query: str = "drug",
    chembl_target: str | None = None,
    chembl_activity_type: str | None = None,
    limit_per_source: int = 5000,
    drugbank_file: str | None = None,
    fallback_to_approved_drugs: bool = True,
) -> pd.DataFrame:
    """Collect and merge public molecule records suitable for local LM fine-tuning."""
    selected_sources = normalize_molecule_sources(sources)
    frames: list[pd.DataFrame] = []

    if "pubchem" in selected_sources:
        frames.append(collector.collect_from_pubchem(query=pubchem_query, limit=limit_per_source))

    if "chembl" in selected_sources:
        frames.append(
            collector.collect_from_chembl(
                target=chembl_target,
                limit=limit_per_source,
                activity_type=chembl_activity_type,
            )
        )

    if "approved_drugs" in selected_sources:
        frames.append(collector.collect_approved_drugs())

    if "drugbank" in selected_sources:
        frames.append(collector.collect_from_drugbank(file_path=drugbank_file, limit=limit_per_source))

    merged = collector.merge_datasets(frames)
    if merged.empty and fallback_to_approved_drugs and "approved_drugs" not in selected_sources:
        merged = collector.collect_approved_drugs()
    return merged


def format_molecule_record(row: Mapping[str, Any] | pd.Series) -> str:
    """Convert a molecule row into a text document for causal LM training."""
    source = _clean_text(row.get("source", "unknown"))
    name = _clean_text(row.get("name", ""))
    smiles = _clean_text(row.get("smiles", ""))

    lines = [
        "Molecule record",
        f"Source: {source or 'unknown'}",
    ]
    if name:
        lines.append(f"Name: {name}")
    lines.append(f"SMILES: {smiles}")
    lines.append(
        "Purpose: Continue learning public chemistry records for local medicinal chemistry fine-tuning."
    )
    return "\n".join(lines)


def build_training_texts(df: pd.DataFrame) -> list[str]:
    """Turn a merged molecule dataframe into text samples for a causal LM."""
    if df is None or df.empty:
        return []

    texts: list[str] = []
    for _, row in df.iterrows():
        text = format_molecule_record(row)
        if text.strip():
            texts.append(text)
    return texts


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a one-column dataframe ready for tokenization."""
    texts = build_training_texts(df)
    return pd.DataFrame({"text": texts})
