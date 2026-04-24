"""NVIDIA NIM LLM integration for drug design.

Provides:
- NvidiaNIMClient: OpenAI-compatible client for NVIDIA NIM-hosted LLMs
  (e.g., Llama-3.1-Nemotron-70B) and the BioNeMo MolMIM molecular generation
  endpoint.
- ChemicalDatabaseHub: Multi-database connector (PubChem, ChEMBL, ZINC20,
  BindingDB, UniProt, RCSB PDB, KEGG, HMDB, and more) used to build rich
  chemical context for the LLM.
- NvidiaLLMDrugDesigner: High-level designer that combines database retrieval
  with LLM generation to design, combine, and optimize drug candidates.

Quick start
-----------
Set ``NVIDIA_NIM_API_KEY`` in your environment (obtain a key at
https://build.nvidia.com/explore/discover) then::

    from drug_discovery.nvidia_nim import NvidiaLLMDrugDesigner, DrugDesignRequest

    designer = NvidiaLLMDrugDesigner()
    result = designer.design(DrugDesignRequest(
        target="EGFR kinase inhibitor for NSCLC",
        num=5,
    ))
    print(result.molecules)

CLI usage::

    python -m drug_discovery.cli nvidia-gen \\
        --target "EGFR kinase inhibitor" \\
        --num 5 \\
        --databases pubchem chembl zinc20
"""

from __future__ import annotations

from drug_discovery.nvidia_nim.chem_databases import ChemicalDatabaseHub, DatabaseRecord
from drug_discovery.nvidia_nim.client import NvidiaNIMClient, NvidiaNIMConfig
from drug_discovery.nvidia_nim.llm_drug_designer import (
    DrugDesignRequest,
    DrugDesignResult,
    NvidiaLLMDrugDesigner,
)

__all__ = [
    "NvidiaNIMClient",
    "NvidiaNIMConfig",
    "ChemicalDatabaseHub",
    "DatabaseRecord",
    "DrugDesignRequest",
    "DrugDesignResult",
    "NvidiaLLMDrugDesigner",
]
