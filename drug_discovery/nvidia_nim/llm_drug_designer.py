"""NVIDIA LLM-aided drug designer.

Combines multi-database chemical knowledge retrieval with NVIDIA NIM-hosted
LLMs (Nemotron, Llama, Mistral, …) and the BioNeMo MolMIM microservice to
design, optimize, and combine drug candidates from scratch.

Workflow
--------
1.  Query multiple public chemical databases (PubChem, ChEMBL, ZINC20,
    BindingDB, UniProt, RCSB PDB, KEGG, HMDB, …) for relevant compounds
    and target information.
2.  Condense the retrieved records into a compact context block.
3.  Send the context + user design prompt to a NVIDIA NIM-hosted LLM
    (e.g., Llama-3.1-Nemotron-70B) specialised in computational chemistry.
4.  Parse SMILES strings from the LLM response.
5.  Optionally call BioNeMo MolMIM to generate additional candidates via
    latent-space optimisation around seed SMILES.

Example::

    from drug_discovery.nvidia_nim import NvidiaLLMDrugDesigner, DrugDesignRequest

    designer = NvidiaLLMDrugDesigner()
    result = designer.design(DrugDesignRequest(
        target="selective CDK2 inhibitor, oral bioavailability",
        num=8,
        seed_smiles=["c1ccc2ncccc2c1"],
        use_molmim=True,
    ))
    for smi in result.molecules:
        print(smi)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from drug_discovery.nvidia_nim.chem_databases import ChemicalDatabaseHub
from drug_discovery.nvidia_nim.client import NvidiaNIMClient, NvidiaNIMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Result data classes
# ---------------------------------------------------------------------------


@dataclass
class DrugDesignRequest:
    """Parameters for a single LLM-guided drug design run.

    Attributes:
        target: High-level design objective (e.g. ``"EGFR kinase inhibitor for NSCLC"``).
        prompt: Additional free-form design instructions (merged with *target*).
        seed_smiles: Scaffold / reference SMILES to guide generation.
        num: Number of candidate molecules requested from the LLM.
        databases: Database keys to query.  ``None`` uses all defaults.
        db_limit_per_source: Max records retrieved per database.
        use_molmim: If ``True``, also call BioNeMo MolMIM for each seed SMILES.
        temperature: LLM sampling temperature (higher → more diverse).
        pharmacophore: Optional pharmacophore constraints dict passed into the
            prompt (e.g. ``{"min_hba": 2, "max_rings": 3}``).
    """

    target: str = ""
    prompt: str = ""
    seed_smiles: list[str] = field(default_factory=list)
    num: int = 10
    databases: list[str] | None = None
    db_limit_per_source: int = 5
    use_molmim: bool = False
    temperature: float = 0.5
    pharmacophore: dict[str, Any] | None = None


@dataclass
class DrugDesignResult:
    """Outcome of an LLM-guided drug design run.

    Attributes:
        success: ``True`` when at least the LLM call succeeded.
        molecules: Parsed SMILES strings from the LLM response.
        reasoning: Full LLM response text (includes rationale for each design).
        db_context_summary: Human-readable summary of the database retrieval step.
        molmim_molecules: Additional SMILES from BioNeMo MolMIM (if requested).
        warnings: Non-fatal issues encountered during the run.
        error: Fatal error message (``None`` on success).
    """

    success: bool
    molecules: list[str]
    reasoning: str
    db_context_summary: str
    molmim_molecules: list[str]
    warnings: list[str]
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "molecules": self.molecules,
            "reasoning": self.reasoning,
            "db_context_summary": self.db_context_summary,
            "molmim_molecules": self.molmim_molecules,
            "warnings": self.warnings,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Designer
# ---------------------------------------------------------------------------


class NvidiaLLMDrugDesigner:
    """NVIDIA-powered LLM drug designer backed by multi-database knowledge retrieval.

    Requires the ``openai`` Python package and ``NVIDIA_NIM_API_KEY`` env var.
    Without them, :meth:`design` returns a structured error (no exception).

    Parameters
    ----------
    nim_client:
        Optional pre-configured :class:`~drug_discovery.nvidia_nim.client.NvidiaNIMClient`.
    db_hub:
        Optional pre-configured :class:`~drug_discovery.nvidia_nim.chem_databases.ChemicalDatabaseHub`.
    nim_config:
        Optional :class:`~drug_discovery.nvidia_nim.client.NvidiaNIMConfig`
        used when *nim_client* is not provided.
    """

    def __init__(
        self,
        nim_client: NvidiaNIMClient | None = None,
        db_hub: ChemicalDatabaseHub | None = None,
        nim_config: NvidiaNIMConfig | None = None,
    ) -> None:
        self.nim = nim_client or NvidiaNIMClient(nim_config)
        self.db = db_hub or ChemicalDatabaseHub()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def design(self, request: DrugDesignRequest) -> DrugDesignResult:
        """Main entry-point: retrieve DB context → call LLM → return results.

        Steps:

        1. Query all configured databases using *request.target* (or *request.prompt*)
           as the search query.
        2. Build a compact context block from the retrieved records.
        3. Call the NVIDIA NIM LLM with the context + prompt.
        4. Optionally run BioNeMo MolMIM on each seed SMILES.
        5. Return a :class:`DrugDesignResult`.
        """
        warnings: list[str] = []
        db_context = ""
        db_summary = ""

        # Step 1: database retrieval
        query = request.target or request.prompt or "drug-like molecule"
        try:
            db_results = self.db.search_all(
                query=query,
                limit_per_source=request.db_limit_per_source,
                sources=request.databases,
            )
            db_context = self.db.build_context_string(db_results)
            total = sum(len(v) for v in db_results.values())
            db_summary = (
                f"Retrieved {total} records from "
                f"{len(db_results)} database(s): {', '.join(db_results.keys())}."
            )
        except Exception as exc:
            warnings.append(f"Database retrieval error: {exc}")

        # Step 2: early exit if NIM is unavailable
        if not self.nim.is_available():
            return DrugDesignResult(
                success=False,
                molecules=[],
                reasoning="",
                db_context_summary=db_summary,
                molmim_molecules=[],
                warnings=warnings,
                error=(
                    "NVIDIA NIM not available. "
                    "Set NVIDIA_NIM_API_KEY and install the 'openai' Python package "
                    "(pip install openai).  API keys: https://build.nvidia.com/explore/discover"
                ),
            )

        # Step 3: build prompt
        prompt_parts: list[str] = []
        if request.target:
            prompt_parts.append(f"Target / objective: {request.target}")
        if request.pharmacophore:
            prompt_parts.append(f"Pharmacophore constraints: {request.pharmacophore}")
        if request.prompt:
            prompt_parts.append(request.prompt)
        if not prompt_parts:
            prompt_parts.append("Design novel, diverse, drug-like molecules.")
        full_prompt = "\n".join(prompt_parts)

        # Step 4: LLM call
        llm_result = self.nim.generate_molecules(
            prompt=full_prompt,
            num=request.num,
            context=db_context or None,
            seed_smiles=request.seed_smiles or None,
            temperature=request.temperature,
        )
        if not llm_result["success"]:
            return DrugDesignResult(
                success=False,
                molecules=[],
                reasoning="",
                db_context_summary=db_summary,
                molmim_molecules=[],
                warnings=warnings,
                error=llm_result.get("error"),
            )

        molecules: list[str] = llm_result["molecules"]
        reasoning: str = llm_result["reasoning"]

        # Step 5: optional MolMIM
        molmim_mols: list[str] = []
        if request.use_molmim and request.seed_smiles:
            for seed in request.seed_smiles[:2]:  # cap at 2 to conserve API quota
                mm_result = self.nim.generate_molmim(seed, num_molecules=5)
                if mm_result["success"]:
                    molmim_mols.extend(mm_result["molecules"])
                else:
                    warnings.append(
                        f"MolMIM failed for seed '{seed}': {mm_result.get('error')}"
                    )

        return DrugDesignResult(
            success=True,
            molecules=molecules,
            reasoning=reasoning,
            db_context_summary=db_summary,
            molmim_molecules=molmim_mols,
            warnings=warnings,
            error=None,
        )

    def combine(
        self,
        smiles_list: list[str],
        strategy: str = "scaffold_hop",
        num: int = 5,
    ) -> DrugDesignResult:
        """Ask the LLM to combine, merge, or scaffold-hop a set of molecules.

        Args:
            smiles_list: Input molecules whose features should be merged.
            strategy: Combination strategy hint passed to the LLM
                (e.g. ``"scaffold_hop"``, ``"fragment_merge"``, ``"bioisostere"``).
            num: Number of novel hybrid molecules to produce.

        Returns:
            :class:`DrugDesignResult` with merged/hopped SMILES.
        """
        if not smiles_list:
            return DrugDesignResult(
                success=False,
                molecules=[],
                reasoning="",
                db_context_summary="",
                molmim_molecules=[],
                warnings=[],
                error="smiles_list cannot be empty",
            )
        smiles_block = "\n".join(f"- {s}" for s in smiles_list)
        prompt = (
            f"Strategy: {strategy}\n"
            f"Input molecules:\n{smiles_block}\n\n"
            "Combine, merge, or scaffold-hop the above molecules to produce novel hybrids. "
            f"Output {num} new SMILES with brief reasoning."
        )
        request = DrugDesignRequest(prompt=prompt, num=num, seed_smiles=list(smiles_list))
        return self.design(request)

    def optimize(
        self,
        smiles: str,
        objectives: list[str] | None = None,
        num: int = 5,
    ) -> DrugDesignResult:
        """Ask the LLM to propose optimized analogues of a given lead molecule.

        Args:
            smiles: Lead molecule SMILES to optimise.
            objectives: List of optimisation objectives
                (default: improve QED, reduce MW, maintain target affinity).
            num: Number of optimised analogues to generate.

        Returns:
            :class:`DrugDesignResult` with optimised analogues.
        """
        objs = objectives or ["improve QED", "reduce MW", "maintain target affinity"]
        obj_str = "; ".join(objs)
        prompt = (
            f"Lead molecule: {smiles}\n"
            f"Optimization objectives: {obj_str}\n\n"
            f"Propose {num} optimized analogues that address these objectives."
        )
        request = DrugDesignRequest(prompt=prompt, seed_smiles=[smiles], num=num)
        return self.design(request)
