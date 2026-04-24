"""Multi-database chemical knowledge hub.

Queries publicly available chemical and biological databases to provide rich,
up-to-date context for LLM-aided drug design.  Every database call is
optional and fails gracefully — a slow or unavailable endpoint simply
contributes no records rather than aborting the overall search.

Supported databases
-------------------
- **PubChem**    – general compound repository (pubchempy + PUG REST)
- **ChEMBL**     – bioactivity database (chembl-webresource-client)
- **ZINC20**     – purchasable compound library (REST)
- **BindingDB**  – protein-ligand binding data (REST)
- **UniProt**    – protein / target information (REST)
- **RCSB PDB**   – protein structure data (REST)
- **KEGG**       – pathway / metabolite data (REST)
- **HMDB**       – Human Metabolome Database (REST)

Additional databases can be plugged in by subclassing or by adding entries to
``ChemicalDatabaseHub.DEFAULT_SOURCES`` and corresponding ``search_*`` methods.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15  # seconds per request
_DEFAULT_LIMIT = 20


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DatabaseRecord:
    """A single chemical or biological record retrieved from a database."""

    smiles: str = ""
    name: str = ""
    source: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    target_info: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_context_line(self) -> str:
        """Format the record as a compact, LLM-readable single line."""
        parts: list[str] = []
        if self.name:
            parts.append(f"Name: {self.name}")
        if self.smiles:
            parts.append(f"SMILES: {self.smiles}")
        if self.target_info:
            parts.append(f"Target: {self.target_info}")
        if self.properties:
            prop_str = ", ".join(f"{k}={v}" for k, v in list(self.properties.items())[:5])
            parts.append(f"Properties: {prop_str}")
        parts.append(f"[Source: {self.source}]")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------


class ChemicalDatabaseHub:
    """Aggregator that queries multiple public chemical/biological databases.

    Example::

        hub = ChemicalDatabaseHub()
        results = hub.search_all("EGFR inhibitor", limit_per_source=5)
        context = hub.build_context_string(results)
        print(context)
    """

    #: Sources queried by default in :meth:`search_all`.
    DEFAULT_SOURCES: list[str] = [
        "pubchem",
        "chembl",
        "zinc20",
        "bindingdb",
        "uniprot",
        "pdb",
        "kegg",
        "hmdb",
    ]

    def __init__(
        self,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = 2,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = session

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Resilient GET with retries and exponential back-off."""
        session = self._get_session()
        for attempt in range(max(1, self.max_retries)):
            try:
                resp = session.get(
                    url,
                    params=params,
                    headers=headers or {},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (2**attempt))
                else:
                    logger.debug("Database GET failed (%s): %s", url, exc)
        return None

    # ------------------------------------------------------------------
    # PubChem
    # ------------------------------------------------------------------

    def search_pubchem(self, query: str, limit: int = _DEFAULT_LIMIT) -> list[DatabaseRecord]:
        """Search PubChem by name via pubchempy (PUG REST fallback)."""
        records: list[DatabaseRecord] = []
        try:
            import pubchempy as pcp  # type: ignore[import]

            compounds = pcp.get_compounds(query, "name", listkey_count=limit)[:limit]
            for c in compounds:
                smiles = (
                    getattr(c, "isomeric_smiles", None)
                    or getattr(c, "canonical_smiles", None)
                    or ""
                )
                props: dict[str, Any] = {}
                for attr in (
                    "molecular_weight",
                    "xlogp",
                    "h_bond_donor_count",
                    "h_bond_acceptor_count",
                    "tpsa",
                ):
                    val = getattr(c, attr, None)
                    if val is not None:
                        props[attr] = val
                records.append(
                    DatabaseRecord(
                        smiles=smiles,
                        name=getattr(c, "iupac_name", None) or query,
                        source="PubChem",
                        properties=props,
                        raw={"cid": c.cid},
                    )
                )
        except Exception as exc:
            logger.debug("PubChem search failed: %s", exc)
        return records

    # ------------------------------------------------------------------
    # ChEMBL
    # ------------------------------------------------------------------

    def search_chembl(self, query: str, limit: int = _DEFAULT_LIMIT) -> list[DatabaseRecord]:
        """Search ChEMBL compound database by preferred name."""
        records: list[DatabaseRecord] = []
        try:
            from chembl_webresource_client.new_client import new_client  # type: ignore[import]

            results = (
                new_client.molecule.filter(pref_name__icontains=query)
                .only(["molecule_chembl_id", "pref_name", "molecule_structures", "molecule_properties"])[:limit]
            )
            for res in results:
                structs = res.get("molecule_structures") or {}
                smiles = structs.get("canonical_smiles", "")
                props_raw = res.get("molecule_properties") or {}
                props: dict[str, Any] = {
                    k: props_raw[k]
                    for k in ("mw_freebase", "alogp", "hba", "hbd", "psa", "ro3_pass")
                    if props_raw.get(k) is not None
                }
                records.append(
                    DatabaseRecord(
                        smiles=smiles,
                        name=res.get("pref_name") or res.get("molecule_chembl_id", ""),
                        source="ChEMBL",
                        properties=props,
                        raw={"chembl_id": res.get("molecule_chembl_id")},
                    )
                )
        except Exception as exc:
            logger.debug("ChEMBL search failed: %s", exc)
        return records

    # ------------------------------------------------------------------
    # ZINC20
    # ------------------------------------------------------------------

    def search_zinc(self, query: str, limit: int = _DEFAULT_LIMIT) -> list[DatabaseRecord]:
        """Search ZINC20 via its public REST API."""
        records: list[DatabaseRecord] = []
        try:
            url = "https://zinc20.docking.org/substances/search.json"
            data = self._get(url, params={"q": query, "count": min(limit, 100)})
            if not data:
                return records
            items = data if isinstance(data, list) else data.get("results", [])
            for item in items[:limit]:
                smiles = item.get("smiles", "")
                records.append(
                    DatabaseRecord(
                        smiles=smiles,
                        name=item.get("name") or item.get("zinc_id", ""),
                        source="ZINC20",
                        properties={
                            k: item[k]
                            for k in ("mwt", "logp", "rb", "hba", "hbd")
                            if k in item
                        },
                        raw={"zinc_id": item.get("zinc_id")},
                    )
                )
        except Exception as exc:
            logger.debug("ZINC20 search failed: %s", exc)
        return records

    # ------------------------------------------------------------------
    # BindingDB
    # ------------------------------------------------------------------

    def search_bindingdb(self, query: str, limit: int = _DEFAULT_LIMIT) -> list[DatabaseRecord]:
        """Search BindingDB for ligand-target binding data."""
        records: list[DatabaseRecord] = []
        try:
            url = "https://bindingdb.org/axis2/services/BDBService/getLigandsByName"
            data = self._get(url, params={"ligandname": query, "response": "application/json"})
            if not data:
                return records
            raw_affinities = (
                data.get("getLigandsByNameResponse", {}).get("affinities", [])
            )
            affinities = raw_affinities if isinstance(raw_affinities, list) else [raw_affinities]
            for item in affinities[:limit]:
                ligand = item.get("ligand") if isinstance(item.get("ligand"), dict) else {}
                target_d = item.get("target") if isinstance(item.get("target"), dict) else {}
                smiles = ligand.get("smiles", "")
                target = target_d.get("name", "")
                records.append(
                    DatabaseRecord(
                        smiles=smiles,
                        name=ligand.get("name", query),
                        source="BindingDB",
                        target_info=target,
                        properties={k: item[k] for k in ("IC50", "Ki", "Kd") if k in item},
                        raw=item,
                    )
                )
        except Exception as exc:
            logger.debug("BindingDB search failed: %s", exc)
        return records

    # ------------------------------------------------------------------
    # UniProt (target / protein information)
    # ------------------------------------------------------------------

    def search_uniprot(self, query: str, limit: int = _DEFAULT_LIMIT) -> list[DatabaseRecord]:
        """Search UniProt for protein/target information."""
        records: list[DatabaseRecord] = []
        try:
            url = "https://rest.uniprot.org/uniprotkb/search"
            params: dict[str, Any] = {
                "query": query,
                "format": "json",
                "size": min(limit, 25),
                "fields": "accession,protein_name,gene_names,organism_name,function",
            }
            data = self._get(url, params=params)
            if not data:
                return records
            for entry in (data.get("results") or [])[:limit]:
                protein_desc = (
                    entry.get("proteinDescription", {})
                    .get("recommendedName", {})
                    .get("fullName", {})
                    .get("value", "")
                )
                gene = ""
                genes = entry.get("genes", [])
                if genes:
                    gene = genes[0].get("geneName", {}).get("value", "")
                organism = entry.get("organism", {}).get("scientificName", "")
                records.append(
                    DatabaseRecord(
                        smiles="",
                        name=protein_desc or entry.get("uniProtkbId", ""),
                        source="UniProt",
                        target_info=f"Gene: {gene}; Organism: {organism}",
                        properties={"accession": entry.get("primaryAccession", "")},
                        raw={"accession": entry.get("primaryAccession")},
                    )
                )
        except Exception as exc:
            logger.debug("UniProt search failed: %s", exc)
        return records

    # ------------------------------------------------------------------
    # RCSB PDB
    # ------------------------------------------------------------------

    def search_pdb(self, query: str, limit: int = _DEFAULT_LIMIT) -> list[DatabaseRecord]:
        """Search the RCSB Protein Data Bank for structures related to a query."""
        records: list[DatabaseRecord] = []
        try:
            url = "https://search.rcsb.org/rcsbsearch/v2/query"
            payload: dict[str, Any] = {
                "query": {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": query},
                },
                "return_type": "entry",
                "request_options": {"paginate": {"start": 0, "rows": min(limit, 25)}},
            }
            session = self._get_session()
            resp = session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            for hit in (data.get("result_set") or [])[:limit]:
                pdb_id = hit.get("identifier", "")
                records.append(
                    DatabaseRecord(
                        smiles="",
                        name=pdb_id,
                        source="RCSB PDB",
                        target_info=f"PDB ID: {pdb_id}",
                        properties={"score": hit.get("score", 0)},
                        raw={"pdb_id": pdb_id},
                    )
                )
        except Exception as exc:
            logger.debug("RCSB PDB search failed: %s", exc)
        return records

    # ------------------------------------------------------------------
    # KEGG
    # ------------------------------------------------------------------

    def search_kegg(self, query: str, limit: int = _DEFAULT_LIMIT) -> list[DatabaseRecord]:
        """Search KEGG compound and drug databases."""
        records: list[DatabaseRecord] = []
        try:
            session = self._get_session()
            for db in ("compound", "drug"):
                encoded = requests.utils.quote(query)  # type: ignore[attr-defined]
                url = f"https://rest.kegg.jp/find/{db}/{encoded}"
                resp = session.get(url, timeout=self.timeout)
                if not resp.ok:
                    continue
                lines = resp.text.strip().splitlines()
                for line in lines[: limit // 2 + 1]:
                    parts = line.split("\t", 1)
                    if len(parts) != 2:
                        continue
                    kegg_id, name = parts
                    records.append(
                        DatabaseRecord(
                            smiles="",
                            name=name.split(";")[0].strip(),
                            source=f"KEGG ({db})",
                            properties={"kegg_id": kegg_id.strip()},
                            raw={"kegg_id": kegg_id.strip(), "db": db},
                        )
                    )
                if len(records) >= limit:
                    break
        except Exception as exc:
            logger.debug("KEGG search failed: %s", exc)
        return records[:limit]

    # ------------------------------------------------------------------
    # HMDB (Human Metabolome Database)
    # ------------------------------------------------------------------

    def search_hmdb(self, query: str, limit: int = _DEFAULT_LIMIT) -> list[DatabaseRecord]:
        """Search the Human Metabolome Database (HMDB)."""
        records: list[DatabaseRecord] = []
        try:
            url = "https://hmdb.ca/metabolites/search/get_results"
            params: dict[str, Any] = {"query[query_text]": query, "commit": "Search"}
            headers = {
                "Accept": "application/json",
                "X-Requested-With": "XMLHttpRequest",
            }
            data = self._get(url, params=params, headers=headers)
            if not data:
                return records
            results = data.get("results", data if isinstance(data, list) else [])
            for item in results[:limit]:
                smiles = item.get("smiles", "")
                records.append(
                    DatabaseRecord(
                        smiles=smiles,
                        name=item.get("name") or item.get("common_name", query),
                        source="HMDB",
                        properties={
                            k: item.get(k)
                            for k in ("molecular_formula", "average_molecular_weight", "status")
                            if item.get(k)
                        },
                        raw={"hmdb_id": item.get("accession")},
                    )
                )
        except Exception as exc:
            logger.debug("HMDB search failed: %s", exc)
        return records

    # ------------------------------------------------------------------
    # Aggregator
    # ------------------------------------------------------------------

    def search_all(
        self,
        query: str,
        limit_per_source: int = 5,
        sources: list[str] | None = None,
    ) -> dict[str, list[DatabaseRecord]]:
        """Search all (or selected) databases and return results grouped by source.

        Args:
            query: Search query (target name, disease, compound name, …).
            limit_per_source: Maximum records to retrieve per database.
            sources: Subset of database keys to query (default: all in
                :attr:`DEFAULT_SOURCES`).

        Returns:
            Mapping of ``source_key -> list[DatabaseRecord]``.
        """
        enabled = sources or self.DEFAULT_SOURCES
        dispatch: dict[str, Any] = {
            "pubchem": self.search_pubchem,
            "chembl": self.search_chembl,
            "zinc20": self.search_zinc,
            "bindingdb": self.search_bindingdb,
            "uniprot": self.search_uniprot,
            "pdb": self.search_pdb,
            "kegg": self.search_kegg,
            "hmdb": self.search_hmdb,
        }
        results: dict[str, list[DatabaseRecord]] = {}
        for src in enabled:
            fn = dispatch.get(src)
            if fn is None:
                logger.warning("Unknown database source: %s", src)
                continue
            try:
                records = fn(query, limit_per_source)
                if records:
                    results[src] = records
            except Exception as exc:
                logger.debug("Database '%s' error: %s", src, exc)
        return results

    def build_context_string(
        self,
        results: dict[str, list[DatabaseRecord]],
        max_records: int = 20,
    ) -> str:
        """Convert multi-database results into a compact LLM-readable context block.

        Args:
            results: Output of :meth:`search_all`.
            max_records: Maximum total records to include (keeps the context
                from growing too large for the LLM context window).

        Returns:
            A multi-line string suitable for inclusion in an LLM prompt.
        """
        lines: list[str] = []
        count = 0
        for source, records in results.items():
            lines.append(f"\n=== {source.upper()} ===")
            for rec in records:
                if count >= max_records:
                    break
                lines.append(rec.to_context_line())
                count += 1
            if count >= max_records:
                break
        return "\n".join(lines) if lines else "No database context available."
