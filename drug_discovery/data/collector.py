"""Data collection helpers for molecular and biomedical datasets."""

from __future__ import annotations

import logging
import time
import asyncio
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import pandas as pd

try:
    from playwright.async_api import async_playwright
    from drug_discovery.web_scraping import BiomedicalScraper, WebDataProcessor
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar("T")

def _require_playwright():
    if not _PLAYWRIGHT_AVAILABLE:
        raise ImportError(
            "Web scraping requires playwright. Install with: "
            "pip install 'zane[scraping]' && playwright install chromium"
        )

async def fetch_page(url: str, browser=None) -> str:
    """Fetch rendered HTML content using Playwright (reusable browser)."""
    _require_playwright()
    
    if browser:
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        content = await page.content()
        await page.close()
        return content
        
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        content = await page.content()
        await browser.close()
    return content


class DataCollector:
    """Collect and merge records from multiple sources with safe fallbacks."""

    def __init__(self, cache_dir: str = "./data/cache", api_keys: dict[str, str] | None = None):
        self.cache_dir = cache_dir
        self.api_keys = api_keys or {}
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _empty_molecule_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=["smiles", "name", "source"])

    @staticmethod
    def _sanitize_smiles_frame(df: pd.DataFrame) -> pd.DataFrame:
        if "smiles" not in df.columns:
            return pd.DataFrame(columns=["smiles", "name", "source"])
        out = df.copy()
        out["smiles"] = out["smiles"].astype(str).str.strip()
        out = out[out["smiles"] != ""]
        out = out.loc[~out["smiles"].duplicated()].reset_index(drop=True)
        if "name" not in out.columns:
            out["name"] = ""
        if "source" not in out.columns:
            out["source"] = "unknown"
        return out

    @staticmethod
    def _is_valid_smiles(smiles: str) -> bool:
        if not smiles or not isinstance(smiles, str):
            return False
        if "INVALID" in smiles.upper():
            return False
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(str(smiles))
            return mol is not None and mol.GetNumAtoms() > 0
        except Exception:
            return bool(str(smiles).strip())

    @staticmethod
    def _extract_pubchem_smiles(compound: Any) -> str | None:
        # Prefer current field to avoid deprecation warnings in PubChemPy.
        for attr in ("connectivity_smiles", "isomeric_smiles", "canonical_smiles"):
            value = getattr(compound, attr, None)
            if value:
                return str(value)
        return None

    def _with_retry(self, fn: Callable[[], T], max_retries: int = 3, backoff_seconds: float = 0.5) -> T:
        """Run a call with bounded retries and exponential backoff."""
        retries = max(1, int(max_retries))
        backoff = max(0.0, float(backoff_seconds))
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt < retries - 1 and backoff > 0:
                    time.sleep(backoff * (2**attempt))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Retry helper failed without exception.")

    def collect_from_pubchem(self, query: str = "drug", limit: int = 100, namespace: str = "name") -> pd.DataFrame:
        """Collect compounds from PubChem when available."""
        try:
            import pubchempy as pcp
        except Exception:
            logger.warning("pubchempy not installed; returning empty PubChem result.")
            return self._empty_molecule_frame()

        def _fetch() -> list[Any]:
            return pcp.get_compounds(query, namespace=namespace)[: max(0, int(limit))]

        rows: list[dict[str, str]] = []
        try:
            compounds = self._with_retry(_fetch)
            for compound in compounds:
                smiles = self._extract_pubchem_smiles(compound)
                if smiles:
                    rows.append(
                        {
                            "smiles": smiles,
                            "name": str(getattr(compound, "iupac_name", "") or ""),
                            "source": "pubchem",
                        }
                    )
        except Exception as exc:
            logger.warning("PubChem collection failed: %s", exc)
            return self._empty_molecule_frame()

        out = self._sanitize_smiles_frame(pd.DataFrame(rows))
        if out.empty:
            return out
        mask = out["smiles"].map(self._is_valid_smiles).astype(bool)
        filtered = cast(pd.DataFrame, out.loc[mask].copy().reset_index(drop=True))
        return filtered

    def collect_from_chembl(
        self,
        target: str | None = None,
        limit: int = 100,
        activity_type: str | None = None,
    ) -> pd.DataFrame:
        """Collect compounds from ChEMBL when available."""
        try:
            from chembl_webresource_client.new_client import new_client
        except Exception:
            logger.warning("chembl client not installed; returning empty ChEMBL result.")
            return self._empty_molecule_frame()

        def _fetch() -> Any:
            if target or activity_type:
                query = new_client.activity
                if target:
                    target_results = new_client.target.filter(pref_name__icontains=target)
                    target_results = list(target_results[:1])
                    if target_results:
                        query = query.filter(target_chembl_id=target_results[0].get("target_chembl_id"))
                if activity_type:
                    query = query.filter(standard_type=activity_type)
                return query.only(["molecule_chembl_id", "canonical_smiles", "molecule_pref_name"])[: max(0, int(limit))]
            return new_client.molecule.filter(molecule_structures__isnull=False).only(
                ["molecule_chembl_id", "molecule_structures", "pref_name"]
            )[: max(0, int(limit))]

        rows: list[dict[str, str]] = []
        try:
            records = self._with_retry(_fetch)
            for item in records:
                smiles = None
                if isinstance(item, dict):
                    structures = item.get("molecule_structures") or {}
                    smiles = structures.get("canonical_smiles") or item.get("canonical_smiles")
                if smiles:
                    rows.append(
                        {
                            "smiles": str(smiles),
                            "name": str(item.get("pref_name") or item.get("molecule_pref_name") or item.get("molecule_chembl_id") or ""),
                            "source": "chembl",
                        }
                    )
        except Exception as exc:
            logger.warning("ChEMBL collection failed: %s", exc)
            return self._empty_molecule_frame()

        out = self._sanitize_smiles_frame(pd.DataFrame(rows))
        if out.empty:
            return out
        mask = out["smiles"].map(self._is_valid_smiles).astype(bool)
        filtered = cast(pd.DataFrame, out.loc[mask].copy().reset_index(drop=True))
        return filtered

    def collect_approved_drugs(self) -> pd.DataFrame:
        """Return a compact built-in approved-drug seed set."""
        rows = [
            {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "name": "Aspirin", "source": "builtin"},
            {"smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "name": "Ibuprofen", "source": "builtin"},
            {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "name": "Caffeine", "source": "builtin"},
        ]
        return self._sanitize_smiles_frame(pd.DataFrame(rows))

    def collect_from_drugbank(self, file_path: str | None = None, limit: int = 1000) -> pd.DataFrame:
        """Collect molecules from a DrugBank export file (CSV/TSV)."""
        path_str = (file_path or "").strip() or str(Path(self.cache_dir) / "drugbank.csv")
        path = Path(path_str)
        if not path.exists() or not path.is_file():
            logger.warning("DrugBank file not found at %s; returning empty DrugBank result.", path)
            return self._empty_molecule_frame()

        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        try:
            raw = pd.read_csv(path, sep=sep)
        except Exception as exc:
            logger.warning("Failed to parse DrugBank file %s: %s", path, exc)
            return self._empty_molecule_frame()

        smiles_col = next(
            (
                c
                for c in ["smiles", "SMILES", "canonical_smiles", "drugbank_smiles", "structure_smiles"]
                if c in raw.columns
            ),
            None,
        )
        if smiles_col is None:
            logger.warning("DrugBank file %s has no supported SMILES column.", path)
            return self._empty_molecule_frame()

        name_col = next(
            (c for c in ["name", "drug_name", "generic_name", "drugbank_id", "drugbank-id", "id"] if c in raw.columns),
            None,
        )

        limited = raw.head(max(0, int(limit))).copy()
        rows = pd.DataFrame(
            {
                "smiles": limited[smiles_col].astype(str),
                "name": limited[name_col].astype(str) if name_col else "",
                "source": "drugbank",
            }
        )
        out = self._sanitize_smiles_frame(rows)
        if out.empty:
            return out
        mask = out["smiles"].map(self._is_valid_smiles).astype(bool)
        filtered = cast(pd.DataFrame, out.loc[mask].copy().reset_index(drop=True))
        return filtered

    def collect_from_pdb(
        self,
        pdb_ids: list[str] | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Collect Protein Data Bank metadata."""
        try:
            import requests
        except Exception:
            logger.warning("requests not installed; returning empty PDB result.")
            return pd.DataFrame()

        base_url = "https://data.rcsb.org/rest/v1/core/entry/"
        ids = pdb_ids or []

        if not ids:
            try:
                search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
                payload = {
                    "query": {"type": "terminal", "service": "text", "parameters": {"value": query or "drug"}},
                    "return_type": "entry",
                    "request_options": {"paginate": {"start": 0, "rows": max(0, int(limit))}},
                }
                response = requests.post(search_url, json=payload, timeout=20)
                response.raise_for_status()
                ids = [item.get("identifier") for item in response.json().get("result_set", []) if item.get("identifier")]
            except Exception as exc:
                logger.warning("PDB search failed: %s", exc)
                return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for pdb_id in ids[: max(0, int(limit))]:
            try:
                response = requests.get(base_url + str(pdb_id), timeout=20)
                response.raise_for_status()
                entry = response.json()
                rows.append(
                    {
                        "pdb_id": str(pdb_id),
                        "title": entry.get("struct", {}).get("title", ""),
                        "resolution": (entry.get("refine") or [{}])[0].get("ls_d_res_high"),
                        "deposition_date": entry.get("rcsb_accession_info", {}).get("deposit_date"),
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(rows)

    def collect_from_clinical_trials(
        self,
        condition: str | None = None,
        intervention: str | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Collect clinical trial metadata from ClinicalTrials.gov."""
        try:
            import requests
        except Exception:
            logger.warning("requests not installed; returning empty clinical trials result.")
            return pd.DataFrame()

        params = {"format": "json", "pageSize": min(max(0, int(limit)), 1000)}
        if condition:
            params["query.cond"] = condition
        if intervention:
            params["query.intr"] = intervention

        try:
            response = requests.get("https://clinicaltrials.gov/api/v2/studies", params=params, timeout=20)
            response.raise_for_status()
            studies = response.json().get("studies", [])
        except Exception as exc:
            logger.warning("ClinicalTrials.gov collection failed: %s", exc)
            return pd.DataFrame()

        rows = []
        for study in studies[: max(0, int(limit))]:
            protocol = study.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})
            rows.append(
                {
                    "nct_id": ident.get("nctId"),
                    "title": ident.get("briefTitle"),
                    "status": status.get("overallStatus"),
                    "phase": (protocol.get("designModule", {}).get("phases") or ["N/A"])[0],
                    "enrollment": status.get("enrollmentInfo", {}).get("count"),
                }
            )
        return pd.DataFrame(rows)

    def collect_multi_source(
        self,
        sources: list[str],
        query: str | None = None,
        limit_per_source: int = 100,
    ) -> dict[str, pd.DataFrame]:
        """Collect from multiple sources and return frames keyed by source name."""
        results: dict[str, pd.DataFrame] = {}
        if "chembl" in sources:
            results["chembl"] = self.collect_from_chembl(target=query, limit=limit_per_source)
        if "pubchem" in sources:
            results["pubchem"] = self.collect_from_pubchem(query=query or "aspirin", limit=limit_per_source)
        if "pdb" in sources:
            results["pdb"] = self.collect_from_pdb(query=query, limit=limit_per_source)
        if "clinical_trials" in sources:
            results["clinical_trials"] = self.collect_from_clinical_trials(condition=query, limit=limit_per_source)
        if "approved_drugs" in sources:
            results["approved_drugs"] = self.collect_approved_drugs()
        if "drugbank" in sources:
            results["drugbank"] = self.collect_from_drugbank(limit=limit_per_source)
        if "web_scraping" in sources and _PLAYWRIGHT_AVAILABLE:
            scraper = BiomedicalScraper()
            processor = WebDataProcessor()
            keywords = [query] if query else ["anticancer", "antibiotic"]
            articles = scraper.scrape_drug_research(keywords=keywords, max_per_keyword=limit_per_source // len(keywords))
            
            import pubchempy as pcp
            
            molecules = []
            seen_names = set()
            for art in articles:
                names = processor.extract_molecules(art.get("abstract", ""))
                for name in names:
                    if name.lower() in seen_names:
                        continue
                    seen_names.add(name.lower())
                    
                    smiles = None
                    try:
                        # Attempt to resolve name to SMILES via PubChem
                        pcp_results = pcp.get_compounds(name, "name")
                        if pcp_results:
                            smiles = pcp_results[0].canonical_smiles
                    except Exception:
                        pass
                    
                    if smiles:
                        molecules.append({"smiles": smiles, "name": name, "source": "web_scraping"})
            results["web_scraping"] = pd.DataFrame(molecules)
        return results

    def merge_datasets(self, datasets: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge molecular datasets by unique SMILES."""
        valid = [self._sanitize_smiles_frame(df) for df in datasets if df is not None and not df.empty]
        if not valid:
            return pd.DataFrame(columns=["smiles", "name", "source"])
        merged = pd.concat(valid, ignore_index=True, sort=False)
        merged = self._sanitize_smiles_frame(merged)
        if merged.empty:
            return merged
        mask = merged["smiles"].map(self._is_valid_smiles).astype(bool)
        filtered = cast(pd.DataFrame, merged.loc[mask].copy().reset_index(drop=True))
        return filtered

    def generate_data_quality_report(self, df: pd.DataFrame) -> dict[str, float | int]:
        """Generate quality metrics for molecule tables used in scientific workflows."""
        if df is None or df.empty:
            return {
                "total_rows": 0,
                "valid_smiles_rows": 0,
                "invalid_smiles_rows": 0,
                "duplicate_smiles_rows": 0,
                "unique_smiles": 0,
                "validity_ratio": 0.0,
            }

        total = len(df)
        smiles_series = df["smiles"].astype(str) if "smiles" in df.columns else pd.Series([], dtype=str)
        valid_mask = smiles_series.map(self._is_valid_smiles) if len(smiles_series) else pd.Series([], dtype=bool)
        valid_rows = int(valid_mask.sum()) if len(valid_mask) else 0
        invalid_rows = int(total - valid_rows)
        duplicates = int(total - smiles_series.nunique()) if len(smiles_series) else 0
        unique_smiles = int(smiles_series.nunique()) if len(smiles_series) else 0

        return {
            "total_rows": int(total),
            "valid_smiles_rows": valid_rows,
            "invalid_smiles_rows": invalid_rows,
            "duplicate_smiles_rows": max(0, duplicates),
            "unique_smiles": unique_smiles,
            "validity_ratio": float(valid_rows / total) if total > 0 else 0.0,
        }
