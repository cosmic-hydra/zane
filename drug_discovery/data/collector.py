"""
Multi-Source Data Collector for Biomedical Databases

Supports:
- ChEMBL (bioactivity data)
- PubChem (molecules, properties)
- Protein Data Bank (3D protein structures)
- DrugBank (drug + target info)
- ClinicalTrials.gov (trial outcomes)
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import pubchempy as pcp
from rdkit import Chem

logger = logging.getLogger(__name__)


class DataCollector:
    """Unified data collector for multiple biomedical databases."""

    def __init__(
        self,
        cache_dir: str = "./cache/data",
        api_keys: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the data collector.

        Args:
            cache_dir: Directory for caching downloaded data
            api_keys: Optional API keys for various services
        """
        self.cache_dir = cache_dir
        self.api_keys = api_keys or {}
        self._init_cache()

    def _init_cache(self):
        """Initialize cache directory."""
        import os

        os.makedirs(self.cache_dir, exist_ok=True)

    def collect_from_chembl(
        self,
        target: Optional[str] = None,
        limit: int = 1000,
        activity_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Collect bioactivity data from ChEMBL.

        Args:
            target: Target protein/gene name
            limit: Maximum number of records
            activity_type: Type of bioactivity (e.g., 'IC50', 'Ki')

        Returns:
            DataFrame with bioactivity data
        """
        try:
            from chembl_webresource_client.new_client import new_client

            activities = new_client.activity

            # Build query
            query = activities
            if target:
                targets = new_client.target
                target_results = targets.filter(pref_name__icontains=target)
                if target_results:
                    target_chembl_id = target_results[0]["target_chembl_id"]
                    query = query.filter(target_chembl_id=target_chembl_id)

            if activity_type:
                query = query.filter(standard_type=activity_type)

            results = []
            for i, record in enumerate(query):
                if i >= limit:
                    break
                results.append(record)

            df = pd.DataFrame(results)
            logger.info(f"Collected {len(df)} records from ChEMBL")
            return df

        except Exception as e:
            logger.error(f"ChEMBL collection failed: {e}")
            return pd.DataFrame()

    def collect_from_pubchem(
        self,
        query: str,
        limit: int = 100,
        namespace: str = "name",
    ) -> pd.DataFrame:
        """
        Collect molecular data from PubChem.

        Args:
            query: Search query
            limit: Maximum number of compounds
            namespace: Search namespace ('name', 'smiles', 'formula')

        Returns:
            DataFrame with PubChem data
        """
        try:
            compounds = pcp.get_compounds(query, namespace=namespace)[:limit]

            data = []
            for compound in compounds:
                data.append({
                    "cid": compound.cid,
                    "smiles": compound.canonical_smiles,
                    "molecular_formula": compound.molecular_formula,
                    "molecular_weight": compound.molecular_weight,
                    "iupac_name": compound.iupac_name,
                    "inchi": compound.inchi,
                    "inchikey": compound.inchikey,
                })

            df = pd.DataFrame(data)
            logger.info(f"Collected {len(df)} compounds from PubChem")
            return df

        except Exception as e:
            logger.error(f"PubChem collection failed: {e}")
            return pd.DataFrame()

    def collect_from_pdb(
        self,
        pdb_ids: Optional[List[str]] = None,
        query: Optional[str] = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """
        Collect 3D protein structures from Protein Data Bank.

        Args:
            pdb_ids: List of PDB IDs
            query: Search query for proteins
            limit: Maximum number of structures

        Returns:
            DataFrame with PDB metadata
        """
        try:
            import requests

            base_url = "https://data.rcsb.org/rest/v1/core/entry/"

            if pdb_ids is None:
                # If no IDs provided, perform a search
                search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
                search_query = {
                    "query": {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "value": query if query else "drug",
                        },
                    },
                    "return_type": "entry",
                    "request_options": {"paginate": {"start": 0, "rows": limit}},
                }
                response = requests.post(search_url, json=search_query, timeout=30)
                if response.status_code == 200:
                    result_set = response.json().get("result_set", [])
                    pdb_ids = [item["identifier"] for item in result_set]
                else:
                    logger.warning(f"PDB search failed: {response.status_code}")
                    pdb_ids = []

            data = []
            for pdb_id in pdb_ids[:limit]:
                response = requests.get(base_url + pdb_id, timeout=30)
                if response.status_code == 200:
                    entry = response.json()
                    data.append({
                        "pdb_id": pdb_id,
                        "title": entry.get("struct", {}).get("title", ""),
                        "resolution": entry.get("refine", [{}])[0].get("ls_d_res_high"),
                        "deposition_date": entry.get("rcsb_accession_info", {}).get("deposit_date"),
                    })

            df = pd.DataFrame(data)
            logger.info(f"Collected {len(df)} PDB structures")
            return df

        except Exception as e:
            logger.error(f"PDB collection failed: {e}")
            return pd.DataFrame()

    def collect_from_drugbank(
        self,
        drug_type: Optional[str] = None,
        approved_only: bool = True,
    ) -> pd.DataFrame:
        """
        Collect drug and target information from DrugBank.

        Note: Full DrugBank access requires API key or downloaded XML.

        Args:
            drug_type: Type of drug to filter
            approved_only: Only include approved drugs

        Returns:
            DataFrame with DrugBank data
        """
        logger.warning("DrugBank requires API key or XML download for full access")
        # Placeholder - would need API key or XML parsing
        return pd.DataFrame()

    def collect_from_clinical_trials(
        self,
        condition: Optional[str] = None,
        intervention: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Collect clinical trial data from ClinicalTrials.gov.

        Args:
            condition: Medical condition
            intervention: Drug/intervention name
            limit: Maximum number of trials

        Returns:
            DataFrame with clinical trial data
        """
        try:
            import requests

            base_url = "https://clinicaltrials.gov/api/v2/studies"
            params = {
                "format": "json",
                "pageSize": min(limit, 1000),
            }

            if condition:
                params["query.cond"] = condition
            if intervention:
                params["query.intr"] = intervention

            response = requests.get(base_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                studies = data.get("studies", [])

                results = []
                for study in studies[:limit]:
                    protocol_section = study.get("protocolSection", {})
                    identification = protocol_section.get("identificationModule", {})
                    status = protocol_section.get("statusModule", {})

                    results.append({
                        "nct_id": identification.get("nctId"),
                        "title": identification.get("briefTitle"),
                        "status": status.get("overallStatus"),
                        "phase": protocol_section.get("designModule", {}).get("phases", ["N/A"])[0],
                        "enrollment": status.get("enrollmentInfo", {}).get("count"),
                    })

                df = pd.DataFrame(results)
                logger.info(f"Collected {len(df)} clinical trials")
                return df
            else:
                logger.error(f"ClinicalTrials.gov API error: {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Clinical trials collection failed: {e}")
            return pd.DataFrame()

    def collect_multi_source(
        self,
        sources: List[str],
        query: Optional[str] = None,
        limit_per_source: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect data from multiple sources simultaneously.

        Args:
            sources: List of source names (e.g., ['chembl', 'pubchem', 'pdb'])
            query: Universal query string
            limit_per_source: Maximum records per source

        Returns:
            Dictionary mapping source names to DataFrames
        """
        results = {}

        if "chembl" in sources:
            results["chembl"] = self.collect_from_chembl(
                target=query, limit=limit_per_source
            )

        if "pubchem" in sources:
            results["pubchem"] = self.collect_from_pubchem(
                query=query or "aspirin", limit=limit_per_source
            )

        if "pdb" in sources:
            results["pdb"] = self.collect_from_pdb(
                query=query, limit=limit_per_source
            )

        if "clinical_trials" in sources:
            results["clinical_trials"] = self.collect_from_clinical_trials(
                condition=query, limit=limit_per_source
            )

        logger.info(f"Collected data from {len(results)} sources")
        return results
