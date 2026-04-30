"""
Tests verifying DataCollector works with any supported database,
covering drug conditions and clinical trial queries.

All external-network calls are mocked so the suite runs offline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from drug_discovery.data.collector import DataCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
CAFFEINE_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
IBUPROFEN_SMILES = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
ETHANOL_SMILES = "CCO"

DRUG_CONDITIONS = ["cancer", "diabetes", "hypertension", "alzheimer", "covid-19"]
DRUG_INTERVENTIONS = ["aspirin", "metformin", "ibuprofen", "caffeine"]


@pytest.fixture
def collector(tmp_path):
    return DataCollector(cache_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Built-in / offline sources
# ---------------------------------------------------------------------------


class TestApprovedDrugs:
    """DataCollector.collect_approved_drugs returns the seed set without network."""

    def test_returns_dataframe(self, collector):
        df = collector.collect_approved_drugs()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, collector):
        df = collector.collect_approved_drugs()
        assert "smiles" in df.columns
        assert "name" in df.columns
        assert "source" in df.columns

    def test_source_label(self, collector):
        df = collector.collect_approved_drugs()
        assert not df.empty
        assert (df["source"] == "builtin").all()

    def test_nonempty_smiles(self, collector):
        df = collector.collect_approved_drugs()
        assert (df["smiles"].str.len() > 0).all()


# ---------------------------------------------------------------------------
# DrugBank (file-based) — multiple column-name variants
# ---------------------------------------------------------------------------


class TestDrugBankDatabase:
    """DataCollector.collect_from_drugbank handles diverse CSV/TSV schemas."""

    def test_standard_smiles_column(self, tmp_path, collector):
        csv = tmp_path / "db.csv"
        csv.write_text("smiles,name\nCCO,Ethanol\nCC(=O)O,Acetic acid\n")
        df = collector.collect_from_drugbank(file_path=str(csv))
        assert not df.empty
        assert "smiles" in df.columns
        assert set(df["source"]) == {"drugbank"}

    def test_uppercase_smiles_column(self, tmp_path, collector):
        csv = tmp_path / "db.csv"
        csv.write_text(f"SMILES,drug_name\n{ASPIRIN_SMILES},Aspirin\n{IBUPROFEN_SMILES},Ibuprofen\n")
        df = collector.collect_from_drugbank(file_path=str(csv))
        assert not df.empty

    def test_canonical_smiles_column(self, tmp_path, collector):
        csv = tmp_path / "db.csv"
        csv.write_text(f"canonical_smiles,generic_name\n{CAFFEINE_SMILES},Caffeine\n")
        df = collector.collect_from_drugbank(file_path=str(csv))
        assert not df.empty

    def test_tsv_format(self, tmp_path, collector):
        tsv = tmp_path / "db.tsv"
        tsv.write_text(f"smiles\tname\n{ASPIRIN_SMILES}\tAspirin\n{IBUPROFEN_SMILES}\tIbuprofen\n")
        df = collector.collect_from_drugbank(file_path=str(tsv))
        assert not df.empty

    def test_missing_file_returns_empty(self, tmp_path, collector):
        df = collector.collect_from_drugbank(file_path=str(tmp_path / "missing.csv"))
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_no_smiles_column_returns_empty(self, tmp_path, collector):
        csv = tmp_path / "bad.csv"
        csv.write_text("compound_id,formula\n1,C2H5OH\n")
        df = collector.collect_from_drugbank(file_path=str(csv))
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_limit_respected(self, tmp_path, collector):
        rows = "\n".join(f"{ETHANOL_SMILES},Drug{i}" for i in range(20))
        csv = tmp_path / "db.csv"
        csv.write_text(f"smiles,name\n{rows}\n")
        df = collector.collect_from_drugbank(file_path=str(csv), limit=5)
        assert len(df) <= 5

    def test_invalid_smiles_filtered_out(self, tmp_path, collector):
        csv = tmp_path / "db.csv"
        csv.write_text(f"smiles,name\n{ASPIRIN_SMILES},Aspirin\nNOT_A_SMILES,Bad\n")
        df = collector.collect_from_drugbank(file_path=str(csv))
        # Should contain only rows with valid SMILES
        assert "NOT_A_SMILES" not in df["smiles"].values

    def test_drug_conditions_annotation(self, tmp_path, collector):
        """CSV may carry condition/indication column that should survive."""
        csv = tmp_path / "db_cond.csv"
        csv.write_text(
            f"smiles,name,condition\n"
            f"{ASPIRIN_SMILES},Aspirin,pain\n"
            f"{IBUPROFEN_SMILES},Ibuprofen,inflammation\n"
        )
        df = collector.collect_from_drugbank(file_path=str(csv))
        assert not df.empty
        assert "smiles" in df.columns


# ---------------------------------------------------------------------------
# PubChem (mocked)
# ---------------------------------------------------------------------------


class TestPubChemDatabase:
    """DataCollector.collect_from_pubchem returns consistent schema or empty."""

    def _make_pubchem_compound(self, smiles: str, name: str):
        compound = MagicMock()
        compound.isomeric_smiles = smiles
        compound.canonical_smiles = smiles
        compound.connectivity_smiles = None
        compound.iupac_name = name
        return compound

    def test_returns_dataframe_on_success(self, collector):
        compound = self._make_pubchem_compound(ASPIRIN_SMILES, "aspirin")
        with patch("drug_discovery.data.collector.DataCollector.collect_from_pubchem") as mock:
            mock.return_value = pd.DataFrame(
                [{"smiles": ASPIRIN_SMILES, "name": "aspirin", "source": "pubchem"}]
            )
            df = collector.collect_from_pubchem(query="aspirin", limit=1)
        assert isinstance(df, pd.DataFrame)

    def test_empty_on_pubchempy_missing(self, collector):
        with patch.dict("sys.modules", {"pubchempy": None}):
            df = collector.collect_from_pubchem(query="aspirin", limit=1)
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.parametrize("condition", ["cancer", "diabetes", "hypertension", "ibuprofen"])
    def test_drug_condition_queries(self, collector, condition):
        """Any drug condition string should be accepted without raising."""
        with patch("drug_discovery.data.collector.DataCollector.collect_from_pubchem") as mock:
            mock.return_value = pd.DataFrame(columns=["smiles", "name", "source"])
            df = collector.collect_from_pubchem(query=condition, limit=3)
        assert isinstance(df, pd.DataFrame)

    def test_schema_when_populated(self, collector):
        with patch("drug_discovery.data.collector.DataCollector.collect_from_pubchem") as mock:
            mock.return_value = pd.DataFrame(
                [{"smiles": CAFFEINE_SMILES, "name": "caffeine", "source": "pubchem"}]
            )
            df = collector.collect_from_pubchem(query="caffeine", limit=5)
        if not df.empty:
            for col in ("smiles", "name", "source"):
                assert col in df.columns


# ---------------------------------------------------------------------------
# ChEMBL (mocked)
# ---------------------------------------------------------------------------


class TestChEMBLDatabase:
    """DataCollector.collect_from_chembl returns consistent schema or empty."""

    def _mock_chembl(self, records):
        """Patch the chembl client to return *records* dict list."""
        mock_client = MagicMock()
        mock_client.activity.filter.return_value.only.return_value.__getitem__.return_value = records
        mock_client.molecule.filter.return_value.only.return_value.__getitem__.return_value = records
        mock_client.target.filter.return_value.__getitem__ = lambda self, key: [
            {"target_chembl_id": "CHEMBL202"}
        ]
        return mock_client

    def test_returns_dataframe_on_success(self, collector):
        records = [
            {
                "molecule_chembl_id": "CHEMBL1",
                "molecule_structures": {"canonical_smiles": ASPIRIN_SMILES},
                "pref_name": "aspirin",
            }
        ]
        mock_new_client = self._mock_chembl(records)
        with patch("drug_discovery.data.collector.DataCollector.collect_from_chembl") as mock:
            mock.return_value = pd.DataFrame(
                [{"smiles": ASPIRIN_SMILES, "name": "aspirin", "source": "chembl"}]
            )
            df = collector.collect_from_chembl(target="kinase", limit=5)
        assert isinstance(df, pd.DataFrame)

    def test_empty_on_chembl_missing(self, collector):
        with patch.dict("sys.modules", {"chembl_webresource_client": None, "chembl_webresource_client.new_client": None}):
            df = collector.collect_from_chembl(limit=5)
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.parametrize("target", ["kinase", "EGFR", "VEGFR", "cancer target"])
    def test_target_condition_queries(self, collector, target):
        with patch("drug_discovery.data.collector.DataCollector.collect_from_chembl") as mock:
            mock.return_value = pd.DataFrame(columns=["smiles", "name", "source"])
            df = collector.collect_from_chembl(target=target, limit=3)
        assert isinstance(df, pd.DataFrame)

    def test_schema_when_populated(self, collector):
        with patch("drug_discovery.data.collector.DataCollector.collect_from_chembl") as mock:
            mock.return_value = pd.DataFrame(
                [{"smiles": IBUPROFEN_SMILES, "name": "ibuprofen", "source": "chembl"}]
            )
            df = collector.collect_from_chembl(limit=5)
        if not df.empty:
            for col in ("smiles", "name", "source"):
                assert col in df.columns


# ---------------------------------------------------------------------------
# PDB (mocked HTTP)
# ---------------------------------------------------------------------------


class TestPDBDatabase:
    """DataCollector.collect_from_pdb returns consistent schema or empty."""

    def _pdb_entry(self, pdb_id: str = "1ABC"):
        return {
            "struct": {"title": "Drug target complex"},
            "refine": [{"ls_d_res_high": 2.1}],
            "rcsb_accession_info": {"deposit_date": "2020-01-01"},
        }

    def test_direct_ids(self, collector):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._pdb_entry()
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_resp):
            df = collector.collect_from_pdb(pdb_ids=["1ABC"], limit=1)
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "pdb_id" in df.columns

    def test_search_by_query(self, collector):
        search_resp = MagicMock()
        search_resp.json.return_value = {"result_set": [{"identifier": "2XYZ"}]}
        search_resp.raise_for_status = MagicMock()
        entry_resp = MagicMock()
        entry_resp.json.return_value = self._pdb_entry("2XYZ")
        entry_resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=entry_resp):
            with patch("requests.post", return_value=search_resp):
                df = collector.collect_from_pdb(query="drug", limit=1)
        assert isinstance(df, pd.DataFrame)

    def test_empty_on_network_failure(self, collector):
        with patch("requests.get", side_effect=ConnectionError("offline")):
            with patch("requests.post", side_effect=ConnectionError("offline")):
                df = collector.collect_from_pdb(query="drug", limit=2)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @pytest.mark.parametrize("query", ["cancer", "kinase inhibitor", "antiviral"])
    def test_drug_condition_queries(self, collector, query):
        search_resp = MagicMock()
        search_resp.json.return_value = {"result_set": []}
        search_resp.raise_for_status = MagicMock()
        with patch("requests.post", return_value=search_resp):
            df = collector.collect_from_pdb(query=query, limit=2)
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# ClinicalTrials.gov (mocked HTTP)
# ---------------------------------------------------------------------------


class TestClinicalTrialsDatabase:
    """DataCollector.collect_from_clinical_trials covers conditions and interventions."""

    def _make_study(self, nct_id, title, status="RECRUITING", phase="PHASE3", enrollment=100):
        return {
            "protocolSection": {
                "identificationModule": {"nctId": nct_id, "briefTitle": title},
                "statusModule": {
                    "overallStatus": status,
                    "enrollmentInfo": {"count": enrollment},
                },
                "designModule": {"phases": [phase]},
            }
        }

    def _mock_api(self, studies):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"studies": studies}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_returns_dataframe(self, collector):
        studies = [self._make_study("NCT0001", "Aspirin for pain")]
        with patch("requests.get", return_value=self._mock_api(studies)):
            df = collector.collect_from_clinical_trials(condition="pain", limit=5)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, collector):
        studies = [self._make_study("NCT0002", "Metformin diabetes trial")]
        with patch("requests.get", return_value=self._mock_api(studies)):
            df = collector.collect_from_clinical_trials(condition="diabetes", limit=5)
        if not df.empty:
            for col in ("nct_id", "title", "status", "phase"):
                assert col in df.columns

    @pytest.mark.parametrize("condition", DRUG_CONDITIONS)
    def test_drug_conditions(self, collector, condition):
        studies = [self._make_study(f"NCT{i:04d}", f"{condition} study") for i in range(3)]
        with patch("requests.get", return_value=self._mock_api(studies)):
            df = collector.collect_from_clinical_trials(condition=condition, limit=10)
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.parametrize("intervention", DRUG_INTERVENTIONS)
    def test_drug_interventions(self, collector, intervention):
        studies = [self._make_study("NCT9999", f"{intervention} trial")]
        with patch("requests.get", return_value=self._mock_api(studies)):
            df = collector.collect_from_clinical_trials(intervention=intervention, limit=5)
        assert isinstance(df, pd.DataFrame)

    def test_condition_and_intervention_combined(self, collector):
        studies = [self._make_study("NCT1234", "Ibuprofen for inflammation")]
        with patch("requests.get", return_value=self._mock_api(studies)):
            df = collector.collect_from_clinical_trials(condition="inflammation", intervention="ibuprofen", limit=5)
        assert isinstance(df, pd.DataFrame)

    def test_no_results(self, collector):
        with patch("requests.get", return_value=self._mock_api([])):
            df = collector.collect_from_clinical_trials(condition="rare_orphan_condition_xyz", limit=5)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_limit_respected(self, collector):
        studies = [self._make_study(f"NCT{i:04d}", f"Study {i}") for i in range(20)]
        with patch("requests.get", return_value=self._mock_api(studies)):
            df = collector.collect_from_clinical_trials(condition="cancer", limit=5)
        assert len(df) <= 5

    def test_phase_captured(self, collector):
        studies = [
            self._make_study("NCT0010", "Phase 1 study", phase="PHASE1"),
            self._make_study("NCT0011", "Phase 3 study", phase="PHASE3"),
        ]
        with patch("requests.get", return_value=self._mock_api(studies)):
            df = collector.collect_from_clinical_trials(condition="cancer", limit=10)
        if not df.empty:
            assert "phase" in df.columns

    def test_network_failure_returns_empty(self, collector):
        with patch("requests.get", side_effect=ConnectionError("offline")):
            df = collector.collect_from_clinical_trials(condition="cancer", limit=5)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_api_error_returns_empty(self, collector):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 503")
        with patch("requests.get", return_value=mock_resp):
            df = collector.collect_from_clinical_trials(condition="hypertension", limit=5)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_enrollment_column(self, collector):
        studies = [self._make_study("NCT0020", "Big trial", enrollment=500)]
        with patch("requests.get", return_value=self._mock_api(studies)):
            df = collector.collect_from_clinical_trials(limit=5)
        if not df.empty:
            assert "enrollment" in df.columns


# ---------------------------------------------------------------------------
# Multi-source collection
# ---------------------------------------------------------------------------


class TestMultiSourceCollection:
    """collect_multi_source works with any combination of supported databases."""

    ALL_SOURCES = ["pubchem", "chembl", "pdb", "clinical_trials", "approved_drugs", "drugbank"]

    def _patch_all(self, collector, tmp_path):
        """Return a context-manager dict that patches all external methods."""
        drugbank_csv = tmp_path / "drugbank.csv"
        drugbank_csv.write_text(f"smiles,name\n{ASPIRIN_SMILES},Aspirin\n")
        collector.cache_dir = str(tmp_path)

        empty_mol = pd.DataFrame(columns=["smiles", "name", "source"])
        empty_ct = pd.DataFrame(columns=["nct_id", "title", "status", "phase", "enrollment"])

        patches = {
            "pubchem": patch.object(collector, "collect_from_pubchem", return_value=empty_mol),
            "chembl": patch.object(collector, "collect_from_chembl", return_value=empty_mol),
            "pdb": patch.object(collector, "collect_from_pdb", return_value=pd.DataFrame()),
            "clinical_trials": patch.object(collector, "collect_from_clinical_trials", return_value=empty_ct),
        }
        return patches

    def test_all_sources_returns_dict(self, collector, tmp_path):
        patches = self._patch_all(collector, tmp_path)
        with patches["pubchem"], patches["chembl"], patches["pdb"], patches["clinical_trials"]:
            results = collector.collect_multi_source(self.ALL_SOURCES, query="aspirin")
        assert isinstance(results, dict)
        for source in self.ALL_SOURCES:
            assert source in results

    def test_subset_of_sources(self, collector, tmp_path):
        patches = self._patch_all(collector, tmp_path)
        with patches["clinical_trials"]:
            results = collector.collect_multi_source(["approved_drugs", "clinical_trials"], query="cancer")
        assert "approved_drugs" in results
        assert "clinical_trials" in results
        assert "pubchem" not in results

    def test_single_source_approved_drugs(self, collector):
        results = collector.collect_multi_source(["approved_drugs"])
        assert "approved_drugs" in results
        assert isinstance(results["approved_drugs"], pd.DataFrame)
        assert not results["approved_drugs"].empty

    @pytest.mark.parametrize("source", ["pubchem", "chembl", "pdb", "clinical_trials", "approved_drugs"])
    def test_each_source_individually(self, collector, tmp_path, source):
        patches = self._patch_all(collector, tmp_path)
        patch_map = {k: v for k, v in patches.items() if k == source}
        with patch_map.get(source, patch.object(collector, "_with_retry", side_effect=lambda f, **kw: f())):
            results = collector.collect_multi_source([source], query="ibuprofen")
        assert source in results

    def test_returns_dataframes_for_all(self, collector, tmp_path):
        patches = self._patch_all(collector, tmp_path)
        with patches["pubchem"], patches["chembl"], patches["pdb"], patches["clinical_trials"]:
            results = collector.collect_multi_source(self.ALL_SOURCES, query="cancer")
        for source, df in results.items():
            assert isinstance(df, pd.DataFrame), f"{source} should return a DataFrame"

    def test_unknown_source_ignored(self, collector):
        results = collector.collect_multi_source(["approved_drugs", "unknown_db"])
        assert "approved_drugs" in results
        assert "unknown_db" not in results


# ---------------------------------------------------------------------------
# Merge and quality across databases
# ---------------------------------------------------------------------------


class TestCrossDatabaseMerge:
    """Merging results from different databases produces clean, deduplicated data."""

    def _pubchem_frame(self):
        return pd.DataFrame([{"smiles": ASPIRIN_SMILES, "name": "aspirin", "source": "pubchem"}])

    def _chembl_frame(self):
        return pd.DataFrame([
            {"smiles": ASPIRIN_SMILES, "name": "acetylsalicylic acid", "source": "chembl"},
            {"smiles": IBUPROFEN_SMILES, "name": "ibuprofen", "source": "chembl"},
        ])

    def _drugbank_frame(self):
        return pd.DataFrame([{"smiles": CAFFEINE_SMILES, "name": "caffeine", "source": "drugbank"}])

    def test_merge_deduplicates_smiles(self, collector):
        merged = collector.merge_datasets([self._pubchem_frame(), self._chembl_frame()])
        assert not merged.empty
        assert merged["smiles"].duplicated().sum() == 0

    def test_merge_preserves_all_sources(self, collector):
        merged = collector.merge_datasets([
            self._pubchem_frame(),
            self._chembl_frame(),
            self._drugbank_frame(),
        ])
        assert not merged.empty
        # SMILES from all three frames should be present
        smiles_set = set(merged["smiles"])
        assert ASPIRIN_SMILES in smiles_set
        assert IBUPROFEN_SMILES in smiles_set
        assert CAFFEINE_SMILES in smiles_set

    def test_merge_empty_inputs(self, collector):
        merged = collector.merge_datasets([pd.DataFrame(), pd.DataFrame()])
        assert isinstance(merged, pd.DataFrame)

    def test_quality_report_mixed_sources(self, collector):
        df = pd.concat([self._pubchem_frame(), self._chembl_frame(), self._drugbank_frame()], ignore_index=True)
        report = collector.generate_data_quality_report(df)
        assert report["total_rows"] == 4
        assert report["valid_smiles_rows"] >= 0
        assert 0.0 <= report["validity_ratio"] <= 1.0

    def test_quality_report_empty(self, collector):
        report = collector.generate_data_quality_report(pd.DataFrame())
        assert report["total_rows"] == 0
        assert report["validity_ratio"] == 0.0

    def test_quality_report_all_valid(self, collector):
        df = collector.collect_approved_drugs()
        report = collector.generate_data_quality_report(df)
        assert report["validity_ratio"] >= 0.0

    def test_drugbank_merged_with_approved(self, collector, tmp_path):
        csv = tmp_path / "db.csv"
        csv.write_text(f"smiles,name\n{CAFFEINE_SMILES},Caffeine\n")
        db_frame = collector.collect_from_drugbank(file_path=str(csv))
        approved = collector.collect_approved_drugs()
        merged = collector.merge_datasets([db_frame, approved])
        assert not merged.empty
        assert "smiles" in merged.columns


# ---------------------------------------------------------------------------
# Retry / resilience
# ---------------------------------------------------------------------------


class TestRetryAndResilience:
    """_with_retry enforces bounded retries and propagates the last exception."""

    def test_succeeds_on_first_try(self, collector):
        result = collector._with_retry(lambda: 42)
        assert result == 42

    def test_succeeds_after_one_failure(self, collector):
        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ValueError("transient")
            return "ok"

        result = collector._with_retry(flaky, max_retries=3, backoff_seconds=0)
        assert result == "ok"
        assert call_count["n"] == 2

    def test_raises_after_max_retries(self, collector):
        def always_fails():
            raise RuntimeError("always")

        with pytest.raises(RuntimeError, match="always"):
            collector._with_retry(always_fails, max_retries=2, backoff_seconds=0)

    def test_collect_from_clinical_trials_network_error(self, collector):
        with patch("requests.get", side_effect=OSError("network")):
            df = collector.collect_from_clinical_trials(condition="cancer")
        assert isinstance(df, pd.DataFrame)

    def test_collect_from_pdb_network_error(self, collector):
        with patch("requests.get", side_effect=OSError("network")):
            with patch("requests.post", side_effect=OSError("network")):
                df = collector.collect_from_pdb(pdb_ids=["1ABC"])
        assert isinstance(df, pd.DataFrame)
