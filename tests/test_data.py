"""
Tests for Data Collection Module
"""

from pathlib import Path

import pandas as pd
from drug_discovery.data import DataCollector


class TestDataCollector:
    """Test DataCollector functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.collector = DataCollector(cache_dir="./test_cache")

    def test_initialization(self):
        """Test DataCollector initialization"""
        assert self.collector is not None
        assert self.collector.cache_dir == "./test_cache"

    def test_pubchem_collection(self):
        """Test PubChem data collection"""
        # Small test - just collect a few compounds
        df = self.collector.collect_from_pubchem(query="aspirin", limit=5)

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "smiles" in df.columns
            assert len(df) <= 5

    def test_merge_datasets(self):
        """Test dataset merging"""
        df1 = pd.DataFrame({"smiles": ["C", "CC", "CCC"]})
        df2 = pd.DataFrame({"smiles": ["CC", "CCCC", "CCCCC"]})

        merged = self.collector.merge_datasets([df1, df2])

        assert isinstance(merged, pd.DataFrame)
        assert "smiles" in merged.columns
        # Should remove duplicates
        assert len(merged) < len(df1) + len(df2)

    def test_collect_from_drugbank_csv(self, tmp_path):
        """Test DrugBank CSV parsing with normalized output schema."""
        csv_path = tmp_path / "drugbank.csv"
        csv_path.write_text(
            "SMILES,drug_name\nCCO,Ethanol\nCC(=O)O,Acetic acid\n",
            encoding="utf-8",
        )

        df = self.collector.collect_from_drugbank(file_path=str(csv_path), limit=10)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "smiles" in df.columns
        assert "name" in df.columns
        assert "source" in df.columns
        assert set(df["source"].unique()) == {"drugbank"}

    def test_collect_from_drugbank_missing_file(self, tmp_path):
        """Test DrugBank behavior when file is absent."""
        missing_file = str(Path(tmp_path) / "not_found.csv")
        df = self.collector.collect_from_drugbank(file_path=missing_file, limit=10)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_data_quality_report_metrics(self):
        """Quality report should summarize validity and duplicate metrics."""
        df = pd.DataFrame(
            {
                "smiles": ["CCO", "CCO", "INVALID_SMILES", "CC(=O)O"],
                "name": ["a", "b", "c", "d"],
                "source": ["x", "x", "x", "x"],
            }
        )

        report = self.collector.generate_data_quality_report(df)

        assert report["total_rows"] == 4
        assert report["invalid_smiles_rows"] >= 1
        assert report["duplicate_smiles_rows"] >= 1
        assert 0.0 <= report["validity_ratio"] <= 1.0

    def test_merge_filters_invalid_smiles(self):
        """Merged datasets should not keep invalid SMILES rows."""
        df1 = pd.DataFrame({"smiles": ["CCO", "INVALID_SMILES"], "name": ["a", "b"], "source": ["s", "s"]})
        df2 = pd.DataFrame({"smiles": ["CC(=O)O"], "name": ["c"], "source": ["s"]})

        merged = self.collector.merge_datasets([df1, df2])

        assert isinstance(merged, pd.DataFrame)
        assert "INVALID_SMILES" not in set(merged["smiles"].tolist())
