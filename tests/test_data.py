"""
Tests for Data Collection Module
"""

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
