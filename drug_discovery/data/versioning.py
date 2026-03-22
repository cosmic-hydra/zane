"""
Dataset Versioning - Track Data Lineage and Changes

Provides version control for datasets to ensure reproducibility and
track data evolution over time.
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class DatasetVersioning:
    """Version control system for molecular datasets."""

    def __init__(self, versions_dir: str = "./data/versions"):
        """
        Initialize dataset versioning.

        Args:
            versions_dir: Directory for storing version metadata
        """
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.versions_dir / "manifest.json"

        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load the version manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {"versions": []}

    def _save_manifest(self) -> None:
        """Save the version manifest."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _compute_hash(self, data: pd.DataFrame) -> str:
        """Compute hash of dataset for change detection."""
        # Create a deterministic hash of the DataFrame
        data_str = pd.util.hash_pandas_object(data).sum()
        return hashlib.sha256(str(data_str).encode()).hexdigest()[:16]

    def create_version(
        self,
        dataset: pd.DataFrame,
        version_name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Create a new dataset version.

        Args:
            dataset: DataFrame to version
            version_name: Name for this version
            description: Optional description
            metadata: Optional metadata dictionary

        Returns:
            Version ID
        """
        # Generate version ID
        timestamp = datetime.now().isoformat()
        data_hash = self._compute_hash(dataset)
        version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{data_hash}"

        # Save dataset
        dataset_path = self.versions_dir / f"{version_id}.parquet"
        dataset.to_parquet(dataset_path, index=False)

        # Create version entry
        version_entry = {
            "version_id": version_id,
            "version_name": version_name,
            "timestamp": timestamp,
            "description": description or "",
            "data_hash": data_hash,
            "num_records": len(dataset),
            "columns": list(dataset.columns),
            "metadata": metadata or {},
            "dataset_path": str(dataset_path),
        }

        self.manifest["versions"].append(version_entry)
        self._save_manifest()

        logger.info(f"Created dataset version: {version_id} ({version_name})")
        return version_id

    def load_version(self, version_id: str) -> Optional[pd.DataFrame]:
        """
        Load a specific dataset version.

        Args:
            version_id: Version ID to load

        Returns:
            DataFrame or None if not found
        """
        # Find version in manifest
        version_entry = None
        for entry in self.manifest["versions"]:
            if entry["version_id"] == version_id:
                version_entry = entry
                break

        if version_entry is None:
            logger.error(f"Version not found: {version_id}")
            return None

        # Load dataset
        dataset_path = Path(version_entry["dataset_path"])
        if not dataset_path.exists():
            logger.error(f"Dataset file not found: {dataset_path}")
            return None

        dataset = pd.read_parquet(dataset_path)
        logger.info(f"Loaded dataset version: {version_id}")
        return dataset

    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all available versions.

        Returns:
            List of version metadata dictionaries
        """
        return self.manifest["versions"]

    def get_latest_version(self) -> Optional[str]:
        """
        Get the ID of the most recent version.

        Returns:
            Version ID or None if no versions exist
        """
        if not self.manifest["versions"]:
            return None
        return self.manifest["versions"][-1]["version_id"]

    def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str,
    ) -> Dict[str, Any]:
        """
        Compare two dataset versions.

        Args:
            version_id_1: First version ID
            version_id_2: Second version ID

        Returns:
            Dictionary with comparison results
        """
        # Find versions
        v1_entry = None
        v2_entry = None

        for entry in self.manifest["versions"]:
            if entry["version_id"] == version_id_1:
                v1_entry = entry
            if entry["version_id"] == version_id_2:
                v2_entry = entry

        if v1_entry is None or v2_entry is None:
            logger.error("One or both versions not found")
            return {}

        comparison = {
            "version_1": version_id_1,
            "version_2": version_id_2,
            "record_count_diff": v2_entry["num_records"] - v1_entry["num_records"],
            "is_same_hash": v1_entry["data_hash"] == v2_entry["data_hash"],
            "columns_v1": v1_entry["columns"],
            "columns_v2": v2_entry["columns"],
            "new_columns": set(v2_entry["columns"]) - set(v1_entry["columns"]),
            "removed_columns": set(v1_entry["columns"]) - set(v2_entry["columns"]),
        }

        return comparison

    def tag_version(self, version_id: str, tag: str) -> bool:
        """
        Add a tag to a version (e.g., 'production', 'baseline').

        Args:
            version_id: Version ID to tag
            tag: Tag name

        Returns:
            True if successful, False otherwise
        """
        for entry in self.manifest["versions"]:
            if entry["version_id"] == version_id:
                if "tags" not in entry:
                    entry["tags"] = []
                if tag not in entry["tags"]:
                    entry["tags"].append(tag)
                    self._save_manifest()
                    logger.info(f"Tagged version {version_id} with '{tag}'")
                    return True

        logger.error(f"Version not found: {version_id}")
        return False

    def get_versions_by_tag(self, tag: str) -> List[str]:
        """
        Get all version IDs with a specific tag.

        Args:
            tag: Tag name

        Returns:
            List of version IDs
        """
        result = []
        for entry in self.manifest["versions"]:
            if "tags" in entry and tag in entry["tags"]:
                result.append(entry["version_id"])
        return result
