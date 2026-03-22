"""
Feature Store - Embeddings for Molecules, Proteins, and Assays

Provides persistent storage and retrieval of computed molecular features
and embeddings for efficient reuse across training runs.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Any, List
import numpy as np
import torch

logger = logging.getLogger(__name__)


class FeatureStore:
    """Persistent storage for molecular embeddings and features."""

    def __init__(self, store_path: str = "./cache/features"):
        """
        Initialize the feature store.

        Args:
            store_path: Directory for storing features
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._cache: Dict[str, Any] = {}

    def _get_feature_path(self, feature_key: str, feature_type: str) -> Path:
        """Get filesystem path for a feature."""
        return self.store_path / f"{feature_type}_{feature_key}.pkl"

    def store_embedding(
        self,
        key: str,
        embedding: np.ndarray,
        feature_type: str = "molecule",
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Store an embedding vector.

        Args:
            key: Unique identifier (e.g., SMILES, InChIKey)
            embedding: Embedding vector
            feature_type: Type of feature ('molecule', 'protein', 'assay')
            metadata: Optional metadata dictionary
        """
        feature_path = self._get_feature_path(key, feature_type)

        data = {
            "embedding": embedding,
            "metadata": metadata or {},
        }

        with open(feature_path, "wb") as f:
            pickle.dump(data, f)

        # Update cache
        cache_key = f"{feature_type}:{key}"
        self._cache[cache_key] = data

        logger.debug(f"Stored {feature_type} embedding for key: {key}")

    def retrieve_embedding(
        self,
        key: str,
        feature_type: str = "molecule",
    ) -> Optional[np.ndarray]:
        """
        Retrieve an embedding vector.

        Args:
            key: Unique identifier
            feature_type: Type of feature

        Returns:
            Embedding vector or None if not found
        """
        cache_key = f"{feature_type}:{key}"

        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]["embedding"]

        # Check filesystem
        feature_path = self._get_feature_path(key, feature_type)
        if not feature_path.exists():
            return None

        try:
            with open(feature_path, "rb") as f:
                data = pickle.load(f)
                self._cache[cache_key] = data
                return data["embedding"]
        except Exception as e:
            logger.error(f"Failed to load embedding for {key}: {e}")
            return None

    def store_batch(
        self,
        keys: List[str],
        embeddings: np.ndarray,
        feature_type: str = "molecule",
        metadata_list: Optional[List[Dict]] = None,
    ) -> None:
        """
        Store multiple embeddings efficiently.

        Args:
            keys: List of unique identifiers
            embeddings: Array of embeddings (N, D)
            feature_type: Type of features
            metadata_list: Optional list of metadata dicts
        """
        if metadata_list is None:
            metadata_list = [{}] * len(keys)

        for key, embedding, metadata in zip(keys, embeddings, metadata_list):
            self.store_embedding(key, embedding, feature_type, metadata)

        logger.info(f"Stored batch of {len(keys)} {feature_type} embeddings")

    def retrieve_batch(
        self,
        keys: List[str],
        feature_type: str = "molecule",
    ) -> Dict[str, np.ndarray]:
        """
        Retrieve multiple embeddings.

        Args:
            keys: List of unique identifiers
            feature_type: Type of features

        Returns:
            Dictionary mapping keys to embeddings
        """
        results = {}
        for key in keys:
            embedding = self.retrieve_embedding(key, feature_type)
            if embedding is not None:
                results[key] = embedding

        logger.info(f"Retrieved {len(results)}/{len(keys)} {feature_type} embeddings")
        return results

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
        logger.info("Cleared feature store cache")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored features."""
        stats = {
            "cache_size": len(self._cache),
            "disk_files": len(list(self.store_path.glob("*.pkl"))),
            "store_path": str(self.store_path),
        }
        return stats
