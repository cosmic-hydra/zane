"""
Feature Store - Embeddings for Molecules, Proteins, and Assays

Provides persistent storage and retrieval of computed molecular features
and embeddings for efficient reuse across training runs.
"""

import logging
import pickle
import os
from pathlib import Path
from typing import Dict, Optional, Any, List
import numpy as np
import torch
try:
    from pymongo import MongoClient
    _PYMONGO = True
except ImportError:
    _PYMONGO = False
    MongoClient = None

logger = logging.getLogger(__name__)


class FeatureStore:
    """Persistent storage for molecular embeddings and features."""

    def __init__(self, store_path: str = "./cache/features", use_mongodb: bool = True):
        """
        Initialize the feature store.

        Args:
            store_path: Directory for storing features
            use_mongodb: Whether to use MongoDB for storage
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._cache: Dict[str, Any] = {}
        
        self.use_mongodb = use_mongodb and _PYMONGO
        if self.use_mongodb and MongoClient:
            mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            self.mongo_client = MongoClient(mongodb_uri)
            self.db = self.mongo_client["zane_discovery"]
            self.collection = self.db["embeddings"]
            # Ensure index
            self.collection.create_index([("key", 1), ("feature_type", 1)], unique=True)

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
        data = {
            "key": key,
            "feature_type": feature_type,
            "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            "metadata": metadata or {},
        }

        if self.use_mongodb:
            try:
                self.collection.update_one(
                    {"key": key, "feature_type": feature_type},
                    {"$set": data},
                    upsert=True
                )
            except Exception as e:
                logger.error(f"Failed to store embedding in MongoDB for {key}: {e}")
        else:
            feature_path = self._get_feature_path(key, feature_type)
            with open(feature_path, "wb") as f:
                pickle.dump(data, f)

        # Update cache
        cache_key = f"{feature_type}:{key}"
        self._cache[cache_key] = {"embedding": embedding, "metadata": metadata or {}}

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

        if self.use_mongodb:
            try:
                doc = self.collection.find_one({"key": key, "feature_type": feature_type})
                if doc:
                    embedding = np.array(doc["embedding"])
                    self._cache[cache_key] = {"embedding": embedding, "metadata": doc.get("metadata", {})}
                    return embedding
            except Exception as e:
                logger.error(f"Failed to retrieve embedding from MongoDB for {key}: {e}")
        else:
            # Check filesystem
            feature_path = self._get_feature_path(key, feature_type)
            if not feature_path.exists():
                return None

            try:
                with open(feature_path, "rb") as f:
                    data = pickle.load(f)
                    # Handle legacy format where 'key' and 'feature_type' might not be in the pkl
                    emb = data["embedding"]
                    self._cache[cache_key] = data
                    return emb
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

        if self.use_mongodb:
            operations = []
            from pymongo import UpdateOne
            for key, embedding, metadata in zip(keys, embeddings, metadata_list):
                data = {
                    "key": key,
                    "feature_type": feature_type,
                    "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    "metadata": metadata or {},
                }
                operations.append(UpdateOne(
                    {"key": key, "feature_type": feature_type},
                    {"$set": data},
                    upsert=True
                ))
            if operations:
                try:
                    self.collection.bulk_write(operations)
                except Exception as e:
                    logger.error(f"Failed to bulk store embeddings in MongoDB: {e}")
        else:
            for key, embedding, metadata in zip(keys, embeddings, metadata_list):
                self.store_embedding(key, embedding, feature_type, metadata)

        # Update cache for all
        for key, embedding, metadata in zip(keys, embeddings, metadata_list):
            cache_key = f"{feature_type}:{key}"
            self._cache[cache_key] = {"embedding": embedding, "metadata": metadata or {}}

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
        
        # Check cache first and identify missing keys
        missing_keys = []
        for key in keys:
            cache_key = f"{feature_type}:{key}"
            if cache_key in self._cache:
                results[key] = self._cache[cache_key]["embedding"]
            else:
                missing_keys.append(key)
        
        if not missing_keys:
            return results

        if self.use_mongodb:
            try:
                docs = self.collection.find({"key": {"$in": missing_keys}, "feature_type": feature_type})
                for doc in docs:
                    key = doc["key"]
                    embedding = np.array(doc["embedding"])
                    results[key] = embedding
                    cache_key = f"{feature_type}:{key}"
                    self._cache[cache_key] = {"embedding": embedding, "metadata": doc.get("metadata", {})}
            except Exception as e:
                logger.error(f"Failed to bulk retrieve embeddings from MongoDB: {e}")
        else:
            for key in missing_keys:
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
            "store_path": str(self.store_path),
            "use_mongodb": self.use_mongodb,
        }
        if not self.use_mongodb:
            stats["disk_files"] = len(list(self.store_path.glob("*.pkl")))
        else:
            try:
                stats["db_count"] = self.collection.count_documents({})
            except:
                stats["db_count"] = "error"
        return stats
