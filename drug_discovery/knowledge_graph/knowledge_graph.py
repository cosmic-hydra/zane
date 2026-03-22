"""
Knowledge Graph with Vector Database Integration

Implements a hybrid knowledge graph combining:
- Structured graph (Neo4j-style nodes and edges)
- Vector embeddings for semantic search
- Hybrid retrieval (graph traversal + vector similarity)
- Entity linking and disambiguation
- Multi-hop reasoning

Nodes: Molecules, Proteins, Diseases, Pathways, Genes, Assays
Edges: treats, binds, inhibits, causes, participates_in, etc.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import pandas as pd
from enum import Enum

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    MOLECULE = "molecule"
    PROTEIN = "protein"
    DISEASE = "disease"
    PATHWAY = "pathway"
    GENE = "gene"
    ASSAY = "assay"
    PUBLICATION = "publication"


class EdgeType(Enum):
    """Types of edges in the knowledge graph."""
    TREATS = "treats"
    BINDS = "binds"
    INHIBITS = "inhibits"
    ACTIVATES = "activates"
    CAUSES = "causes"
    PARTICIPATES_IN = "participates_in"
    REGULATES = "regulates"
    CITED_BY = "cited_by"
    SIMILAR_TO = "similar_to"


@dataclass
class KGNode:
    """Node in the knowledge graph."""
    node_id: str
    node_type: NodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class KGEdge:
    """Edge in the knowledge graph."""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0


class VectorDatabase:
    """Simple vector database for semantic search."""

    def __init__(self, embedding_dim: int = 512):
        """
        Initialize vector database.

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def add_vector(
        self,
        vector_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add vector to database.

        Args:
            vector_id: Unique ID
            embedding: Embedding vector
            metadata: Optional metadata
        """
        if embedding.shape[0] != self.embedding_dim:
            logger.warning(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
            return

        self.vectors[vector_id] = embedding
        self.metadata[vector_id] = metadata or {}

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_func: Optional[callable] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_func: Optional filter function on metadata

        Returns:
            List of (vector_id, similarity_score) tuples
        """
        if len(self.vectors) == 0:
            return []

        # Compute cosine similarity
        similarities = []
        for vector_id, embedding in self.vectors.items():
            # Apply filter if provided
            if filter_func and not filter_func(self.metadata.get(vector_id, {})):
                continue

            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-10
            )
            similarities.append((vector_id, float(similarity)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def batch_search(
        self,
        query_embeddings: List[np.ndarray],
        top_k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch search for similar vectors.

        Args:
            query_embeddings: List of query vectors
            top_k: Number of results per query

        Returns:
            List of result lists
        """
        results = []
        for query_emb in query_embeddings:
            results.append(self.search_similar(query_emb, top_k=top_k))
        return results


class KnowledgeGraph:
    """Hybrid knowledge graph with vector embeddings."""

    def __init__(self, embedding_dim: int = 512):
        """
        Initialize knowledge graph.

        Args:
            embedding_dim: Dimension for embeddings
        """
        self.nodes: Dict[str, KGNode] = {}
        self.edges: Dict[str, KGEdge] = {}

        # Adjacency lists for efficient traversal
        self.outgoing_edges: Dict[str, List[str]] = defaultdict(list)
        self.incoming_edges: Dict[str, List[str]] = defaultdict(list)

        # Vector database for semantic search
        self.vector_db = VectorDatabase(embedding_dim=embedding_dim)

        # Indexes
        self.nodes_by_type: Dict[NodeType, Set[str]] = defaultdict(set)

    def add_node(self, node: KGNode) -> None:
        """
        Add node to graph.

        Args:
            node: Node to add
        """
        self.nodes[node.node_id] = node
        self.nodes_by_type[node.node_type].add(node.node_id)

        # Add to vector DB if embedding exists
        if node.embedding is not None:
            self.vector_db.add_vector(
                node.node_id,
                node.embedding,
                metadata={
                    "node_type": node.node_type.value,
                    "name": node.name,
                },
            )

        logger.debug(f"Added node: {node.node_id} ({node.node_type.value})")

    def add_edge(self, edge: KGEdge) -> None:
        """
        Add edge to graph.

        Args:
            edge: Edge to add
        """
        # Validate nodes exist
        if edge.source_id not in self.nodes:
            logger.warning(f"Source node not found: {edge.source_id}")
            return
        if edge.target_id not in self.nodes:
            logger.warning(f"Target node not found: {edge.target_id}")
            return

        self.edges[edge.edge_id] = edge
        self.outgoing_edges[edge.source_id].append(edge.edge_id)
        self.incoming_edges[edge.target_id].append(edge.edge_id)

        logger.debug(f"Added edge: {edge.edge_id} ({edge.edge_type.value})")

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "outgoing",
    ) -> List[KGNode]:
        """
        Get neighboring nodes.

        Args:
            node_id: Node ID
            edge_type: Optional edge type filter
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of neighboring nodes
        """
        neighbors = []
        edge_lists = []

        if direction in ["outgoing", "both"]:
            edge_lists.append(self.outgoing_edges.get(node_id, []))
        if direction in ["incoming", "both"]:
            edge_lists.append(self.incoming_edges.get(node_id, []))

        for edge_list in edge_lists:
            for edge_id in edge_list:
                edge = self.edges[edge_id]

                # Filter by edge type
                if edge_type and edge.edge_type != edge_type:
                    continue

                # Get neighbor node
                if edge.source_id == node_id:
                    neighbor_id = edge.target_id
                else:
                    neighbor_id = edge.source_id

                if neighbor_id in self.nodes:
                    neighbors.append(self.nodes[neighbor_id])

        return neighbors

    def find_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 5,
    ) -> Optional[List[Tuple[str, str]]]:
        """
        Find shortest path between two nodes using BFS.

        Args:
            start_node_id: Starting node ID
            end_node_id: Target node ID
            max_depth: Maximum path length

        Returns:
            List of (node_id, edge_id) tuples or None
        """
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            return None

        # BFS
        queue = [(start_node_id, [])]
        visited = {start_node_id}

        while queue:
            current_id, path = queue.pop(0)

            if len(path) >= max_depth:
                continue

            if current_id == end_node_id:
                return path

            # Explore neighbors
            for edge_id in self.outgoing_edges.get(current_id, []):
                edge = self.edges[edge_id]
                neighbor_id = edge.target_id

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_path = path + [(neighbor_id, edge_id)]
                    queue.append((neighbor_id, new_path))

        return None

    def semantic_search(
        self,
        query_embedding: np.ndarray,
        node_type: Optional[NodeType] = None,
        top_k: int = 10,
    ) -> List[Tuple[KGNode, float]]:
        """
        Search for semantically similar nodes.

        Args:
            query_embedding: Query embedding
            node_type: Optional node type filter
            top_k: Number of results

        Returns:
            List of (node, similarity_score) tuples
        """
        # Define filter function
        def filter_func(metadata):
            if node_type:
                return metadata.get("node_type") == node_type.value
            return True

        # Search vector DB
        results = self.vector_db.search_similar(
            query_embedding,
            top_k=top_k,
            filter_func=filter_func,
        )

        # Convert to nodes
        node_results = []
        for node_id, similarity in results:
            if node_id in self.nodes:
                node_results.append((self.nodes[node_id], similarity))

        return node_results

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        start_node_ids: Optional[List[str]] = None,
        max_hops: int = 2,
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> List[Tuple[KGNode, float]]:
        """
        Hybrid search combining graph structure and vector similarity.

        Args:
            query_embedding: Query embedding
            start_node_ids: Optional starting nodes for graph traversal
            max_hops: Maximum graph traversal hops
            top_k: Number of results
            alpha: Balance between graph (0) and vector (1) scores

        Returns:
            List of (node, combined_score) tuples
        """
        # Get candidates from vector search
        vector_candidates = self.semantic_search(query_embedding, top_k=top_k * 2)
        vector_scores = {node.node_id: score for node, score in vector_candidates}

        # Get candidates from graph traversal
        graph_candidates = set()
        if start_node_ids:
            for start_id in start_node_ids:
                # BFS traversal
                queue = [(start_id, 0)]
                visited = {start_id}

                while queue:
                    node_id, depth = queue.pop(0)

                    if depth > max_hops:
                        continue

                    graph_candidates.add(node_id)

                    # Explore neighbors
                    for edge_id in self.outgoing_edges.get(node_id, []):
                        edge = self.edges[edge_id]
                        neighbor_id = edge.target_id

                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            queue.append((neighbor_id, depth + 1))

        # Compute graph scores (inverse of distance)
        graph_scores = {}
        for node_id in graph_candidates:
            if start_node_ids:
                min_dist = float('inf')
                for start_id in start_node_ids:
                    path = self.find_path(start_id, node_id, max_depth=max_hops)
                    if path:
                        min_dist = min(min_dist, len(path))

                if min_dist < float('inf'):
                    graph_scores[node_id] = 1.0 / (1.0 + min_dist)
                else:
                    graph_scores[node_id] = 0.0
            else:
                graph_scores[node_id] = 0.0

        # Combine scores
        all_node_ids = set(vector_scores.keys()) | graph_candidates
        combined_scores = []

        for node_id in all_node_ids:
            vector_score = vector_scores.get(node_id, 0.0)
            graph_score = graph_scores.get(node_id, 0.0)

            combined_score = alpha * vector_score + (1 - alpha) * graph_score

            if node_id in self.nodes:
                combined_scores.append((self.nodes[node_id], combined_score))

        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)

        return combined_scores[:top_k]

    def get_subgraph(
        self,
        node_ids: List[str],
        include_edges: bool = True,
    ) -> Tuple[List[KGNode], List[KGEdge]]:
        """
        Extract subgraph containing specified nodes.

        Args:
            node_ids: Node IDs to include
            include_edges: Whether to include edges between nodes

        Returns:
            Tuple of (nodes, edges)
        """
        nodes = [self.nodes[nid] for nid in node_ids if nid in self.nodes]

        edges = []
        if include_edges:
            node_id_set = set(node_ids)
            for edge_id, edge in self.edges.items():
                if edge.source_id in node_id_set and edge.target_id in node_id_set:
                    edges.append(edge)

        return nodes, edges

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes_by_type": {
                node_type.value: len(node_ids)
                for node_type, node_ids in self.nodes_by_type.items()
            },
            "avg_degree": len(self.edges) * 2 / len(self.nodes) if len(self.nodes) > 0 else 0,
            "nodes_with_embeddings": sum(1 for n in self.nodes.values() if n.embedding is not None),
        }

        return stats

    def export_to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export graph to DataFrames.

        Returns:
            Tuple of (nodes_df, edges_df)
        """
        # Nodes
        nodes_data = []
        for node in self.nodes.values():
            row = {
                "node_id": node.node_id,
                "node_type": node.node_type.value,
                "name": node.name,
                **node.properties,
            }
            nodes_data.append(row)

        nodes_df = pd.DataFrame(nodes_data)

        # Edges
        edges_data = []
        for edge in self.edges.values():
            row = {
                "edge_id": edge.edge_id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "edge_type": edge.edge_type.value,
                "weight": edge.weight,
                "confidence": edge.confidence,
                **edge.properties,
            }
            edges_data.append(row)

        edges_df = pd.DataFrame(edges_data)

        return nodes_df, edges_df
