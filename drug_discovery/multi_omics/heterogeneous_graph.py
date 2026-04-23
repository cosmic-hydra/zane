"""
Heterogeneous Graph Construction for Drug-Target-PPI Networks.

Maps drug molecules against massive, multi-layered Protein-Protein Interaction
and gene regulatory networks for comprehensive pharmacogenomics modeling.

References:
    - Wang et al., "Graph Neural Networks for Heterogeneous Graphs"
    - Zitnik et al., "Modeling Multiplex Networks"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, GATConv, SAGEConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    torch = None
    nn = object
    HeteroData = None
    logger.warning("PyTorch Geometric not available. Using simplified graph model.")


class NodeType(Enum):
    """Node types in heterogeneous drug-target graph."""

    DRUG = "drug"
    PROTEIN = "protein"
    GENE = "gene"
    CELL = "cell"
    PATHWAY = "pathway"
    SIDE_EFFECT = "side_effect"
    DISEASE = "disease"


class EdgeType(Enum):
    """Edge types in heterogeneous graph."""

    DRUG_PROTEIN = "drug_protein"  # Drug binds to protein
    PROTEIN_PROTEIN = "protein_protein"  # PPI
    GENE_REGULATES = "gene_regulates"  # Gene regulatory
    PROTEIN_PATHWAY = "protein_pathway"  # Protein in pathway
    DRUG_SIDE_EFFECT = "drug_side_effect"  # Drug causes side effect
    DISEASE_GENE = "disease_gene"  # Disease associated


@dataclass
class GraphNode:
    """Node in heterogeneous graph.

    Attributes:
        node_id: Unique node identifier.
        node_type: Type of node.
        features: Node feature vector.
        metadata: Additional node information.
    """

    node_id: str
    node_type: NodeType
    features: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "features": self.features.tolist() if self.features is not None else None,
            "metadata": self.metadata,
        }


@dataclass
class GraphEdge:
    """Edge in heterogeneous graph.

    Attributes:
        source: Source node ID.
        target: Target node ID.
        edge_type: Type of edge.
        weight: Edge weight.
        metadata: Additional edge information.
    """

    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class DrugTargetInteraction:
    """Drug-protein target interaction.

    Attributes:
        drug_id: Drug identifier.
        protein_id: Protein identifier.
        affinity: Binding affinity (Ki, IC50).
        interaction_type: Type of interaction (agonist, antagonist, etc.).
        source: Database source.
    """

    drug_id: str
    protein_id: str
    affinity: float = 0.0  # pKi or -log(IC50)
    interaction_type: str = "binder"
    source: str = "unknown"


class HeterogeneousGraph:
    """
    Heterogeneous graph for drug-target-PPI network.

    Constructs and manages multi-relational graphs connecting:
    - Drugs to target proteins
    - Proteins to each other (PPI)
    - Genes to diseases
    - Cells to drug responses

    Example::

        graph = HeterogeneousGraph()
        graph.add_drug("DRUG_001", smiles="CCO", features=...)
        graph.add_protein("PROTEIN_001", sequence=..., features=...)
        graph.add_interaction(
            DrugTargetInteraction(drug_id="DRUG_001", protein_id="PROTEIN_001", affinity=8.5)
        )
        pyg_graph = graph.to_pyg_hetero_data()
    """

    def __init__(
        self,
        node_feature_dim: int = 128,
        edge_feature_dim: int = 32,
    ):
        """
        Initialize heterogeneous graph.

        Args:
            node_feature_dim: Dimension of node features.
            edge_feature_dim: Dimension of edge features.
        """
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim

        # Graph storage
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

        # Node/edge type indices
        self.node_type_index: dict[NodeType, list[str]] = {
            nt: [] for nt in NodeType
        }
        self.edge_type_index: dict[EdgeType, list[tuple[str, str]]] = {
            et: [] for et in EdgeType
        }

        logger.info(f"HeterogeneousGraph initialized: feature_dim={node_feature_dim}")

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        features: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a node to the graph.

        Args:
            node_id: Unique node identifier.
            node_type: Type of node.
            features: Feature vector.
            metadata: Additional information.
        """
        if features is None:
            features = np.random.randn(self.node_feature_dim).astype(np.float32)

        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            features=features,
            metadata=metadata or {},
        )

        self.nodes[node_id] = node
        self.node_type_index[node_type].append(node_id)

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add an edge to the graph.

        Args:
            source: Source node ID.
            target: Target node ID.
            edge_type: Type of edge.
            weight: Edge weight.
            metadata: Additional information.
        """
        if source not in self.nodes or target not in self.nodes:
            logger.warning(f"Cannot add edge: node not found ({source} -> {target})")
            return

        edge = GraphEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )

        self.edges.append(edge)
        self.edge_type_index[edge_type].append((source, target))

    def add_drug(
        self,
        drug_id: str,
        smiles: str | None = None,
        features: np.ndarray | None = None,
        **metadata,
    ) -> None:
        """Add a drug node."""
        self.add_node(
            node_id=drug_id,
            node_type=NodeType.DRUG,
            features=features,
            metadata={"smiles": smiles, **metadata},
        )

    def add_protein(
        self,
        protein_id: str,
        sequence: str | None = None,
        features: np.ndarray | None = None,
        **metadata,
    ) -> None:
        """Add a protein node."""
        self.add_node(
            node_id=protein_id,
            node_type=NodeType.PROTEIN,
            features=features,
            metadata={"sequence": sequence, **metadata},
        )

    def add_interaction(
        self,
        interaction: DrugTargetInteraction,
    ) -> None:
        """Add drug-protein interaction."""
        # Ensure nodes exist
        if interaction.drug_id not in self.nodes:
            self.add_drug(interaction.drug_id)
        if interaction.protein_id not in self.nodes:
            self.add_protein(interaction.protein_id)

        self.add_edge(
            source=interaction.drug_id,
            target=interaction.protein_id,
            edge_type=EdgeType.DRUG_PROTEIN,
            weight=interaction.affinity,
            metadata={
                "interaction_type": interaction.interaction_type,
                "source": interaction.source,
            },
        )

    def to_pyg_hetero_data(self) -> Any:
        """
        Convert to PyTorch Geometric HeteroData.

        Returns:
            HeteroData object ready for GNN training.
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.warning("PyTorch Geometric not available. Returning None.")
            return None

        data = HeteroData()

        # Add nodes for each type
        for node_type, node_ids in self.node_type_index.items():
            if not node_ids:
                continue

            type_str = node_type.value
            features = []
            for node_id in node_ids:
                node = self.nodes[node_id]
                features.append(node.features)

            data[type_str].x = torch.tensor(
                np.stack(features), dtype=torch.float32
            )

        # Add edges for each relation
        for edge_type, edges in self.edge_type_index.items():
            if not edges:
                continue

            # Parse relation
            source_type, target_type = self._parse_relation(edge_type)

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

            # Edge attributes
            edge_attrs = []
            edge_weights = []
            for src, tgt in edges:
                for edge in self.edges:
                    if edge.source == src and edge.target == tgt:
                        edge_attrs.append([edge.weight])
                        edge_weights.append(edge.weight)

            data[source_type, edge_type.value, target_type].edge_index = edge_index
            data[source_type, edge_type.value, target_type].edge_attr = torch.tensor(
                edge_attrs, dtype=torch.float32
            )

        return data

    def _parse_relation(
        self,
        edge_type: EdgeType,
    ) -> tuple[str, str]:
        """Parse edge type to source and target node types."""
        mapping = {
            EdgeType.DRUG_PROTEIN: (NodeType.DRUG.value, NodeType.PROTEIN.value),
            EdgeType.PROTEIN_PROTEIN: (NodeType.PROTEIN.value, NodeType.PROTEIN.value),
            EdgeType.GENE_REGULATES: (NodeType.GENE.value, NodeType.GENE.value),
            EdgeType.PROTEIN_PATHWAY: (NodeType.PROTEIN.value, NodeType.PATHWAY.value),
            EdgeType.DRUG_SIDE_EFFECT: (NodeType.DRUG.value, NodeType.SIDE_EFFECT.value),
            EdgeType.DISEASE_GENE: (NodeType.DISEASE.value, NodeType.GENE.value),
        }
        return mapping.get(edge_type, (NodeType.DRUG.value, NodeType.PROTEIN.value))

    def get_subgraph(
        self,
        center_node: str,
        radius: int = 2,
    ) -> HeterogeneousGraph:
        """
        Extract subgraph around a center node.

        Args:
            center_node: Center node ID.
            radius: BFS radius.

        Returns:
            Subgraph as new HeterogeneousGraph.
        """
        # BFS to find nodes within radius
        visited = {center_node}
        frontier = [center_node]

        for _ in range(radius):
            next_frontier = []
            for node_id in frontier:
                for edge in self.edges:
                    if edge.source == node_id and edge.target not in visited:
                        visited.add(edge.target)
                        next_frontier.append(edge.target)
                    if edge.target == node_id and edge.source not in visited:
                        visited.add(edge.source)
                        next_frontier.append(edge.source)
            frontier = next_frontier

        # Create subgraph
        subgraph = HeterogeneousGraph(
            node_feature_dim=self.node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
        )

        # Add nodes
        for node_id in visited:
            node = self.nodes[node_id]
            subgraph.add_node(
                node_id=node.node_id,
                node_type=node.node_type,
                features=node.features.copy() if node.features is not None else None,
                metadata=node.metadata.copy(),
            )

        # Add edges (within subgraph)
        for edge in self.edges:
            if edge.source in visited and edge.target in visited:
                subgraph.add_edge(
                    source=edge.source,
                    target=edge.target,
                    edge_type=edge.edge_type,
                    weight=edge.weight,
                    metadata=edge.metadata.copy(),
                )

        return subgraph

    def to_networkx(self) -> Any:
        """Convert to NetworkX MultiGraph."""
        try:
            import networkx as nx

            G = nx.MultiGraph()

            # Add nodes
            for node_id, node in self.nodes.items():
                G.add_node(
                    node_id,
                    node_type=node.node_type.value,
                    **node.metadata,
                )

            # Add edges
            for edge in self.edges:
                G.add_edge(
                    edge.source,
                    edge.target,
                    edge_type=edge.edge_type.value,
                    weight=edge.weight,
                    **edge.metadata,
                )

            return G

        except ImportError:
            logger.warning("NetworkX not available. Returning None.")
            return None

    def summary(self) -> dict[str, Any]:
        """Get graph summary statistics."""
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "nodes_by_type": {
                nt.value: len(ids) for nt, ids in self.node_type_index.items()
            },
            "edges_by_type": {
                et.value: len(edges) for et, edges in self.edge_type_index.items()
            },
            "feature_dim": self.node_feature_dim,
        }
