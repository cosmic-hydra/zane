"""
Drug Discovery Knowledge Graph
Stores entities and relationships for reasoning
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
import networkx as nx

logger = logging.getLogger(__name__)


class DrugKnowledgeGraph:
    """
    Knowledge graph for drug discovery
    Nodes: Molecules, Proteins, Diseases, Pathways
    Edges: binds_to, inhibits, treats, associated_with
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_types = {'molecule', 'protein', 'disease', 'pathway'}
        self.relation_types = {'binds_to', 'inhibits', 'activates', 'treats', 'associated_with'}

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Optional[Dict] = None
    ):
        """
        Add an entity to the graph

        Args:
            entity_id: Unique identifier
            entity_type: Type of entity
            properties: Additional properties
        """
        if entity_type not in self.entity_types:
            logger.warning(f"Unknown entity type: {entity_type}")

        properties = properties or {}
        properties['type'] = entity_type

        self.graph.add_node(entity_id, **properties)

    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: Optional[Dict] = None
    ):
        """
        Add a relationship between entities

        Args:
            source: Source entity ID
            target: Target entity ID
            relation_type: Type of relationship
            properties: Additional properties
        """
        if relation_type not in self.relation_types:
            logger.warning(f"Unknown relation type: {relation_type}")

        properties = properties or {}
        properties['relation'] = relation_type

        self.graph.add_edge(source, target, **properties)

    def query_relations(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = 'outgoing'  # 'outgoing', 'incoming', or 'both'
    ) -> List[Tuple[str, str, Dict]]:
        """
        Query relationships for an entity

        Args:
            entity_id: Entity to query
            relation_type: Filter by relation type
            direction: Direction of relationships

        Returns:
            List of (source, target, properties) tuples
        """
        results = []

        if direction in ['outgoing', 'both']:
            for target in self.graph.successors(entity_id):
                edges = self.graph[entity_id][target]
                for key, data in edges.items():
                    if relation_type is None or data.get('relation') == relation_type:
                        results.append((entity_id, target, data))

        if direction in ['incoming', 'both']:
            for source in self.graph.predecessors(entity_id):
                edges = self.graph[source][entity_id]
                for key, data in edges.items():
                    if relation_type is None or data.get('relation') == relation_type:
                        results.append((source, entity_id, data))

        return results

    def find_shortest_path(
        self,
        source: str,
        target: str
    ) -> Optional[List[str]]:
        """
        Find shortest path between entities

        Args:
            source: Source entity
            target: Target entity

        Returns:
            List of entity IDs in path
        """
        try:
            path = nx.shortest_path(self.graph, source, target)
            return path
        except nx.NetworkXNoPath:
            return None

    def get_neighbors(
        self,
        entity_id: str,
        max_hops: int = 1
    ) -> Set[str]:
        """
        Get neighbors within max_hops

        Args:
            entity_id: Entity to query
            max_hops: Maximum number of hops

        Returns:
            Set of neighbor entity IDs
        """
        neighbors = set()

        if max_hops == 1:
            neighbors.update(self.graph.successors(entity_id))
            neighbors.update(self.graph.predecessors(entity_id))
        else:
            # BFS to find all neighbors within max_hops
            visited = {entity_id}
            current_level = {entity_id}

            for _ in range(max_hops):
                next_level = set()
                for node in current_level:
                    for neighbor in self.graph.successors(node):
                        if neighbor not in visited:
                            next_level.add(neighbor)
                            visited.add(neighbor)
                    for neighbor in self.graph.predecessors(node):
                        if neighbor not in visited:
                            next_level.add(neighbor)
                            visited.add(neighbor)

                neighbors.update(next_level)
                current_level = next_level

        return neighbors

    def get_entity_properties(
        self,
        entity_id: str
    ) -> Optional[Dict]:
        """
        Get properties of an entity

        Args:
            entity_id: Entity ID

        Returns:
            Dictionary of properties
        """
        if entity_id in self.graph:
            return dict(self.graph.nodes[entity_id])
        return None


class KnowledgeGraphBuilder:
    """
    Build knowledge graph from various data sources
    """

    def __init__(self, kg: DrugKnowledgeGraph):
        self.kg = kg

    def build_from_chembl(
        self,
        bioactivity_data: List[Dict]
    ):
        """
        Build graph from ChEMBL bioactivity data

        Args:
            bioactivity_data: List of bioactivity records
        """
        for record in bioactivity_data:
            # Add molecule
            mol_id = record.get('molecule_chembl_id')
            if mol_id:
                self.kg.add_entity(
                    mol_id,
                    'molecule',
                    {'smiles': record.get('canonical_smiles')}
                )

            # Add target
            target_id = record.get('target_chembl_id')
            if target_id:
                self.kg.add_entity(
                    target_id,
                    'protein',
                    {'name': record.get('target_pref_name')}
                )

            # Add interaction
            if mol_id and target_id:
                self.kg.add_relation(
                    mol_id,
                    target_id,
                    'binds_to',
                    {
                        'pchembl_value': record.get('pchembl_value'),
                        'assay_type': record.get('assay_type')
                    }
                )

        logger.info(f"Built graph: {self.kg.graph.number_of_nodes()} nodes, "
                   f"{self.kg.graph.number_of_edges()} edges")

    def build_from_literature(
        self,
        articles: List[Dict]
    ):
        """
        Build graph from literature mentions

        Args:
            articles: List of scientific articles
        """
        # Placeholder for NER-based graph construction
        # Would extract entities and relationships from text
        logger.info(f"Processing {len(articles)} articles for graph construction")
