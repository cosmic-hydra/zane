from typing import List, Dict, Tuple, Set, Any
import networkx as nx

class KnowledgeGraphReasoner:
    """
    Symbolic and path-based reasoning for Drug Discovery Knowledge Graphs.
    Can answer queries like "Find all proteins regulated by genes associated with disease X".
    """
    
    def __init__(self, nx_graph: nx.MultiDiGraph):
        self.graph = nx_graph
        
    def find_causal_paths(self, start_node: str, end_node: str, max_length: int = 3) -> List[List[Tuple[str, str, str]]]:
        """
        Find all directed paths between two nodes up to a certain length.
        Returns paths as list of (u, v, edge_type) tuples.
        """
        paths = []
        if start_node not in self.graph or end_node not in self.graph:
            return []
            
        for path in nx.all_simple_paths(self.graph, start_node, end_node, cutoff=max_length):
            path_with_types = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # Get the edge type (assuming it's stored in 'edge_type' attribute)
                edge_data = self.graph.get_edge_data(u, v)
                # MultiDiGraph might have multiple edges between same nodes
                etype = "unknown"
                if edge_data:
                    # Just take the first one for simplicity
                    key = list(edge_data.keys())[0]
                    etype = edge_data[key].get('edge_type', 'unknown')
                path_with_types.append((u, v, etype))
            paths.append(path_with_types)
            
        return paths

    def query_by_relation_chain(self, start_node: str, relations: List[str]) -> Set[str]:
        """
        Multi-hop query: Start at start_node, follow the sequence of relations.
        Example: relations=['ASSOCIATED_WITH_GENE', 'CODES_FOR_PROTEIN']
        """
        current_nodes = {start_node}
        for rel in relations:
            next_nodes = set()
            for node in current_nodes:
                if node not in self.graph:
                    continue
                for _, neighbor, data in self.graph.out_edges(node, data=True):
                    if data.get('edge_type') == rel:
                        next_nodes.add(neighbor)
            current_nodes = next_nodes
            if not current_nodes:
                break
        return current_nodes

    def check_consistency(self) -> List[str]:
        """
        Check for logical inconsistencies in the KG.
        Example: A drug cannot both inhibit and activate the same protein simultaneously.
        """
        issues = []
        for u, v, data in self.graph.edges(data=True):
            etype = data.get('edge_type')
            if etype == 'INHIBITS':
                # Check if there's also an 'ACTIVATES' edge
                if self.graph.has_edge(u, v):
                    edge_data = self.graph.get_edge_data(u, v)
                    for key in edge_data:
                        if edge_data[key].get('edge_type') == 'ACTIVATES':
                            issues.append(f"Contradictory relationship between {u} and {v}: INHIBITS and ACTIVATES")
        return issues
