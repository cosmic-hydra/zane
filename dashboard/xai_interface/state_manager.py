import uuid
from typing import Any

import networkx as nx


class MoleculeStateManager:
    """
    Refinement 3: Molecular Git-Tree (State Management)
    Replaces linear Undo/Redo with a directed acyclic graph (DAG) of molecular states.
    Every time the chemist alters the structure, save the new SMILES and its associated scores
    as a node in a tree.
    """

    def __init__(self, initial_smiles: str, initial_scores: dict[str, Any]):
        self.dag = nx.DiGraph()
        self.current_node_id = str(uuid.uuid4())

        # Add root node
        self.dag.add_node(self.current_node_id, smiles=initial_smiles, scores=initial_scores, label="Initial")
        self.root_id = self.current_node_id

    def add_state(self, new_smiles: str, scores: dict[str, Any], action_description: str = "Modified") -> str:
        """
        Adds a new state to the DAG branching off the current state.
        Returns the ID of the new node.
        """
        new_node_id = str(uuid.uuid4())

        self.dag.add_node(new_node_id, smiles=new_smiles, scores=scores, label=action_description)

        self.dag.add_edge(self.current_node_id, new_node_id)
        self.current_node_id = new_node_id

        return new_node_id

    def checkout_state(self, node_id: str):
        """
        Moves the current pointer to a specific state in the DAG (like git checkout).
        """
        if node_id in self.dag:
            self.current_node_id = node_id
        else:
            raise ValueError(f"State {node_id} not found in the state graph.")

    def get_current_state(self) -> dict[str, Any]:
        """Returns the data of the current node."""
        return self.dag.nodes[self.current_node_id]

    def get_all_states(self) -> list[dict[str, Any]]:
        """Returns all states for frontend rendering of the tree."""
        states = []
        for node_id in self.dag.nodes:
            node_data = self.dag.nodes[node_id].copy()
            node_data["id"] = node_id
            node_data["is_current"] = node_id == self.current_node_id

            # Find parent (assuming tree structure)
            parents = list(self.dag.predecessors(node_id))
            node_data["parent_id"] = parents[0] if parents else None

            states.append(node_data)
        return states

    def render_tree_graphviz(self):
        """
        Returns a graphviz/dot representation of the DAG for visualization in the frontend.
        Allows the chemist to effortlessly click between branching exploratory pathways.
        """
        dot_str = "digraph DAG {\n"
        dot_str += '  node [shape=box, style="rounded,filled", fontname="Helvetica"];\n'

        # Nodes
        for node_id in self.dag.nodes:
            node_data = self.dag.nodes[node_id]
            smiles = node_data["smiles"]
            # Truncate SMILES for display
            display_smiles = smiles if len(smiles) < 15 else smiles[:12] + "..."

            color = "#a5d6a7" if node_id == self.current_node_id else "#e0e0e0"

            label = f"{display_smiles}\\nΔG: {node_data['scores'].get('dg_mean', 0):.2f}"

            dot_str += f'  "{node_id}" [label="{label}", fillcolor="{color}"];\n'

        # Edges
        for u, v in self.dag.edges:
            dot_str += f'  "{u}" -> "{v}";\n'

        dot_str += "}"
        return dot_str
