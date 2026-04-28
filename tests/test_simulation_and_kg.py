"""
Test Suite for Biological Simulation and Knowledge Graph

Tests for:
- ADME prediction
- Dose-response simulation
- Cellular response simulation
- Knowledge graph operations
- Vector database
"""

import numpy as np
import pytest

from drug_discovery.knowledge_graph.knowledge_graph import (
    EdgeType,
    KGEdge,
    KGNode,
    KnowledgeGraph,
    NodeType,
    VectorDatabase,
)
from drug_discovery.simulation.biological_response import (
    ADMEPredictor,
    BiologicalResponseSimulator,
    CellularResponseSimulator,
    DoseResponseSimulator,
)


class TestADMEPredictor:
    """Test ADME prediction."""

    def setup_method(self):
        """Setup test fixtures."""
        self.predictor = ADMEPredictor()
        self.test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

    def test_predict_adme(self):
        """Test ADME property prediction."""
        adme = self.predictor.predict_adme(self.test_smiles)

        assert adme is not None
        assert 0 <= adme.absorption <= 1
        assert adme.distribution > 0
        assert 0 <= adme.metabolism <= 1
        assert adme.excretion > 0
        assert adme.half_life > 0
        assert 0 <= adme.bioavailability <= 1

    def test_check_drug_likeness(self):
        """Test drug-likeness assessment."""
        result = self.predictor.check_drug_likeness(self.test_smiles)

        assert "drug_like" in result
        assert "violations" in result
        assert "properties" in result
        assert isinstance(result["drug_like"], bool)

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        result = self.predictor.predict_adme("INVALID")
        assert result is None


class TestDoseResponseSimulator:
    """Test dose-response simulation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.simulator = DoseResponseSimulator()

    def test_simulate_dose_response(self):
        """Test dose-response curve simulation."""
        ec50 = 1.0
        dose_response = self.simulator.simulate_dose_response(
            ec50=ec50,
            emax=1.0,
            hill_coefficient=1.5,
        )

        assert dose_response.ec50 == ec50
        assert len(dose_response.doses) > 0
        assert len(dose_response.responses) == len(dose_response.doses)
        assert all(0 <= r <= 1 for r in dose_response.responses)

    def test_estimate_effective_dose(self):
        """Test effective dose estimation."""
        dose_response = self.simulator.simulate_dose_response(
            ec50=1.0,
            emax=1.0,
        )

        effective_dose = self.simulator.estimate_effective_dose(
            dose_response,
            target_effect=0.9,
        )

        assert effective_dose > 0

    def test_therapeutic_window(self):
        """Test therapeutic window calculation."""
        result = self.simulator.compute_therapeutic_window(
            efficacy_ec50=1.0,
            toxicity_ec50=10.0,
        )

        assert "therapeutic_index" in result
        assert result["therapeutic_index"] == 10.0
        assert "safety_margin" in result


class TestCellularResponseSimulator:
    """Test cellular response simulation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.simulator = CellularResponseSimulator()
        self.test_smiles = "CCO"

    def test_simulate_cellular_response(self):
        """Test cellular response simulation."""
        response = self.simulator.simulate_cellular_response(
            self.test_smiles,
            dose=10.0,
            treatment_time=24.0,
        )

        assert response is not None
        assert 0 <= response.cell_viability <= 1
        assert response.proliferation_rate >= 0
        assert 0 <= response.apoptosis_rate <= 1
        assert len(response.gene_expression_changes) > 0
        assert len(response.pathway_activation) > 0


class TestBiologicalResponseSimulator:
    """Test comprehensive biological response simulator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.simulator = BiologicalResponseSimulator()
        self.test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    def test_simulate_full_response(self):
        """Test full biological response simulation."""
        result = self.simulator.simulate_full_response(
            self.test_smiles,
            initial_dose=10.0,
        )

        assert "smiles" in result
        assert "adme" in result
        assert "drug_likeness" in result
        assert "dose_response" in result

    def test_batch_simulate(self):
        """Test batch simulation."""
        smiles_list = ["CCO", "CC(C)O", "CCCO"]

        df = self.simulator.batch_simulate(smiles_list, dose=10.0)

        assert len(df) == len(smiles_list)
        assert "drug_like" in df.columns
        assert "bioavailability" in df.columns


class TestVectorDatabase:
    """Test vector database."""

    def setup_method(self):
        """Setup test fixtures."""
        self.db = VectorDatabase(embedding_dim=128)

    def test_add_and_search_vector(self):
        """Test adding and searching vectors."""
        embedding1 = np.random.rand(128)
        embedding2 = np.random.rand(128)
        embedding3 = embedding1 + np.random.rand(128) * 0.01  # Similar to embedding1

        self.db.add_vector("vec1", embedding1)
        self.db.add_vector("vec2", embedding2)
        self.db.add_vector("vec3", embedding3)

        # Search for similar vectors
        results = self.db.search_similar(embedding1, top_k=2)

        assert len(results) == 2
        assert results[0][0] == "vec1"  # Most similar should be itself

    def test_filter_search(self):
        """Test filtered search."""
        embedding = np.random.rand(128)

        self.db.add_vector("mol1", embedding, metadata={"type": "drug"})
        self.db.add_vector("mol2", embedding, metadata={"type": "protein"})

        # Filter for drugs only
        results = self.db.search_similar(
            embedding,
            top_k=10,
            filter_func=lambda m: m.get("type") == "drug",
        )

        assert len(results) == 1
        assert results[0][0] == "mol1"


class TestKnowledgeGraph:
    """Test knowledge graph."""

    def setup_method(self):
        """Setup test fixtures."""
        self.kg = KnowledgeGraph(embedding_dim=128)

    def test_add_node(self):
        """Test adding nodes."""
        node = KGNode(
            node_id="mol1",
            node_type=NodeType.MOLECULE,
            name="Aspirin",
            embedding=np.random.rand(128),
        )

        self.kg.add_node(node)

        assert "mol1" in self.kg.nodes
        assert self.kg.nodes["mol1"].name == "Aspirin"

    def test_add_edge(self):
        """Test adding edges."""
        # Create nodes
        mol_node = KGNode(
            node_id="mol1",
            node_type=NodeType.MOLECULE,
            name="Drug A",
        )
        disease_node = KGNode(
            node_id="disease1",
            node_type=NodeType.DISEASE,
            name="Cancer",
        )

        self.kg.add_node(mol_node)
        self.kg.add_node(disease_node)

        # Create edge
        edge = KGEdge(
            edge_id="edge1",
            source_id="mol1",
            target_id="disease1",
            edge_type=EdgeType.TREATS,
        )

        self.kg.add_edge(edge)

        assert "edge1" in self.kg.edges

    def test_get_neighbors(self):
        """Test getting neighbors."""
        # Setup graph
        node1 = KGNode(node_id="n1", node_type=NodeType.MOLECULE, name="A")
        node2 = KGNode(node_id="n2", node_type=NodeType.PROTEIN, name="B")

        self.kg.add_node(node1)
        self.kg.add_node(node2)

        edge = KGEdge(
            edge_id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=EdgeType.BINDS,
        )

        self.kg.add_edge(edge)

        # Get neighbors
        neighbors = self.kg.get_neighbors("n1", direction="outgoing")

        assert len(neighbors) == 1
        assert neighbors[0].node_id == "n2"

    def test_find_path(self):
        """Test path finding."""
        # Create linear graph: n1 -> n2 -> n3
        for i in range(1, 4):
            node = KGNode(
                node_id=f"n{i}",
                node_type=NodeType.MOLECULE,
                name=f"Node{i}",
            )
            self.kg.add_node(node)

        edge1 = KGEdge(
            edge_id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=EdgeType.BINDS,
        )
        edge2 = KGEdge(
            edge_id="e2",
            source_id="n2",
            target_id="n3",
            edge_type=EdgeType.BINDS,
        )

        self.kg.add_edge(edge1)
        self.kg.add_edge(edge2)

        # Find path
        path = self.kg.find_path("n1", "n3", max_depth=5)

        assert path is not None
        assert len(path) == 2

    def test_semantic_search(self):
        """Test semantic search."""
        embedding1 = np.random.rand(128)
        embedding2 = embedding1 + np.random.rand(128) * 0.01

        node1 = KGNode(
            node_id="n1",
            node_type=NodeType.MOLECULE,
            name="Similar1",
            embedding=embedding1,
        )
        node2 = KGNode(
            node_id="n2",
            node_type=NodeType.MOLECULE,
            name="Similar2",
            embedding=embedding2,
        )

        self.kg.add_node(node1)
        self.kg.add_node(node2)

        # Search
        results = self.kg.semantic_search(embedding1, top_k=2)

        assert len(results) == 2
        assert results[0][0].node_id == "n1"

    def test_get_statistics(self):
        """Test graph statistics."""
        node = KGNode(
            node_id="n1",
            node_type=NodeType.MOLECULE,
            name="Test",
        )
        self.kg.add_node(node)

        stats = self.kg.get_statistics()

        assert "total_nodes" in stats
        assert stats["total_nodes"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
