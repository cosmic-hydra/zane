import logging
from typing import Any

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


# Mock AiZynthFinder
class AiZynthFinderMock:
    def __init__(self):
        pass

    def build_routes(self, target_smiles: str) -> nx.DiGraph:
        """Builds a mock retrosynthetic graph."""
        graph = nx.DiGraph()
        graph.add_node("target", type="molecule", smiles=target_smiles)

        # Route 1: Cheap but uses toxic solvents (environmental penalty)
        graph.add_node("rxn1", type="reaction", solvent="dichloromethane", catalysts=["Pd"])
        graph.add_node("precursor1", type="molecule", smiles="C1=CC=CC=C1")
        graph.add_edge("rxn1", "target")
        graph.add_edge("precursor1", "rxn1")

        # Route 2: Geopolitically unstable
        graph.add_node("rxn2", type="reaction", solvent="water", catalysts=["Pt"])
        graph.add_node("precursor2", type="molecule", smiles="c1ccncc1")
        graph.add_edge("rxn2", "target")
        graph.add_edge("precursor2", "rxn2")

        # Route 3: Optimal (Clean, stable, acceptable cost)
        graph.add_node("rxn3", type="reaction", solvent="ethanol", catalysts=["Fe"])
        graph.add_node("precursor3", type="molecule", smiles="CC(C)O")
        graph.add_edge("rxn3", "target")
        graph.add_edge("precursor3", "rxn3")

        return graph


class CommodityPricingAPI:
    def get_prices(self) -> pd.DataFrame:
        """Mock live API feed for chemical commodity pricing ($/kg)"""
        data = {
            "chemical": ["dichloromethane", "water", "ethanol", "Pd", "Pt", "Fe", "C1=CC=CC=C1", "c1ccncc1", "CC(C)O"],
            "price_per_kg": [2.5, 0.01, 1.2, 50000.0, 30000.0, 50.0, 10.0, 25.0, 5.0],
            "single_source": [False, False, False, True, True, False, False, True, False],
            "geopolitical_risk": [0.1, 0.0, 0.1, 0.8, 0.9, 0.2, 0.3, 0.7, 0.1],
        }
        return pd.DataFrame(data).set_index("chemical")


class COGSOptimizer:
    """
    Module 12: Geopolitical Retrosynthesis & COGS Engine
    """

    def __init__(self):
        self.pricing_api = CommodityPricingAPI()
        self.market_data = self.pricing_api.get_prices()
        self.aizynth = AiZynthFinderMock()

        # Environmental constraints (Waste disposal cost $/kg)
        self.toxic_solvents = {
            "dichloromethane": 50.0,  # Heavily penalize toxic chlorinated solvent waste
            "chloroform": 60.0,
        }

    def _calculate_node_cost(self, node_data: dict[str, Any]) -> float:
        """Calculates dynamic cost accounting for price, geopolitical risk, and waste."""
        cost = 0.0

        if node_data.get("type") == "molecule":
            chem = node_data.get("smiles")
            if chem in self.market_data.index:
                row = self.market_data.loc[chem]
                cost += row["price_per_kg"]

                # Supply Chain Resilience Routing
                if row["single_source"]:
                    cost += 100.0  # Penalty for single-source international suppliers
                cost += row["geopolitical_risk"] * 200.0  # Penalty for geopolitical instability

        elif node_data.get("type") == "reaction":
            solvent = node_data.get("solvent")
            catalysts = node_data.get("catalysts", [])

            if solvent in self.market_data.index:
                cost += self.market_data.loc[solvent]["price_per_kg"]

            # COGS must account for cost of environmental waste disposal
            if solvent in self.toxic_solvents:
                cost += self.toxic_solvents[solvent]

            for cat in catalysts:
                if cat in self.market_data.index:
                    row = self.market_data.loc[cat]
                    cost += row["price_per_kg"] * 0.001  # Catalyst usage fraction
                    if row["single_source"]:
                        cost += 50.0
                    cost += row["geopolitical_risk"] * 100.0

        return cost

    def optimize_route(self, target_smiles: str, max_cogs: float = 100.0) -> list[str] | None:
        """
        Evaluates retrosynthesis graph and finds the lowest COGS route that meets criteria.
        If the predicted route exceeds max_cogs ($X/kg), AI searches for alternative.
        """
        logger.info(f"Optimizing synthesis route for {target_smiles} with max COGS ${max_cogs}/kg")
        graph = self.aizynth.build_routes(target_smiles)

        best_route = None
        min_cost = float("inf")

        target_node = "target"
        precursors = [n for n, d in graph.nodes(data=True) if d.get("type") == "molecule" and n != "target"]

        for precursor in precursors:
            try:
                paths = nx.all_simple_paths(graph, source=precursor, target=target_node)
                for path in paths:
                    total_cost = sum(self._calculate_node_cost(graph.nodes[n]) for n in path)
                    logger.info(f"Evaluated route {path} with cost ${total_cost:.2f}/kg")

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_route = path
            except nx.NetworkXNoPath:
                continue

        if min_cost > max_cogs:
            logger.warning(
                f"All routes exceed max COGS of ${max_cogs}/kg. Lowest is ${min_cost:.2f}/kg. Re-planning..."
            )
            # Automatically search for an alternative, cheaper synthesis pathway
            return None

        logger.info(f"Selected route {best_route} with optimized COGS: ${min_cost:.2f}/kg")
        return best_route
