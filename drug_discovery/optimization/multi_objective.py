"""
Multi-Objective Optimization Module
Pareto optimization for drug candidate selection
"""

import torch
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationObjective:
    """Single optimization objective"""
    name: str
    weight: float = 1.0
    minimize: bool = True  # True for minimization, False for maximization
    target: Optional[float] = None
    threshold: Optional[float] = None


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for drug discovery
    Optimizes binding affinity, ADMET properties, toxicity, and synthesis
    """

    def __init__(
        self,
        objectives: Optional[List[OptimizationObjective]] = None
    ):
        """
        Args:
            objectives: List of optimization objectives
        """
        if objectives is None:
            # Default objectives for drug discovery
            self.objectives = [
                OptimizationObjective('binding_affinity', weight=2.0, minimize=True),
                OptimizationObjective('qed_score', weight=1.5, minimize=False),
                OptimizationObjective('toxicity', weight=1.0, minimize=True),
                OptimizationObjective('synthetic_accessibility', weight=1.0, minimize=True),
                OptimizationObjective('lipinski_violations', weight=1.0, minimize=True)
            ]
        else:
            self.objectives = objectives

    def calculate_fitness(
        self,
        candidate: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate weighted fitness score

        Args:
            candidate: Dictionary of objective values
            weights: Optional custom weights

        Returns:
            Fitness score (lower is better)
        """
        total_score = 0.0
        total_weight = 0.0

        for obj in self.objectives:
            if obj.name not in candidate:
                continue

            value = candidate[obj.name]
            weight = weights.get(obj.name, obj.weight) if weights else obj.weight

            # Normalize: convert to minimization problem
            if obj.minimize:
                score = value
            else:
                score = -value  # Flip for maximization

            # Apply threshold if specified
            if obj.threshold is not None:
                if obj.minimize and value > obj.threshold:
                    score += 100.0  # Heavy penalty
                elif not obj.minimize and value < obj.threshold:
                    score += 100.0

            total_score += weight * score
            total_weight += weight

        if total_weight > 0:
            return total_score / total_weight
        return float('inf')

    def rank_candidates(
        self,
        candidates: List[Dict[str, float]]
    ) -> List[Tuple[int, float, Dict[str, float]]]:
        """
        Rank candidates by multi-objective fitness

        Args:
            candidates: List of candidate dictionaries

        Returns:
            List of (rank, score, candidate) tuples
        """
        ranked = []

        for idx, candidate in enumerate(candidates):
            score = self.calculate_fitness(candidate)
            ranked.append((idx, score, candidate))

        # Sort by score (lower is better)
        ranked.sort(key=lambda x: x[1])

        return ranked


class ParetoOptimizer:
    """
    Pareto front optimization for multi-objective drug design
    """

    def __init__(self):
        pass

    def is_dominated(
        self,
        candidate1: np.ndarray,
        candidate2: np.ndarray,
        minimize: List[bool]
    ) -> bool:
        """
        Check if candidate1 is dominated by candidate2

        Args:
            candidate1: First candidate objective values
            candidate2: Second candidate objective values
            minimize: List of booleans indicating minimization

        Returns:
            True if candidate1 is dominated
        """
        better_in_any = False
        worse_in_any = False

        for i, (v1, v2, is_min) in enumerate(zip(candidate1, candidate2, minimize)):
            if is_min:
                if v2 < v1:
                    better_in_any = True
                elif v2 > v1:
                    worse_in_any = True
            else:
                if v2 > v1:
                    better_in_any = True
                elif v2 < v1:
                    worse_in_any = True

        return better_in_any and not worse_in_any

    def find_pareto_front(
        self,
        candidates: np.ndarray,
        minimize: List[bool]
    ) -> np.ndarray:
        """
        Find Pareto-optimal solutions

        Args:
            candidates: Array of shape (n_candidates, n_objectives)
            minimize: List indicating minimization for each objective

        Returns:
            Indices of Pareto-optimal candidates
        """
        n_candidates = candidates.shape[0]
        is_pareto = np.ones(n_candidates, dtype=bool)

        for i in range(n_candidates):
            if not is_pareto[i]:
                continue

            for j in range(n_candidates):
                if i == j or not is_pareto[j]:
                    continue

                if self.is_dominated(candidates[i], candidates[j], minimize):
                    is_pareto[i] = False
                    break

        return np.where(is_pareto)[0]

    def select_diverse_subset(
        self,
        pareto_candidates: np.ndarray,
        n_select: int,
        method: str = 'crowding'
    ) -> np.ndarray:
        """
        Select diverse subset from Pareto front

        Args:
            pareto_candidates: Pareto-optimal candidates
            n_select: Number to select
            method: Selection method ('crowding', 'random')

        Returns:
            Indices of selected candidates
        """
        if len(pareto_candidates) <= n_select:
            return np.arange(len(pareto_candidates))

        if method == 'crowding':
            # Use crowding distance for diversity
            distances = self._calculate_crowding_distance(pareto_candidates)
            selected_indices = np.argsort(distances)[-n_select:]
        else:
            # Random selection
            selected_indices = np.random.choice(
                len(pareto_candidates),
                size=n_select,
                replace=False
            )

        return selected_indices

    def _calculate_crowding_distance(self, candidates: np.ndarray) -> np.ndarray:
        """
        Calculate crowding distance for diversity

        Args:
            candidates: Array of candidates

        Returns:
            Crowding distances
        """
        n_candidates, n_objectives = candidates.shape
        distances = np.zeros(n_candidates)

        for obj_idx in range(n_objectives):
            # Sort by objective
            sorted_indices = np.argsort(candidates[:, obj_idx])

            # Assign infinite distance to boundary points
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # Calculate crowding distance for middle points
            obj_range = candidates[sorted_indices[-1], obj_idx] - candidates[sorted_indices[0], obj_idx]

            if obj_range > 0:
                for i in range(1, n_candidates - 1):
                    idx = sorted_indices[i]
                    idx_next = sorted_indices[i + 1]
                    idx_prev = sorted_indices[i - 1]

                    distance = (candidates[idx_next, obj_idx] - candidates[idx_prev, obj_idx]) / obj_range
                    distances[idx] += distance

        return distances


class ConstraintFilter:
    """
    Filter candidates based on hard constraints
    """

    def __init__(self):
        self.constraints = []

    def add_constraint(
        self,
        name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ):
        """
        Add a constraint

        Args:
            name: Property name
            min_value: Minimum allowed value
            max_value: Maximum allowed value
        """
        self.constraints.append({
            'name': name,
            'min': min_value,
            'max': max_value
        })

    def filter_candidates(
        self,
        candidates: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """
        Filter candidates by constraints

        Args:
            candidates: List of candidate dictionaries

        Returns:
            Filtered candidates
        """
        filtered = []

        for candidate in candidates:
            passes = True

            for constraint in self.constraints:
                prop_name = constraint['name']
                if prop_name not in candidate:
                    continue

                value = candidate[prop_name]

                if constraint['min'] is not None and value < constraint['min']:
                    passes = False
                    break

                if constraint['max'] is not None and value > constraint['max']:
                    passes = False
                    break

            if passes:
                filtered.append(candidate)

        logger.info(f"Filtered {len(filtered)}/{len(candidates)} candidates")
        return filtered
