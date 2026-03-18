"""
Retrosynthesis Planning and Synthesis Feasibility Scoring
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import logging

import numpy as np

from drug_discovery.web_scraping import AISynthesisChat, InternetSearchClient, OnlineResourceReader

logger = logging.getLogger(__name__)


class RetrosynthesisPlanner:
    """
    Retrosynthesis planning for drug candidates
    Identifies synthetic routes and assesses feasibility
    """

    def __init__(self):
        self.reaction_templates = []
        self.internet_search = InternetSearchClient()
        self.resource_reader = OnlineResourceReader()

    def plan_synthesis(self, target_smiles: str, max_depth: int = 5) -> dict:
        """
        Plan synthetic route for a target molecule

        Args:
            target_smiles: Target molecule SMILES
            max_depth: Maximum retrosynthetic depth

        Returns:
            Dictionary with synthesis plan
        """
        try:
            logger.info(f"Planning synthesis for {target_smiles}")

            # Placeholder for actual retrosynthesis implementation
            # Would integrate with tools like RXNMapper, AiZynthFinder, etc.

            result = {
                "target": target_smiles,
                "success": True,
                "num_steps": np.random.randint(2, max_depth + 1),
                "estimated_yield": np.random.uniform(0.3, 0.9),
                "complexity_score": np.random.uniform(1.0, 10.0),
                "available_building_blocks": True,
                "novel_chemistry": np.random.random() > 0.8,
            }

            return result

        except Exception as e:
            logger.error(f"Retrosynthesis planning error: {e}")
            return {"success": False, "error": str(e)}

    def plan_synthesis_with_research(
        self,
        target_smiles: str,
        target_protein: str | None = None,
        max_depth: int = 5,
        max_research_results: int = 5,
        use_internet: bool = True,
        use_ai_chat: bool = True,
        read_online_resources: bool = True,
        max_resource_reads: int = 3,
    ) -> dict:
        """
        Plan synthesis and enrich output with web research and AI guidance.

        Args:
            target_smiles: Target molecule SMILES
            target_protein: Optional target protein context
            max_depth: Maximum retrosynthetic depth
            max_research_results: Number of web results to include
            use_internet: Enable internet research
            use_ai_chat: Enable LLM-generated synthesis brief
            read_online_resources: Read and summarize fetched resources (HTML/PDF)
            max_resource_reads: Number of URLs to fetch and parse

        Returns:
            Extended synthesis plan dictionary
        """
        plan = self.plan_synthesis(target_smiles=target_smiles, max_depth=max_depth)
        if not plan.get("success"):
            return plan

        research_hits: list[dict[str, str]] = []
        query = f"{target_smiles} synthesis route medicinal chemistry"
        if target_protein:
            query = f"{target_smiles} {target_protein} synthesis route medicinal chemistry"

        if use_internet:
            research_hits = self.internet_search.search_web(query=query, max_results=max_research_results)

        if use_internet and read_online_resources and research_hits:
            research_hits = self.resource_reader.enrich_search_hits(research_hits, max_reads=max_resource_reads)

        plan["research_query"] = query
        plan["research_hits"] = research_hits

        if use_ai_chat:
            try:
                ai_chat = AISynthesisChat()
                ai_output = ai_chat.generate_synthesis_brief(
                    smiles=target_smiles,
                    target_protein=target_protein,
                    research_hits=research_hits,
                )
                plan["ai_synthesis_guidance"] = ai_output.get("brief", "")
                plan["ai_model_id"] = ai_output.get("model_id", "")
            except Exception as exc:
                logger.warning(f"AI synthesis guidance unavailable: {exc}")
                plan["ai_synthesis_guidance"] = ""
                plan["ai_error"] = str(exc)

        return plan

    def score_synthetic_accessibility(self, smiles: str) -> float:
        """
        Score synthetic accessibility (1-10, lower is easier)

        Args:
            smiles: Molecule SMILES

        Returns:
            SA score
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 10.0  # Worst score

            # Use RDKit's built-in SA score if available
            # Otherwise, use simple heuristics
            ring_count = getattr(Descriptors, "RingCount")
            num_rotatable_bonds = getattr(Descriptors, "NumRotatableBonds")
            mol_wt_func = getattr(Descriptors, "MolWt")

            num_rings = ring_count(mol)
            num_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            num_rotatable = num_rotatable_bonds(mol)
            mol_wt = mol_wt_func(mol)

            # Simple heuristic (replace with actual SAScore)
            score = 1.0
            score += min(num_rings * 0.5, 3.0)
            score += min(num_stereo * 0.8, 3.0)
            score += min(num_rotatable * 0.1, 2.0)
            score += min((mol_wt - 200) / 100, 1.0) if mol_wt > 200 else 0

            return min(max(score, 1.0), 10.0)

        except Exception as e:
            logger.error(f"SA score calculation error: {e}")
            return 10.0


class SynthesisFeasibilityScorer:
    """
    Score synthesis feasibility based on multiple criteria
    """

    def __init__(self):
        pass

    def score_feasibility(self, smiles: str, retro_plan: dict | None = None) -> dict[str, float]:
        """
        Comprehensive synthesis feasibility scoring

        Args:
            smiles: Molecule SMILES
            retro_plan: Optional retrosynthesis plan

        Returns:
            Dictionary of feasibility scores
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"overall": 0.0}

            scores = {}

            # Synthetic accessibility (1-10, lower better)
            planner = RetrosynthesisPlanner()
            sa_score = planner.score_synthetic_accessibility(smiles)
            scores["sa_score"] = 11 - sa_score  # Convert to 1-10, higher better

            # Molecular complexity
            ring_count = getattr(Descriptors, "RingCount")
            num_heteroatoms_func = getattr(Lipinski, "NumHeteroatoms")
            num_rings = ring_count(mol)
            num_heteroatoms = num_heteroatoms_func(mol)
            complexity = (num_rings + num_heteroatoms) / 10.0
            scores["complexity"] = max(0, 10 - complexity * 10)

            # Functional group diversity (penalize exotic groups)
            num_aliphatic_rings = getattr(Descriptors, "NumAliphaticRings")
            num_aromatic_rings = getattr(Descriptors, "NumAromaticRings")
            num_functional_groups = num_aliphatic_rings(mol) + num_aromatic_rings(mol)
            scores["fg_score"] = min(10, num_functional_groups)

            # Retrosynthesis plan quality
            if retro_plan and retro_plan.get("success"):
                num_steps = retro_plan.get("num_steps", 5)
                scores["retro_steps"] = max(0, 10 - num_steps)
                scores["estimated_yield"] = retro_plan.get("estimated_yield", 0.5) * 10
            else:
                scores["retro_steps"] = 5.0
                scores["estimated_yield"] = 5.0

            # Overall score (average)
            scores["overall"] = np.mean(list(scores.values()))

            return scores

        except Exception as e:
            logger.error(f"Feasibility scoring error: {e}")
            return {"overall": 0.0}

    def filter_synthesizable(self, smiles_list: list[str], threshold: float = 5.0) -> list[tuple[str, float]]:
        """
        Filter molecules by synthesis feasibility

        Args:
            smiles_list: List of SMILES strings
            threshold: Minimum feasibility score

        Returns:
            List of (SMILES, score) tuples
        """
        feasible = []

        for smiles in smiles_list:
            scores = self.score_feasibility(smiles)
            overall_score = scores.get("overall", 0.0)

            if overall_score >= threshold:
                feasible.append((smiles, overall_score))

        # Sort by score (highest first)
        feasible.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Found {len(feasible)} synthesizable molecules from {len(smiles_list)}")

        return feasible
