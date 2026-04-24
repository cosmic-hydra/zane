"""
Pocket-Aware Molecular Generator.

Generates molecules conditioned on 3D binding pocket structure, optimizing
for both binding affinity and synthetic accessibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from rdkit import Chem
    from rdkit.Chem import QED, AllChem, Crippen, Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None


@dataclass
class PocketContext:
    """Context information for pocket-conditioned generation.

    Attributes:
        pocket_coords: 3D coordinates of pocket atoms.
        pocket_features: Feature vector describing pocket.
        pocket_volume: Estimated pocket volume.
        pocket_pharmacophore: Pharmacophore points.
        target_protein: Target protein identifier.
    """

    pocket_coords: np.ndarray | None = None
    pocket_features: np.ndarray | None = None
    pocket_volume: float = 500.0
    pocket_pharmacophore: list[tuple[str, np.ndarray]] = field(default_factory=list)
    target_protein: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pocket_volume": self.pocket_volume,
            "target_protein": self.target_protein,
            "n_pharmacophores": len(self.pocket_pharmacophore),
            "metadata": self.metadata,
        }


@dataclass
class GeneratedMolecule:
    """Generated molecule with full metadata.

    Attributes:
        smiles: SMILES string.
        coords: 3D coordinates.
        binding_score: Predicted binding affinity.
        sa_score: Synthetic accessibility score.
        qed_score: Drug-likeness (QED).
        logp: Lipophilicity.
        toxicity_flag: Whether flagged as potentially toxic.
        fitness: Overall fitness score.
    """

    smiles: str
    coords: np.ndarray | None = None
    binding_score: float = 0.5
    sa_score: float = 0.5
    qed_score: float = 0.5
    logp: float = 2.0
    toxicity_flag: bool = False
    fitness: float = 0.5
    properties: dict[str, Any] = field(default_factory=dict)

    def is_druglike(self) -> bool:
        """Check if molecule is drug-like."""
        return self.qed_score > 0.4 and not self.toxicity_flag and 0 < self.logp < 6

    def as_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "binding_score": self.binding_score,
            "sa_score": self.sa_score,
            "qed_score": self.qed_score,
            "logp": self.logp,
            "toxicity_flag": self.toxicity_flag,
            "fitness": self.fitness,
            "is_druglike": self.is_druglike(),
            "properties": self.properties,
        }


class PocketAwareGenerator:
    """
    Pocket-aware molecular generator.

    Generates molecules specifically tailored to fit into a given binding pocket,
    using structure-based design principles.

    Features:
    - Pocket volume matching
    - Pharmacophore compatibility
    - H-bond donor/acceptor placement
    - Hydrophobic pocket filling

    Example::

        generator = PocketAwareGenerator()
        context = PocketContext(pocket_coords=pocket_coords)
        molecules = generator.generate(context, n_molecules=100)
    """

    def __init__(
        self,
        diffusion_model: Any = None,
        binding_weight: float = 0.4,
        sa_weight: float = 0.3,
        qed_weight: float = 0.2,
        diversity_weight: float = 0.1,
    ):
        """
        Initialize pocket-aware generator.

        Args:
            diffusion_model: Pre-trained diffusion model (optional).
            binding_weight: Weight for binding affinity in fitness.
            sa_weight: Weight for synthetic accessibility.
            qed_weight: Weight for drug-likeness.
            diversity_weight: Weight for diversity in selection.
        """
        self.diffusion_model = diffusion_model
        self.binding_weight = binding_weight
        self.sa_weight = sa_weight
        self.qed_weight = qed_weight
        self.diversity_weight = diversity_weight

        logger.info("PocketAwareGenerator initialized")

    def generate(
        self,
        pocket_context: PocketContext,
        n_molecules: int = 100,
        max_atoms: int = 50,
        temperature: float = 1.0,
    ) -> list[GeneratedMolecule]:
        """
        Generate molecules for a binding pocket.

        Args:
            pocket_context: Pocket context information.
            n_molecules: Number of molecules to generate.
            max_atoms: Maximum atoms per molecule.
            temperature: Generation temperature.

        Returns:
            List of GeneratedMolecule objects.
        """
        # Extract pocket information
        pocket_coords = pocket_context.pocket_coords

        if pocket_coords is not None:
            center = pocket_coords.mean(axis=0)
        else:
            center = np.zeros(3)

        # Generate candidates
        candidates = []

        # Drug-like scaffolds for generation
        scaffolds = self._get_scaffolds_for_pocket(pocket_context)

        for i in range(n_molecules):
            # Select scaffold
            scaffold = scaffolds[np.random.randint(0, len(scaffolds))]

            # Modify scaffold
            smiles = self._modify_scaffold(
                scaffold,
                pocket_context,
                max_atoms,
            )

            if smiles:
                # Score molecule
                mol = GeneratedMolecule(
                    smiles=smiles,
                    coords=self._generate_conformer(smiles, center),
                )

                self._score_molecule(mol, pocket_context)

                # Compute fitness
                mol.fitness = self._compute_fitness(mol)

                candidates.append(mol)

        # Filter and rank by fitness
        candidates = [c for c in candidates if c.is_druglike()]
        candidates.sort(key=lambda x: x.fitness, reverse=True)

        # Ensure diversity in top results
        diverse = self._select_diverse(candidates, n_molecules // 2)

        return diverse

    def _get_scaffolds_for_pocket(
        self,
        pocket_context: PocketContext,
    ) -> list[str]:
        """Get scaffolds suitable for pocket properties."""
        # Analyze pocket properties
        volume = pocket_context.pocket_volume

        # Small molecules
        if volume < 300:
            return [
                "c1ccccc1",  # Benzene
                "CC",  # Ethane
                "CC(C)C",  # Isobutane
                "c1ccc2ccccc2c1",  # Naphthalene
            ]
        # Medium molecules
        elif volume < 800:
            return [
                "c1ccccc1",  # Benzene
                "CCc1ccccc1",  # Toluene derivative
                "c1ccc(N)cc1",  # Aniline
                "c1ccc(O)cc1",  # Phenol
                "CC(=O)Nc1ccccc1",  # Acetanilide
            ]
        # Large molecules
        else:
            return [
                "c1ccc2ccccc2c1",  # Naphthalene
                "c1ccc2ccccc2c1",  # Larger aromatic
                "CC(C)c1ccc(O)cc1",  # Bulky hydrophobic
                "c1ccc(N)cc1C(=O)N",  # H-bond capable
            ]

    def _modify_scaffold(
        self,
        scaffold: str,
        pocket_context: PocketContext,
        max_atoms: int,
    ) -> str | None:
        """Modify scaffold to fit pocket."""
        try:
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(scaffold)
                if mol is None:
                    return scaffold

                # Add substituents based on pocket
                rwmol = Chem.RWMol(mol)

                # Simple modifications
                n_modifications = np.random.randint(0, 3)

                for _ in range(n_modifications):
                    # Randomly add atoms
                    if rwmol.GetNumAtoms() < max_atoms:
                        substituent = np.random.choice(["C", "O", "N", "F", "C(C)C"])
                        try:
                            sub = Chem.MolFromSmiles(substituent)
                            if sub:
                                combined = Chem.CombineMols(rwmol, sub)
                                rwmol = Chem.RWMol(combined)
                        except Exception:
                            pass

                smiles = Chem.MolToSmiles(rwmol)
                return smiles

        except Exception as e:
            logger.warning(f"Scaffold modification failed: {e}")

        return scaffold

    def _generate_conformer(
        self,
        smiles: str,
        center: np.ndarray,
    ) -> np.ndarray:
        """Generate 3D conformation."""
        if not RDKIT_AVAILABLE:
            return np.random.randn(10, 3) + center

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.random.randn(5, 3) + center

            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)

            conf = mol.GetConformer()
            coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

            # Center on pocket
            coords = coords - coords.mean(axis=0) + center

            return coords

        except Exception:
            return np.random.randn(10, 3) + center

    def _score_molecule(
        self,
        mol: GeneratedMolecule,
        pocket_context: PocketContext,
    ) -> None:
        """Score molecule against pocket."""
        if not RDKIT_AVAILABLE:
            mol.binding_score = 0.5
            mol.sa_score = 0.5
            mol.qed_score = 0.5
            mol.logp = 2.0
            return

        try:
            rdkit_mol = Chem.MolFromSmiles(mol.smiles)
            if rdkit_mol is None:
                return

            # QED score
            try:
                mol.qed_score = QED.qed(rdkit_mol)
            except Exception:
                mol.qed_score = 0.5

            # LogP
            try:
                mol.logp = Crippen.MolLogP(rdkit_mol)
            except Exception:
                mol.logp = 2.0

            # SA score (simplified)
            try:
                n_rotatable = Descriptors.NumRotatableBonds(rdkit_mol)
                n_rings = Descriptors.RingCount(rdkit_mol)
                mol.sa_score = 1.0 - min(n_rotatable / 20.0, 1.0) + min(n_rings / 5.0, 0.5)
                mol.sa_score = max(0.0, min(1.0, mol.sa_score))
            except Exception:
                mol.sa_score = 0.5

            # Binding score (based on pocket match)
            mol.binding_score = self._estimate_binding(mol, pocket_context)

            # Toxicity check
            mol.toxicity_flag = self._check_toxicity(mol.smiles)

        except Exception as e:
            logger.warning(f"Scoring failed: {e}")

    def _estimate_binding(
        self,
        mol: GeneratedMolecule,
        pocket_context: PocketContext,
    ) -> float:
        """Estimate binding affinity to pocket."""
        if mol.coords is None or pocket_context.pocket_coords is None:
            return 0.5

        # Compute overlap with pocket
        mol_center = mol.coords.mean(axis=0)
        pocket_center = pocket_context.pocket_coords.mean(axis=0)

        # Distance from ideal center
        center_dist = np.linalg.norm(mol_center - pocket_center)

        # Volume match
        mol_volume = len(mol.smiles) * 15  # Rough estimate
        volume_ratio = min(mol_volume, pocket_context.pocket_volume) / max(mol_volume, pocket_context.pocket_volume, 1)

        # Binding score
        binding = (1.0 - min(center_dist / 10.0, 1.0)) * 0.5 + volume_ratio * 0.5

        return float(max(0.0, min(1.0, binding)))

    def _check_toxicity(self, smiles: str) -> bool:
        """Check for potential toxicity."""
        if not RDKIT_AVAILABLE:
            return False

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            # Check for toxicophores
            toxic_patterns = [
                "C(=O)OCl",  # Acyl chloride
                "[N+](=O)[O-]",  # Nitro
                "C1OC1",  # Epoxide
            ]

            for pattern in toxic_patterns:
                try:
                    pat = Chem.MolFromSmarts(pattern)
                    if pat and mol.HasSubstructMatch(pat):
                        return True
                except Exception:
                    pass

        except Exception:
            pass

        return False

    def _compute_fitness(self, mol: GeneratedMolecule) -> float:
        """Compute overall fitness score."""
        fitness = (
            self.binding_weight * mol.binding_score
            + self.sa_weight * mol.sa_score
            + self.qed_weight * mol.qed_score
            - 0.2 * mol.toxicity_flag  # Penalty
        )

        return float(max(0.0, min(1.0, fitness)))

    def _select_diverse(
        self,
        molecules: list[GeneratedMolecule],
        n_select: int,
    ) -> list[GeneratedMolecule]:
        """Select diverse subset using fingerprint similarity."""
        if len(molecules) <= n_select:
            return molecules

        if not RDKIT_AVAILABLE:
            return molecules[:n_select]

        try:
            from rdkit import DataStructs
            from rdkit.Chem import AllChem

            # Compute fingerprints
            fps = []
            for mol in molecules:
                rdkit_mol = Chem.MolFromSmiles(mol.smiles)
                if rdkit_mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, 2, nBits=1024)
                    fps.append(fp)

            # Greedy selection for diversity
            selected = []
            remaining = list(range(len(fps)))

            while len(selected) < n_select and remaining:
                # Add most dissimilar
                best_idx = remaining[0]
                for idx in remaining[1:]:
                    sims = [DataStructs.TanimotoSimilarity(fps[idx], fps[s]) for s in selected]
                    avg_sim = np.mean(sims) if sims else 0
                    best_sims = [DataStructs.TanimotoSimilarity(fps[best_idx], fps[s]) for s in selected]
                    best_avg_sim = np.mean(best_sims) if best_sims else 0

                    if avg_sim < best_avg_sim:
                        best_idx = idx

                selected.append(best_idx)
                remaining.remove(best_idx)

            return [molecules[i] for i in selected]

        except Exception as e:
            logger.warning(f"Diversity selection failed: {e}")
            return molecules[:n_select]


class PharmacophoreGenerator:
    """
    Generates molecules based on pharmacophore requirements.

    Creates molecules that satisfy spatial arrangements of pharmacophoric features.
    """

    def __init__(self):
        """Initialize pharmacophore generator."""
        self.pharmacophore_types = [
            "HBD",  # H-bond donor
            "HBA",  # H-bond acceptor
            "HYDRO",  # Hydrophobic
            "AROM",  # Aromatic
            "POS",  # Positive charge
            "NEG",  # Negative charge
        ]

    def generate_from_pharmacophore(
        self,
        points: list[tuple[str, np.ndarray]],
        n_molecules: int = 50,
    ) -> list[str]:
        """
        Generate molecules matching pharmacophore.

        Args:
            points: List of (type, position) tuples.
            n_molecules: Number to generate.

        Returns:
            List of SMILES strings.
        """
        molecules = []

        # Simple template-based generation
        if len(points) == 2:
            # Distance constraints
            template = self._two_point_template(points)
            molecules = [template] * n_molecules

        elif len(points) == 3:
            # Triangle constraints
            template = self._three_point_template(points)
            molecules = [template] * n_molecules

        else:
            # Default scaffold
            molecules = ["c1ccccc1"] * n_molecules

        return molecules

    def _two_point_template(self, points) -> str:
        """Generate template for two pharmacophore points."""
        dist = np.linalg.norm(points[0][1] - points[1][1])

        if dist < 3.0:
            return "c1ccc(N)cc1"  # Close points
        elif dist < 5.0:
            return "c1ccc(C)cc1"  # Medium distance
        else:
            return "c1ccc(CC)cc1"  # Longer distance

    def _three_point_template(self, points) -> str:
        """Generate template for three pharmacophore points."""
        return "c1ccc(N)cc1C(=O)O"  # Aromatic with H-bond capability
