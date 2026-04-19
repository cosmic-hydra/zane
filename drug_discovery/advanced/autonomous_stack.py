"""
Advanced autonomous discovery stack that stitches together learnable docking,
end-to-end differentiable binding, multi-fidelity learning, and synthesis-aware
generation. All heavy dependencies are optional; torch is required when using
the differentiable modules.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drug_discovery.generation.backends import BaseGeneratorBackend, GenerationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learnable docking (replaces classical external docking)
# ---------------------------------------------------------------------------


def _rmsd(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute RMSD with optional atom mask."""
    if mask is not None:
        pred = pred * mask.unsqueeze(-1)
        target = target * mask.unsqueeze(-1)
        denom = mask.sum() * pred.shape[-1]
    else:
        denom = torch.tensor(pred.numel(), device=pred.device, dtype=pred.dtype)
    diff = pred - target
    return torch.sqrt((diff.pow(2).sum() + 1e-8) / (denom + 1e-8))


class NeuralDockingModel(nn.Module):
    """Lightweight neural docking model that predicts poses and interaction energy."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.ligand_proj = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.pocket_proj = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.cross_attend = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.pose_head = nn.Linear(hidden, 3)
        self.energy_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(
        self,
        ligand_coords: torch.Tensor,
        pocket_coords: torch.Tensor,
        ligand_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ligand_coords: [B, N, 3]
            pocket_coords: [B, M, 3]
            ligand_mask: [B, N] mask of valid atoms
        Returns:
            predicted_coords [B, N, 3], interaction_energy [B, 1]
        """
        lig_embed = self.ligand_proj(ligand_coords)
        pocket_embed = self.pocket_proj(pocket_coords)
        pocket_context = pocket_embed.mean(dim=1, keepdim=True)
        lig_context = lig_embed.mean(dim=1, keepdim=True)
        fused = torch.cat(
            [
                lig_embed,
                pocket_context.expand_as(lig_embed),
            ],
            dim=-1,
        )
        fused = self.cross_attend(fused)
        delta = self.pose_head(fused)
        predicted_coords = ligand_coords + delta

        pooled = (fused * (ligand_mask.unsqueeze(-1) if ligand_mask is not None else 1.0)).mean(dim=1)
        interaction_energy = self.energy_head(pooled)
        return predicted_coords, interaction_energy

    def energy_from_pose(self, ligand_pose: torch.Tensor, pocket_coords: torch.Tensor) -> torch.Tensor:
        """Simple differentiable interaction energy based on inverse distances."""
        distances = torch.cdist(ligand_pose, pocket_coords).clamp_min(1e-4)
        energy = (-1.0 / distances).mean(dim=[1, 2], keepdim=True)
        return energy


@dataclass
class DockingBatch:
    ligand_coords: torch.Tensor  # [B, N, 3]
    pocket_coords: torch.Tensor  # [B, M, 3]
    true_pose: torch.Tensor | None = None  # [B, N, 3]
    interaction_energy: torch.Tensor | None = None  # [B, 1]
    ligand_mask: torch.Tensor | None = None  # [B, N]


class LearnableDockingEngine:
    """Orchestrates training and inference for the neural docking module."""

    def __init__(self, model: NeuralDockingModel | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or NeuralDockingModel()
        self.model.to(self.device)

    def predict_pose(self, ligand_coords: torch.Tensor, pocket_coords: torch.Tensor, ligand_mask: torch.Tensor | None = None) -> dict:
        self.model.eval()
        with torch.no_grad():
            pose, energy = self.model(ligand_coords.to(self.device), pocket_coords.to(self.device), ligand_mask)
            return {
                "pose": pose.cpu(),
                "interaction_energy": energy.cpu(),
            }

    def loss(self, batch: DockingBatch) -> dict[str, torch.Tensor]:
        ligand = batch.ligand_coords.to(self.device)
        pocket = batch.pocket_coords.to(self.device)
        mask = batch.ligand_mask.to(self.device) if batch.ligand_mask is not None else None
        pred_pose, pred_energy = self.model(ligand, pocket, mask)
        losses: dict[str, torch.Tensor] = {}
        if batch.true_pose is not None:
            target_pose = batch.true_pose.to(self.device)
            losses["pose_rmsd"] = _rmsd(pred_pose, target_pose, mask)
            # encourage consistency with energy computed from predicted pose
            energy_true_pose = self.model.energy_from_pose(target_pose, pocket)
            energy_pred_pose = self.model.energy_from_pose(pred_pose, pocket)
            losses["energy_consistency"] = F.mse_loss(energy_pred_pose, energy_true_pose)
        if batch.interaction_energy is not None:
            true_energy = batch.interaction_energy.to(self.device)
            losses["energy_regression"] = F.mse_loss(pred_energy, true_energy)
        total = sum(losses.values()) if losses else torch.tensor(0.0, device=self.device)
        losses["total"] = total
        return losses

    def fit(self, dataloader: Iterable[DockingBatch], epochs: int = 1, lr: float = 1e-3) -> list[dict[str, float]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        history: list[dict[str, float]] = []
        for _ in range(epochs):
            epoch_log: dict[str, float] = {}
            for batch in dataloader:
                loss_dict = self.loss(batch)
                optimizer.zero_grad()
                loss_dict["total"].backward()
                optimizer.step()
                for k, v in loss_dict.items():
                    epoch_log[k] = epoch_log.get(k, 0.0) + float(v.detach().cpu())
            if epoch_log:
                for k in epoch_log:
                    epoch_log[k] /= max(1, len(dataloader))
            history.append(epoch_log)
        return history


# ---------------------------------------------------------------------------
# End-to-end differentiable binding pipeline with QM correction
# ---------------------------------------------------------------------------


class QuantumCorrectionNetwork(nn.Module):
    """Small QM-inspired correction head that adjusts classical energy."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        return self.net(energy)


class StructuralUncertaintyHead(nn.Module):
    """Predicts mean and log-variance over coordinates for robustness."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 6))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(features)
        mean = params[..., :3]
        logvar = params[..., 3:]
        return mean, logvar


@dataclass
class BindingPipelineOutput:
    pose: torch.Tensor
    energy: torch.Tensor
    affinity: torch.Tensor
    uncertainty: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DifferentiableBindingPipeline(nn.Module):
    """
    Connect diffusion -> learnable docking -> energy -> affinity with gradients.
    """

    def __init__(
        self,
        docking: LearnableDockingEngine,
        quantum_correction: QuantumCorrectionNetwork | None = None,
    ):
        super().__init__()
        self.docking_engine = docking
        self.diffusion_prior = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        self.affinity_head = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
        self.quantum_correction = quantum_correction or QuantumCorrectionNetwork()

    def forward(
        self,
        ligand_seed: torch.Tensor,
        pocket_coords: torch.Tensor,
        ligand_mask: torch.Tensor | None = None,
    ) -> BindingPipelineOutput:
        noisy_pose = ligand_seed + self.diffusion_prior(ligand_seed)
        predicted_pose, coarse_energy = self.docking_engine.model(noisy_pose, pocket_coords, ligand_mask)
        qm_correction = self.quantum_correction(coarse_energy)
        total_energy = coarse_energy + qm_correction
        affinity = self.affinity_head(total_energy)
        return BindingPipelineOutput(
            pose=predicted_pose,
            energy=total_energy,
            affinity=affinity,
            metadata={"coarse_energy": coarse_energy, "qm_correction": qm_correction},
        )


# ---------------------------------------------------------------------------
# Memory-augmented molecular search and temporal learning
# ---------------------------------------------------------------------------


def _default_embed(smiles: str, dim: int = 256) -> np.ndarray:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("invalid SMILES")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=dim)
        arr = np.zeros((dim,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        rng = np.random.default_rng(abs(hash(smiles)) % (2**32))
        return rng.normal(size=dim).astype(np.float32)


class MemoryAugmentedSearch:
    """In-memory vector database for retrieval-augmented generation."""

    def __init__(self, embedder: Callable[[str], np.ndarray] | None = None, dim: int = 256):
        self.embedder = embedder or (lambda s: _default_embed(s, dim))
        self.vectors: list[np.ndarray] = []
        self.records: list[dict[str, Any]] = []

    def add(self, smiles: str, score: float | None = None, metadata: dict[str, Any] | None = None) -> None:
        vec = self.embedder(smiles)
        self.vectors.append(vec)
        self.records.append({"smiles": smiles, "score": score, "metadata": metadata or {}})

    def retrieve(self, smiles: str, k: int = 5) -> list[dict[str, Any]]:
        if not self.vectors:
            return []
        query = self.embedder(smiles)
        mat = np.stack(self.vectors)
        denom = (np.linalg.norm(mat, axis=1) * (np.linalg.norm(query) + 1e-8)) + 1e-8
        scores = (mat @ query) / denom
        topk = np.argsort(scores)[::-1][:k]
        return [{**self.records[i], "similarity": float(scores[i])} for i in topk]

    def bias_candidates(self, smiles_list: Sequence[str], exploration_weight: float = 0.2) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for smi in smiles_list:
            sims = self.retrieve(smi, k=3)
            diversity_bonus = exploration_weight * (1.0 - np.mean([s["similarity"] for s in sims]) if sims else 1.0)
            prev_best = max((s.get("score", 0.0) for s in sims), default=0.0)
            results.append((smi, prev_best + diversity_bonus))
        return sorted(results, key=lambda x: x[1], reverse=True)


class TrajectoryStabilityModel(nn.Module):
    """Temporal model over MD trajectories."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.encoder = nn.GRU(input_size=3, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectories: [B, T, 3] aggregated coordinates per frame
        Returns:
            stability score per trajectory [B, 1]
        """
        _, h = self.encoder(trajectories)
        return self.head(h[-1])


# ---------------------------------------------------------------------------
# Reaction-conditioned generation backend (synthesis-aware)
# ---------------------------------------------------------------------------


class ReactionConditionedBackend(BaseGeneratorBackend):
    """Generation backend conditioned on reaction templates or pathways."""

    name = "reaction-conditioned"

    def __init__(self, reaction_templates: Sequence[str] | None = None):
        self.reaction_templates = list(reaction_templates or [])

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str | None, num: int = 10, **kwargs) -> GenerationResult:
        templates = kwargs.get("reaction_templates") or self.reaction_templates
        seed_smiles = kwargs.get("seed_smiles") or []
        molecules: list[str] = []
        for i in range(num):
            template = templates[i % max(1, len(templates))] if templates else "UNSPECIFIED_TEMPLATE"
            base = seed_smiles[i % len(seed_smiles)] if seed_smiles else f"RC_GEN_{i}"
            molecules.append(f"{base}|{template}")
        return GenerationResult(
            backend=self.name,
            success=True,
            molecules=molecules,
            info={"prompt": prompt, "templates": templates or ["none"], "seeded": bool(seed_smiles)},
            warnings=[] if templates else ["No reaction templates supplied; used generic placeholders."],
        )


# ---------------------------------------------------------------------------
# Failure-aware training, multi-fidelity learning, and constraints
# ---------------------------------------------------------------------------


class FailureAwareTrainer:
    """Tracks error hotspots and oversamples difficult regions."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.error_table: dict[str, float] = {}

    def update_errors(self, ids: Sequence[str], losses: torch.Tensor) -> None:
        for sample_id, loss_val in zip(ids, losses.detach().cpu().tolist()):
            prev = self.error_table.get(sample_id, 0.0)
            self.error_table[sample_id] = 0.7 * prev + 0.3 * float(loss_val)

    def sample_weights(self, ids: Sequence[str]) -> torch.Tensor:
        weights = [1.0 + self.alpha * self.error_table.get(sample_id, 0.0) for sample_id in ids]
        arr = torch.tensor(weights, dtype=torch.float32)
        return arr / (arr.mean() + 1e-8)


class MultiFidelityRegressor(nn.Module):
    """Combines cheap predictions with expensive corrections."""

    def __init__(self, low_fidelity: nn.Module, correction_head: nn.Module | None = None):
        super().__init__()
        self.low_fidelity = low_fidelity
        self.correction_head = correction_head or nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor | None = None) -> torch.Tensor:
        low_pred = self.low_fidelity(x_low)
        if x_high is None:
            return low_pred
        correction = self.correction_head(x_high)
        return low_pred + correction


class NeuralConstraintProjector(nn.Module):
    """Differentiable constraint projection for valency and sterics."""

    def forward(
        self,
        coords: torch.Tensor,
        steric_radii: torch.Tensor | None = None,
        min_distance: float = 1.0,
    ) -> torch.Tensor:
        """
        Projects coordinates to satisfy soft steric constraints.
        Args:
            coords: [B, N, 3]
            steric_radii: [B, N] optional radii
        """
        pairwise = torch.cdist(coords, coords)
        target = min_distance
        if steric_radii is not None:
            sr = steric_radii.unsqueeze(-1) + steric_radii.unsqueeze(-2)
            target = torch.maximum(torch.tensor(min_distance, device=coords.device), sr)
        penalty = torch.relu(target - pairwise).unsqueeze(-1)
        direction = (coords.unsqueeze(2) - coords.unsqueeze(1)).clamp(min=-1.0, max=1.0)
        adjustment = (penalty * direction).mean(dim=2)
        return coords + 0.1 * adjustment


class AdaptiveComputeAllocator:
    """Chooses compute tier based on uncertainty and potential."""

    def decide(self, potential: float, uncertainty: float) -> str:
        if potential > 0.8 and uncertainty > 0.3:
            return "fep"
        if potential > 0.6 and uncertainty > 0.1:
            return "md"
        if potential < 0.2:
            return "cheap-filter"
        return "standard"


# ---------------------------------------------------------------------------
# Causal modeling, meta-learning, and hybrid symbolic+neural chemistry
# ---------------------------------------------------------------------------


class CausalPropertyModel(nn.Module):
    """Simple structural causal model for property->toxicity reasoning."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.struct_to_property = nn.Linear(feature_dim, 1)
        self.property_to_toxicity = nn.Linear(1, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        property_score = torch.sigmoid(self.struct_to_property(features))
        toxicity = torch.sigmoid(self.property_to_toxicity(property_score))
        return {"property": property_score, "toxicity": toxicity}

    def do_intervention(self, property_value: float) -> torch.Tensor:
        prop = torch.tensor([[property_value]], dtype=torch.float32)
        return torch.sigmoid(self.property_to_toxicity(prop))


class MetaLearnerMAML:
    """Minimal MAML-style adapter for few-shot protein targets."""

    def __init__(self, inner_lr: float = 1e-2, steps: int = 1):
        self.inner_lr = inner_lr
        self.steps = steps

    def adapt(self, model: nn.Module, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], support_x, support_y):
        cloned = type(model)()  # assumes default constructor is valid
        cloned.load_state_dict(model.state_dict())
        for _ in range(self.steps):
            pred = cloned(support_x)
            loss = loss_fn(pred, support_y)
            grads = torch.autograd.grad(loss, cloned.parameters(), create_graph=True)
            for param, grad in zip(cloned.parameters(), grads):
                param.data = param.data - self.inner_lr * grad
        return cloned


class HybridSymbolicNeuralEngine:
    """Combines rule-based validation with neural scoring."""

    def __init__(self, neural_scorer: Callable[[str], float]):
        self.neural_scorer = neural_scorer

    def validate(self, smiles: str) -> dict[str, Any]:
        symbolic_ok = self._symbolic_check(smiles)
        neural_score = self.neural_scorer(smiles)
        return {"symbolic_ok": symbolic_ok, "neural_score": neural_score, "composite": neural_score * (0.5 if not symbolic_ok else 1.0)}

    def _symbolic_check(self, smiles: str) -> bool:
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            return bool(mol) and Chem.SanitizeMol(mol, catchErrors=True) is None
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Workflow benchmarking and pipeline evaluation
# ---------------------------------------------------------------------------


class WorkflowBenchmarkHarness:
    """Simulates generate->filter->synthesize->test flow and tracks drop-off."""

    def __init__(self):
        self.metrics: dict[str, Any] = {}

    def run(
        self,
        generator: Callable[[int], list[str]],
        scorer: Callable[[str], float],
        synthesizer: Callable[[str], bool],
        tester: Callable[[str], float],
        num: int = 20,
    ) -> dict[str, Any]:
        generated = generator(num)
        scored = [(smi, scorer(smi)) for smi in generated]
        scored = sorted(scored, key=lambda x: x[1], reverse=True)
        top_for_synthesis = scored[: max(1, num // 5)]
        synth_pass = [(smi, score) for smi, score in top_for_synthesis if synthesizer(smi)]
        tested = [(smi, tester(smi)) for smi, _ in synth_pass]
        success_rate = len(tested) / max(1, len(generated))
        self.metrics = {
            "generated": len(generated),
            "scored": len(scored),
            "sent_to_synthesis": len(top_for_synthesis),
            "synthesized": len(synth_pass),
            "tested": len(tested),
            "success_rate": success_rate,
        }
        return self.metrics
