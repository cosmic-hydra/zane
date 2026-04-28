"""GFlowNet-Based Molecular Generator for ZANE.

Generative Flow Networks for diverse, reward-proportional molecular sampling.
References: Bengio et al. "GFlowNet Foundations" (JMLR 2023)

The reward function supports two modes:

1. **Data-matching** (default) -- a user-supplied callable scores molecules.
2. **Physics-RLHF** -- the :class:`PhysicsRewardFunction` combines delta-G
   scores from the :class:`~drug_discovery.polyglot_integration.PhysicsOracle`
   with ADMET toxicity penalties, steering the GFlowNet toward low-energy,
   non-toxic candidates.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GFlowNetConfig:
    hidden_dim: int = 256
    num_layers: int = 4
    max_atoms: int = 38
    atom_vocab: int = 10
    bond_types: int = 4
    temperature: float = 1.0
    reward_exponent: float = 2.0
    learning_rate: float = 5e-4


# ---------------------------------------------------------------------------
# Physics-aware reward function
# ---------------------------------------------------------------------------
@dataclass
class PhysicsRewardConfig:
    """Tuneable knobs for the physics-driven reward."""

    delta_g_weight: float = 1.0
    toxicity_weight: float = 0.5
    delta_g_threshold: float = -5.0  # kcal/mol; scores above this get penalised
    toxicity_threshold: float = 0.5  # probability above which we penalise
    min_reward: float = 1e-6  # clamp floor (GFlowNet requires R > 0)


class PhysicsRewardFunction:
    """Reward function combining Physics Oracle delta-G with ADMET toxicity.

    The reward is designed so that:

    * **Negative delta-G** (strong binding) is rewarded.
    * **High toxicity probability** is penalised.
    * The output is always positive (GFlowNet requirement).

    Usage::

        from drug_discovery.polyglot_integration import PhysicsOracle

        oracle = PhysicsOracle(protein_pdb_path="target.pdb")
        reward_fn = PhysicsRewardFunction(oracle=oracle)

        # Inside the GFlowNet trainer:
        trainer = GFlowNetTrainer(config, reward_fn=reward_fn)
    """

    def __init__(
        self,
        oracle: Any | None = None,
        admet_predictor: Any | None = None,
        config: PhysicsRewardConfig | None = None,
        atoms_to_smiles: Callable[..., str] | None = None,
    ):
        """
        Args:
            oracle: :class:`~drug_discovery.polyglot_integration.PhysicsOracle`
                instance.  When *None* a mock delta-G is used.
            admet_predictor: Optional ADMET/toxicity predictor with a
                ``predict(smiles) -> dict`` interface.
            config: Reward configuration.
            atoms_to_smiles: Callable that converts a trajectory dict
                (``{"atoms": Tensor, ...}``) into a SMILES string.  A simple
                placeholder is used when *None*.
        """
        self.oracle = oracle
        self.admet_predictor = admet_predictor
        self.cfg = config or PhysicsRewardConfig()
        self._atoms_to_smiles = atoms_to_smiles or self._default_atoms_to_smiles

    # ------------------------------------------------------------------
    # Main entry -- called by GFlowNetTrainer.train_step
    # ------------------------------------------------------------------
    def __call__(self, trajectory: dict[str, Any]) -> float:
        """Compute the reward for a single generated molecule trajectory.

        Args:
            trajectory: Dict produced by
                :meth:`GFlowNetTrainer._sample_trajectory` containing at
                least ``"atoms"`` and ``"num_atoms"``.

        Returns:
            Positive float reward.
        """
        smiles = self._atoms_to_smiles(trajectory)

        delta_g = self._get_delta_g(smiles)
        toxicity = self._get_toxicity(smiles)

        reward = self._combine(delta_g, toxicity)
        return max(reward, self.cfg.min_reward)

    # ------------------------------------------------------------------
    # Delta-G scoring
    # ------------------------------------------------------------------
    def _get_delta_g(self, smiles: str) -> float:
        """Obtain binding free energy from the Physics Oracle."""
        if self.oracle is None:
            # Mock: hash-based pseudo-random in [-12, 0] range
            h = int(hash(smiles) % 10000) / 10000.0
            return -12.0 * h

        try:
            results = self.oracle.score_batch_sync([smiles])
            if results and results[0].success and results[0].delta_g is not None:
                return float(results[0].delta_g)
        except Exception as exc:
            logger.warning("PhysicsOracle failed for %s: %s", smiles, exc)

        return 0.0  # neutral reward on failure

    # ------------------------------------------------------------------
    # Toxicity scoring
    # ------------------------------------------------------------------
    def _get_toxicity(self, smiles: str) -> float:
        """Return a toxicity probability in [0, 1]."""
        if self.admet_predictor is None:
            return 0.0  # assume non-toxic when no predictor

        try:
            result = self.admet_predictor.predict(smiles)
            if isinstance(result, dict):
                # Support various key names
                for key in ("toxicity", "tox_probability", "cytotoxicity", "overall_toxicity"):
                    if key in result:
                        return float(result[key])
            if isinstance(result, (int, float)):
                return float(result)
        except Exception as exc:
            logger.warning("ADMET predictor failed for %s: %s", smiles, exc)

        return 0.0

    # ------------------------------------------------------------------
    # Reward combination
    # ------------------------------------------------------------------
    def _combine(self, delta_g: float, toxicity: float) -> float:
        """Combine delta-G and toxicity into a single positive reward.

        Strategy:
        - Binding component: ``exp(-w * delta_g)`` so more negative delta_g
          yields higher reward.
        - Toxicity penalty: ``(1 - tox)^w_tox`` scales the reward down when
          toxicity is high.
        """
        # Binding reward -- more negative delta_g -> larger positive value
        binding_reward = math.exp(-self.cfg.delta_g_weight * delta_g)

        # Toxicity penalty -- 1.0 when non-toxic, approaching 0 as tox -> 1
        tox_penalty = max(1.0 - toxicity, 0.0) ** self.cfg.toxicity_weight

        # Extra hard penalty if toxicity exceeds threshold
        if toxicity > self.cfg.toxicity_threshold:
            tox_penalty *= 0.1

        return binding_reward * tox_penalty

    # ------------------------------------------------------------------
    # Multi-objective scoring (via ToxicityGate + ParetoRanker)
    # ------------------------------------------------------------------
    def score_multi_objective(self, smiles: str) -> dict[str, float]:
        """Compute a full multi-objective score dict for a molecule.

        Returns dict with keys: delta_g, toxicity, drug_likeness, sa_score.
        Useful for feeding into the ParetoRanker.
        """
        delta_g = self._get_delta_g(smiles)
        toxicity = self._get_toxicity(smiles)
        drug_likeness = 0.5  # default
        sa_score = 3.0  # default (lower is more synthesizable)

        # Use ToxicityGate for richer evaluation if available
        try:
            from drug_discovery.safety.toxicity_gate import ToxicityGate

            gate = ToxicityGate()
            verdict = gate.evaluate(smiles)
            toxicity = verdict.overall_toxicity
            drug_likeness = verdict.drug_likeness
        except Exception:
            pass

        return {
            "smiles": smiles,
            "delta_g": delta_g,
            "toxicity": toxicity,
            "drug_likeness": drug_likeness,
            "sa_score": sa_score,
        }

    # ------------------------------------------------------------------
    # Atom-to-SMILES converter
    # ------------------------------------------------------------------
    @staticmethod
    def _default_atoms_to_smiles(trajectory: dict[str, Any]) -> str:
        """Placeholder converter: maps atom indices to element symbols.

        A production system would reconstruct the molecular graph from
        ``trajectory["atoms"]`` and ``trajectory["edges"]`` and convert
        via RDKit.  This stub produces a deterministic string usable by
        the oracle.
        """
        ELEMENT_MAP = {0: "C", 1: "N", 2: "O", 3: "S", 4: "F", 5: "Cl", 6: "Br", 7: "P", 8: "I", 9: "B"}
        atoms = trajectory.get("atoms")
        if atoms is None or (hasattr(atoms, "numel") and atoms.numel() == 0):
            return "C"  # single carbon fallback
        if hasattr(atoms, "tolist"):
            atoms = atoms.tolist()
        elements = [ELEMENT_MAP.get(a % len(ELEMENT_MAP), "C") for a in atoms]
        # Build a simple linear SMILES
        return "".join(elements) if elements else "C"


# ---------------------------------------------------------------------------
# Forward policy
# ---------------------------------------------------------------------------
class GFlowNetPolicy(nn.Module):
    def __init__(self, config: GFlowNetConfig):
        super().__init__()
        h = config.hidden_dim
        self.atom_embed = nn.Embedding(config.atom_vocab + 1, h)
        self.graph_encoder = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "msg": nn.Sequential(nn.Linear(2 * h + 1, h), nn.SiLU(), nn.Linear(h, h)),
                        "upd": nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h)),
                        "ln": nn.LayerNorm(h),
                    }
                )
                for _ in range(config.num_layers)
            ]
        )
        self.add_atom_head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, config.atom_vocab))
        self.stop_head = nn.Sequential(nn.Linear(h, h // 2), nn.SiLU(), nn.Linear(h // 2, 1))
        self.config = config

    def forward(self, state):
        z, edge_index = state["atoms"], state["edge_index"]
        h = self.atom_embed(z)
        if edge_index.numel() > 0:
            row, col = edge_index
            for layer in self.graph_encoder:
                m = layer["msg"](torch.cat([h[row], h[col], torch.ones(row.size(0), 1, device=h.device)], -1))
                agg = torch.zeros_like(h)
                agg.index_add_(0, row, m)
                h = layer["ln"](h + layer["upd"](torch.cat([h, agg], -1)))
        g = h.mean(0) if h.size(0) > 0 else torch.zeros(self.config.hidden_dim, device=h.device)
        return {
            "atom_logits": self.add_atom_head(g) / self.config.temperature,
            "stop_logit": self.stop_head(g),
            "graph_embedding": g,
        }


# ---------------------------------------------------------------------------
# Backward policy
# ---------------------------------------------------------------------------
class GFlowNetBackwardPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.hidden_dim
        self.net = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, g):
        return self.net(g)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class GFlowNetTrainer:
    """Trajectory Balance training for GFlowNet molecular generation.

    Supports two reward modes:

    * Pass a plain callable as *reward_fn* for data-matching.
    * Pass a :class:`PhysicsRewardFunction` for physics-RLHF training
      driven by OpenMM delta-G and ADMET toxicity penalties.

    Example::

        trainer = GFlowNetTrainer(config, reward_fn=fn, device="cuda")
    """

    def __init__(self, config: GFlowNetConfig, reward_fn: Callable[..., float] | None = None, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.forward_policy = GFlowNetPolicy(config).to(self.device)
        self.backward_policy = GFlowNetBackwardPolicy(config).to(self.device)
        self.log_Z = nn.Parameter(torch.zeros(1, device=self.device))
        self.reward_fn = reward_fn or (lambda m: 1.0)
        self.optimizer = torch.optim.Adam(
            list(self.forward_policy.parameters()) + list(self.backward_policy.parameters()) + [self.log_Z],
            lr=config.learning_rate,
        )

    def train_step(self) -> float:
        self.optimizer.zero_grad()
        traj, log_pf, log_pb = self._sample_trajectory()
        r = self.reward_fn(traj)
        loss = (self.log_Z + log_pf - log_pb - math.log(max(r, 1e-8)) * self.config.reward_exponent) ** 2
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _sample_trajectory(self):
        atoms = torch.zeros(0, dtype=torch.long, device=self.device)
        edges = torch.zeros(2, 0, dtype=torch.long, device=self.device)
        log_pf = torch.tensor(0.0, device=self.device)
        log_pb = torch.tensor(0.0, device=self.device)
        for step in range(self.config.max_atoms):
            out = self.forward_policy({"atoms": atoms, "edge_index": edges})
            sp = torch.sigmoid(out["stop_logit"])
            if step > 0 and torch.rand(1, device=self.device) < sp:
                log_pf = log_pf + torch.log(sp + 1e-8)
                break
            if step > 0:
                log_pf = log_pf + torch.log(1 - sp + 1e-8)
            probs = F.softmax(out["atom_logits"], -1)
            atom = torch.multinomial(probs, 1)
            log_pf = log_pf + torch.log(probs[atom] + 1e-8)
            atoms = torch.cat([atoms, atom.view(-1)])
            if atoms.size(0) > 1:
                ni = atoms.size(0) - 1
                edges = torch.cat([edges, torch.tensor([[ni, ni - 1], [ni - 1, ni]], device=self.device)], 1)
            log_pb = log_pb + self.backward_policy(out["graph_embedding"]).squeeze()
        return {"atoms": atoms, "edges": edges, "num_atoms": atoms.size(0)}, log_pf, log_pb

    @torch.no_grad()
    def sample(self, n: int = 10) -> list[dict[str, Any]]:
        self.forward_policy.eval()
        results = []
        for _ in range(n):
            t, _, _ = self._sample_trajectory()
            results.append({"atoms": t["atoms"].cpu().numpy(), "num_atoms": t["num_atoms"]})
        self.forward_policy.train()
        return results
