"""
ADMET Prediction via Message Passing Neural Networks.

Implements a Message Passing Neural Network (MPNN) to evaluate Absorption,
Distribution, Metabolism, Excretion, and Toxicity (ADMET) profiles, flagging
off-target interactions that would cause clinical failure.

References:
    - Gilmer et al., "Neural Message Passing for Quantum Chemistry"
    - Stokes et al., "A Deep Learning Approach to Antibiotic Discovery"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from torch_geometric.data import Data
    from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = object
    MessagePassing = object
    logger.warning("PyTorch not available. Using heuristic ADMET prediction.")

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    logger.warning("RDKit not available. Using simplified molecular features.")


@dataclass
class ADMETProfile:
    """ADMET prediction profile.

    Attributes:
        absorption: Absorption score (0-1, higher is better).
        distribution: Distribution score.
        metabolism: Metabolism score.
        excretion: Excretion score.
        toxicity: Toxicity score (0-1, lower is better).
        off_target_risks: List of off-target interaction risks.
        overall_druglikeness: Overall drug-likeness score.
        clinical_risk_flags: List of clinical failure flags.
    """

    absorption: float = 0.5
    distribution: float = 0.5
    metabolism: float = 0.5
    excretion: float = 0.5
    toxicity: float = 0.5
    off_target_risks: list[str] = field(default_factory=list)
    overall_druglikeness: float = 0.5
    clinical_risk_flags: list[str] = field(default_factory=list)

    def is_druglike(self, threshold: float = 0.4) -> bool:
        """Check if molecule meets drug-likeness threshold."""
        return self.overall_druglikeness >= threshold and len(self.clinical_risk_flags) == 0 and self.toxicity < 0.6

    def as_dict(self) -> dict[str, Any]:
        return {
            "absorption": self.absorption,
            "distribution": self.distribution,
            "metabolism": self.metabolism,
            "excretion": self.excretion,
            "toxicity": self.toxicity,
            "off_target_risks": self.off_target_risks,
            "overall_druglikeness": self.overall_druglikeness,
            "clinical_risk_flags": self.clinical_risk_flags,
        }


@dataclass
class ADMETConfig:
    """Configuration for ADMET prediction.

    Attributes:
        hidden_dim: Hidden layer dimension.
        num_layers: Number of message passing layers.
        dropout: Dropout rate.
        use_3d_features: Use 3D molecular conformations.
        toxicity_threshold: Toxicity risk threshold.
        druglikeness_threshold: Drug-likeness threshold.
    """

    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.2
    use_3d_features: bool = False
    toxicity_threshold: float = 0.6
    druglikeness_threshold: float = 0.4


class MPNNMessagePassing(MessagePassing):
    """Message Passing layer for molecular graphs."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.message_lin = nn.Linear(in_channels, out_channels)
        self.update_lin = nn.Linear(in_channels + out_channels, out_channels)

    def message(self, x_j, edge_attr):
        """Compute message from neighbor."""
        return F.relu(self.message_lin(torch.cat([x_j, edge_attr], dim=-1)))

    def update(self, aggr_out, x):
        """Update node features."""
        return F.relu(self.update_lin(torch.cat([x, aggr_out], dim=-1)))


class ADMETPredictor(nn.Module if TORCH_AVAILABLE else object):
    """
    Message Passing Neural Network for ADMET prediction.

    Architecture:
    1. Molecular graph construction (atoms + bonds)
    2. Multiple message passing layers
    3. Global pooling
    4. Task-specific heads for each ADMET endpoint

    Example::

        predictor = ADMETPredictor(config=ADMETConfig(hidden_dim=128))
        profile = predictor.predict("CCO")  # ethanol
        print(f"Toxicity: {profile.toxicity}")
    """

    def __init__(self, config: ADMETConfig | None = None):
        """
        Initialize ADMET predictor.

        Args:
            config: ADMET model configuration.
        """
        super().__init__()

        self.config = config or ADMETConfig()

        if not TORCH_AVAILABLE:
            self.device = "cpu"
            logger.info("ADMETPredictor initialized in heuristic mode")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding for atom types
        self.atom_embedding = nn.Embedding(100, self.config.hidden_dim)

        # Edge encoding
        self.edge_lin = nn.Linear(4, self.config.hidden_dim)  # Bond type + stereo

        # Message passing layers
        self.mp_layers = nn.ModuleList(
            [MPNNMessagePassing(self.config.hidden_dim, self.config.hidden_dim) for _ in range(self.config.num_layers)]
        )

        # Global pooling
        self.global_lin = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)

        # ADMET prediction heads
        self.absorption_head = nn.Linear(self.config.hidden_dim, 1)
        self.distribution_head = nn.Linear(self.config.hidden_dim, 1)
        self.metabolism_head = nn.Linear(self.config.hidden_dim, 1)
        self.excretion_head = nn.Linear(self.config.hidden_dim, 1)
        self.toxicity_head = nn.Linear(self.config.hidden_dim, 1)

        self.dropout = nn.Dropout(self.config.dropout)

        self.to(self.device)
        logger.info(f"ADMETPredictor initialized on {self.device}")

    def forward(self, batch: Any) -> tuple[torch.Tensor, ...]:
        """Forward pass through MPNN."""
        if not TORCH_AVAILABLE:
            return self._forward_heuristic()

        x = self.atom_embedding(batch.x.squeeze().long())
        edge_index = batch.edge_index
        edge_attr = self.edge_lin(batch.edge_attr)

        # Message passing
        for mp_layer in self.mp_layers:
            x = mp_layer(x, edge_index, edge_attr)
            x = self.dropout(x)

        # Global pooling
        batch_idx = batch.batch
        x_mean = global_mean_pool(x, batch_idx)
        x_max = global_max_pool(x, batch_idx)
        h = F.relu(self.global_lin(torch.cat([x_mean, x_max], dim=-1)))
        h = self.dropout(h)

        # ADMET predictions
        absorption = torch.sigmoid(self.absorption_head(h))
        distribution = torch.sigmoid(self.distribution_head(h))
        metabolism = torch.sigmoid(self.metabolism_head(h))
        excretion = torch.sigmoid(self.excretion_head(h))
        toxicity = torch.sigmoid(self.toxicity_head(h))

        return absorption, distribution, metabolism, excretion, toxicity

    def predict(self, smiles: str) -> ADMETProfile:
        """
        Predict ADMET profile for a molecule.

        Args:
            smiles: SMILES string.

        Returns:
            ADMETProfile with predictions.
        """
        if not TORCH_AVAILABLE or not RDKIT_AVAILABLE:
            return self._predict_heuristic(smiles)

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ADMETProfile(clinical_risk_flags=["Invalid SMILES"])

            # Build graph
            graph = self._smiles_to_graph(mol)

            # Forward pass
            self.eval()
            with torch.no_grad():
                absorption, distribution, metabolism, excretion, toxicity = self(graph)

            # Extract predictions
            absorption = absorption.item()
            distribution = distribution.item()
            metabolism = metabolism.item()
            excretion = excretion.item()
            toxicity = toxicity.item()

            # Off-target risk assessment
            off_target_risks = self._assess_off_target_risks(mol)

            # Clinical risk flags
            risk_flags = []
            if toxicity > self.config.toxicity_threshold:
                risk_flags.append(f"High toxicity risk ({toxicity:.2f})")
            if absorption < 0.3:
                risk_flags.append(f"Poor absorption ({absorption:.2f})")
            if distribution < 0.3:
                risk_flags.append(f"Poor distribution ({distribution:.2f})")

            # Overall drug-likeness (weighted average)
            druglikeness = (
                0.2 * absorption
                + 0.15 * distribution
                + 0.15 * metabolism
                + 0.1 * excretion
                + 0.4 * (1 - toxicity)  # Lower toxicity is better
            )

            return ADMETProfile(
                absorption=absorption,
                distribution=distribution,
                metabolism=metabolism,
                excretion=excretion,
                toxicity=toxicity,
                off_target_risks=off_target_risks,
                overall_druglikeness=druglikeness,
                clinical_risk_flags=risk_flags,
            )

        except Exception as e:
            logger.error(f"ADMET prediction failed: {e}")
            return ADMETProfile(clinical_risk_flags=[f"Prediction error: {str(e)}"])

    def _smiles_to_graph(self, mol) -> Data:
        """Convert RDKit molecule to PyTorch Geometric Data."""
        if not TORCH_AVAILABLE:
            return None

        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = self._atom_features(atom)
            atom_features.append(features)

        x = torch.tensor(atom_features, dtype=torch.float32)

        # Edge features
        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_index.append([i, j])
            edge_index.append([j, i])

            bond_feat = self._bond_features(bond)
            edge_attr.append(bond_feat)
            edge_attr.append(bond_feat)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        # Create data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data = data.to(self.device)

        return data

    def _atom_features(self, atom) -> list[float]:
        """Extract atom-level features."""
        return [
            atom.GetAtomicNum() / 100.0,
            atom.GetDegree() / 6.0,
            atom.GetFormalCharge() / 2.0,
            atom.GetNumRadicalElectrons() / 4.0,
            atom.GetHybridization().real / 6.0,
            atom.GetIsAromatic() * 1.0,
            atom.IsInRing() * 1.0,
            atom.GetTotalNumHs() / 6.0,
            atom.GetChiralTag().real / 6.0,
        ]

    def _bond_features(self, bond) -> list[float]:
        """Extract bond-level features."""
        return [
            bond.GetBondTypeAsDouble() / 3.0,
            bond.GetIsAromatic() * 1.0,
            bond.GetIsConjugated() * 1.0,
            bond.GetStereo().real / 6.0,
        ]

    def _assess_off_target_risks(self, mol) -> list[str]:
        """Assess off-target interaction risks."""
        risks = []

        # Check for known toxicophores
        smarts_patterns = {
            "aniline": "[c]N",
            "nitro": "[N+](=O)[O-]",
            "epoxide": "C1OC1",
            "alkyl_halide": "C[F,Cl,Br,I]",
            "thiol": "S",
        }

        if RDKIT_AVAILABLE:
            for name, smarts in smarts_patterns.items():
                try:
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern and mol.HasSubstructMatch(pattern):
                        risks.append(f"Contains {name} motif")
                except Exception:
                    pass

        return risks

    def _forward_heuristic(self):
        """Fallback forward pass without PyTorch."""
        return (
            torch.tensor([0.5]),
            torch.tensor([0.5]),
            torch.tensor([0.5]),
            torch.tensor([0.5]),
            torch.tensor([0.5]),
        )

    def _predict_heuristic(self, smiles: str) -> ADMETProfile:
        """Heuristic ADMET prediction without ML model."""
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return ADMETProfile(clinical_risk_flags=["Invalid SMILES"])

                # Basic properties
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                num_h_donors = Descriptors.NumHDonors(mol)
                num_h_acceptors = Descriptors.NumHAcceptors(mol)
                num_rotatable = Descriptors.NumRotatableBonds(mol)
                aromatic_rings = Descriptors.RingCount(mol)

                # Absorption (based on Lipinski's rule)
                absorption = 1.0
                if mw > 500:
                    absorption -= 0.2
                if logp > 5:
                    absorption -= 0.1
                if num_h_donors > 5:
                    absorption -= 0.15
                if num_h_acceptors > 10:
                    absorption -= 0.15
                if tpsa > 140:
                    absorption -= 0.1
                absorption = max(0.0, min(1.0, absorption))

                # Distribution (TPSA-based)
                distribution = 1.0 - (tpsa / 200)
                distribution = max(0.0, min(1.0, distribution))

                # Metabolism (rotatable bonds, aromatic rings)
                metabolism = 1.0
                if num_rotatable > 10:
                    metabolism -= 0.2
                if aromatic_rings > 4:
                    metabolism -= 0.2
                metabolism = max(0.0, min(1.0, metabolism))

                # Excretion (simple heuristic)
                excretion = 1.0
                if mw > 600:
                    excretion -= 0.3
                excretion = max(0.0, min(1.0, excretion))

                # Toxicity (basic check)
                toxicity = 0.3  # Base rate
                risks = self._assess_off_target_risks(mol)
                if risks:
                    toxicity += 0.1 * len(risks)
                toxicity = max(0.0, min(1.0, toxicity))

                # Drug-likeness
                druglikeness = (
                    0.3 * absorption + 0.2 * distribution + 0.2 * metabolism + 0.1 * excretion + 0.2 * (1 - toxicity)
                )

                # Risk flags
                risk_flags = []
                if mw > 500:
                    risk_flags.append("MW > 500 (Poor absorption)")
                if logp > 5:
                    risk_flags.append("High logP (Low solubility)")
                if num_h_donors > 5 or num_h_acceptors > 10:
                    risk_flags.append("Lipinski violation")
                if toxicity > self.config.toxicity_threshold:
                    risk_flags.append("Potential toxicity")

                return ADMETProfile(
                    absorption=absorption,
                    distribution=distribution,
                    metabolism=metabolism,
                    excretion=excretion,
                    toxicity=toxicity,
                    off_target_risks=risks,
                    overall_druglikeness=druglikeness,
                    clinical_risk_flags=risk_flags,
                )

            except Exception as e:
                logger.warning(f"Heuristic prediction failed: {e}")

        # Default fallback
        return ADMETProfile(
            absorption=0.5,
            distribution=0.5,
            metabolism=0.5,
            excretion=0.5,
            toxicity=0.5,
            overall_druglikeness=0.5,
        )

    def predict_batch(self, smiles_list: list[str]) -> list[ADMETProfile]:
        """
        Predict ADMET profiles for a batch of molecules.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            List of ADMETProfile objects.
        """
        return [self.predict(smiles) for smiles in smiles_list]
