"""
Molecular Dataset - 2D/3D Featurization for Training

Provides PyTorch Dataset classes for molecular data with support for
various featurization strategies.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

logger = logging.getLogger(__name__)


class MolecularDataset(Dataset):
    """PyTorch Dataset for molecular data with flexible featurization."""

    def __init__(
        self,
        data: pd.DataFrame,
        smiles_column: str = "smiles",
        target_column: Optional[str] = None,
        featurization: str = "fingerprint",
        use_3d: bool = False,
    ):
        """
        Initialize molecular dataset.

        Args:
            data: DataFrame with molecular data
            smiles_column: Name of SMILES column
            target_column: Name of target/label column
            featurization: Type of featurization ('fingerprint', 'graph', 'descriptors')
            use_3d: Whether to generate 3D conformers
        """
        self.data = data
        self.smiles_column = smiles_column
        self.target_column = target_column
        self.featurization = featurization
        self.use_3d = use_3d

        # Precompute features
        self.features = []
        self.targets = []
        self._preprocess()

    def _preprocess(self) -> None:
        """Precompute all molecular features."""
        for idx, row in self.data.iterrows():
            smiles = row[self.smiles_column]
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                continue

            # Generate features
            if self.featurization == "fingerprint":
                feature = self._generate_fingerprint(mol)
            elif self.featurization == "graph":
                feature = self._generate_graph_features(mol)
            elif self.featurization == "descriptors":
                feature = self._generate_descriptors(mol)
            else:
                feature = self._generate_fingerprint(mol)

            if feature is not None:
                self.features.append(feature)

                # Extract target if available
                if self.target_column and self.target_column in row:
                    self.targets.append(float(row[self.target_column]))
                else:
                    self.targets.append(0.0)

        logger.info(f"Preprocessed {len(self.features)} molecules")

    def _generate_fingerprint(
        self,
        mol: Chem.Mol,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> Optional[np.ndarray]:
        """Generate Morgan fingerprint."""
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fp)
        except Exception as e:
            logger.warning(f"Fingerprint generation failed: {e}")
            return None

    def _generate_graph_features(self, mol: Chem.Mol) -> Optional[Dict[str, Any]]:
        """Generate graph-based features (nodes + edges)."""
        try:
            # Atom features
            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    atom.GetHybridization().real,
                    int(atom.GetIsAromatic()),
                ]
                atom_features.append(features)

            # Bond features (adjacency)
            num_atoms = mol.GetNumAtoms()
            adjacency = np.zeros((num_atoms, num_atoms))

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                adjacency[i, j] = 1
                adjacency[j, i] = 1

            return {
                "atom_features": np.array(atom_features),
                "adjacency": adjacency,
            }
        except Exception as e:
            logger.warning(f"Graph feature generation failed: {e}")
            return None

    def _generate_descriptors(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        """Generate RDKit molecular descriptors."""
        try:
            descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.FractionCSSP3(mol),
                Descriptors.NumSaturatedRings(mol),
            ]
            return np.array(descriptors, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Descriptor generation failed: {e}")
            return None

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        feature = self.features[idx]
        target = self.targets[idx]

        # Convert to tensors
        if isinstance(feature, dict):
            # Graph features
            feature_tensor = {
                "atom_features": torch.FloatTensor(feature["atom_features"]),
                "adjacency": torch.FloatTensor(feature["adjacency"]),
            }
        else:
            feature_tensor = torch.FloatTensor(feature)

        target_tensor = torch.FloatTensor([target])

        return feature_tensor, target_tensor

    def get_feature_dim(self) -> int:
        """Get dimensionality of features."""
        if len(self.features) == 0:
            return 0

        feature = self.features[0]
        if isinstance(feature, dict):
            return feature["atom_features"].shape[1]
        else:
            return len(feature)
