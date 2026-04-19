"""Molecular featurization and dataset abstractions."""

from __future__ import annotations

from dataclasses import dataclass
import random
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


@dataclass
class _GraphFallback:
    """Fallback graph container if torch-geometric is unavailable."""

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor | None = None


class MolecularFeaturizer:
    """Convert SMILES strings to graph/fingerprint/descriptors."""

    @staticmethod
    def _is_valid_smiles_like(smiles: str) -> bool:
        if not smiles or not isinstance(smiles, str):
            return False
        if "INVALID" in smiles.upper():
            return False
        try:
            from rdkit import Chem

            return Chem.MolFromSmiles(smiles) is not None
        except Exception:
            allowed = [c for c in smiles.strip() if c.isalnum() or c in "=#()[]+-@"]
            return len(allowed) > 0

    @staticmethod
    def _build_graph_from_rdkit(smiles: str):
        try:
            from rdkit import Chem
        except Exception:
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None

        node_features = []
        for atom in mol.GetAtoms():
            hyb = atom.GetHybridization()
            node_features.append(
                [
                    float(atom.GetAtomicNum()) / 100.0,
                    float(atom.GetTotalDegree()) / 4.0,
                    float(atom.GetFormalCharge() + 5) / 10.0,
                    float(atom.GetIsAromatic()),
                    float(hyb == Chem.rdchem.HybridizationType.SP),
                    float(hyb == Chem.rdchem.HybridizationType.SP2),
                    float(hyb == Chem.rdchem.HybridizationType.SP3),
                    float(atom.IsInRing()),
                ]
            )

        edges: list[list[int]] = []
        edge_features: list[list[float]] = []
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            btype = bond.GetBondType()
            edge_feat = [
                float(btype == Chem.rdchem.BondType.SINGLE),
                float(btype == Chem.rdchem.BondType.DOUBLE),
                float(btype in {Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC}),
            ]
            edges.append([begin, end])
            edges.append([end, begin])
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)

        if not edges:
            edges = [[0, 0]]
            edge_features = [[0.0, 0.0, 1.0]]

        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        try:
            from torch_geometric.data import Data

            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        except Exception:
            return _GraphFallback(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def smiles_to_graph(self, smiles: str):
        """Build a graph representation from SMILES."""
        if not self._is_valid_smiles_like(smiles):
            return None

        rdkit_graph = self._build_graph_from_rdkit(smiles)
        if rdkit_graph is not None:
            return rdkit_graph

        tokens = [c for c in smiles.strip() if c.isalnum() or c in "=#()[]+-@"]
        n = len(tokens)
        if n == 0:
            return None

        node_features = []
        for token in tokens:
            node_features.append(
                [
                    float(token.isalpha()),
                    float(token.isdigit()),
                    float(token.isupper()),
                    float(token.islower()),
                    float(token in "=#"),
                    float(token in "()[]"),
                    float(token in "+-"),
                    float((ord(token) % 31) / 30.0),
                ]
            )

        edges: list[list[int]] = []
        edge_features: list[list[float]] = []
        for i in range(n - 1):
            nxt = tokens[i + 1]
            edges.append([i, i + 1])
            edges.append([i + 1, i])
            edge_features.append([float(nxt == "="), float(nxt == "#"), 1.0])
            edge_features.append([float(nxt == "="), float(nxt == "#"), 1.0])

        if not edges:
            edges = [[0, 0]]
            edge_features = [[0.0, 0.0, 1.0]]

        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        try:
            from torch_geometric.data import Data

            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        except Exception:
            return _GraphFallback(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def smiles_to_fingerprint(self, smiles: str, n_bits: int = 2048) -> np.ndarray | None:
        """Return a deterministic fingerprint vector as numpy array."""
        if not self._is_valid_smiles_like(smiles) or n_bits <= 0:
            return None

        try:
            from rdkit import Chem
            from rdkit.Chem import rdFingerprintGenerator

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
                fp = fp_gen.GetFingerprint(mol)
                arr = np.zeros(n_bits, dtype=np.float32)
                for bit_idx in fp.GetOnBits():
                    arr[int(bit_idx)] = 1.0
                return arr
        except Exception:
            pass

        fp = np.zeros(n_bits, dtype=np.float32)
        for i, ch in enumerate(smiles):
            fp[(ord(ch) * 131 + i * 17) % n_bits] = 1.0
        return fp

    def compute_molecular_descriptors(self, smiles: str) -> dict[str, float] | None:
        """Compute descriptor set for heuristic modeling."""
        if not self._is_valid_smiles_like(smiles):
            return None

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return {
                    "mw": float(Descriptors.MolWt(mol)),
                    "logp": float(Descriptors.MolLogP(mol)),
                    "h_donors": float(Descriptors.NumHDonors(mol)),
                    "h_acceptors": float(Descriptors.NumHAcceptors(mol)),
                    "tpsa": float(Descriptors.TPSA(mol)),
                    "rot_bonds": float(Descriptors.NumRotatableBonds(mol)),
                }
        except Exception:
            pass

        s = smiles.strip()
        return {
            "length": float(len(s)),
            "num_rings_like": float(sum(s.count(d) for d in "123456789")),
            "num_branches": float(s.count("(")),
            "num_double_bonds": float(s.count("=")),
            "num_triple_bonds": float(s.count("#")),
            "hetero_ratio": float(sum(ch in "NOSPFClBrI" for ch in s) / max(len(s), 1)),
        }


class MolecularDataset(Dataset):
    """Dataset wrapper for molecular features and optional targets."""

    def __init__(
        self,
        data,
        smiles_col: str = "smiles",
        target_col: str | None = None,
        featurization: str = "graph",
        smiles_column: str | None = None,
        target_column: str | None = None,
    ):
        targets = None
        if not isinstance(smiles_col, str):
            targets = smiles_col
            smiles_col = "smiles"
            target_col = target_col or "target"

        self.smiles_col = smiles_column or smiles_col
        self.target_col = target_column or target_col
        self.featurization = featurization
        self.featurizer = MolecularFeaturizer()
        self._dict_graph_output = smiles_column is not None

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame({self.smiles_col: list(data)})
            if targets is not None:
                data[self.target_col or "target"] = list(targets)
        elif targets is not None:
            data = data.copy()
            data[self.target_col or "target"] = list(targets)

        self.data = data.reset_index(drop=True)

        self._valid_indices: list[int] = []
        for i in range(len(self.data)):
            smiles = str(self.data.loc[i, self.smiles_col])
            if self.featurization == "graph":
                feat = self.featurizer.smiles_to_graph(smiles)
            elif self.featurization == "descriptors":
                feat = self.featurizer.compute_molecular_descriptors(smiles)
            else:
                feat = self.featurizer.smiles_to_fingerprint(smiles)
            if feat is not None:
                self._valid_indices.append(i)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int):
        row_idx = self._valid_indices[idx]
        smiles = str(self.data.loc[row_idx, self.smiles_col])
        target = 0.0
        if self.target_col and self.target_col in self.data.columns:
            target = float(self.data.loc[row_idx, self.target_col])

        if self.featurization == "graph":
            graph = self.featurizer.smiles_to_graph(smiles)
            if graph is None:
                raise IndexError("Invalid graph feature at index")
            if self._dict_graph_output:
                feature = {
                    "atom_features": graph.x.detach().cpu().numpy(),
                    "adjacency": self._edge_index_to_adjacency(graph.edge_index, graph.x.shape[0]),
                }
                y = torch.tensor([target], dtype=torch.float32)
                return feature, y
            graph.y = torch.tensor([target], dtype=torch.float32)
            return graph, graph.y

        if self.featurization == "descriptors":
            desc = self.featurizer.compute_molecular_descriptors(smiles)
            if desc is None:
                raise IndexError("Invalid descriptor feature at index")
            x = torch.tensor(list(desc.values()), dtype=torch.float32)
            y = torch.tensor([target], dtype=torch.float32)
            return x, y

        fp = self.featurizer.smiles_to_fingerprint(smiles)
        if fp is None:
            raise IndexError("Invalid fingerprint feature at index")
        y = torch.tensor([target], dtype=torch.float32)
        return fp, y

    @staticmethod
    def _edge_index_to_adjacency(edge_index: torch.Tensor, n_nodes: int):
        adjacency = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
        if edge_index.numel() == 0:
            return adjacency.numpy()
        for src, dst in edge_index.t().tolist():
            if 0 <= src < n_nodes and 0 <= dst < n_nodes:
                adjacency[src, dst] = 1.0
        return adjacency.numpy()

    def get_feature_dim(self) -> int:
        """Return dimensionality of the current featurization output."""
        if len(self) == 0:
            return 0
        first_row_idx = self._valid_indices[0]
        smiles = str(self.data.loc[first_row_idx, self.smiles_col])

        if self.featurization == "graph":
            graph = self.featurizer.smiles_to_graph(smiles)
            if graph is not None and hasattr(graph, "x"):
                return int(graph.x.shape[1])
            return 0

        if self.featurization == "descriptors":
            desc = self.featurizer.compute_molecular_descriptors(smiles)
            return len(desc) if desc is not None else 0

        fp = self.featurizer.smiles_to_fingerprint(smiles)
        if fp is not None and hasattr(fp, "shape"):
            return int(fp.shape[0])
        return 0


def train_test_split_molecular(dataset: MolecularDataset, test_size: float = 0.2, seed: int | None = None):
    """Split dataset into train and test subsets with optional seeding."""
    n = len(dataset)
    if n == 0:
        return Subset(dataset, []), Subset(dataset, [])

    test_n = int(round(n * float(test_size)))
    if n > 1:
        test_n = max(1, test_n)
    test_n = min(test_n, n)
    train_n = max(0, n - test_n)

    idx = list(range(n))
    if seed is not None:
        rng = random.Random(int(seed))
        rng.shuffle(idx)

    return Subset(dataset, idx[:train_n]), Subset(dataset, idx[train_n:])


def murcko_scaffold_split_molecular(dataset: MolecularDataset, test_size: float = 0.2, seed: int | None = None):
    """Split by Bemis-Murcko scaffolds to reduce scaffold leakage.

    Falls back to random split when RDKit scaffold extraction is unavailable.
    """
    n = len(dataset)
    if n == 0:
        return Subset(dataset, []), Subset(dataset, [])

    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        return train_test_split_molecular(dataset, test_size=test_size, seed=seed)

    scaffold_to_indices: dict[str, list[int]] = {}
    for local_idx, row_idx in enumerate(dataset._valid_indices):
        smiles = str(dataset.data.loc[row_idx, dataset.smiles_col])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            scaffold = f"__invalid__{local_idx}"
        else:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or f"__acyclic__{local_idx}"
        scaffold_to_indices.setdefault(scaffold, []).append(local_idx)

    scaffold_groups = list(scaffold_to_indices.values())
    scaffold_groups.sort(key=len, reverse=True)
    if seed is not None:
        rng = random.Random(int(seed))
        rng.shuffle(scaffold_groups)
        scaffold_groups.sort(key=len, reverse=True)

    target_test_n = min(n, max(1 if n > 1 else 0, int(round(n * float(test_size)))))
    test_indices: list[int] = []
    train_indices: list[int] = []

    for group in scaffold_groups:
        if len(test_indices) + len(group) <= target_test_n:
            test_indices.extend(group)
        else:
            train_indices.extend(group)

    # Ensure non-empty train set for non-trivial datasets.
    if n > 1 and len(train_indices) == 0 and len(test_indices) > 0:
        train_indices.append(test_indices.pop())

    # Deterministic ordering for reproducibility.
    train_indices.sort()
    test_indices.sort()
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def murcko_scaffold_kfold_split_molecular(
    dataset: MolecularDataset,
    n_splits: int = 5,
    seed: int | None = None,
):
    """Create reproducible scaffold-based k-fold splits.

    Returns a list of ``(train_subset, test_subset)`` tuples.
    Falls back to seeded random k-fold when RDKit scaffold extraction is unavailable.
    """
    n = len(dataset)
    if n == 0:
        return [(Subset(dataset, []), Subset(dataset, []))]

    k = max(2, int(n_splits))
    k = min(k, n)

    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        idx = list(range(n))
        if seed is not None:
            rng = random.Random(int(seed))
            rng.shuffle(idx)
        fold_sizes = [n // k] * k
        for i in range(n % k):
            fold_sizes[i] += 1

        folds: list[list[int]] = []
        start = 0
        for size in fold_sizes:
            folds.append(sorted(idx[start : start + size]))
            start += size

        out = []
        for i in range(k):
            test_idx = folds[i]
            train_idx = sorted(j for f, fold in enumerate(folds) if f != i for j in fold)
            out.append((Subset(dataset, train_idx), Subset(dataset, test_idx)))
        return out

    scaffold_to_indices: dict[str, list[int]] = {}
    for local_idx, row_idx in enumerate(dataset._valid_indices):
        smiles = str(dataset.data.loc[row_idx, dataset.smiles_col])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            scaffold = f"__invalid__{local_idx}"
        else:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or f"__acyclic__{local_idx}"
        scaffold_to_indices.setdefault(scaffold, []).append(local_idx)

    groups = list(scaffold_to_indices.values())
    groups.sort(key=len, reverse=True)
    if seed is not None:
        rng = random.Random(int(seed))
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)

    fold_bins: list[list[int]] = [[] for _ in range(k)]
    fold_sizes = [0 for _ in range(k)]

    # Greedy balancing: place each scaffold group in the smallest current fold.
    for group in groups:
        tgt = min(range(k), key=lambda i: fold_sizes[i])
        fold_bins[tgt].extend(group)
        fold_sizes[tgt] += len(group)

    out = []
    for i in range(k):
        test_idx = sorted(fold_bins[i])
        train_idx = sorted(j for f, fold in enumerate(fold_bins) if f != i for j in fold)
        out.append((Subset(dataset, train_idx), Subset(dataset, test_idx)))
    return out
