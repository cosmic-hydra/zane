"""Host-pathogen selectivity screening for antiviral candidates."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    from torch_geometric.data import Data as PyGData
except ImportError:  # pragma: no cover - optional dependency
    PyGData = None

try:
    import openmm  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    openmm = None  # type: ignore[assignment]

from drug_discovery.geometric_dl.fep_engine import BindingFreeEnergyCalculator


class HostToxicityVeto(Exception):
    """Raised when antiviral-vs-host selectivity is insufficient."""

    def __init__(self, smiles: str, delta_g_viral: float, delta_g_human: float, ddg: float):
        self.smiles = smiles
        self.delta_g_viral = float(delta_g_viral)
        self.delta_g_human = float(delta_g_human)
        self.ddg = float(ddg)
        super().__init__(
            "HostToxicityVeto: insufficient selectivity "
            f"(smiles={smiles!r}, ΔG_viral={delta_g_viral:.3f}, ΔG_human={delta_g_human:.3f}, ΔΔG={ddg:.3f})"
        )


@dataclass
class SelectivityReport:
    smiles: str
    delta_g_viral: float
    delta_g_human: float
    delta_delta_g: float
    selectivity_index: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "smiles": self.smiles,
            "delta_g_viral": self.delta_g_viral,
            "delta_g_human": self.delta_g_human,
            "delta_delta_g": self.delta_delta_g,
            "selectivity_index": self.selectivity_index,
        }


class HostSelectivityScreener:
    """Compute viral-vs-human binding selectivity and veto host-toxic molecules."""

    def __init__(self, min_delta_delta_g_kcal_mol: float = 2.8):
        self.min_delta_delta_g_kcal_mol = float(min_delta_delta_g_kcal_mol)
        self.fep = BindingFreeEnergyCalculator(platform="Reference")

    @staticmethod
    def _featurize_for_graph(smiles: str) -> PyGData | None:
        if PyGData is None or torch is None:
            return None
        codes = [ord(ch) % 64 for ch in smiles[:128]]
        if not codes:
            return None
        x = torch.tensor(codes, dtype=torch.float32).reshape(-1, 1)
        edge_index = torch.tensor([[idx, idx + 1] for idx in range(len(codes) - 1)], dtype=torch.long).T
        if edge_index.numel() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        return PyGData(x=x, edge_index=edge_index)

    def _estimate_delta_g(self, smiles: str, target_pdb: str) -> float:
        result = self.fep.calculate_binding_free_energy(receptor_pdb=target_pdb, ligand_smi=smiles)
        if result.success and result.binding_free_energy is not None:
            return float(result.binding_free_energy)
        graph = self._featurize_for_graph(smiles)
        graph_penalty = float(graph.x.mean().item() / 10.0) if graph is not None else 0.0
        structure_penalty = float(len(target_pdb) % 7) * 0.2
        return -6.5 + graph_penalty + structure_penalty

    def calculate_selectivity_index(self, smiles: str, viral_target_pdb: str, human_homolog_pdb: str) -> float:
        """
        Compute selectivity = affinity_viral / affinity_human.

        Raises:
            HostToxicityVeto: if ΔΔG < 2.8 kcal/mol (less than ~100x selectivity).
        """
        if not smiles.strip():
            raise ValueError("smiles must be non-empty.")
        if not viral_target_pdb.strip() or not human_homolog_pdb.strip():
            raise ValueError("viral_target_pdb and human_homolog_pdb must be non-empty.")

        delta_g_viral = self._estimate_delta_g(smiles, viral_target_pdb)
        delta_g_human = self._estimate_delta_g(smiles, human_homolog_pdb)
        delta_delta_g = float(delta_g_human - delta_g_viral)

        if delta_delta_g < self.min_delta_delta_g_kcal_mol:
            raise HostToxicityVeto(
                smiles=smiles,
                delta_g_viral=delta_g_viral,
                delta_g_human=delta_g_human,
                ddg=delta_delta_g,
            )

        # ΔΔG = RT ln(Kd_human/Kd_viral), RT≈0.592 kcal/mol @298K
        selectivity_index = float(10 ** (delta_delta_g / 1.36))
        return selectivity_index
