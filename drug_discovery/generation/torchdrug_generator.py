import torch
import logging
try:
    import torchdrug as td
    from torchdrug import data, models, tasks, utils
    TORCHDRUG_AVAILABLE = True
except ImportError:
    TORCHDRUG_AVAILABLE = False
    td = None

from typing import List, Optional
from ..data.rdkit_utils import smiles_to_sdf

logger = logging.getLogger(__name__)

class TorchDrugGenerator:
    def __init__(self, model_name: str = "VAE", num_layers: int = 3):
        self._fallback_mode = not TORCHDRUG_AVAILABLE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._fallback_mode:
            self.dataset = None
            self.task = None
        else:
            self.dataset = td.data.MoleculeDataset("./data/zinc_standard_agent", rdkit=True)
            self.task = self._build_model(model_name, num_layers)

    def _build_model(self, model_name: str, num_layers: int):
        if model_name == "VAE":
            model = models.MoleculeVAEGNN(
                td.data.MoleculeTensor(self.dataset[0]),
                encoder_layers=num_layers,
                decoder_layers=num_layers,
                latent_size=128,
                use_layer_norm=True
            )
            task = tasks.MoleculeGeneration(model, self.dataset, td.metrics.MoleculeMetrics("valid"))
        # Add GNN etc.
        task = task.to(self.device)
        return task

    def generate(self, num: int = 1000, scaffold: Optional[str] = None) -> List[str]:
        if self._fallback_mode or self.task is None:
            logger.warning("TorchDrug unavailable; returning placeholder molecules in fallback mode.")
            if scaffold and scaffold.strip():
                seeds = [scaffold, "CCO", "CCN", "CCC"]
            else:
                seeds = ["CCO", "CCN", "CCC", "c1ccccc1"]
            return [seeds[i % len(seeds)] for i in range(num)]
        torch.manual_seed(42)
        generated = self.task.generate(num_samples=num)
        smiles_list = [mol.smiles() for mol in generated]
        return smiles_list

    def save_to_sdf(self, smiles_list: List[str], path: str):
        smiles_to_sdf(smiles_list, path)
