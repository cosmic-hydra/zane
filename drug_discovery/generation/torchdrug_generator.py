import torch
try:
    import torchdrug as td
    from torchdrug import data, models, tasks, utils
    TORCHDRUG_AVAILABLE = True
except ImportError:
    TORCHDRUG_AVAILABLE = False
    td = None

from typing import List, Optional
from ..data.rdkit_utils import smiles_to_sdf

class TorchDrugGenerator:
    def __init__(self, model_name: str = &quot;VAE&quot;, num_layers: int = 3):
        if not TORCHDRUG_AVAILABLE:
            raise ImportError(&quot;TorchDrug not available. Install torchdrug.&quot;)
        self.device = torch.device(&quot;cuda&quot; if torch.cuda.is_available() else &quot;cpu&quot;)
        self.dataset = td.data.MoleculeDataset(&quot;./data/zinc_standard_agent&quot;, rdkit=True)
        self.task = self._build_model(model_name, num_layers)

    def _build_model(self, model_name: str, num_layers: int):
        if model_name == &quot;VAE&quot;:
            model = models.MoleculeVAEGNN(
                td.data.MoleculeTensor(self.dataset[0]),
                encoder_layers=num_layers,
                decoder_layers=num_layers,
                latent_size=128,
                use_layer_norm=True
            )
            task = tasks.MoleculeGeneration(model, self.dataset, td.metrics.MoleculeMetrics(&quot;valid&quot;))
        # Add GNN etc.
        task = task.to(self.device)
        return task

    def generate(self, num: int = 1000, scaffold: Optional[str] = None) -> List[str]:
        torch.manual_seed(42)
        generated = self.task.generate(num_samples=num)
        smiles_list = [mol.smiles() for mol in generated]
        return smiles_list

    def save_to_sdf(self, smiles_list: List[str], path: str):
        smiles_to_sdf(smiles_list, path)