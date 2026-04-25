import os

import numpy as np
import torch
import torch.nn as nn

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class SurrogateModel(nn.Module):
    """
    Mock PyTorch surrogate model to export to ONNX.
    In practice, this would be a trained GNN or MLP for ADMET and Binding Affinity.
    Includes Dropout for Monte Carlo Dropout (MC Dropout) during inference.
    """

    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Using functional dropout to ensure it can be triggered or we can enable train mode
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.2)

        # Outputs: [Binding Affinity (ΔG), ADMET toxicity gauge]
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, ligand_features, protein_embedding):
        # Concatenate cached protein embedding and new ligand features
        x = ligand_features + protein_embedding  # Simplified fusion
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)


class SubMillisecondSurrogateEngine:
    """
    Refinement 1: Sub-Millisecond Surrogate Inference (ONNX)
    Refinement 2: Epistemic Uncertainty Quantification (UQ) via MC Dropout
    """

    def __init__(self, model_path="surrogate_model.onnx"):
        self.model_path = model_path
        self.ort_session = None
        self.protein_embedding_cache = None

    def export_to_onnx(self):
        """Exports the PyTorch model to ONNX format."""
        model = SurrogateModel()
        # To keep dropout in ONNX for MC Dropout inference, we must export in training mode
        model.train()

        dummy_ligand = torch.randn(1, 1024)
        dummy_protein = torch.randn(1, 1024)

        torch.onnx.export(
            model,
            (dummy_ligand, dummy_protein),
            self.model_path,
            input_names=["ligand_features", "protein_embedding"],
            output_names=["predictions"],
            dynamic_axes={
                "ligand_features": {0: "batch_size"},
                "protein_embedding": {0: "batch_size"},
                "predictions": {0: "batch_size"},
            },
            training=torch.onnx.TrainingMode.TRAINING,  # Crucial for MC Dropout
            do_constant_folding=False,
        )

    def load_onnx_model(self):
        if ort is None:
            return

        if not os.path.exists(self.model_path):
            self.export_to_onnx()

        # Initialize ONNX Runtime session
        self.ort_session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])

    def cache_target_protein(self, protein_sequence: str):
        """Cache the target protein embeddings so only the ligand graph is re-computed."""
        # Mock protein embedding generation
        np.random.seed(hash(protein_sequence) % (2**32))
        self.protein_embedding_cache = np.random.randn(1, 1024).astype(np.float32)

    def predict_with_mc_dropout(self, ligand_smiles: str, num_samples: int = 30) -> dict:
        """
        Executes the forward pass using onnxruntime.
        Applies Monte Carlo Dropout to calculate mean and variance (epistemic uncertainty).
        """
        if self.ort_session is None:
            self.load_onnx_model()

        if self.protein_embedding_cache is None:
            self.cache_target_protein("MOCK_PROTEIN_SEQ")

        if not ligand_smiles:
            raise ValueError("Invalid SMILES")

        # Mock ligand feature extraction (e.g., Morgan fingerprints in reality)
        # Using a deterministic hash just for the mock
        np.random.seed(hash(ligand_smiles) % (2**32))
        ligand_features = np.random.randn(1, 1024).astype(np.float32)

        predictions = []
        for _ in range(num_samples):
            ort_inputs = {"ligand_features": ligand_features, "protein_embedding": self.protein_embedding_cache}
            ort_outs = self.ort_session.run(None, ort_inputs)
            predictions.append(ort_outs[0])

        predictions = np.concatenate(predictions, axis=0)  # Shape: (num_samples, 2)

        means = np.mean(predictions, axis=0)
        variances = np.var(predictions, axis=0)

        # means[0] -> Binding Affinity (ΔG)
        # means[1] -> ADMET toxicity

        return {
            "dg_mean": float(means[0]) - 8.0,  # Shift to realistic ΔG range
            "dg_variance": float(variances[0]),
            "admet_mean": float(means[1]),
            "admet_variance": float(variances[1]),
        }
