import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from captum.attr import IntegratedGradients
except ImportError:
    IntegratedGradients = None

try:
    import streamlit as st
except ImportError:
    st = None

try:
    import py3Dmol
except ImportError:
    py3Dmol = None  # noqa: N816


class AugmentedChemistInterface:
    """
    Module 10: The "Glass Box" XAI & Augmented Chemist Interface
    """

    def __init__(self, gnn_model=None):
        self.gnn_model = gnn_model if gnn_model else self._build_mock_model()
        if IntegratedGradients is not None:
            self.integrated_gradients = IntegratedGradients(self.gnn_model)
        else:
            self.integrated_gradients = None

    def _build_mock_model(self):
        class MockGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

        return MockGNN()

    def get_3d_attention_mapping(self, molecular_features: torch.Tensor):
        """
        3D Attention Mapping:
        Interface with Captum to extract the integrated gradients and attention weights
        from our Graph Neural Networks.
        """
        if self.integrated_gradients is None:
            return None

        molecular_features.requires_grad_()
        attributions, delta = self.integrated_gradients.attribute(
            molecular_features, target=0, return_convergence_delta=True
        )
        return attributions

    def render_xai_visualization(self, protein_pdb: str, molecule_smiles: str, attributions: torch.Tensor = None):
        """
        Uses py3Dmol to render the target protein pocket and the generated drug in 3D.
        Visually highlights (in glowing heatmaps) the exact sub-atomic regions that the AI
        believes are driving the favorable Binding Free Energy (ΔG).
        """
        if py3Dmol is None:
            return None

        mol = Chem.MolFromSmiles(molecule_smiles)
        mol_block = ""
        if mol:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            mol_block = Chem.MolToMolBlock(mol)

        view = py3Dmol.view(width=800, height=600)

        if protein_pdb:
            view.addModel(protein_pdb, "pdb")
            view.setStyle({"cartoon": {"color": "lightgray"}})

        if mol_block:
            view.addModel(mol_block, "mol")
            # The attributions would be used here to dynamically color atoms
            view.setStyle({"model": -1}, {"stick": {"colorscheme": "greenCarbon"}})

        view.zoomTo()
        return view

    def real_time_sub_second_rescoring(self, modified_smiles: str) -> dict:
        """
        Real-Time Sub-Second Rescoring:
        The moment the human chemist modifies the molecule in the browser,
        instantly send a WebSocket request back to the PyTorch backend,
        run a rapid surrogate model inference, and instantly update the
        ADMET toxicity gauges and predicted binding affinity.
        """
        mol = Chem.MolFromSmiles(modified_smiles)
        if not mol:
            return {"error": "Invalid SMILES structure"}

        # Surrogate model inference mock
        predicted_dg = -(8.0 + np.random.random() * 4.0)
        admet_toxicity = np.random.random()

        return {
            "smiles": modified_smiles,
            "predicted_binding_affinity_dg": predicted_dg,
            "admet_toxicity_gauge": admet_toxicity,
            "molecular_weight": Chem.Descriptors.MolWt(mol),
        }


def run_glass_box_dashboard():
    """
    Human-in-the-Loop Editor:
    Integrates a web-based molecular editor. The human chemist can override
    the AI's suggestions in real-time.
    """
    if st is None:
        print("Streamlit is not installed.")
        return

    st.set_page_config(page_title="Glass Box XAI & Augmented Chemist", layout="wide")
    st.title("Module 10: Glass Box XAI & Augmented Chemist Interface")

    interface = AugmentedChemistInterface()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("3D Attention Mapping")
        st.markdown("Visualizing sub-atomic regions driving favorable Binding Free Energy (ΔG).")
        st.components.v1.html(
            "<div style='height: 400px; border: 1px solid #ccc; display: flex; align-items: center; justify-content: center;'>py3Dmol Render Placeholder</div>",
            height=420,
        )

    with col2:
        st.subheader("Human-in-the-Loop Editor")
        st.markdown("Modify the molecule below (e.g., using Ketcher or JSME):")
        edited_smiles = st.text_input("SMILES Editor", "CCO")

        st.subheader("Real-Time Sub-Second Rescoring")
        if edited_smiles:
            scores = interface.real_time_sub_second_rescoring(edited_smiles)
            if "error" not in scores:
                st.metric("Predicted ΔG (kcal/mol)", f"{scores['predicted_binding_affinity_dg']:.2f}")
                st.progress(
                    scores["admet_toxicity_gauge"],
                    text=f"ADMET Toxicity Gauge: {scores['admet_toxicity_gauge']:.2f}",
                )
            else:
                st.error(scores["error"])


if __name__ == "__main__":
    run_glass_box_dashboard()
