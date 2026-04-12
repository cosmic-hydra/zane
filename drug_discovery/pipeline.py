"""
Main Drug Discovery Pipeline
Orchestrates the entire AI drug discovery process
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

from .data import (
    DataCollector,
    MolecularDataset,
    MolecularFeaturizer,
    murcko_scaffold_split_molecular,
    train_test_split_molecular,
)
from .evaluation import ADMETPredictor, ModelEvaluator, PropertyPredictor
from .evaluation.torchdrug_scorer import TorchDrugScorer
from .models import EnsembleModel, MolecularGNN, MolecularTransformer
from .physics.diffdock_adapter import DiffDockAdapter
from .physics.openmm_adapter import OpenMMAdapter
from .physics.protein_structure import ProteinStructurePredictor
from .synthesis.pistachio_datasets import PistachioDatasets
from .synthesis.reaction_prediction import ReactionPredictor
from .training import SelfLearningTrainer


class DrugDiscoveryPipeline:
    """
    Complete AI-powered drug discovery pipeline
    """

    def __init__(
        self,
        model_type: str = "gnn",  # 'gnn', 'transformer', or 'ensemble'
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: str = "./data/cache",
        checkpoint_dir: str = "./checkpoints",
    ):
        """
        Args:
            model_type: Type of model to use
            device: Device for training/inference
            cache_dir: Directory for cached data
            checkpoint_dir: Directory for model checkpoints
        """
        self.model_type = model_type
        self.device = device
        self.cache_dir = cache_dir
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize components
        self.data_collector = DataCollector(cache_dir=cache_dir)
        self.featurizer = MolecularFeaturizer()
        self.admet_predictor = ADMETPredictor()
        self.evaluator = ModelEvaluator()

        # Models and trainers (initialized during training)
        self.model = None
        self.trainer = None
        self.property_predictor = None

        # Elite external-tool adapters (lazy-initialized on first use)
        self._torchdrug_scorer: TorchDrugScorer | None = None
        self._reaction_predictor: ReactionPredictor | None = None
        self._diffdock_adapter: DiffDockAdapter | None = None
        self._protein_predictor: ProteinStructurePredictor | None = None
        self._openmm_adapter: OpenMMAdapter | None = None
        self._pistachio_datasets: PistachioDatasets | None = None

        print("Drug Discovery Pipeline initialized")
        print(f"Model type: {model_type}")
        print(f"Device: {device}")

    def collect_data(
        self,
        sources: list[str] = ["pubchem", "chembl", "approved_drugs"],
        limit_per_source: int = 1000,
        drugbank_file: str | None = None,
    ) -> pd.DataFrame:
        """
        Collect molecular data from multiple sources

        Args:
            sources: List of data sources
            limit_per_source: Maximum samples per source

        Returns:
            Combined DataFrame
        """
        print("\n=== Data Collection Phase ===")

        datasets = []

        if "pubchem" in sources:
            print("\nCollecting from PubChem...")
            df = self.data_collector.collect_from_pubchem(limit=limit_per_source)
            if not df.empty:
                datasets.append(df)

        if "chembl" in sources:
            print("\nCollecting from ChEMBL...")
            df = self.data_collector.collect_from_chembl(limit=limit_per_source)
            if not df.empty:
                datasets.append(df)

        if "approved_drugs" in sources:
            print("\nCollecting approved drugs...")
            df = self.data_collector.collect_approved_drugs()
            if not df.empty:
                datasets.append(df)

        if "drugbank" in sources:
            print("\nCollecting from DrugBank...")
            df = self.data_collector.collect_from_drugbank(file_path=drugbank_file, limit=limit_per_source)
            if not df.empty:
                datasets.append(df)

        # Merge datasets
        if datasets:
            merged_data = self.data_collector.merge_datasets(datasets)
            quality = self.data_collector.generate_data_quality_report(merged_data)
            print(f"\nTotal unique molecules collected: {len(merged_data)}")
            print(
                "Data quality: "
                f"valid={quality['valid_smiles_rows']}/{quality['total_rows']} "
                f"({quality['validity_ratio']:.2%}), "
                f"duplicates_removed={quality['duplicate_smiles_rows']}"
            )
            return merged_data
        else:
            print("No data collected!")
            return pd.DataFrame()

    def prepare_datasets(
        self,
        data: pd.DataFrame,
        smiles_col: str = "smiles",
        target_col: str | None = None,
        test_size: float = 0.2,
        batch_size: int = 32,
        seed: int | None = None,
        split_strategy: str = "random",
        num_workers: int | None = None,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Prepare train and test dataloaders

        Args:
            data: DataFrame with molecular data
            smiles_col: Column name for SMILES
            target_col: Column name for target variable
            test_size: Fraction for test set
            batch_size: Batch size

        Returns:
            Train and test dataloaders
        """
        print("\n=== Data Preparation Phase ===")

        # Determine featurization based on model type
        if self.model_type == "gnn":
            featurization = "graph"
        else:
            featurization = "fingerprint"

        # Create dataset
        dataset = MolecularDataset(data=data, smiles_col=smiles_col, target_col=target_col, featurization=featurization)

        # Split dataset
        if split_strategy == "scaffold":
            train_dataset, test_dataset = murcko_scaffold_split_molecular(dataset, test_size=test_size, seed=seed)
        else:
            train_dataset, test_dataset = train_test_split_molecular(dataset, test_size=test_size, seed=seed)

        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        resolved_workers = num_workers
        if resolved_workers is None:
            cpu = os.cpu_count() or 2
            resolved_workers = max(0, min(6, cpu // 2))
        resolved_workers = int(max(0, resolved_workers))
        use_pin_memory = self.device.startswith("cuda")

        loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": resolved_workers,
            "pin_memory": use_pin_memory,
        }
        if resolved_workers > 0:
            loader_kwargs["persistent_workers"] = True

        # Create dataloaders
        if featurization == "graph":
            train_loader = GeometricDataLoader(cast(Any, train_dataset), shuffle=True, **loader_kwargs)
            test_loader = GeometricDataLoader(cast(Any, test_dataset), shuffle=False, **loader_kwargs)
        else:
            train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
            test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

        return train_loader, test_loader

    def build_model(self, **model_kwargs):
        """
        Build the model based on model_type

        Args:
            **model_kwargs: Model-specific arguments

        Returns:
            Built model
        """
        print("\n=== Model Building Phase ===")

        if self.model_type == "gnn":
            self.model = MolecularGNN(**model_kwargs)
            print("Built Graph Neural Network model")

        elif self.model_type == "transformer":
            self.model = MolecularTransformer(**model_kwargs)
            print("Built Transformer model")

        elif self.model_type == "ensemble":
            # Create ensemble of GNN and Transformer
            gnn = MolecularGNN()
            transformer = MolecularTransformer()
            self.model = EnsembleModel([gnn, transformer])
            print("Built Ensemble model (GNN + Transformer)")

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return self.model

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        **trainer_kwargs,
    ) -> dict[str, Any]:
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            learning_rate: Learning rate
            **trainer_kwargs: Additional trainer arguments

        Returns:
            Training history
        """
        print("\n=== Training Phase ===")

        # Build model if not already built
        if self.model is None:
            self.build_model()
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        # Initialize trainer
        self.trainer = SelfLearningTrainer(
            model=self.model,
            device=self.device,
            learning_rate=learning_rate,
            save_dir=self.checkpoint_dir,
            **trainer_kwargs,
        )

        # Train
        is_graph = self.model_type == "gnn"
        history = self.trainer.train(
            train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs, is_graph=is_graph
        )

        # Initialize property predictor
        self.property_predictor = PropertyPredictor(model=self.model, device=self.device)

        print("\n✓ Training complete!")
        return history

    def predict_properties(self, smiles: str, include_admet: bool = True) -> dict:
        """
        Predict properties for a molecule

        Args:
            smiles: SMILES string
            include_admet: Whether to include ADMET predictions

        Returns:
            Dictionary of predicted properties
        """
        if self.property_predictor is None:
            raise RuntimeError("Model not trained yet. Call train() first.")

        results: dict[str, Any] = {"smiles": smiles}

        # Model predictions
        if self.model_type == "gnn":
            graph_data = self.featurizer.smiles_to_graph(smiles)
            if graph_data is not None:
                graph_data = graph_data.to(self.device)
                with torch.no_grad():
                    if self.model is None:
                        raise RuntimeError("Model is not initialized.")
                    prediction = self.model(graph_data).cpu().numpy()
                results["predicted_property"] = float(prediction[0])
        else:
            fingerprint = self.featurizer.smiles_to_fingerprint(smiles)
            if fingerprint is not None:
                prediction = self.property_predictor.predict_from_smiles(smiles, self.featurizer)
                results["predicted_property"] = prediction

        # ADMET predictions
        if include_admet:
            lipinski = self.admet_predictor.check_lipinski_rule(smiles)
            qed = self.admet_predictor.calculate_qed(smiles)
            sa_score = self.admet_predictor.calculate_synthetic_accessibility(smiles)
            toxicity = self.admet_predictor.predict_toxicity_flags(smiles)

            results["lipinski_pass"] = lipinski["passes"] if lipinski else None
            results["lipinski_violations"] = lipinski["num_violations"] if lipinski else None
            results["qed_score"] = qed
            results["synthetic_accessibility"] = sa_score
            results["toxicity_flags"] = toxicity

        return results

    def generate_candidates(
        self, target_protein: str | None = None, num_candidates: int = 10, filter_criteria: dict | None = None
    ) -> pd.DataFrame:
        """
        Generate drug candidate molecules

        Args:
            target_protein: Target protein name
            num_candidates: Number of candidates to generate
            filter_criteria: Filtering criteria (e.g., Lipinski rules)

        Returns:
            DataFrame of candidate molecules
        """
        print("\n=== Generating Drug Candidates ===")
        print(f"Target: {target_protein or 'General'}")

        # For demonstration, we'll use molecules from the database
        # In a real implementation, this would use generative models
        print("Note: Using existing molecules. Generative models not yet implemented.")

        # Load some molecules
        cache_file = os.path.join(self.cache_dir, "approved_drugs.csv")
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            candidates = df.head(num_candidates).copy()

            # Add predictions for each candidate
            predictions = []
            for smiles in candidates["smiles"]:
                try:
                    pred = self.predict_properties(smiles, include_admet=True)
                    predictions.append(pred)
                except Exception:
                    predictions.append({})

            # Merge predictions
            for i, pred in enumerate(predictions):
                for key, value in pred.items():
                    if key != "smiles":
                        candidates.loc[i, key] = value

            return candidates
        else:
            print("No cached data available. Run collect_data() first.")
            return pd.DataFrame()

    def evaluate(self, test_loader: DataLoader, is_graph: bool | None = None) -> dict[str, float]:
        """
        Evaluate the model

        Args:
            test_loader: Test data loader
            is_graph: Whether data is graph-structured

        Returns:
            Evaluation metrics
        """
        if is_graph is None:
            is_graph = self.model_type == "gnn"
        is_graph = bool(is_graph)

        print("\n=== Evaluation Phase ===")

        # Get predictions
        if self.trainer is None:
            raise RuntimeError("Trainer is not initialized. Call train() first.")
        y_pred = self.trainer.predict(test_loader, is_graph=is_graph)

        # Get true values
        y_true = []
        for batch in test_loader:
            if is_graph:
                y_true.append(batch.y.cpu().numpy())
            else:
                if isinstance(batch, (list, tuple)):
                    _, targets = batch
                    y_true.append(targets.cpu().numpy())

        y_true = np.vstack(y_true)

        # Filter out missing values
        mask = y_true != -1
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        # Evaluate
        metrics = self.evaluator.evaluate_regression(y_true_filtered, y_pred_filtered)
        self.evaluator.print_metrics()

        return metrics

    def save(self, filepath: str):
        """Save the pipeline"""
        if self.model is not None:
            save_path = os.path.abspath(filepath)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "model_type": self.model_type,
                    "checkpoint_version": 2,
                },
                save_path,
            )
            print(f"Pipeline saved to {save_path}")

    def load(self, filepath: str):
        """Load the pipeline"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model_type = checkpoint["model_type"]
        self.build_model()
        if self.model is None:
            raise RuntimeError("Failed to build model during load().")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.property_predictor = PropertyPredictor(self.model, self.device)
        print(f"Pipeline loaded from {filepath}")

    # ------------------------------------------------------------------
    # Integrated elite-pipeline methods (TorchDrug → MolecularTransformer
    #   → DiffDock → OpenMM)
    # ------------------------------------------------------------------

    @property
    def torchdrug_scorer(self) -> TorchDrugScorer:
        if self._torchdrug_scorer is None:
            self._torchdrug_scorer = TorchDrugScorer(device=self.device)
        return self._torchdrug_scorer

    @property
    def reaction_predictor(self) -> ReactionPredictor:
        if self._reaction_predictor is None:
            self._reaction_predictor = ReactionPredictor()
        return self._reaction_predictor

    @property
    def diffdock_adapter(self) -> DiffDockAdapter:
        if self._diffdock_adapter is None:
            self._diffdock_adapter = DiffDockAdapter()
        return self._diffdock_adapter

    @property
    def protein_predictor(self) -> ProteinStructurePredictor:
        if self._protein_predictor is None:
            self._protein_predictor = ProteinStructurePredictor()
        return self._protein_predictor

    @property
    def openmm_adapter(self) -> OpenMMAdapter:
        if self._openmm_adapter is None:
            self._openmm_adapter = OpenMMAdapter(temperature_K=300.0)
        return self._openmm_adapter

    @property
    def pistachio_datasets(self) -> PistachioDatasets:
        if self._pistachio_datasets is None:
            self._pistachio_datasets = PistachioDatasets()
        return self._pistachio_datasets

    def run_integrated_pipeline(
        self,
        smiles_list: list[str],
        target_protein_sequence: str | None = None,
        target_protein_pdb: str | None = None,
        target_protein_id: str = "target",
        md_steps: int = 10_000,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Run the full elite drug-discovery pipeline:

        1. **TorchDrug** – score candidates by predicted properties
           (QED, solubility, bioactivity, toxicity)
        2. **MolecularTransformer** – validate synthetic accessibility via
           reaction-outcome prediction
        3. **OpenFold** – predict target protein 3D structure (optional, if a
           sequence is provided and no PDB is available)
        4. **DiffDock** – predict protein–ligand binding poses
        5. **OpenMM** – simulate MD stability of top candidates

        Each step degrades gracefully when the corresponding external
        submodule is not installed.

        Args:
            smiles_list: List of candidate molecule SMILES strings.
            target_protein_sequence: Amino-acid sequence for structure
                prediction with OpenFold (optional).
            target_protein_pdb: Path to a PDB file for docking (optional).
            target_protein_id: Human-readable identifier for the target.
            md_steps: Number of MD steps for OpenMM simulation.
            top_n: Number of top candidates to carry through to docking and MD.

        Returns:
            Dictionary with per-step results and a ranked summary DataFrame.
        """
        print("\n=== Integrated Elite Pipeline ===")
        results: dict[str, Any] = {
            "input_count": len(smiles_list),
            "target_protein_id": target_protein_id,
        }

        # ── Step 1: TorchDrug property scoring ────────────────────────
        print("\n[1/5] TorchDrug – property scoring …")
        property_scores = self.torchdrug_scorer.score_batch(smiles_list)
        ranked_smiles = self.torchdrug_scorer.rank(smiles_list)
        results["property_scores"] = [s.as_dict() for s in property_scores]
        top_smiles = [smi for smi, _ in ranked_smiles[:top_n]]
        print(f"  ✓ Scored {len(smiles_list)} molecules; top {top_n} selected.")

        # ── Step 2: MolecularTransformer reaction validation ──────────
        print("\n[2/5] MolecularTransformer – reaction validation …")
        reaction_results = self.reaction_predictor.predict_batch(top_smiles)
        results["reaction_predictions"] = [r.as_dict() for r in reaction_results]
        print(f"  ✓ Predicted reaction outcomes for {len(top_smiles)} molecules.")

        # ── Step 3: OpenFold protein structure (if sequence given) ────
        protein_structure = None
        if target_protein_sequence and target_protein_pdb is None:
            print("\n[3/5] OpenFold – protein structure prediction …")
            protein_structure = self.protein_predictor.predict(
                sequence=target_protein_sequence,
                protein_id=target_protein_id,
            )
            results["protein_structure"] = protein_structure.as_dict()
            if protein_structure.pdb_string:
                target_protein_pdb = protein_structure.pdb_string
            print(f"  ✓ Structure prediction: {protein_structure.backend}.")
        else:
            print("\n[3/5] OpenFold – skipped (PDB provided or no sequence given).")
            results["protein_structure"] = None

        # ── Step 4: DiffDock binding prediction ──────────────────────
        print("\n[4/5] DiffDock – binding pose prediction …")
        docking_poses_by_smiles: dict[str, list[dict[str, Any]]] = {}
        for smi in top_smiles:
            poses = self.diffdock_adapter.dock(
                ligand_smiles=smi,
                protein_id=target_protein_id,
                protein_pdb=target_protein_pdb,
            )
            docking_poses_by_smiles[smi] = [p.as_dict() for p in poses]
        results["docking_poses"] = docking_poses_by_smiles
        print(f"  ✓ Docking complete for {len(top_smiles)} molecules.")

        # ── Step 5: OpenMM MD stability ───────────────────────────────
        print("\n[5/5] OpenMM – molecular dynamics stability …")
        md_results = self.openmm_adapter.batch_simulate(top_smiles, steps=md_steps)
        results["md_simulations"] = [r.as_dict() for r in md_results]
        print(f"  ✓ MD simulation complete for {len(top_smiles)} molecules.")

        # ── Summary DataFrame ─────────────────────────────────────────
        rows = []
        for smi, prop_score in ranked_smiles[:top_n]:
            md = next((r for r in md_results if r.smiles == smi), None)
            docking = docking_poses_by_smiles.get(smi, [])
            top_confidence = max((d.get("confidence", 0.0) for d in docking), default=None) if docking else None
            rows.append({
                "smiles": smi,
                "composite_property_score": prop_score,
                "docking_top_confidence": top_confidence,
                "md_stable": md.stable if md else None,
                "md_stability_score": md.stability_score if md else None,
            })
        results["summary"] = pd.DataFrame(rows)

        print("\n=== Pipeline Complete ===")
        return results

    def run_boltzgen_design(
        self,
        design_spec: str | Path,
        output_dir: str | Path | None = None,
        protocol: str = "protein-anything",
        num_designs: int = 50,
        budget: int = 10,
        steps: Sequence[str] | None = None,
        devices: int | None = None,
        reuse: bool = True,
        cache_dir: str | Path | None = None,
        top_k: int = 5,
        score_key: str | None = None,
        runner: Any | None = None,
    ) -> dict[str, Any]:
        """
        Launch a BoltzGen design run and return parsed results.

        Args:
            design_spec: Path to BoltzGen design YAML.
            output_dir: Destination for BoltzGen artifacts.
            protocol: BoltzGen protocol (e.g., protein-anything, peptide-anything).
            num_designs: Intermediate designs to generate.
            budget: Final designs to keep after filtering.
            steps: Optional subset of steps to run.
            devices: Number of accelerators to request.
            reuse: Reuse intermediate files when present.
            cache_dir: Optional cache directory for BoltzGen downloads.
            top_k: Number of ranked designs to summarize.
            score_key: Optional metric key used to sort summaries.
            runner: Optional BoltzGenRunner for injection/testing.

        Returns:
            Dictionary with run status, command, parsed metrics, and a top-k summary.
        """
        if runner is None:
            from .boltzgen_adapter import BoltzGenRunner

            runner = BoltzGenRunner(cache_dir=cache_dir, work_dir=output_dir or self.checkpoint_dir)

        result = runner.run(
            design_spec=design_spec,
            output_dir=output_dir or self.checkpoint_dir,
            protocol=protocol,
            num_designs=num_designs,
            budget=budget,
            steps=steps,
            devices=devices,
            reuse=reuse,
            parse_results=True,
        )

        summary = runner.summarize_metrics(result.metrics, top_k=top_k, score_key=score_key)

        return {
            "success": result.success,
            "command": result.command,
            "output_dir": str(result.output_dir),
            "metrics_file": str(result.metrics_file) if result.metrics_file else None,
            "metrics": result.metrics,
            "summary": summary,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
