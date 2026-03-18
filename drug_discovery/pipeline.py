"""
Main Drug Discovery Pipeline
Orchestrates the entire AI drug discovery process
"""

import os
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

from .data import DataCollector, MolecularDataset, MolecularFeaturizer, train_test_split_molecular
from .evaluation import ADMETPredictor, ModelEvaluator, PropertyPredictor
from .models import EnsembleModel, MolecularGNN, MolecularTransformer
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

        # Initialize components
        self.data_collector = DataCollector(cache_dir=cache_dir)
        self.featurizer = MolecularFeaturizer()
        self.admet_predictor = ADMETPredictor()
        self.evaluator = ModelEvaluator()

        # Models and trainers (initialized during training)
        self.model = None
        self.trainer = None
        self.property_predictor = None

        print("Drug Discovery Pipeline initialized")
        print(f"Model type: {model_type}")
        print(f"Device: {device}")

    def collect_data(
        self, sources: list[str] = ["pubchem", "chembl", "approved_drugs"], limit_per_source: int = 1000
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

        # Merge datasets
        if datasets:
            merged_data = self.data_collector.merge_datasets(datasets)
            print(f"\nTotal unique molecules collected: {len(merged_data)}")
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
        train_dataset, test_dataset = train_test_split_molecular(dataset, test_size=test_size)

        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        # Create dataloaders
        if featurization == "graph":
            train_loader = GeometricDataLoader(cast(Any, train_dataset), batch_size=batch_size, shuffle=True)
            test_loader = GeometricDataLoader(cast(Any, test_dataset), batch_size=batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "model_type": self.model_type,
                },
                filepath,
            )
            print(f"Pipeline saved to {filepath}")

    def load(self, filepath: str):
        """Load the pipeline"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model_type = checkpoint["model_type"]
        self.build_model()
        if self.model is None:
            raise RuntimeError("Failed to build model during load().")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.property_predictor = PropertyPredictor(self.model, self.device)
        print(f"Pipeline loaded from {filepath}")
