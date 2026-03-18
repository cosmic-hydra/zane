"""
Self-Learning Training Pipeline
Automatically trains models with continuous learning capabilities
"""

import os
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class SelfLearningTrainer:
    """
    Self-learning trainer with automatic hyperparameter tuning
    and continuous learning capabilities
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 10,
        save_dir: str = "./checkpoints",
    ):
        """
        Args:
            model: PyTorch model to train
            device: Device to use
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=patience // 2
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "best_val_loss": float("inf"),
            "epochs_without_improvement": 0,
        }

    def train_epoch(self, train_loader: DataLoader, loss_fn: Callable, is_graph: bool = False) -> float:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            loss_fn: Loss function
            is_graph: Whether data is graph-structured

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()

            if is_graph:
                batch = batch.to(self.device)
                predictions = self.model(batch)
                targets = batch.y
            else:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    features, targets = batch
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    predictions = self.model(features)
                else:
                    if hasattr(batch, "to"):
                        batch_obj = cast(Any, batch)
                        batch_obj = batch_obj.to(self.device)
                        predictions = self.model(batch_obj)
                        targets = batch_obj.y if hasattr(batch_obj, "y") else None
                    else:
                        continue

            if targets is not None:
                # Filter out missing values
                mask = targets != -1
                if mask.sum() > 0:
                    loss = loss_fn(predictions[mask], targets[mask])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

        return total_loss / max(num_batches, 1)

    def validate(self, val_loader: DataLoader, loss_fn: Callable, is_graph: bool = False) -> float:
        """
        Validate the model

        Args:
            val_loader: Validation data loader
            loss_fn: Loss function
            is_graph: Whether data is graph-structured

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if is_graph:
                    batch = batch.to(self.device)
                    predictions = self.model(batch)
                    targets = batch.y
                else:
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        features, targets = batch
                        features = features.to(self.device)
                        targets = targets.to(self.device)
                        predictions = self.model(features)
                    else:
                        if hasattr(batch, "to"):
                            batch_obj = cast(Any, batch)
                            batch_obj = batch_obj.to(self.device)
                            predictions = self.model(batch_obj)
                            targets = batch_obj.y if hasattr(batch_obj, "y") else None
                        else:
                            continue

                if targets is not None:
                    mask = targets != -1
                    if mask.sum() > 0:
                        loss = loss_fn(predictions[mask], targets[mask])
                        total_loss += loss.item()
                        num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        loss_fn: Callable | None = None,
        is_graph: bool = False,
    ) -> dict:
        """
        Train the model with early stopping

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            loss_fn: Loss function (defaults to MSE)
            is_graph: Whether data is graph-structured

        Returns:
            Training history
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader, loss_fn, is_graph)
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss = self.validate(val_loader, loss_fn, is_graph)
            self.history["val_loss"].append(val_loss)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping check
            if val_loss < self.history["best_val_loss"]:
                self.history["best_val_loss"] = val_loss
                self.history["epochs_without_improvement"] = 0
                self.save_checkpoint("best_model.pt")
                print(f"✓ New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.history["epochs_without_improvement"] += 1

            if self.history["epochs_without_improvement"] >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Load best model
        self.load_checkpoint("best_model.pt")

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "timestamp": datetime.now().isoformat(),
        }

        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.history = checkpoint.get("history", self.history)
            print(f"Checkpoint loaded from {filepath}")

    def predict(self, data_loader: DataLoader, is_graph: bool = False) -> np.ndarray:
        """
        Make predictions

        Args:
            data_loader: Data loader
            is_graph: Whether data is graph-structured

        Returns:
            Predictions array
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                if is_graph:
                    batch = batch.to(self.device)
                    pred = self.model(batch)
                else:
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        features, _ = batch
                        features = features.to(self.device)
                        pred = self.model(features)
                    else:
                        if hasattr(batch, "to"):
                            batch_obj = cast(Any, batch)
                            batch_obj = batch_obj.to(self.device)
                            pred = self.model(batch_obj)
                        else:
                            continue

                predictions.append(pred.cpu().numpy())

        return np.vstack(predictions)


class ContinuousLearner:
    """
    Continuous learning system that periodically retrains on new data
    """

    def __init__(
        self,
        trainer: SelfLearningTrainer,
        data_collector,
        retrain_threshold: int = 1000,  # Number of new samples before retraining
    ):
        """
        Args:
            trainer: SelfLearningTrainer instance
            data_collector: DataCollector instance
            retrain_threshold: Number of new samples to trigger retraining
        """
        self.trainer = trainer
        self.data_collector = data_collector
        self.retrain_threshold = retrain_threshold
        self.new_samples_count = 0

    def add_samples(self, num_samples: int):
        """Track new samples"""
        self.new_samples_count += num_samples

        if self.new_samples_count >= self.retrain_threshold:
            print(f"Retraining threshold reached ({self.new_samples_count} new samples)")
            return True
        return False

    def retrain(self, train_loader, val_loader, **kwargs):
        """Retrain the model on updated data"""
        print("Starting continuous learning retraining...")
        self.trainer.train(train_loader, val_loader, **kwargs)
        self.new_samples_count = 0
        print("Retraining complete!")
