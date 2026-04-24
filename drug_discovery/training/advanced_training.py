"""
Advanced Training Pipeline for ZANE.

Modern training utilities to improve model performance:
- Cosine annealing with warm restarts
- Learning rate warmup
- Gradient clipping + Mixed precision (AMP)
- EMA of model weights
- Plateau-aware early stopping with checkpointing
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau

logger = logging.getLogger(__name__)


@dataclass
class AdvancedTrainingConfig:
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.999)
    scheduler: str = "cosine_warm_restarts"
    warmup_epochs: int = 5
    cosine_t0: int = 10
    cosine_t_mult: int = 2
    cosine_eta_min: float = 1e-6
    plateau_patience: int = 10
    plateau_factor: float = 0.5
    epochs: int = 100
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    accumulation_steps: int = 1
    use_ema: bool = True
    ema_decay: float = 0.999
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True


class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.current_epoch = 0
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch=None, metrics=None):
        self.current_epoch = epoch if epoch is not None else self.current_epoch + 1
        if self.current_epoch <= self.warmup_epochs:
            factor = self.current_epoch / max(1, self.warmup_epochs)
            for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = blr * factor
        else:
            if isinstance(self.base_scheduler, ReduceLROnPlateau) and metrics is not None:
                self.base_scheduler.step(metrics)
            else:
                self.base_scheduler.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metric):
        improved = (
            (metric < self.best - self.min_delta) if self.mode == "min" else (metric > self.best + self.min_delta)
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class AdvancedTrainer:
    """Production-grade trainer with AMP, EMA, warmup, grad clipping.

    Example::
        config = AdvancedTrainingConfig(epochs=100, use_amp=True)
        trainer = AdvancedTrainer(model, config, device="cuda")
        history = trainer.fit(train_loader, val_loader)
    """

    def __init__(self, model, config, device="cpu", loss_fn=None):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.loss_fn = loss_fn or nn.MSELoss()
        if config.optimizer == "adamw":
            self.optimizer = AdamW(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=config.betas
            )
        else:
            self.optimizer = Adam(model.parameters(), lr=config.learning_rate, betas=config.betas)
        if config.scheduler == "cosine_warm_restarts":
            base = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=config.cosine_t0, T_mult=config.cosine_t_mult, eta_min=config.cosine_eta_min
            )
        elif config.scheduler == "plateau":
            base = ReduceLROnPlateau(self.optimizer, patience=config.plateau_patience, factor=config.plateau_factor)
        else:
            base = OneCycleLR(self.optimizer, max_lr=config.learning_rate, epochs=config.epochs, steps_per_epoch=1)
        self.scheduler = WarmupScheduler(self.optimizer, config.warmup_epochs, base)
        self.scaler = torch.amp.GradScaler("cuda") if config.use_amp and device != "cpu" else None
        self.ema = EMA(model, config.ema_decay) if config.use_ema else None
        self.early_stopping = EarlyStopping(config.early_stopping_patience, config.early_stopping_min_delta)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def fit(self, train_loader, val_loader=None):
        history = {"train_loss": [], "val_loss": [], "lr": [], "epoch_time": []}
        best_val = float("inf")
        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            tl = self._train_epoch(train_loader)
            history["train_loss"].append(tl)
            vl = None
            if val_loader:
                if self.ema:
                    self.ema.apply(self.model)
                vl = self._eval_epoch(val_loader)
                if self.ema:
                    self.ema.restore(self.model)
                history["val_loss"].append(vl)
            self.scheduler.step(epoch=epoch, metrics=vl)
            lr = self.scheduler.get_last_lr()[0]
            history["lr"].append(lr)
            history["epoch_time"].append(time.time() - t0)
            logger.info(
                f"Epoch {epoch}/{self.config.epochs} | train={tl:.6f} | val={vl:.6f if vl else 'N/A'} | lr={lr:.2e}"
            )
            if vl is not None and vl < best_val:
                best_val = vl
                self._save(epoch, vl, "best_model.pt")
            if vl is not None and self.early_stopping.step(vl):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        return history

    def _train_epoch(self, loader):
        self.model.train()
        total, n = 0.0, 0
        self.optimizer.zero_grad()
        for i, batch in enumerate(loader):
            batch = self._to_dev(batch)
            amp_on = self.scaler is not None
            with torch.amp.autocast("cuda", enabled=amp_on):
                pred = self.model(**batch["inputs"])
                loss = self.loss_fn(pred, batch["targets"]) / self.config.accumulation_steps
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if (i + 1) % self.config.accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.ema:
                    self.ema.update(self.model)
            total += loss.item() * self.config.accumulation_steps
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader):
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            batch = self._to_dev(batch)
            pred = self.model(**batch["inputs"])
            total += self.loss_fn(pred, batch["targets"]).item()
            n += 1
        return total / max(n, 1)

    def _to_dev(self, batch):
        if isinstance(batch, dict):
            return {k: self._to_dev(v) for k, v in batch.items()}
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch

    def _save(self, epoch, val_loss, fname):
        path = Path(self.config.checkpoint_dir) / fname
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )
        logger.info(f"Checkpoint: {path} (val_loss={val_loss:.6f})")
