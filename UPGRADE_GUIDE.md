## 2026 Q2 — SOTA Drug Discovery AI Upgrades

Seven new modules were added implementing state-of-the-art 2025-2026 techniques:

### New Modules Summary

| Module | Path | Capability |
|--------|------|------------|
| E(3)-Equivariant GNN | `drug_discovery/models/equivariant_gnn.py` | EGNN/SchNet with 3D rotational invariance |
| Diffusion Generator | `drug_discovery/models/diffusion_generator.py` | SE(3) denoising diffusion for molecule generation |
| Advanced Training | `drug_discovery/training/advanced_training.py` | AMP, EMA, cosine warmup, gradient clipping |
| Uncertainty Quant. | `drug_discovery/evaluation/uncertainty.py` | MC Dropout, Conformal Prediction, ECE |
| ML-FEP Pipeline | `drug_discovery/simulation/free_energy.py` | GNN surrogate for ΔΔG prediction |
| Advanced ADMET | `drug_discovery/evaluation/advanced_admet.py` | 16-endpoint ensemble GNN+Transformer |
| Multi-Obj Optimizer | `drug_discovery/optimization/multi_objective.py` | EHVI Bayesian optimization |

### Quick Usage Examples

**E(3)-Equivariant GNN:**
```python
from drug_discovery.models.equivariant_gnn import EquivariantGNN, EquivariantGNNConfig

config = EquivariantGNNConfig(variant="egnn", hidden_dim=128, num_layers=6)
model = EquivariantGNN(config)
predictions = model(atomic_numbers, positions, edge_index, batch)
```

**Advanced Training:**
```python
from drug_discovery.training.advanced_training import AdvancedTrainer, AdvancedTrainingConfig

config = AdvancedTrainingConfig(epochs=100, use_amp=True, use_ema=True)
trainer = AdvancedTrainer(model, config, device="cuda")
history = trainer.fit(train_loader, val_loader)
```

**Uncertainty Quantification:**
```python
from drug_discovery.evaluation.uncertainty import MCDropoutPredictor, ConformalPredictor

# MC Dropout
mc = MCDropoutPredictor(model, mc_samples=30)
mean, std, samples = mc.predict(**inputs)

# Conformal Prediction
cp = ConformalPredictor(alpha=0.05)
cp.calibrate(val_preds, val_targets)
lower, upper = cp.predict_interval(test_preds)
```

**Multi-Objective Optimization:**
```python
from drug_discovery.optimization.multi_objective import (
    MultiObjectiveBayesianOptimizer, MOBOConfig
)

config = MOBOConfig(
    objective_names=["potency", "selectivity", "solubility"],
    objective_directions=["maximize", "maximize", "maximize"],
    ref_point=[0.0, 0.0, -5.0],
)
optimizer = MultiObjectiveBayesianOptimizer(config)
optimizer.tell(X_init, Y_init)
selected_idx, acq_values = optimizer.ask(candidates, n_select=5)
```

For full details see [CHANGELOG.md](./CHANGELOG.md).