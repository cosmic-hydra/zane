# ZANE API Reference

> Complete reference for all public classes, functions, and configurations.

---

## Model Registry (`drug_discovery.models`)

### `get_model(name, **kwargs) -> nn.Module`
Instantiate a model by name.

```python
from drug_discovery.models import get_model, list_models

print(list_models())  # ['diffusion', 'dmpnn', 'egnn', 'gflownet', 'gnn', 'schnet', 'transformer']

model = get_model("egnn", hidden_dim=128, num_layers=6)
model = get_model("schnet", hidden_dim=256, rbf_cutoff=6.0)
model = get_model("dmpnn")  # uses DMPNNConfig defaults
```

---

## Models

### EquivariantGNN
```python
from drug_discovery.models.equivariant_gnn import EquivariantGNN, EquivariantGNNConfig

config = EquivariantGNNConfig(
    variant="egnn",       # "egnn" or "schnet"
    hidden_dim=128,       # Hidden dimension
    num_layers=6,         # Message passing layers
    num_rbf=50,           # Radial basis functions
    rbf_cutoff=5.0,       # Distance cutoff (Angstroms)
    max_atomic_num=100,   # Max atomic number
    output_dim=1,         # Output dimensions
    dropout=0.0,          # Dropout rate
    num_tasks=1,          # Multi-task heads
)
model = EquivariantGNN(config)
output = model(z, pos, edge_index, batch)  # (B, output_dim*num_tasks)
```

### DMPNN (Directed Message Passing)
```python
from drug_discovery.models.dmpnn import DMPNN, DMPNNConfig

config = DMPNNConfig(
    atom_fdim=133,     # Atom feature dimension
    bond_fdim=14,      # Bond feature dimension
    hidden_dim=300,    # Hidden size
    depth=8,           # Message passing iterations
    aggregation="mean" # "mean", "sum", or "attention"
)
model = DMPNN(config)
output = model(atom_feats, bond_feats, a2b, b2a, b2revb, batch)
```

### DiffusionMoleculeGenerator
```python
from drug_discovery.models.diffusion_generator import DiffusionMoleculeGenerator, DiffusionConfig

config = DiffusionConfig(hidden_dim=256, noise_steps=1000, num_atom_types=10)
gen = DiffusionMoleculeGenerator(config, device="cuda")
molecules = gen.sample(num_molecules=10, num_atoms=20)
# Returns: {"positions": (10, 20, 3), "atom_types": (10, 20)}
```

### GFlowNetTrainer
```python
from drug_discovery.models.gflownet import GFlowNetTrainer, GFlowNetConfig

config = GFlowNetConfig(hidden_dim=256, max_atoms=38, temperature=1.0)
trainer = GFlowNetTrainer(config, reward_fn=my_reward, device="cuda")
for step in range(10000):
    loss = trainer.train_step()
molecules = trainer.sample(n=50)
```

---

## Training

### AdvancedTrainer
```python
from drug_discovery.training.advanced_training import AdvancedTrainer, AdvancedTrainingConfig

config = AdvancedTrainingConfig(
    epochs=100,
    learning_rate=3e-4,
    scheduler="cosine_warm_restarts",  # "cosine_warm_restarts", "plateau", "one_cycle"
    warmup_epochs=5,
    use_amp=True,
    use_ema=True,
    ema_decay=0.999,
    grad_clip_norm=1.0,
    early_stopping_patience=20,
)
trainer = AdvancedTrainer(model, config, device="cuda")
history = trainer.fit(train_loader, val_loader)
# Returns: {"train_loss": [...], "val_loss": [...], "lr": [...]}
```

### ContrastivePretrainer
```python
from drug_discovery.training.contrastive_pretraining import ContrastivePretrainer, ContrastiveConfig

config = ContrastiveConfig(hidden_dim=256, temperature=0.1)
pretrainer = ContrastivePretrainer(config)
for z, pos, edge_index in unlabeled_data:
    loss = pretrainer(z, pos, edge_index)
    loss.backward()
encoder = pretrainer.get_pretrained_encoder()  # Use for finetuning
```

---

## Evaluation & Safety

### ClinicalSuccessPredictor
```python
from drug_discovery.evaluation.clinical_success_predictor import ClinicalSuccessPredictor

predictor = ClinicalSuccessPredictor()
profile = predictor.assess("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin

print(profile.clinical_success_score)  # 0.0-1.0
print(profile.risk_level)              # "low"/"medium"/"high"/"critical"
print(profile.mpo_scores)             # {"cns_mpo": 4.2, "oral_mpo": 5.1}
print(profile.toxicity_flags)         # {"cardiotoxicity": 0.1, ...}
print(profile.recommendations)        # ["CNS-PENETRANT: suitable for CNS targets"]

# Batch + rank
profiles = predictor.batch_assess(smiles_list)
ranked = predictor.rank_by_success(profiles)
```

### FailFastPipeline
```python
from drug_discovery.evaluation.failfast_pipeline import FailFastPipeline, FailFastConfig

config = FailFastConfig(min_qed=0.3, min_oral_mpo=3.0, max_tox_risk=0.7)
pipeline = FailFastPipeline(config)
results = pipeline.run(smiles_list)

survivors = pipeline.get_survivors(results)
report = pipeline.attrition_report(results)
# {"total": 1000, "survivors": 342, "pass_rate": 0.342,
#  "eliminated_by_stage": {"druglikeness": 280, "admet": 198, "toxicity": 180}}
```

### StructuralAlertScreener
```python
from drug_discovery.evaluation.structural_alerts import StructuralAlertScreener

screener = StructuralAlertScreener()
report = screener.screen("O=C1CSC(=S)N1")  # Rhodanine (known PAINS)
print(report.is_clean)        # False
print(report.pains_alerts)    # [AlertResult("rhodanine", ...)]
print(report.risk_score)      # 0.25

# Batch: get only clean molecules
clean = screener.get_clean_molecules(screener.batch_screen(smiles_list))
```

### DeepToxPanel
```python
from drug_discovery.evaluation.deep_tox_panel import DeepToxPanel

panel = DeepToxPanel()
report = panel.screen("c1ccccc1")  # Benzene
for name, result in report.endpoints.items():
    print(f"{name}: {result.risk_level} (p={result.probability:.2f})")
# herg: low (p=0.12), ames: low (p=0.15), dili: low (p=0.08), ...
```

### Uncertainty Quantification
```python
from drug_discovery.evaluation.uncertainty import MCDropoutPredictor, ConformalPredictor

# MC Dropout
mc = MCDropoutPredictor(model, mc_samples=30, dropout_rate=0.1)
mean, std, samples = mc.predict(z=z, pos=pos, edge_index=ei, batch=batch)

# Conformal Prediction (95% coverage)
cp = ConformalPredictor(alpha=0.05)
cp.calibrate(val_predictions, val_true)
lower, upper = cp.predict_interval(test_predictions)
print(cp.coverage(test_predictions, test_true))  # ~0.95
```

---

## Optimization

### MultiObjectiveBayesianOptimizer
```python
from drug_discovery.optimization.multi_objective import MultiObjectiveBayesianOptimizer, MOBOConfig

config = MOBOConfig(
    objective_names=["potency", "selectivity", "solubility"],
    objective_directions=["maximize", "maximize", "maximize"],
    ref_point=[0.0, 0.0, -5.0],
)
opt = MultiObjectiveBayesianOptimizer(config)
opt.tell(X_observed, Y_observed)
indices, acq_values = opt.ask(X_candidates, n_select=5)
front = opt.get_pareto_front()
```

### HPOptimizer
```python
from drug_discovery.optimization.hyperparameter_optimization import HPOptimizer, HPOConfig, SearchSpace

space = SearchSpace()
space.add_float("lr", 1e-5, 1e-2, log=True)
space.add_int("num_layers", 2, 8)
space.add_categorical("model", ["egnn", "schnet", "dmpnn"])

def train_fn(params):
    model = get_model(params["model"], num_layers=params["num_layers"])
    # ... train and return val_loss
    return val_loss

best = HPOptimizer(HPOConfig(n_trials=50)).optimize(space, train_fn)
print(best.params, best.metric_value)
```

### ActiveLearner
```python
from drug_discovery.optimization.active_learning import ActiveLearner, ActiveLearningConfig

config = ActiveLearningConfig(acquisition="thompson", n_select=10)
learner = ActiveLearner(config)
learner.fit(X_train, y_train, model_fn=my_model_factory)
next_indices = learner.select(X_pool, fingerprints=fp_pool)
# After lab results:
learner.update(X_new, y_new)
```

---

## Data Pipeline

### MolecularDataset
```python
from drug_discovery.data.pipeline import (
    MolecularDataset, validate_smiles, compute_descriptors,
    compute_morgan_fingerprint, smiles_to_graph, lipinski_filter
)

ds = MolecularDataset(smiles=["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
ds.featurize(methods=["descriptors", "fingerprints", "graphs"])
report = ds.quality_report()
# {"total": 3, "valid": 3, "lipinski_pass_rate": 1.0, "mean_qed": 0.45}
```

### Scientific Validation
```python
from drug_discovery.validation.scientific_validation import (
    scaffold_kfold, compute_metrics, ExperimentReport, set_global_seed
)

set_global_seed(42)
folds = scaffold_kfold(smiles_list, n_folds=5)
report = ExperimentReport(model_name="egnn", dataset="chembl")
for train_idx, val_idx in folds:
    # ... train and evaluate
    report.fold_metrics.append(compute_metrics(y_true, y_pred, "regression"))
report.compute_aggregates()
report.save("outputs/reports/egnn_chembl.json")
```
