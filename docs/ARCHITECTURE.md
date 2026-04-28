# ZANE Architecture Guide

> A complete map of ZANE's layered architecture, data flows, and module interactions.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INTERFACE LAYER                            │
│  CLI (drug_discovery/cli.py, cli_extension.py)                  │
│  Terminal Dashboard (dashboard.py)                              │
│  Python API (import drug_discovery)                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    ORCHESTRATION LAYER                           │
│  Fail-Fast Pipeline    │  Agent Orchestrator    │  Active Learn  │
│  (failfast_pipeline)   │  (agents/orchestrator) │  (active_learn)│
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    INTELLIGENCE LAYER                            │
│  Models          │ Training        │ Evaluation      │ Optim    │
│  ├─ EGNN/SchNet  │ ├─ Advanced     │ ├─ Uncertainty  │ ├─ MOBO  │
│  ├─ D-MPNN       │ ├─ Contrastive  │ ├─ ADMET (16ep) │ ├─ HPO   │
│  ├─ Diffusion    │ └─ EMA/AMP      │ ├─ Clinical CSS │ └─ Active│
│  ├─ GFlowNet     │                 │ ├─ Tox Panel    │          │
│  └─ GNN/Trans    │                 │ └─ Struct Alert  │          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      SCIENCE LAYER                              │
│  Docking (Vina/DiffDock)  │  ML-FEP (free energy)              │
│  MD Simulation (OpenMM)   │  Retrosynthesis (AiZynthFinder)    │
│  Protein Structure        │  Reaction Prediction               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                       DATA LAYER                                │
│  Collection (PubChem, ChEMBL, DrugBank)                         │
│  Pipeline (SMILES validation, fingerprints, 3D graphs)          │
│  Featurization (Morgan FP, descriptors, graph construction)     │
│  Validation (scaffold splits, metrics, statistical tests)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: End-to-End Drug Discovery

```
Raw SMILES ──► Validation ──► Featurization ──► Model Training
                  │                │                   │
            (reject bad)    (FP, graphs,          (EGNN, D-MPNN,
                             descriptors)          Transformer)
                                                      │
                                                      ▼
                                              Property Prediction
                                                      │
                              ┌────────────────────────┼─────────────────┐
                              ▼                        ▼                 ▼
                     Structural Alerts          Tox Panel (12ep)   ADMET (16ep)
                     (PAINS/Brenk/reactive)     (hERG,Ames,DILI)  (solubility,BBB)
                              │                        │                 │
                              └────────────┬───────────┘                 │
                                           ▼                             │
                                   Fail-Fast Pipeline ◄──────────────────┘
                                           │
                                    ┌──────┴──────┐
                                    ▼             ▼
                              Survivors      Eliminated
                                    │        (with reason)
                                    ▼
                           Clinical Success Score
                           (57% efficacy + 17% safety + 20% ADMET)
                                    │
                                    ▼
                           Multi-Obj Optimization
                           (EHVI, Pareto ranking)
                                    │
                                    ▼
                            Ranked Candidates
                            (for expert review)
```

---

## Module Map

### `drug_discovery/models/` — Neural Architectures

| Module | Class | Purpose | Key Config |
|--------|-------|---------|------------|
| `equivariant_gnn.py` | `EquivariantGNN` | E(3)-invariant 3D property prediction | `EquivariantGNNConfig(variant="egnn")` |
| `diffusion_generator.py` | `DiffusionMoleculeGenerator` | SE(3) denoising 3D generation | `DiffusionConfig(noise_steps=1000)` |
| `gflownet.py` | `GFlowNetTrainer` | Diverse reward-proportional sampling | `GFlowNetConfig(temperature=1.0)` |
| `dmpnn.py` | `DMPNN` | Directed bond-centric message passing | `DMPNNConfig(depth=8, hidden_dim=300)` |
| `gnn.py` | `MolecularGNN` | Standard graph neural network | `node_features=133` |
| `transformer.py` | `MolecularTransformer` | Fingerprint/sequence transformer | `hidden_dim=256` |
| `ensemble.py` | `EnsembleModel` | Weighted model ensemble | `learnable_weights=True` |
| `__init__.py` | `get_model("egnn")` | Unified model registry | Auto-discovers available models |

### `drug_discovery/training/` — Training Pipelines

| Module | Class | Purpose |
|--------|-------|---------|
| `advanced_training.py` | `AdvancedTrainer` | AMP, EMA, cosine warmup, grad clip, early stopping |
| `contrastive_pretraining.py` | `ContrastivePretrainer` | 2D-3D self-supervised pretraining |

### `drug_discovery/evaluation/` — Assessment & Safety

| Module | Class | Purpose |
|--------|-------|---------|
| `uncertainty.py` | `MCDropoutPredictor`, `ConformalPredictor` | Uncertainty quantification |
| `advanced_admet.py` | `AdvancedADMETPredictor` | 16-endpoint GNN+Transformer ADMET |
| `clinical_success_predictor.py` | `ClinicalSuccessPredictor` | CSS scoring, CNS/Oral MPO |
| `failfast_pipeline.py` | `FailFastPipeline` | Multi-stage elimination pipeline |
| `structural_alerts.py` | `StructuralAlertScreener` | PAINS/Brenk/reactive metabolite alerts |
| `deep_tox_panel.py` | `DeepToxPanel` | 12-endpoint organ-level toxicity |

### `drug_discovery/optimization/` — Lead Optimization

| Module | Class | Purpose |
|--------|-------|---------|
| `multi_objective.py` | `MultiObjectiveBayesianOptimizer` | EHVI Pareto optimization |
| `hyperparameter_optimization.py` | `HPOptimizer` | Grid/random/Bayesian HPO |
| `active_learning.py` | `ActiveLearner` | Uncertainty-driven experiment planning |

### `drug_discovery/simulation/` & `drug_discovery/physics/`

| Module | Class | Purpose |
|--------|-------|---------|
| `free_energy.py` | `FEPPipeline` | ML-accelerated ΔΔG prediction |
| `docking.py` | `DockingPipeline` | Vina/DiffDock/GNina interface |
| `md_simulator.py` | `MolecularDynamicsSimulator` | OpenMM molecular dynamics |
| `protein_structure.py` | `ProteinStructurePredictor` | Structure prediction interface |

### `drug_discovery/data/` — Data Processing

| Module | Class | Purpose |
|--------|-------|---------|
| `pipeline.py` | `MolecularDataset` | SMILES validation, fingerprints, 3D graphs |
| `collector.py` | `DataCollector` | PubChem/ChEMBL/DrugBank ingestion |
| `dataset.py` | `MolecularDataset` (torch) | PyTorch Dataset for training |

### `drug_discovery/validation/` — Scientific Rigor

| Module | Class | Purpose |
|--------|-------|---------|
| `scientific_validation.py` | `ExperimentReport` | Scaffold CV, bootstrap CI, significance tests |

---

## Configuration Pattern

All modules use Python `dataclass` configs:

```python
from drug_discovery.models.equivariant_gnn import EquivariantGNN, EquivariantGNNConfig

config = EquivariantGNNConfig(
    variant="egnn",
    hidden_dim=128,
    num_layers=6,
    rbf_cutoff=5.0,
    dropout=0.1,
)
model = EquivariantGNN(config)
```

Or via the registry:

```python
from drug_discovery.models import get_model
model = get_model("egnn", hidden_dim=128, num_layers=6)
```

---

## Import Safety

All `__init__.py` files use guarded imports:

```python
try:
    from drug_discovery.models.equivariant_gnn import EquivariantGNN
except ImportError:
    pass  # torch not installed — module silently unavailable
```

This means ZANE works in environments with partial dependencies.
