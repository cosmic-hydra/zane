# Project Structure

```
zane/
├── LICENSE                          # CC0 1.0 Universal License
├── README.md                        # Main README
├── DOCUMENTATION.md                 # Comprehensive technical documentation
├── setup.py                         # Python package setup
├── requirements.txt                 # Python dependencies
├── pytest.ini                       # Pytest configuration
├── .gitignore                       # Git ignore rules
│
├── configs/
│   └── config.py                    # Configuration settings
│
├── drug_discovery/                  # Main package
│   ├── __init__.py                  # Package initialization
│   ├── pipeline.py                  # Main DrugDiscoveryPipeline class
│   ├── cli.py                       # Command-line interface
│   │
│   ├── data/                        # Data collection and processing
│   │   ├── __init__.py
│   │   ├── collector.py             # DataCollector - PubChem, ChEMBL, etc.
│   │   └── dataset.py               # MolecularDataset, MolecularFeaturizer
│   │
│   ├── models/                      # AI/ML models
│   │   ├── __init__.py
│   │   ├── gnn.py                   # Graph Neural Networks (GNN, MPNN)
│   │   ├── transformer.py           # Transformer models
│   │   └── ensemble.py              # Ensemble and multi-task models
│   │
│   ├── training/                    # Training infrastructure
│   │   ├── __init__.py
│   │   └── trainer.py               # SelfLearningTrainer, ContinuousLearner
│   │
│   ├── evaluation/                  # Evaluation and prediction
│   │   ├── __init__.py
│   │   └── predictor.py             # PropertyPredictor, ADMETPredictor, ModelEvaluator
│   │
│   └── utils/                       # Utility functions
│       └── __init__.py              # Common utilities
│
├── examples/                        # Usage examples
│   ├── basic_usage.py               # Basic pipeline usage
│   ├── continuous_learning.py      # Continuous learning example
│   └── admet_prediction.py          # ADMET analysis example
│
└── tests/                           # Unit and integration tests
    ├── conftest.py                  # Pytest fixtures
    ├── test_data.py                 # Data collection tests
    ├── test_dataset.py              # Dataset tests
    ├── test_models.py               # Model tests
    ├── test_evaluation.py           # Evaluation tests
    └── test_pipeline.py             # Pipeline integration tests
```

## Component Overview

### 1. Data Collection (`drug_discovery/data/`)
- **collector.py**: Automated data collection from PubChem, ChEMBL, and DrugBank
- **dataset.py**: Molecular featurization (graphs, fingerprints, descriptors)

### 2. Models (`drug_discovery/models/`)
- **gnn.py**: Graph Neural Networks with attention mechanisms
- **transformer.py**: Transformer-based models for molecular sequences
- **ensemble.py**: Ensemble methods and multi-task learning

### 3. Training (`drug_discovery/training/`)
- **trainer.py**: Self-learning training with automatic hyperparameter tuning
- Continuous learning capabilities
- Early stopping and learning rate scheduling

### 4. Evaluation (`drug_discovery/evaluation/`)
- **predictor.py**: Property prediction and ADMET analysis
- Lipinski's Rule of Five
- Drug-likeness (QED)
- Synthetic accessibility
- Toxicity screening

### 5. Pipeline (`drug_discovery/pipeline.py`)
- Main orchestrator class
- End-to-end drug discovery workflow
- Model training, evaluation, and deployment

## Key Technologies

- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural networks
- **RDKit**: Cheminformatics toolkit
- **Transformers**: State-of-the-art NLP models adapted for chemistry
- **scikit-learn**: Machine learning utilities
- **PubChemPy**: PubChem API client
- **ChEMBL Web Services**: Bioactivity data

## Features Implemented

✅ Multi-source data collection (PubChem, ChEMBL, DrugBank)
✅ Graph Neural Networks (GAT, MPNN)
✅ Transformer models for molecular fingerprints
✅ Ensemble methods
✅ Self-learning and continuous training
✅ ADMET property prediction
✅ Drug-likeness assessment
✅ Comprehensive testing suite
✅ CLI interface
✅ Extensive documentation
✅ Configuration management

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from drug_discovery import DrugDiscoveryPipeline

pipeline = DrugDiscoveryPipeline(model_type='gnn')
data = pipeline.collect_data()
train_loader, test_loader = pipeline.prepare_datasets(data)
pipeline.train(train_loader, test_loader)
properties = pipeline.predict_properties("CC(=O)OC1=CC=CC=C1C(=O)O")
```

## Testing

```bash
pytest tests/ -v
```

## CLI Usage

```bash
# Analyze ADMET properties
python -m drug_discovery.cli admet "CC(=O)OC1=CC=CC=C1C(=O)O"

# Train a model
python -m drug_discovery.cli train --model gnn --epochs 100

# Collect data
python -m drug_discovery.cli collect --sources pubchem chembl --limit 1000
```

## 2026-03-23 Structure Addendum

- Dashboard runtime upgrades are implemented in `drug_discovery/dashboard.py`.
- Dashboard CLI controls are exposed in `drug_discovery/cli.py` with:
    - `--theme` (`lab`, `neon`, `classic`)
    - `--motion-intensity` (1-3)
- Reproducible benchmark reports are stored in `outputs/reports/`.
- Latest benchmark artifact: `outputs/reports/scientific_benchmark_20260323.json`.

## Unicorn Platform Modules 14-16

```
infrastructure/
│   ├── cryptography/ (Module 14: ZKP Federated Data Bourse)
│   │   ├── __init__.py
│   │   └── zkp_marketplace.py  # PySyft federation, snarkjs proofs, Fabric royalties, FPGA offload
│   └── cloud_lab/ (Module 15: Cloud-Lab OS Kernel)
│       ├── __init__.py
│       └── os_kernel.py        # LabOP compilation, Bacalhau dispatch, closed-loop
models/biologics/
│   ├── crispr_foundry.py       # Module 16: De Novo CRISPR (ESM3, RoseTTAFold, RMSD verify)
│   └── mrna_designer.py (existing)

Root level:
└── unicorn_platform_orchestrator.py  # CLI orchestrator &amp; integration
```

**Install &amp; Run:**
```bash
pip install -e .[unicorn]
unicorn-orchestrator run --help
unicorn-orchestrator run &quot;ATCG...&quot; --fpga-device /dev/fpga0
```

## 2024 Breakthroughs Integration

- `drug_discovery/alphafold3/alphafold3_docking.py`: AF3 ligand binding proxy (DiffDock+OpenFold)
- `drug_discovery/rfdiffusion/protein_design.py`: RFdiffusion motif scaffolding
- `models/biologics/crispr_base_editor.py`: Casgevy base editing
- `models/nextgen_adcs/adc_optimizer.py`: ADC linker/BBB optimization
- `models/delivery/bbb_shuttles.py`: BBB shuttle design
- `drug_discovery/generation/enhanced_retrosynth.py`: Insilico forward synth
- Enhanced `glp_tox_panel.py`: ADC/mRNA tox endpoints
- `validation/breakthrough_metrics.py`: New benchmarks (RMSD&lt;2Å, etc.)
- `tests/test_2024_breakthroughs.py`: Unit tests
- `examples/2024_breakthroughs.py`: Usage examples
- Integrated with `DrugDiscoveryPipeline`, `PolyglotPipeline`, Ray, Unicorn CLI
