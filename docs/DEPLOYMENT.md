# ZANE Deployment Guide

> Installation, configuration, GPU setup, Docker, and CI/CD.

---

## Quick Install

```bash
# Clone
git clone https://github.com/cosmic-hydra/zane.git
cd zane

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install core
pip install -e .

# Install with all optional dependencies
pip install -e ".[integrations]"
```

---

## Dependency Tiers

| Tier | Install | What You Get |
|------|---------|--------------|
| **Core** | `pip install -e .` | CLI, data pipeline, descriptors, evaluation |
| **ML** | `pip install torch` | All neural models (EGNN, D-MPNN, GFlowNet, etc.) |
| **Chemistry** | `pip install rdkit` | SMILES validation, fingerprints, 3D conformers |
| **Full** | `pip install -e ".[integrations]"` | All external tools + optional backends |

ZANE works with **partial dependencies** — missing packages are silently skipped.

---

## GPU Setup

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install PyTorch with CUDA (adjust CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Training with GPU

```python
from drug_discovery.training.advanced_training import AdvancedTrainer, AdvancedTrainingConfig

config = AdvancedTrainingConfig(use_amp=True)  # Mixed precision for faster GPU training
trainer = AdvancedTrainer(model, config, device="cuda")
history = trainer.fit(train_loader, val_loader)
```

---

## Environment Variables

Create `.env` from the template:

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
|----------|----------|---------|
| `GOOGLE_CSE_API_KEY` | Optional | Web search in synthesis research |
| `GOOGLE_CSE_ID` | Optional | Google Custom Search Engine ID |
| `NCBI_API_KEY` | Optional | PubChem/NCBI data access |
| `CEREBRAS_API_KEY` | Optional | Cerebras AI guidance |
| `HF_TOKEN` | Optional | Hugging Face model access (Llama) |
| `AIZYNTH_CONFIG` | Optional | AiZynthFinder config path |

---

## Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e ".[integrations]" && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000
CMD ["python", "-m", "drug_discovery.cli", "dashboard", "--static"]
```

Build and run:

```bash
docker build -t zane .
docker run -p 8000:8000 zane
```

### GPU Docker

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[integrations]"

CMD ["python", "-m", "drug_discovery.cli", "dashboard", "--static"]
```

---

## CI/CD (GitHub Actions)

The repo includes workflows in `.github/workflows/`:

- `publish-testpypi.yml` — Publish to TestPyPI on main branch changes
- `publish-pypi.yml` — Publish to PyPI on GitHub Release

### Required Secrets

Set these in GitHub → Settings → Secrets:

| Secret | Purpose |
|--------|---------|
| `TEST_PYPI_API_TOKEN` | TestPyPI upload |
| `PYPI_API_TOKEN` | PyPI upload |

### Running Tests

```bash
# All tests
pytest -q

# Just the 2026 Q2 upgrade tests
pytest tests/test_2026q2_upgrades.py -v

# With coverage
pytest --cov=drug_discovery --cov-report=html
```

---

## External Integrations

Check what's available in your environment:

```bash
python -m drug_discovery.cli integrations           # Original integrations
python -m drug_discovery.cli integrations-extended   # 10 new tools
```

### Install Optional Tools

```bash
pip install fair-esm          # ESM-2 protein language model
pip install mace-torch        # MACE equivariant potentials
pip install boltz             # Boltz protein structure
pip install deepchem          # DeepChem framework
pip install chemprop          # D-MPNN (Chemprop)
pip install torchdrug         # TorchDrug platform
```

---

## Production Checklist

- [ ] Python 3.11+ installed
- [ ] Virtual environment created
- [ ] `pip install -e .` succeeds
- [ ] `pytest -q` passes (102+ tests)
- [ ] GPU detected (if available): `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] `.env` file created from `.env.example`
- [ ] Dashboard renders: `zane dashboard --static`
