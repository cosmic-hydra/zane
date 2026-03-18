# ZANE: Autonomous AI Drug Discovery Platform

A production-minded, research-first platform for molecular intelligence workflows, from data acquisition and model training to simulation-aware candidate prioritization and AI-assisted decision support.

## Executive Summary

ZANE unifies the core layers of computational drug discovery:

- Molecular data ingestion and harmonization
- Learning pipelines (GNN, Transformer, Ensemble)
- Property and ADMET assessment
- Physics-informed simulation hooks
- Synthesis feasibility tooling
- Terminal-native operational dashboard
- Meta Llama-powered AI support

The repository is intended for scientific teams that need a repeatable, extensible, and operator-friendly environment for accelerating discovery iterations.

## Table of Contents

1. Platform Scope
2. Key Capabilities
3. Architecture
4. Repository Layout
5. Installation
6. Quick Start
7. Operations Guide
8. AI Support (Meta Llama)
9. Dashboard Operations
10. Workflow Blueprints
11. Quality and CI/CD
12. Security and Responsible Use
13. Troubleshooting
14. Contribution Standards
15. License

## 1. Platform Scope

ZANE is designed to support the full loop of computational triage:

1. Gather molecules from multiple sources.
2. Convert molecules to model-ready representations.
3. Train and evaluate predictive models.
4. Estimate ADMET and related quality signals.
5. Incorporate simulation evidence where available.
6. Rank and export candidates for expert review.

## 2. Key Capabilities

### Data Intelligence

- Multi-source collection pipelines (PubChem, ChEMBL, approved drugs)
- Deduplicated dataset merging and caching
- Structured molecular featurization workflows

### Modeling

- Graph Neural Networks for structure-aware learning
- Transformer pipelines for sequence/fingerprint modeling
- Ensemble mode for robust aggregate scoring

### Evaluation and Ranking

- Property prediction support
- ADMET and quality indicators (including QED and SA)
- Candidate-level result aggregation for triage

### Operations

- Unified CLI command surface
- Rich terminal dashboard for run visibility
- Artifact-friendly execution model

### AI Assistance

- Meta Llama-backed assistant for strategy and interpretation support
- Context-injected prompting for research workflows

## 3. Architecture

ZANE follows a layered architecture for maintainability and extension safety:

- Interface Layer: CLI and terminal dashboard
- Orchestration Layer: pipeline and agent coordination
- Intelligence Layer: models, predictors, evaluators, optimizers
- Science Layer: docking, molecular dynamics, retrosynthesis
- Data Layer: collection, featurization, datasets
- Platform Layer: tests, linting, CI/CD, packaging

### Runtime Flow

1. Collect and merge molecular inputs.
2. Build train/test datasets.
3. Train selected model architecture.
4. Evaluate model behavior and prediction quality.
5. Score and rank candidate molecules.
6. Monitor via dashboard and export artifacts.

## 4. Repository Layout

Primary modules:

- drug_discovery/data: collection, featurization, dataset logic
- drug_discovery/models: GNN, Transformer, Ensemble, equivariant components
- drug_discovery/training: training loop and closed-loop utilities
- drug_discovery/evaluation: property/ADMET prediction and model evaluation
- drug_discovery/physics: docking and MD simulation utilities
- drug_discovery/synthesis: retrosynthesis and feasibility support
- drug_discovery/optimization: Bayesian and multi-objective optimization
- drug_discovery/agents: multi-agent orchestration framework
- drug_discovery/dashboard.py: terminal dashboard implementation
- drug_discovery/ai_support.py: Meta Llama integration

## 5. Installation

### Standard Setup

```bash
pip install -r requirements.txt
```

### Recommended Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional GPU Validation

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Optional Multi-Language Accelerator (Go)

ZANE includes an optional Go backend for faster web search retrieval in synthesis workflows.

Build the Go helper:

```bash
make build-go-fastsearch
export ZANE_GO_SEARCH_BIN="$PWD/tools/bin/zane-fastsearch"
```

This binary is used by synthesis research flows as a fast fallback when Google CSE is not configured.

## 6. Quick Start

### Train a Baseline Model

```bash
python -m drug_discovery.cli train --model transformer --epochs 20 --batch-size 32
```

### Run Property Prediction

```bash
python -m drug_discovery.cli predict "CC(=O)OC1=CC=CC=C1C(=O)O" \
  --model gnn \
  --checkpoint ./checkpoints/gnn_model.pt
```

### Run ADMET Check

```bash
python -m drug_discovery.cli admet "CC(=O)OC1=CC=CC=C1C(=O)O"
```

## 7. Operations Guide

### Data Collection

```bash
python -m drug_discovery.cli collect --sources pubchem chembl --limit 500
```

### Dashboard (Static)

```bash
python -m drug_discovery.cli dashboard --static
```

### Dashboard (Live)

```bash
python -m drug_discovery.cli dashboard --refresh 1.0 --iterations 60
```

### Synthesis Research (Internet + AI)

```bash
python -m drug_discovery.cli synthesis-research "CCO" --target EGFR --max-results 5
```

Read linked resources (including PDFs) with controlled depth:

```bash
python -m drug_discovery.cli synthesis-research "CCO" \
  --max-results 5 \
  --max-resource-reads 3
```

Disable URL/PDF reading if needed:

```bash
python -m drug_discovery.cli synthesis-research "CCO" --no-resource-read
```

Offline-safe mode:

```bash
python -m drug_discovery.cli synthesis-research "CCO" --no-internet --no-ai
```

### Operational Notes

- Keep checkpoints versioned by experiment intent.
- Use consistent splits for model-to-model comparisons.
- Persist run outputs under dedicated artifact directories.

## 8. AI Support (Meta Llama)

### Basic Command

```bash
python -m drug_discovery.cli assist "Summarize risk factors in the current candidate shortlist"
```

### Advanced Command with Context

```bash
python -m drug_discovery.cli assist "Draft next assay plan" \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --context "Top candidates: Caffeine, Warfarin" \
  --max-new-tokens 300 \
  --temperature 0.7 \
  --top-p 0.9
```

### Access Requirements

- Meta Llama checkpoints may be gated.
- Ensure model access is approved in your Hugging Face account.
- Provide an auth token in environment variables (for example, HF_TOKEN).

## 9. Dashboard Operations

The terminal dashboard is optimized for operator awareness during active runs.

Displayed signal groups:

- Run metadata and model mode
- KPI panel (throughput, hit rate, quality metrics)
- Training monitor (epoch and loss behavior)
- Candidate queue preview
- Alerts and operational status

## 10. Workflow Blueprints

### Baseline Discovery Workflow

1. Collect 200 to 1000 molecules.
2. Train a transformer baseline.
3. Evaluate and shortlist by quality metrics.
4. Run ADMET checks on the shortlist.
5. Review status in dashboard and export outputs.

### Comparative Workflow

1. Train GNN and transformer with aligned splits.
2. Compare metrics and top-k overlap.
3. Use ensemble mode for consensus ranking.

### Human-in-the-Loop Workflow

1. Export top candidates.
2. Use AI support to draft test priorities.
3. Finalize shortlist with domain experts.

## 11. Quality and CI/CD

Recommended pre-push checks:

```bash
python -m pytest -q
python -m ruff check .
python -m black --check .
```

Expected quality posture:

- Tests must pass for core modules.
- Lint and format checks should be clean.
- User-facing behavior changes should be documented.

## 12. Security and Responsible Use

This repository is intended for research and decision support.

- Do not treat outputs as direct clinical recommendations.
- Validate predictions experimentally.
- Apply governance and provenance controls for data and results.
- Ensure expert review before any high-impact downstream use.

## 13. Troubleshooting

### Llama Model Fails to Load

Potential causes:

- Missing or invalid Hugging Face token
- Access not granted for selected model
- Restricted network environment

### Training Instability

Actions:

- Lower learning rate
- Reduce batch size
- Inspect data quality and target distributions

### CLI Runtime Errors

Actions:

- Confirm virtual environment activation
- Reinstall dependencies
- Re-run with explicit model/checkpoint arguments

## 14. Contribution Standards

Recommended development flow:

1. Create a focused branch.
2. Implement minimal, scoped changes.
3. Run tests and static checks locally.
4. Update documentation for behavior changes.
5. Open PR with validation evidence.

## 15. License

CC0 1.0 Universal
