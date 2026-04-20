# Implementation Status - ZANE Project

## Current Status: Re-Benchmarking in Progress

Following a transition to the MoleculeNet benchmarking suite, historical performance metrics have been deprecated.

### Done
- [x] Standardized on Playwright for web scraping.
- [x] Implemented chemistry-aware SMILES tokenization (ChemBERTa).
- [x] Integrated MoleculeNet (BACE, BBBP, Tox21) evaluation suite.
- [x] Added weekly automated CI benchmarking.
- [x] Removed invalid synthetic benchmark artifacts.

### In Progress
- [ ] Reproducing SOTA benchmarks across 3 random seeds.
- [ ] Validating featurization-normalization pipeline integrity for GNNs.
- [ ] Publishing new verified metrics to README.

### Expected Targets (MoleculeNet)
| Dataset | Metric | Target Range |
| :--- | :--- | :--- |
| BACE | R² | 0.70 - 0.85 |
| BBBP | ROC-AUC | 0.85 - 0.92 |
| Tox21 | ROC-AUC | 0.75 - 0.82 |
