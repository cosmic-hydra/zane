# Modular End-to-End Drug Discovery Pipeline

## Phases
1. Data Ingestion: RDKit utils for SMILES/SDF
2. Generative: TorchDrug VAE/GNN de novo
3. Screening: DeepChem ADMET (hERG, hepatotox, BBB), filtering
4. Docking: Vina wrapper, DiffDock stub
5. Structure Analysis: CIF parsing, XRPD peak picking
6. Orchestration: `DrugDiscoveryPipeline.run_full_pipeline`

## CLI Example
```bash
python -m drug_discovery.cli drug-discovery full --target &quot;c1ccccc1&quot; --receptor 1abc.pdb --output ./results
```

Generates generated.sdf, screened.sdf, docking results in results/.

## Validation
pytest tests/test_drug_phases.py