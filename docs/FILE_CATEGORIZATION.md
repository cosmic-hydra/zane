# ZANE File Categorization and Setup

This project now includes a standard folder layout for generated assets and runtime outputs.

## Category Layout

- Source code: `drug_discovery/`, `tests/`, `examples/`, `tools/`
- Configuration: `configs/`, `.env.example`, `pyproject.toml`, `setup.cfg`, `pytest.ini`, `ruff.toml`
- Documentation: `README.md`, `DOCUMENTATION.md`, `PROJECT_STRUCTURE.md`, `docs/`
- Data and checkpoints: `data/`, `checkpoints/`, `artifacts/`
- Operational output: `logs/`, `outputs/`

## Setup Commands

```bash
make setup-workspace
make setup-venv
```

## Notes

- The setup script is non-destructive: it creates missing folders only.
- Existing source file paths are preserved to avoid breaking imports and test references.

## 2026-03-23 Addendum

- Benchmark report outputs are tracked under `outputs/reports/`.
- Current reference artifact: `outputs/reports/scientific_benchmark_20260323.json`.
- Dashboard runtime controls are exposed via CLI options in `drug_discovery/cli.py` and rendered in `drug_discovery/dashboard.py`.
