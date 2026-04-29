from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

BREAKTHROUGH_BENCHMARKS = {
    "af3_rmsd": 1.8,  # <2Å target
    "rf_pose_recovery": 75.0,  # >70%
    "adc_dar_uniformity": 0.95,
    "crispr_edit_eff": 0.90,
    "bbb_penetration": 0.70,
}

def validate_breakthroughs(results_dir: str = "outputs/validation/2024") -> Dict[str, Any]:
    """Validate 2024 breakthrough metrics against benchmarks."""
    report = {}
    path = Path(results_dir)
    for benchmark, threshold in BREAKTHROUGH_BENCHMARKS.items():
        file = path / f"{benchmark}.json"
        if file.exists():
            data = json.loads(file.read_text())
            achieved = data.get("mean", 0)
            passed = achieved <= threshold if "rmsd" in benchmark else achieved >= threshold
            report[benchmark] = {"achieved": achieved, "threshold": threshold, "passed": passed}
    return report