from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

BREAKTHROUGH_BENCHMARKS = {
    &quot;af3_rmsd&quot;: 1.8,  # &lt;2Å target
    &quot;rf_pose_recovery&quot;: 75.0,  # &gt;70%
    &quot;adc_dar_uniformity&quot;: 0.95,
    &quot;crispr_edit_eff&quot;: 0.90,
    &quot;bbb_penetration&quot;: 0.70,
}

def validate_breakthroughs(results_dir: str = &quot;outputs/validation/2024&quot;) -&gt; Dict[str, Any]:
    &quot;&quot;&quot;Validate 2024 breakthrough metrics against benchmarks.&quot;&quot;&quot;
    report = {}
    path = Path(results_dir)
    for benchmark, threshold in BREAKTHROUGH_BENCHMARKS.items():
        file = path / f&quot;{benchmark}.json&quot;
        if file.exists():
            data = json.loads(file.read_text())
            achieved = data.get(&quot;mean&quot;, 0)
            passed = achieved &lt;= threshold if &quot;rmsd&quot; in benchmark else achieved &gt;= threshold
            report[benchmark] = {&quot;achieved&quot;: achieved, &quot;threshold&quot;: threshold, &quot;passed&quot;: passed}
    return report