from pathlib import Path

import pytest

from drug_discovery.boltzgen_adapter import BoltzGenRunner


def test_build_run_command_supports_options(tmp_path: Path):
    runner = BoltzGenRunner(executable="boltzgen", cache_dir=tmp_path / "cache", work_dir=tmp_path / "work")
    command = runner.build_run_command(
        design_spec=tmp_path / "design.yaml",
        output_dir=tmp_path / "out",
        protocol="peptide-anything",
        num_designs=20,
        budget=4,
        steps=["design", "analysis"],
        devices=2,
        reuse=False,
        config_overrides=["folding num_workers=2"],
        extra_args=["--alpha", "0.1"],
    )

    assert command[0] == "boltzgen"
    assert "--steps" in command and "design" in command and "analysis" in command
    assert "--devices" in command and "2" in command
    assert "--reuse" not in command  # disabled in call above
    assert "--cache" in command and str(tmp_path / "cache") in command
    assert "--config" in command and "folding num_workers=2" in command
    assert "--alpha" in command and "0.1" in command


def test_parse_metrics_prefers_budget_file(tmp_path: Path):
    out_dir = tmp_path / "out"
    metrics_dir = out_dir / "final_ranked_designs"
    metrics_dir.mkdir(parents=True)
    metrics_path = metrics_dir / "final_designs_metrics_3.csv"
    metrics_path.write_text("design_id,refolding_rmsd,filter_rank\n1,2.5,0\n2,1.5,1\n")

    runner = BoltzGenRunner(work_dir=tmp_path / "work")
    metrics, chosen_path = runner.parse_metrics(out_dir, budget=3)

    assert chosen_path == metrics_path
    assert metrics is not None
    assert metrics[0]["design_id"] == 1
    assert metrics[0]["refolding_rmsd"] == 2.5
    assert metrics[1]["filter_rank"] == 1


def test_summarize_metrics_orders_by_score_key():
    metrics = [
        {"design_id": "a", "score": 0.4},
        {"design_id": "b", "score": 0.1},
        {"design_id": "c", "score": 0.2},
    ]

    top = BoltzGenRunner.summarize_metrics(metrics, top_k=2, score_key="score")

    assert len(top) == 2
    assert top[0]["design_id"] == "b"  # lowest score first


def test_run_raises_when_executable_missing(tmp_path: Path):
    runner = BoltzGenRunner(executable="definitely_missing_command", work_dir=tmp_path / "work")
    with pytest.raises(FileNotFoundError):
        runner.run(design_spec=tmp_path / "design.yaml", output_dir=tmp_path / "out", parse_results=False)
