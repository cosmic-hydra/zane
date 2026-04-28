"""Tests for the Titan Architecture modules.

Covers:
1. Off-target safety veto — ToxPanelScorer raises HighToxicityVeto on a
   known hERG-blocking scaffold.
2. QM/MM metabolite screening — calculate_homo_lumo_gap returns a valid float.
3. ONNX inference latency — ONNXInferenceServer processes a request in < 150 ms.
"""

from __future__ import annotations

import time

import pytest


# ---------------------------------------------------------------------------
# 1. Off-target veto
# ---------------------------------------------------------------------------
def test_off_target_veto() -> None:
    """ToxPanelScorer raises HighToxicityVeto for a high-hERG-risk molecule.

    Verapamil (a well-known hERG blocker, logP ≈ 5.7, MW ≈ 454) is expected
    to score above the hERG safety threshold (0.5) based on its
    physicochemical descriptors.
    """
    from drug_discovery.toxicity.off_target_interactome import HighToxicityVeto, ToxPanelScorer

    # Verapamil: potent hERG blocker and CYP3A4 inhibitor
    verapamil = "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC"

    scorer = ToxPanelScorer()
    with pytest.raises(HighToxicityVeto) as exc_info:
        scorer.score_off_targets(verapamil)

    veto = exc_info.value
    assert veto.score > veto.threshold, (
        f"Expected veto.score ({veto.score:.3f}) > veto.threshold ({veto.threshold:.3f})"
    )
    assert veto.smiles == verapamil


# ---------------------------------------------------------------------------
# 2. QM/MM HOMO-LUMO gap execution
# ---------------------------------------------------------------------------
def test_qm_mm_execution() -> None:
    """calculate_homo_lumo_gap returns a valid finite float for a simple molecule."""
    import math

    from drug_discovery.toxicity.qm_mm_metabolites import ReactiveMetaboliteScreener

    screener = ReactiveMetaboliteScreener(use_pyscf=False)  # use heuristic for speed

    # Benzene: well-known narrow gap aromatic
    gap = screener.calculate_homo_lumo_gap("c1ccccc1")

    assert isinstance(gap, float), f"Expected float, got {type(gap).__name__}"
    assert not math.isnan(gap), "HOMO-LUMO gap should not be NaN for a valid SMILES"
    assert gap > 0.0, f"HOMO-LUMO gap must be positive, got {gap:.3f} eV"


# ---------------------------------------------------------------------------
# 3. ONNX inference latency
# ---------------------------------------------------------------------------
def test_onnx_latency() -> None:
    """ONNXInferenceServer processes a mock SMILES request in under 150 ms."""
    from dashboard.xai_core import ONNXInferenceServer

    server = ONNXInferenceServer()
    server.load()  # export + load model (one-time cost, excluded from timing)

    smiles = "CCO"  # ethanol

    t_start = time.perf_counter()
    result = server.predict(smiles)
    elapsed_s = time.perf_counter() - t_start

    assert elapsed_s < 0.15, (
        f"Inference took {elapsed_s * 1000:.1f} ms, expected < 150 ms"
    )
    assert "mean_score" in result
    assert "variance" in result
    assert "confidence_warning" in result
    assert isinstance(result["mean_score"], float)
    assert isinstance(result["variance"], float)
    assert isinstance(result["confidence_warning"], bool)
    assert result["variance"] >= 0.0, "Variance must be non-negative"
