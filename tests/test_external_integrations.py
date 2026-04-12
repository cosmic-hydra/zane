"""Tests for the elite external-repo integration adapters.

Each test validates the public API and graceful fallback behaviour when the
underlying submodule (MolecularTransformer, DiffDock, TorchDrug, OpenFold,
OpenMM, Pistachio) is not installed.
"""

import pytest

# ---------------------------------------------------------------------------
# drug_discovery/integrations.py
# ---------------------------------------------------------------------------


def test_new_integrations_registered():
    """All six new external repos must appear in the INTEGRATIONS registry."""
    from drug_discovery.integrations import INTEGRATIONS

    expected_keys = {
        "molecular_transformer",
        "diffdock",
        "torchdrug",
        "openfold",
        "openmm",
        "pistachio",
    }
    assert expected_keys.issubset(set(INTEGRATIONS.keys()))


def test_new_integration_specs_have_required_fields():
    """Each new integration spec must have a non-empty URL and submodule_path."""
    from drug_discovery.integrations import INTEGRATIONS

    for key in ("molecular_transformer", "diffdock", "torchdrug", "openfold", "openmm", "pistachio"):
        spec = INTEGRATIONS[key]
        assert spec.url.startswith("https://")
        assert spec.submodule_path is not None
        assert spec.submodule_path.startswith("external/")


def test_get_integration_status_returns_status_for_new_keys():
    """get_integration_status must not raise for any of the new keys."""
    from drug_discovery.integrations import get_integration_status

    for key in ("molecular_transformer", "diffdock", "torchdrug", "openfold", "openmm", "pistachio"):
        status = get_integration_status(key)
        assert status.key == key
        # Submodule dirs exist as empty directories → registered and present.
        assert isinstance(status.submodule_registered, bool)
        assert isinstance(status.local_checkout_present, bool)


# ---------------------------------------------------------------------------
# drug_discovery/external_tooling.py
# ---------------------------------------------------------------------------


def test_predict_reaction_returns_list_without_submodule():
    """predict_reaction must return an empty list (not raise) when submodule absent."""
    from drug_discovery.external_tooling import predict_reaction

    result = predict_reaction("CCO.CC(=O)O")
    assert isinstance(result, list)


def test_torchdrug_predict_properties_returns_dict_without_submodule():
    from drug_discovery.external_tooling import torchdrug_predict_properties

    result = torchdrug_predict_properties("CCO")
    assert isinstance(result, dict)


def test_openfold_predict_structure_returns_dict_without_submodule():
    from drug_discovery.external_tooling import openfold_predict_structure

    result = openfold_predict_structure("MKTAY")
    assert isinstance(result, dict)
    assert "sequence_length" in result


def test_openmm_minimize_energy_fallback_with_valid_smiles():
    """openmm_minimize_energy must fall back to RDKit for a valid SMILES."""
    from drug_discovery.external_tooling import openmm_minimize_energy

    result = openmm_minimize_energy("CCO", steps=5)
    assert isinstance(result, dict)
    # Either backend (openmm or rdkit_fallback) should indicate a result.
    assert "backend" in result


def test_pistachio_load_reactions_missing_file_returns_empty():
    from drug_discovery.external_tooling import pistachio_load_reactions

    result = pistachio_load_reactions("/nonexistent/path.jsonl", limit=10)
    assert isinstance(result, list)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# drug_discovery/synthesis/reaction_prediction.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reaction_predictor_predict_returns_reaction_prediction():
    from drug_discovery.synthesis.reaction_prediction import ReactionPrediction, ReactionPredictor

    predictor = ReactionPredictor(beam_size=3)
    result = predictor.predict("CCO.CC(=O)O")

    assert isinstance(result, ReactionPrediction)
    assert result.reactants == "CCO.CC(=O)O"
    assert isinstance(result.products, list)


@pytest.mark.unit
def test_reaction_predictor_predict_batch():
    from drug_discovery.synthesis.reaction_prediction import ReactionPredictor

    predictor = ReactionPredictor()
    results = predictor.predict_batch(["CCO", "CCCO"])

    assert len(results) == 2


@pytest.mark.unit
def test_reaction_predictor_validate_reaction():
    from drug_discovery.synthesis.reaction_prediction import ReactionPredictor

    predictor = ReactionPredictor()
    outcome = predictor.validate_reaction("CCO", "CC=O")

    assert "matched" in outcome
    assert "predicted_products" in outcome
    assert isinstance(outcome["matched"], bool)


@pytest.mark.unit
def test_reaction_predictor_integration_status():
    from drug_discovery.synthesis.reaction_prediction import ReactionPredictor

    status = ReactionPredictor.integration_status()
    assert status["key"] == "molecular_transformer"


# ---------------------------------------------------------------------------
# drug_discovery/physics/diffdock_adapter.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_diffdock_adapter_dock_without_submodule_returns_list():
    from drug_discovery.physics.diffdock_adapter import DiffDockAdapter

    adapter = DiffDockAdapter(fallback_to_vina=False)
    poses = adapter.dock("CCO", "test_protein")

    assert isinstance(poses, list)


@pytest.mark.unit
def test_diffdock_adapter_rank_poses_sorts_by_confidence():
    from drug_discovery.physics.diffdock_adapter import DiffDockAdapter, DockingPose

    adapter = DiffDockAdapter()
    poses = [
        DockingPose(ligand_smiles="CCO", protein_id="p", confidence=0.3),
        DockingPose(ligand_smiles="CCCO", protein_id="p", confidence=0.9),
        DockingPose(ligand_smiles="CCCCO", protein_id="p", confidence=0.6),
    ]
    ranked = adapter.rank_poses(poses)

    assert ranked[0].confidence == 0.9
    assert ranked[-1].confidence == 0.3


@pytest.mark.unit
def test_diffdock_adapter_integration_status():
    from drug_discovery.physics.diffdock_adapter import DiffDockAdapter

    status = DiffDockAdapter.integration_status()
    assert status["key"] == "diffdock"


# ---------------------------------------------------------------------------
# drug_discovery/evaluation/torchdrug_scorer.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_torchdrug_scorer_score_returns_property_score():
    torch = pytest.importorskip("torch")  # noqa: F841
    from drug_discovery.evaluation.torchdrug_scorer import PropertyScore, TorchDrugScorer

    scorer = TorchDrugScorer()
    result = scorer.score("CCO")

    assert isinstance(result, PropertyScore)
    assert result.smiles == "CCO"
    assert isinstance(result.scores, dict)
    assert 0.0 <= result.composite_score <= 1.0


@pytest.mark.unit
def test_torchdrug_scorer_rank_sorts_descending():
    pytest.importorskip("torch")
    from drug_discovery.evaluation.torchdrug_scorer import TorchDrugScorer

    scorer = TorchDrugScorer()
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
    ranked = scorer.rank(smiles_list)

    assert len(ranked) == len(smiles_list)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_torchdrug_scorer_invalid_smiles_returns_zero_composite():
    pytest.importorskip("torch")
    from drug_discovery.evaluation.torchdrug_scorer import TorchDrugScorer

    scorer = TorchDrugScorer()
    result = scorer.score("NOT_VALID_SMILES_###")

    assert result.composite_score == 0.0


@pytest.mark.unit
def test_torchdrug_scorer_integration_status():
    pytest.importorskip("torch")
    from drug_discovery.evaluation.torchdrug_scorer import TorchDrugScorer

    status = TorchDrugScorer.integration_status()
    assert status["key"] == "torchdrug"


# ---------------------------------------------------------------------------
# drug_discovery/physics/protein_structure.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_protein_structure_predictor_returns_structure():
    from drug_discovery.physics.protein_structure import ProteinStructure, ProteinStructurePredictor

    predictor = ProteinStructurePredictor()
    structure = predictor.predict("MKTAY", protein_id="test_protein")

    assert isinstance(structure, ProteinStructure)
    assert structure.protein_id == "test_protein"
    assert structure.num_residues == 5


@pytest.mark.unit
def test_protein_structure_predictor_batch():
    from drug_discovery.physics.protein_structure import ProteinStructurePredictor

    predictor = ProteinStructurePredictor()
    sequences = ["MKTAY", "ACDEF"]
    structures = predictor.predict_batch(sequences)

    assert len(structures) == 2
    assert structures[0].num_residues == len(sequences[0])


@pytest.mark.unit
def test_protein_structure_predictor_integration_status():
    from drug_discovery.physics.protein_structure import ProteinStructurePredictor

    status = ProteinStructurePredictor.integration_status()
    assert status["key"] == "openfold"


# ---------------------------------------------------------------------------
# drug_discovery/physics/openmm_adapter.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_openmm_adapter_simulate_ligand_fallback():
    from drug_discovery.physics.openmm_adapter import MDResult, OpenMMAdapter

    adapter = OpenMMAdapter()
    result = adapter.simulate_ligand("CCO", steps=100)

    assert isinstance(result, MDResult)
    assert result.smiles == "CCO"
    assert isinstance(result.success, bool)


@pytest.mark.unit
def test_openmm_adapter_stability_screen_returns_list():
    from drug_discovery.physics.openmm_adapter import OpenMMAdapter

    adapter = OpenMMAdapter()
    results = adapter.stability_screen(["CCO", "CCCO"], steps=50, threshold=0.0)

    assert isinstance(results, list)


@pytest.mark.unit
def test_openmm_adapter_integration_status():
    from drug_discovery.physics.openmm_adapter import OpenMMAdapter

    status = OpenMMAdapter.integration_status()
    assert status["key"] == "openmm"


# ---------------------------------------------------------------------------
# drug_discovery/synthesis/pistachio_datasets.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pistachio_datasets_load_empty_file(tmp_path):
    from drug_discovery.synthesis.pistachio_datasets import PistachioDatasets

    ds = PistachioDatasets()
    records = ds.load(str(tmp_path / "nonexistent.jsonl"))
    assert isinstance(records, list)
    assert len(records) == 0


@pytest.mark.unit
def test_pistachio_datasets_load_jsonl(tmp_path):
    import json

    from drug_discovery.synthesis.pistachio_datasets import PistachioDatasets

    data_file = tmp_path / "reactions.jsonl"
    records_data = [
        {"id": "r1", "reactants": ["CCO", "CC(=O)O"], "products": ["CC(=O)OCC"]},
        {"id": "r2", "reactants": ["c1ccccc1", "Br"], "products": ["c1ccccc1Br"]},
    ]
    with open(data_file, "w") as fh:
        for rec in records_data:
            fh.write(json.dumps(rec) + "\n")

    ds = PistachioDatasets()
    records = ds.load(str(data_file))

    assert len(records) == 2
    assert records[0].reaction_id == "r1"
    assert records[0].reactants == ["CCO", "CC(=O)O"]
    assert records[0].top_product == "CC(=O)OCC"


@pytest.mark.unit
def test_pistachio_datasets_build_training_dataset(tmp_path):
    import json

    from drug_discovery.synthesis.pistachio_datasets import PistachioDatasets

    data_file = tmp_path / "reactions.jsonl"
    with open(data_file, "w") as fh:
        fh.write(json.dumps({"id": "r1", "reactants": ["CCO"], "products": ["CC=O"]}) + "\n")

    ds = PistachioDatasets()
    records = ds.load(str(data_file))
    training_data = ds.build_training_dataset(records)

    assert len(training_data) == 1
    assert training_data[0]["src"] == "CCO"
    assert training_data[0]["tgt"] == "CC=O"


@pytest.mark.unit
def test_pistachio_datasets_statistics(tmp_path):
    import json

    from drug_discovery.synthesis.pistachio_datasets import PistachioDatasets

    data_file = tmp_path / "reactions.jsonl"
    with open(data_file, "w") as fh:
        for i in range(3):
            fh.write(json.dumps({"id": f"r{i}", "reactants": ["CCO"], "products": ["CC=O"]}) + "\n")

    ds = PistachioDatasets()
    records = ds.load(str(data_file))
    stats = ds.statistics(records)

    assert stats["count"] == 3


@pytest.mark.unit
def test_pistachio_datasets_integration_status():
    from drug_discovery.synthesis.pistachio_datasets import PistachioDatasets

    status = PistachioDatasets.integration_status()
    assert status["key"] == "pistachio"


# ---------------------------------------------------------------------------
# drug_discovery/pipeline.py – integrated pipeline method
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pipeline_has_integrated_pipeline_method():
    """DrugDiscoveryPipeline must expose run_integrated_pipeline."""
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    from drug_discovery.pipeline import DrugDiscoveryPipeline

    assert hasattr(DrugDiscoveryPipeline, "run_integrated_pipeline")


@pytest.mark.unit
def test_pipeline_run_integrated_pipeline_smoke(tmp_path):
    """Smoke test: integrated pipeline must not raise with minimal SMILES input."""
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    from drug_discovery.pipeline import DrugDiscoveryPipeline

    pipeline = DrugDiscoveryPipeline(
        model_type="gnn",
        device="cpu",
        cache_dir=str(tmp_path / "cache"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    results = pipeline.run_integrated_pipeline(
        smiles_list=["CCO", "CC(=O)O"],
        top_n=2,
        md_steps=50,
    )

    assert "property_scores" in results
    assert "reaction_predictions" in results
    assert "docking_poses" in results
    assert "md_simulations" in results
    assert "summary" in results
