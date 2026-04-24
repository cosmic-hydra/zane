"""Unit tests for the NVIDIA NIM LLM drug design integration.

All tests run without network access and without a real NVIDIA API key.
External calls are intercepted via monkeypatching.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from drug_discovery.nvidia_nim.chem_databases import ChemicalDatabaseHub, DatabaseRecord
from drug_discovery.nvidia_nim.client import NvidiaNIMClient, NvidiaNIMConfig
from drug_discovery.nvidia_nim.llm_drug_designer import (
    DrugDesignRequest,
    DrugDesignResult,
    NvidiaLLMDrugDesigner,
)

# ---------------------------------------------------------------------------
# NvidiaNIMConfig
# ---------------------------------------------------------------------------


def test_nim_config_defaults():
    config = NvidiaNIMConfig()
    assert config.max_tokens == 1024
    assert config.temperature == 0.4
    assert config.top_p == 0.9


def test_nim_config_api_key_from_env(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "test-key-123")
    config = NvidiaNIMConfig()
    assert config.api_key == "test-key-123"


# ---------------------------------------------------------------------------
# NvidiaNIMClient – availability
# ---------------------------------------------------------------------------


def test_nim_client_unavailable_without_key():
    config = NvidiaNIMConfig(api_key="")
    client = NvidiaNIMClient(config)
    assert client.is_available() is False


def test_nim_client_available_with_key_and_openai(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "real-key")
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def mock_find_spec(name):
        if name == "openai":
            return MagicMock()
        return original_find_spec(name)

    with patch("importlib.util.find_spec", side_effect=mock_find_spec):
        config = NvidiaNIMConfig(api_key="real-key")
        client = NvidiaNIMClient(config)
        assert client.is_available() is True


# ---------------------------------------------------------------------------
# NvidiaNIMClient – generate_molecules (no key)
# ---------------------------------------------------------------------------


def test_generate_molecules_returns_error_when_unavailable():
    client = NvidiaNIMClient(NvidiaNIMConfig(api_key=""))
    result = client.generate_molecules("Design 5 EGFR inhibitors", num=5)
    assert result["success"] is False
    assert result["molecules"] == []
    assert result["error"] is not None
    assert "NVIDIA_NIM_API_KEY" in result["error"] or "NVIDIA NIM not available" in result["error"]


# ---------------------------------------------------------------------------
# NvidiaNIMClient – generate_molecules (mocked OpenAI)
# ---------------------------------------------------------------------------


def _make_mock_completion(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def test_generate_molecules_parses_smiles():
    raw_response = (
        "Here are 3 drug-like EGFR inhibitors:\n"
        "SMILES: CC1=CC=C(C=C1)NC(=O)C2=CC=CC=C2\n"
        "Reason: Benzamide scaffold with selectivity.\n"
        "SMILES: c1ccc(cc1)NC2=NC=CC=C2\n"
        "Reason: Aminopyridine motif.\n"
        "SMILES: CCc1ccc(cc1)C(=O)O\n"
        "Reason: Simple acid fragment.\n"
    )
    mock_completion = _make_mock_completion(raw_response)

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = mock_completion

    config = NvidiaNIMConfig(api_key="test-key")
    client = NvidiaNIMClient(config)
    client._client = mock_openai_client  # inject mock directly

    # Patch is_available to return True
    with patch.object(client, "is_available", return_value=True):
        result = client.generate_molecules("EGFR inhibitors", num=3)

    assert result["success"] is True
    assert len(result["molecules"]) == 3
    assert "CC1=CC=C" in result["molecules"][0]


# ---------------------------------------------------------------------------
# NvidiaNIMClient – _extract_smiles
# ---------------------------------------------------------------------------


def test_extract_smiles_from_labeled_lines():
    text = (
        "SMILES: CC(=O)Oc1ccccc1C(=O)O\n"
        "Reason: Aspirin scaffold.\n"
        "SMILES: c1ccc2ncccc2c1\n"
        "Reason: Quinoline core.\n"
    )
    smiles = NvidiaNIMClient._extract_smiles(text)
    assert "CC(=O)Oc1ccccc1C(=O)O" in smiles
    assert "c1ccc2ncccc2c1" in smiles


def test_extract_smiles_fallback_bare_lines():
    text = "CC(=O)Oc1ccccc1\nc1ccccc1\nThis is a sentence."
    smiles = NvidiaNIMClient._extract_smiles(text)
    # bare SMILES-like tokens should be captured
    assert any("CC" in s or "c1" in s for s in smiles)


def test_extract_smiles_strips_trailing_punctuation():
    text = "SMILES: c1ccccc1C(=O)O,"
    smiles = NvidiaNIMClient._extract_smiles(text)
    assert smiles[0] == "c1ccccc1C(=O)O"


def test_extract_smiles_case_insensitive():
    text = "smiles: CC(N)=O\nSMILES: c1ccccc1"
    smiles = NvidiaNIMClient._extract_smiles(text)
    assert "CC(N)=O" in smiles


# ---------------------------------------------------------------------------
# NvidiaNIMClient – explain_molecule
# ---------------------------------------------------------------------------


def test_explain_molecule_unavailable():
    client = NvidiaNIMClient(NvidiaNIMConfig(api_key=""))
    result = client.explain_molecule("CC(=O)Oc1ccccc1C(=O)O")
    assert result["success"] is False
    assert result["explanation"] == ""


def test_explain_molecule_success():
    mock_completion = _make_mock_completion("Aspirin is a classic NSAID with COX inhibition.")
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = mock_completion

    config = NvidiaNIMConfig(api_key="test-key")
    client = NvidiaNIMClient(config)
    client._client = mock_openai_client

    with patch.object(client, "is_available", return_value=True):
        result = client.explain_molecule("CC(=O)Oc1ccccc1C(=O)O")

    assert result["success"] is True
    assert "COX" in result["explanation"]


# ---------------------------------------------------------------------------
# DatabaseRecord
# ---------------------------------------------------------------------------


def test_database_record_to_context_line_full():
    rec = DatabaseRecord(
        smiles="c1ccccc1",
        name="Benzene",
        source="PubChem",
        properties={"MW": 78.11},
        target_info="aromatic",
    )
    line = rec.to_context_line()
    assert "Benzene" in line
    assert "c1ccccc1" in line
    assert "PubChem" in line
    assert "MW=78.11" in line


def test_database_record_to_context_line_minimal():
    rec = DatabaseRecord(source="ZINC20")
    line = rec.to_context_line()
    assert "ZINC20" in line


def test_database_record_properties_capped_at_five():
    props = {f"p{i}": i for i in range(10)}
    rec = DatabaseRecord(source="test", properties=props)
    line = rec.to_context_line()
    # Only first 5 properties should appear
    count = sum(1 for i in range(5) if f"p{i}={i}" in line)
    assert count == 5


# ---------------------------------------------------------------------------
# ChemicalDatabaseHub – build_context_string
# ---------------------------------------------------------------------------


def test_build_context_string_basic():
    hub = ChemicalDatabaseHub()
    fake_results = {
        "pubchem": [
            DatabaseRecord(smiles="c1ccccc1", name="Benzene", source="PubChem"),
        ],
        "chembl": [
            DatabaseRecord(smiles="CC(=O)O", name="Acetic acid", source="ChEMBL"),
        ],
    }
    context = hub.build_context_string(fake_results)
    assert "PUBCHEM" in context
    assert "Benzene" in context
    assert "CHEMBL" in context
    assert "Acetic acid" in context


def test_build_context_string_empty():
    hub = ChemicalDatabaseHub()
    context = hub.build_context_string({})
    assert "No database context available." in context


def test_build_context_string_respects_max_records():
    hub = ChemicalDatabaseHub()
    records = [DatabaseRecord(name=f"mol{i}", source="PubChem") for i in range(30)]
    context = hub.build_context_string({"pubchem": records}, max_records=5)
    # At most 5 records should appear
    count = context.count("mol")
    assert count <= 5


# ---------------------------------------------------------------------------
# ChemicalDatabaseHub – search_all with mocked individual searches
# ---------------------------------------------------------------------------


def test_search_all_aggregates_results():
    hub = ChemicalDatabaseHub()
    fake_record = DatabaseRecord(smiles="CC", name="Ethane", source="PubChem")

    with (
        patch.object(hub, "search_pubchem", return_value=[fake_record]),
        patch.object(hub, "search_chembl", return_value=[]),
        patch.object(hub, "search_zinc", return_value=[]),
        patch.object(hub, "search_bindingdb", return_value=[]),
        patch.object(hub, "search_uniprot", return_value=[]),
        patch.object(hub, "search_pdb", return_value=[]),
        patch.object(hub, "search_kegg", return_value=[]),
        patch.object(hub, "search_hmdb", return_value=[]),
    ):
        results = hub.search_all("ethane", limit_per_source=1)

    assert "pubchem" in results
    assert results["pubchem"][0].name == "Ethane"
    # Empty sources should not appear
    assert "chembl" not in results


def test_search_all_handles_source_exception_gracefully():
    hub = ChemicalDatabaseHub()

    def explode(_q, _lim):
        raise RuntimeError("network error")

    with (
        patch.object(hub, "search_pubchem", side_effect=explode),
        patch.object(hub, "search_chembl", return_value=[DatabaseRecord(name="X", source="ChEMBL")]),
        patch.object(hub, "search_zinc", return_value=[]),
        patch.object(hub, "search_bindingdb", return_value=[]),
        patch.object(hub, "search_uniprot", return_value=[]),
        patch.object(hub, "search_pdb", return_value=[]),
        patch.object(hub, "search_kegg", return_value=[]),
        patch.object(hub, "search_hmdb", return_value=[]),
    ):
        results = hub.search_all("test")

    # pubchem exploded — should be absent; chembl present
    assert "pubchem" not in results
    assert "chembl" in results


def test_search_all_unknown_source_is_skipped():
    hub = ChemicalDatabaseHub()
    with patch.object(hub, "search_pubchem", return_value=[]):
        results = hub.search_all("test", sources=["pubchem", "nonexistent_db"])
    assert "nonexistent_db" not in results


# ---------------------------------------------------------------------------
# NvidiaLLMDrugDesigner – design
# ---------------------------------------------------------------------------


def _make_designer_with_mocked_nim(molecules: list[str], success: bool = True) -> NvidiaLLMDrugDesigner:
    """Return a designer where the NIM client always returns the given molecules."""
    mock_nim = MagicMock(spec=NvidiaNIMClient)
    mock_nim.is_available.return_value = success
    mock_nim.generate_molecules.return_value = {
        "success": success,
        "molecules": molecules,
        "reasoning": "mock reasoning",
        "raw_response": "mock",
        "error": None if success else "mock error",
    }
    mock_db = MagicMock(spec=ChemicalDatabaseHub)
    mock_db.search_all.return_value = {}
    mock_db.build_context_string.return_value = ""

    return NvidiaLLMDrugDesigner(nim_client=mock_nim, db_hub=mock_db)


def test_designer_design_success():
    designer = _make_designer_with_mocked_nim(["c1ccccc1", "CC(=O)O"])
    result = designer.design(DrugDesignRequest(target="EGFR inhibitor", num=2))

    assert result.success is True
    assert result.molecules == ["c1ccccc1", "CC(=O)O"]
    assert result.error is None


def test_designer_design_nim_unavailable():
    designer = _make_designer_with_mocked_nim([], success=False)
    designer.nim.is_available.return_value = False
    result = designer.design(DrugDesignRequest(target="test", num=2))

    assert result.success is False
    assert "NVIDIA NIM" in (result.error or "")


def test_designer_design_llm_failure():
    mock_nim = MagicMock(spec=NvidiaNIMClient)
    mock_nim.is_available.return_value = True
    mock_nim.generate_molecules.return_value = {
        "success": False,
        "molecules": [],
        "reasoning": "",
        "raw_response": "",
        "error": "API quota exceeded",
    }
    mock_db = MagicMock(spec=ChemicalDatabaseHub)
    mock_db.search_all.return_value = {}
    mock_db.build_context_string.return_value = ""

    designer = NvidiaLLMDrugDesigner(nim_client=mock_nim, db_hub=mock_db)
    result = designer.design(DrugDesignRequest(target="test"))

    assert result.success is False
    assert result.error == "API quota exceeded"


def test_designer_combine_empty_input():
    designer = _make_designer_with_mocked_nim([])
    result = designer.combine([])
    assert result.success is False
    assert result.error is not None


def test_designer_combine_calls_design():
    designer = _make_designer_with_mocked_nim(["c1ccccc1"])
    result = designer.combine(["CC", "c1ccccc1"], strategy="fragment_merge", num=3)
    assert result.success is True


def test_designer_optimize_calls_design():
    designer = _make_designer_with_mocked_nim(["CC(N)=O"])
    result = designer.optimize("CC(=O)O", objectives=["improve QED"], num=2)
    assert result.success is True


# ---------------------------------------------------------------------------
# NvidiaLLMDrugDesigner – MolMIM integration
# ---------------------------------------------------------------------------


def test_designer_molmim_called_when_requested():
    mock_nim = MagicMock(spec=NvidiaNIMClient)
    mock_nim.is_available.return_value = True
    mock_nim.generate_molecules.return_value = {
        "success": True,
        "molecules": ["c1ccccc1"],
        "reasoning": "ok",
        "raw_response": "ok",
        "error": None,
    }
    mock_nim.generate_molmim.return_value = {
        "success": True,
        "molecules": ["CC(N)=O"],
        "raw": {},
        "error": None,
    }
    mock_db = MagicMock(spec=ChemicalDatabaseHub)
    mock_db.search_all.return_value = {}
    mock_db.build_context_string.return_value = ""

    designer = NvidiaLLMDrugDesigner(nim_client=mock_nim, db_hub=mock_db)
    result = designer.design(
        DrugDesignRequest(target="test", seed_smiles=["c1ccccc1"], use_molmim=True)
    )

    mock_nim.generate_molmim.assert_called_once()
    assert "CC(N)=O" in result.molmim_molecules


def test_designer_molmim_failure_is_a_warning_not_error():
    mock_nim = MagicMock(spec=NvidiaNIMClient)
    mock_nim.is_available.return_value = True
    mock_nim.generate_molecules.return_value = {
        "success": True,
        "molecules": ["c1ccccc1"],
        "reasoning": "",
        "raw_response": "",
        "error": None,
    }
    mock_nim.generate_molmim.return_value = {
        "success": False,
        "molecules": [],
        "raw": {},
        "error": "MolMIM 503",
    }
    mock_db = MagicMock(spec=ChemicalDatabaseHub)
    mock_db.search_all.return_value = {}
    mock_db.build_context_string.return_value = ""

    designer = NvidiaLLMDrugDesigner(nim_client=mock_nim, db_hub=mock_db)
    result = designer.design(
        DrugDesignRequest(target="test", seed_smiles=["c1ccccc1"], use_molmim=True)
    )

    assert result.success is True
    assert any("MolMIM failed" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# DrugDesignResult – as_dict
# ---------------------------------------------------------------------------


def test_drug_design_result_as_dict_keys():
    result = DrugDesignResult(
        success=True,
        molecules=["CC"],
        reasoning="ok",
        db_context_summary="1 record",
        molmim_molecules=[],
        warnings=[],
        error=None,
    )
    d = result.as_dict()
    assert set(d.keys()) == {
        "success",
        "molecules",
        "reasoning",
        "db_context_summary",
        "molmim_molecules",
        "warnings",
        "error",
    }


# ---------------------------------------------------------------------------
# NvidiaChemLLMBackend (via generation.backends)
# ---------------------------------------------------------------------------


def test_nvidia_backend_unavailable_when_no_key():
    from drug_discovery.generation.backends import NvidiaChemLLMBackend

    backend = NvidiaChemLLMBackend()
    # Without API key the inner NvidiaNIMClient is not available
    with patch(
        "drug_discovery.nvidia_nim.client.NvidiaNIMClient.is_available",
        return_value=False,
    ):
        assert backend.is_available() is False


def test_nvidia_backend_generate_failure_message():
    from drug_discovery.generation.backends import NvidiaChemLLMBackend

    backend = NvidiaChemLLMBackend()
    with patch.object(backend, "is_available", return_value=False):
        result = backend.generate("test", num=3)

    assert result.success is False
    assert result.backend == "nvidia-nim"
    assert result.molecules == []
    assert any("NVIDIA_NIM_API_KEY" in w or "openai" in w for w in result.warnings)


def test_nvidia_backend_generate_success():
    from drug_discovery.generation.backends import NvidiaChemLLMBackend

    backend = NvidiaChemLLMBackend(target="EGFR")

    mock_designer = MagicMock()
    mock_result = DrugDesignResult(
        success=True,
        molecules=["c1ccccc1", "CC(=O)O"],
        reasoning="mock",
        db_context_summary="2 records",
        molmim_molecules=[],
        warnings=[],
        error=None,
    )
    mock_designer.design.return_value = mock_result

    with (
        patch.object(backend, "is_available", return_value=True),
        patch("drug_discovery.nvidia_nim.llm_drug_designer.NvidiaNIMClient") as _mock_nim,
        patch("drug_discovery.nvidia_nim.llm_drug_designer.ChemicalDatabaseHub") as _mock_db,
    ):
        _mock_nim.return_value.is_available.return_value = True
        _mock_nim.return_value.generate_molecules.return_value = {
            "success": True,
            "molecules": ["c1ccccc1", "CC(=O)O"],
            "reasoning": "mock",
            "raw_response": "mock",
            "error": None,
        }
        _mock_db.return_value.search_all.return_value = {}
        _mock_db.return_value.build_context_string.return_value = ""

        result = backend.generate("design EGFR inhibitors", num=2)

    assert result.success is True
    assert result.backend == "nvidia-nim"


# ---------------------------------------------------------------------------
# Integrations registry
# ---------------------------------------------------------------------------


def test_nvidia_nim_in_integrations_registry():
    from drug_discovery.integrations import INTEGRATIONS

    assert "nvidia_nim" in INTEGRATIONS
    spec = INTEGRATIONS["nvidia_nim"]
    assert "openai" in spec.python_modules
    assert spec.url == "https://build.nvidia.com/explore/discover"
