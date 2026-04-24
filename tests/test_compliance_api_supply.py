"""Tests for compliance engine, API gateway, and supply chain modules."""

from __future__ import annotations

import json
import time

import pytest

from drug_discovery.compliance.audit_ledger import (
    AuditEntry,
    AuditLedger,
    compliance_log,
    get_default_ledger,
    sha256_hash,
)
from drug_discovery.compliance.rbac import (
    AuthenticationError,
    Permission,
    PermissionError,
    RBACManager,
    Role,
    SignatureError,
    User,
    require_permission,
    require_signature,
)
from infrastructure.api_gateway import (
    GenerateRequest,
    TaskStore,
    _execute_generation_task,
)
from external.supply_chain import (
    SupplyChainAnalyzer,
    EnamineAdapter,
    MculeAdapter,
    Synthon,
    RetrosynthesisResult,
    decompose_molecule,
)


# =========================================================================
# Module 1: Compliance -- Audit Ledger
# =========================================================================
class TestSha256Hash:
    def test_deterministic(self):
        assert sha256_hash({"a": 1}) == sha256_hash({"a": 1})

    def test_different_data(self):
        assert sha256_hash({"a": 1}) != sha256_hash({"a": 2})

    def test_string_input(self):
        h = sha256_hash("test")
        assert isinstance(h, str) and len(h) == 64


class TestAuditLedger:
    def test_log_creates_entry(self):
        ledger = AuditLedger()
        entry = ledger.log(action="test", function_name="fn", input_data={"x": 1})
        assert entry.action == "test"
        assert entry.entry_id != ""
        assert entry.entry_hash != ""

    def test_chain_integrity(self):
        ledger = AuditLedger()
        for i in range(5):
            ledger.log(action=f"action_{i}", input_data={"i": i})
        assert ledger.chain_length == 5
        assert ledger.verify_chain()

    def test_entry_verify(self):
        ledger = AuditLedger()
        entry = ledger.log(action="test", input_data="data")
        assert entry.verify()

    def test_tampered_entry_fails(self):
        ledger = AuditLedger()
        entry = ledger.log(action="test", input_data="data")
        entry.action = "tampered"
        assert not entry.verify()

    def test_genesis_hash(self):
        ledger = AuditLedger()
        assert ledger.last_hash == "0" * 64

    def test_chain_links(self):
        ledger = AuditLedger()
        e1 = ledger.log(action="first")
        e2 = ledger.log(action="second")
        assert e2.previous_hash == e1.entry_hash

    def test_get_entries_filter(self):
        ledger = AuditLedger()
        ledger.log(action="predict", user_id="alice")
        ledger.log(action="train", user_id="bob")
        ledger.log(action="predict", user_id="bob")

        assert len(ledger.get_entries(action="predict")) == 2
        assert len(ledger.get_entries(user_id="bob")) == 2
        assert len(ledger.get_entries(action="predict", user_id="bob")) == 1

    def test_export_json(self):
        ledger = AuditLedger()
        ledger.log(action="test")
        exported = ledger.export_json()
        data = json.loads(exported)
        assert len(data) == 1
        assert data[0]["action"] == "test"

    def test_as_dict(self):
        entry = AuditEntry(entry_id="x", action="test", entry_hash="abc")
        d = entry.as_dict()
        assert d["entry_id"] == "x"
        assert d["action"] == "test"


class TestComplianceLogDecorator:
    def test_decorator_logs(self):
        ledger = AuditLedger()

        @compliance_log(action="my_action", ledger=ledger)
        def my_fn(x):
            return x * 2

        result = my_fn(5)
        assert result == 10
        assert ledger.chain_length == 1
        assert ledger.get_entries()[0].action == "my_action"

    def test_decorator_captures_output(self):
        ledger = AuditLedger()

        @compliance_log(action="compute", ledger=ledger, capture_output=True)
        def add(a, b):
            return a + b

        add(3, 4)
        entry = ledger.get_entries()[0]
        assert entry.output_hash != ""


# =========================================================================
# Module 1: Compliance -- RBAC
# =========================================================================
class TestRBACManager:
    @pytest.fixture
    def rbac(self):
        mgr = RBACManager()
        mgr.create_user("alice", "Alice", "lead", "pass123")
        mgr.create_user("bob", "Bob", "scientist", "secret")
        mgr.create_user("viewer", "Viewer", "viewer", "view")
        return mgr

    def test_create_user(self, rbac):
        users = rbac.list_users()
        assert len(users) == 3

    def test_duplicate_user_raises(self, rbac):
        with pytest.raises(ValueError):
            rbac.create_user("alice", "Alice2", "viewer", "x")

    def test_authenticate_success(self, rbac):
        user = rbac.authenticate("alice", "pass123")
        assert user.token != ""
        assert user.user_id == "alice"

    def test_authenticate_bad_password(self, rbac):
        with pytest.raises(AuthenticationError):
            rbac.authenticate("alice", "wrong")

    def test_authenticate_nonexistent(self, rbac):
        with pytest.raises(AuthenticationError):
            rbac.authenticate("nobody", "x")

    def test_permission_check(self, rbac):
        user = rbac.authenticate("bob", "secret")
        rbac.check_permission(user, Permission.RUN_PREDICTION)  # should pass

        with pytest.raises(PermissionError):
            rbac.check_permission(user, Permission.APPROVE_CHECKPOINT)

    def test_lead_has_signing(self, rbac):
        user = rbac.authenticate("alice", "pass123")
        assert user.has_permission(Permission.SIGN_PREDICTION)

    def test_verify_signature(self, rbac):
        user = rbac.authenticate("alice", "pass123")
        sig = rbac.verify_signature(user, "pass123", "approving checkpoint")
        assert "signature_hash" in sig
        assert sig["user_id"] == "alice"

    def test_signature_bad_password(self, rbac):
        user = rbac.authenticate("alice", "pass123")
        with pytest.raises(SignatureError):
            rbac.verify_signature(user, "wrong", "test")

    def test_get_user_by_token(self, rbac):
        user = rbac.authenticate("alice", "pass123")
        found = rbac.get_user_by_token(user.token)
        assert found.user_id == "alice"

    def test_invalid_token(self, rbac):
        with pytest.raises(AuthenticationError):
            rbac.get_user_by_token("bogus")


class TestRBACDecorators:
    def test_require_permission_passes(self):
        role = Role.from_template("scientist")
        user = User(user_id="u1", name="Test", role=role, is_active=True)

        @require_permission(Permission.RUN_PREDICTION)
        def run(user=None):
            return "ok"

        assert run(user=user) == "ok"

    def test_require_permission_fails(self):
        role = Role.from_template("viewer")
        user = User(user_id="u2", name="Test", role=role, is_active=True)

        @require_permission(Permission.SAVE_CHECKPOINT)
        def save(user=None):
            return "ok"

        with pytest.raises(PermissionError):
            save(user=user)

    def test_require_signature_passes(self):
        role = Role.from_template("lead")
        user = User(
            user_id="u3", name="Lead",
            role=role,
            password_hash=sha256_hash(None),  # we set manually
            is_active=True,
        )
        # Set a real password hash
        import hashlib
        user.password_hash = hashlib.sha256("u3:mypass".encode()).hexdigest()

        @require_signature(reason="test export")
        def export(user=None, password=None):
            return "exported"

        assert export(user=user, password="mypass") == "exported"

    def test_require_signature_no_password(self):
        role = Role.from_template("lead")
        user = User(user_id="u4", name="Lead", role=role, is_active=True)

        @require_signature()
        def export(user=None, password=None):
            return "exported"

        with pytest.raises(SignatureError):
            export(user=user, password="")


# =========================================================================
# Module 2: API Gateway
# =========================================================================
class TestTaskStore:
    def test_create_and_get(self):
        store = TaskStore()
        store.create("t1", {"num_candidates": 10})
        task = store.get("t1")
        assert task is not None
        assert task["status"] == "pending"

    def test_update(self):
        store = TaskStore()
        store.create("t2", {})
        store.update("t2", "completed", {"result": "ok"})
        task = store.get("t2")
        assert task["status"] == "completed"

    def test_list_tasks(self):
        store = TaskStore()
        for i in range(5):
            store.create(f"task_{i}", {})
        tasks = store.list_tasks(limit=3)
        assert len(tasks) == 3

    def test_nonexistent_task(self):
        store = TaskStore()
        assert store.get("nope") is None


class TestExecuteGenerationTask:
    def test_basic_execution(self):
        from infrastructure.api_gateway import _task_store

        task_id = "test_exec_1"
        _task_store.create(task_id, {"num_candidates": 10, "top_k": 3})
        result = _execute_generation_task(task_id, {"num_candidates": 10, "top_k": 3})
        assert result["status"] == "completed"
        assert "candidates" in result
        assert result["elapsed_seconds"] > 0


class TestGenerateRequest:
    def test_defaults(self):
        req = GenerateRequest()
        assert req.num_candidates == 100
        assert req.top_k == 10
        assert req.include_admet is True


# =========================================================================
# Module 3: Supply Chain
# =========================================================================
class TestSynthon:
    def test_viability_available(self):
        s = Synthon(smiles="CCO", available=True, vendors=["enamine"], estimated_cost_usd=50.0, lead_time_days=7)
        assert s.viability > 0.8

    def test_viability_unavailable(self):
        s = Synthon(smiles="XXX", available=False)
        assert s.viability == 0.0

    def test_viability_unknown(self):
        s = Synthon(smiles="CCO", available=None)
        assert s.viability == 0.5

    def test_viability_expensive(self):
        s_cheap = Synthon(smiles="CCO", available=True, vendors=["v"], estimated_cost_usd=30.0)
        s_expensive = Synthon(smiles="CCO", available=True, vendors=["v"], estimated_cost_usd=600.0)
        assert s_cheap.viability > s_expensive.viability


class TestDecomposeMolecule:
    def test_basic_decomposition(self):
        fragments = decompose_molecule("CC(=O)Oc1ccccc1C(=O)O")
        assert len(fragments) >= 1

    def test_simple_molecule(self):
        fragments = decompose_molecule("CCO")
        assert len(fragments) >= 1


class TestVendorAdapters:
    def test_enamine_mock(self):
        adapter = EnamineAdapter()
        result = adapter.search("CCO")
        assert result is not None
        assert "available" in result

    def test_mcule_mock(self):
        adapter = MculeAdapter()
        result = adapter.search("CCO")
        assert result is not None
        assert "available" in result

    def test_deterministic_mock(self):
        adapter = EnamineAdapter()
        r1 = adapter.search("CCO")
        r2 = adapter.search("CCO")
        assert r1 == r2


class TestSupplyChainAnalyzer:
    def test_basic_analysis(self):
        analyzer = SupplyChainAnalyzer()
        result = analyzer.analyze("CCO")
        assert isinstance(result, RetrosynthesisResult)
        assert 0.0 <= result.supply_chain_score <= 1.0
        assert len(result.synthons) >= 1

    def test_complex_molecule(self):
        analyzer = SupplyChainAnalyzer()
        result = analyzer.analyze("CC(=O)Oc1ccccc1C(=O)O")
        assert result.pathway_steps >= 0
        assert result.analysis_method in ("brics", "heuristic")

    def test_batch_analysis(self):
        analyzer = SupplyChainAnalyzer()
        results = analyzer.analyze_batch(["CCO", "c1ccccc1"])
        assert len(results) == 2

    def test_as_dict(self):
        analyzer = SupplyChainAnalyzer()
        result = analyzer.analyze("CCO")
        d = result.as_dict()
        assert "supply_chain_score" in d
        assert "synthons" in d
        assert "target_smiles" in d

    def test_custom_vendors(self):
        analyzer = SupplyChainAnalyzer(vendors=[EnamineAdapter()])
        result = analyzer.analyze("CCO")
        for s in result.synthons:
            if s.available:
                assert "enamine" in s.vendors
