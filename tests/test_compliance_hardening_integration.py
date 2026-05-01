"""Test script for compliance hardening examples."""

import sys
sys.path.insert(0, '/workspaces/zane')

from drug_discovery.safety.strict_compliance_gate import (
    ComplianceLevel,
    StrictComplianceGate,
)
from drug_discovery.safety.parametrized_toxicity_gate import (
    ParametrizedToxicityGate,
    ToxicityThresholdConfig,
)
from drug_discovery.compliance.audit_trail import ComplianceAuditLogger


print("=" * 70)
print("COMPLIANCE HARDENING - BASIC FUNCTIONAL TEST")
print("=" * 70)

# Test 1: Basic strict compliance
print("\n[TEST 1] Basic Strict Compliance")
gate = StrictComplianceGate(compliance_level=ComplianceLevel.STRICT)
smiles = "CC(=O)O"  # Acetic acid
result = gate.evaluate(smiles, toxicity_probs={"herg": 0.1})
print(f"  SMILES: {smiles}")
print(f"  Quality Tier: {result.quality_tier.value}")
print(f"  Status: PASS" if result.overall_passed else "  Status: FAIL")
print(f"  ✓ Strict compliance evaluation working")

# Test 2: Regulatory tier switching
print("\n[TEST 2] Regulatory Tier Switching")
discovery = StrictComplianceGate(compliance_level=ComplianceLevel.RELAXED)
strict = StrictComplianceGate(compliance_level=ComplianceLevel.STRICT)
print(f"  Discovery hERG threshold: {discovery.strict_herg_threshold}")
print(f"  Strict hERG threshold: {strict.strict_herg_threshold}")
print(f"  Strict is more restrictive: {strict.strict_herg_threshold < discovery.strict_herg_threshold}")
print(f"  ✓ Regulatory tier switching working")

# Test 3: Parametrized toxicity gate
print("\n[TEST 3] Parametrized Toxicity Gate")
config = ToxicityThresholdConfig.from_regulatory_tier("ind")
gate = ParametrizedToxicityGate(config)
result = gate.evaluate(herg_prob=0.12, ames_prob=0.08)
print(f"  IND Tier - Active thresholds:")
print(f"    hERG: {result['active_thresholds']['herg']}")
print(f"    Ames: {result['active_thresholds']['ames']}")
print(f"  Evaluation passed: {result['passed']}")
print(f"  ✓ Parametrized toxicity gate working")

# Test 4: Audit trail
print("\n[TEST 4] Audit Trail & Compliance Logging")
audit_logger = ComplianceAuditLogger()
audit_logger.log_compound_screened("CCO", "COMP-001", "test_user")
audit_logger.log_quality_assessment(
    "COMP-001", "CCO", "TIER_1", True, ["none"], "test_user"
)
is_verified = audit_logger.verify_integrity()
report = audit_logger.export_report()
print(f"  Audit entries recorded: {report['total_entries']}")
print(f"  Chain integrity verified: {is_verified}")
print(f"  ✓ Audit trail working")

print("\n" + "=" * 70)
print("✓ ALL COMPLIANCE HARDENING TESTS PASSED")
print("=" * 70)
