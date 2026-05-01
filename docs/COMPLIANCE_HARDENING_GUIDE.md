"""Integration Guide: Strict Compliance and Quality Hardening

This guide demonstrates how to use the new compliance and quality modules:
- StrictComplianceGate (strict_compliance_gate.py)
- ParametrizedToxicityGate (parametrized_toxicity_gate.py)
- ComplianceAuditTrail (audit_trail.py)

These modules replace hardcoded thresholds with parametrized configurations,
enabling regulatory tier switching without code changes.
"""

from drug_discovery.safety.strict_compliance_gate import (
    ComplianceLevel,
    QualityTier,
    RiskFactor,
    StrictComplianceGate,
    evaluate_batch_with_strict_compliance,
)
from drug_discovery.safety.parametrized_toxicity_gate import (
    ParametrizedToxicityGate,
    ToxicityThresholdConfig,
)
from drug_discovery.compliance.audit_trail import (
    ComplianceAuditLogger,
    AuditTrail,
    AuditEventType,
)


# ============================================================================
# EXAMPLE 1: Basic Strict Compliance Evaluation
# ============================================================================

def example_basic_compliance():
    """Evaluate single compound with strict compliance."""
    print("\n=== EXAMPLE 1: Basic Compliance Evaluation ===\n")
    
    # Create gate with STRICT compliance level (IND-ready)
    gate = StrictComplianceGate(
        compliance_level=ComplianceLevel.STRICT,
    )
    
    # Evaluate a compound
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    toxicity_probs = {
        "herg": 0.08,
        "ames": 0.02,
        "hepatotox": 0.1,
    }
    
    assessment = gate.evaluate(smiles, toxicity_probs=toxicity_probs, user_id="chemist_001")
    
    print(f"SMILES            : {assessment.smiles}")
    print(f"Quality Tier      : {assessment.quality_tier.value}")
    print(f"Overall Passed    : {assessment.overall_passed}")
    print(f"Confidence Score  : {assessment.confidence_score:.2f}")
    print(f"Recommendation    : {assessment.recommendation}")
    print(f"Audit ID          : {assessment.audit_id}")
    print(f"Risk Factors      : {[rf.value for rf in assessment.risk_factors]}")
    
    print("\nCompliance Checks:")
    for check in assessment.compliance_checks:
        status = "✓" if check.passed else "✗"
        print(f"  {status} {check.check_name}")
        print(f"      {check.message}")
        if check.remediation:
            print(f"      Remediation: {check.remediation}")


# ============================================================================
# EXAMPLE 2: Regulatory Tier Switching
# ============================================================================

def example_regulatory_tiers():
    """Demonstrate switching between regulatory compliance tiers."""
    print("\n=== EXAMPLE 2: Regulatory Tier Switching ===\n")
    
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    herg_prob = 0.18
    
    # Discovery phase (relaxed)
    discovery_gate = StrictComplianceGate(compliance_level=ComplianceLevel.RELAXED)
    discovery_result = discovery_gate.evaluate(smiles, toxicity_probs={"herg": herg_prob})
    print(f"Discovery Phase (hERG threshold={discovery_gate.strict_herg_threshold}):")
    print(f"  Result: {'PASSED' if discovery_result.overall_passed else 'FAILED'}")
    
    # Lead optimization (standard)
    standard_gate = StrictComplianceGate(compliance_level=ComplianceLevel.STANDARD)
    standard_result = standard_gate.evaluate(smiles, toxicity_probs={"herg": herg_prob})
    print(f"\nLead Optimization (hERG threshold={standard_gate.strict_herg_threshold}):")
    print(f"  Result: {'PASSED' if standard_result.overall_passed else 'FAILED'}")
    
    # IND submission (strict)
    ind_gate = StrictComplianceGate(compliance_level=ComplianceLevel.STRICT)
    ind_result = ind_gate.evaluate(smiles, toxicity_probs={"herg": herg_prob})
    print(f"\nIND Submission (hERG threshold={ind_gate.strict_herg_threshold}):")
    print(f"  Result: {'PASSED' if ind_result.overall_passed else 'FAILED'}")
    
    # NDA submission (hardened)
    nda_gate = StrictComplianceGate(compliance_level=ComplianceLevel.HARDENED)
    nda_result = nda_gate.evaluate(smiles, toxicity_probs={"herg": herg_prob})
    print(f"\nNDA Submission (hERG threshold={nda_gate.strict_herg_threshold}):")
    print(f"  Result: {'PASSED' if nda_result.overall_passed else 'FAILED'}")


# ============================================================================
# EXAMPLE 3: Parametrized Toxicity Gate
# ============================================================================

def example_parametrized_toxicity_gate():
    """Demonstrate parametrized toxicity evaluation."""
    print("\n=== EXAMPLE 3: Parametrized Toxicity Gate ===\n")
    
    # Create config for IND submission
    config = ToxicityThresholdConfig.from_regulatory_tier("ind")
    gate = ParametrizedToxicityGate(config)
    
    # Evaluate molecule
    result = gate.evaluate(
        herg_prob=0.15,
        ames_prob=0.08,
        hepatotox_prob=0.12,
        logp=2.5,
        tpsa=75.0,
        mw=280.0,
        rotatable_bonds=5,
        confidence=0.8,
    )
    
    print(f"Passed: {result['passed']}")
    print(f"Active Thresholds:")
    for endpoint, threshold in result['active_thresholds'].items():
        print(f"  {endpoint}: {threshold}")
    
    if result['reasons']:
        print(f"\nRejection Reasons:")
        for reason in result['reasons']:
            print(f"  - {reason}")
    
    if result['warnings']:
        print(f"\nWarnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")


# ============================================================================
# EXAMPLE 4: Batch Evaluation with Compliance Tracking
# ============================================================================

def example_batch_evaluation():
    """Evaluate multiple compounds and generate compliance report."""
    print("\n=== EXAMPLE 4: Batch Evaluation ===\n")
    
    smiles_list = [
        "CC(=O)O",  # Acetic acid
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CCO",  # Ethanol
    ]
    
    result = evaluate_batch_with_strict_compliance(
        smiles_list,
        compliance_level=ComplianceLevel.STRICT,
    )
    
    print(f"Batch Compliance Summary:")
    print(f"  Total Evaluated: {result['total_evaluated']}")
    print(f"  Passed: {result['passed']}")
    print(f"  Rejected: {result['rejected']}")
    print(f"  Pass Rate: {result['pass_rate']:.1%}")
    
    if result['critical_issues']:
        print(f"\n  Critical Issues:")
        for issue in result['critical_issues']:
            print(f"    {issue['smiles']}: {issue['tier']} - {issue['recommendation'][:50]}...")


# ============================================================================
# EXAMPLE 5: Audit Trail and Compliance Logging
# ============================================================================

def example_audit_trail():
    """Demonstrate audit trail and regulatory compliance logging."""
    print("\n=== EXAMPLE 5: Audit Trail & Compliance Logging ===\n")
    
    # Create audit logger
    audit_logger = ComplianceAuditLogger()
    
    # Log compound screening
    audit_logger.log_compound_screened(
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        compound_id="COMP-001",
        user_id="chemist_001",
    )
    
    # Log toxicity prediction
    audit_logger.log_toxicity_prediction(
        compound_id="COMP-001",
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        predictions={"herg": 0.08, "ames": 0.02, "hepatotox": 0.1},
        user_id="chemist_001",
    )
    
    # Log quality assessment
    audit_logger.log_quality_assessment(
        compound_id="COMP-001",
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        quality_tier="TIER_1",
        overall_passed=True,
        risk_factors=["aromatic_ring"],
        user_id="chemist_001",
    )
    
    # Log approval decision
    audit_logger.log_approval_decision(
        compound_id="COMP-001",
        decision="approved",
        reason="Passed all strict compliance criteria",
        user_id="supervisor_001",
    )
    
    # Verify audit trail integrity
    print(f"Audit Trail Integrity: {audit_logger.verify_integrity()}")
    
    # Generate audit report
    report = audit_logger.export_report()
    print(f"\nAudit Report:")
    print(f"  Total Entries: {report['total_entries']}")
    print(f"  Chain Verified: {report['chain_integrity']}")
    print(f"  Event Counts:")
    for event_type, count in report['events_by_type'].items():
        print(f"    {event_type}: {count}")


# ============================================================================
# EXAMPLE 6: Custom Threshold Configuration
# ============================================================================

def example_custom_configuration():
    """Configure custom thresholds without hardcoding."""
    print("\n=== EXAMPLE 6: Custom Threshold Configuration ===\n")
    
    # Create custom config for early-stage discovery
    custom_config = ToxicityThresholdConfig(
        herg_threshold=0.45,  # Relaxed hERG
        ames_threshold=0.35,
        hepatotox_threshold=0.48,
        min_prediction_confidence=0.3,  # Lower confidence threshold for discovery
        logp_max=5.5,  # Allow slightly higher logp
        require_lipinski_compliance=False,
    )
    
    gate = ParametrizedToxicityGate(custom_config)
    
    # Evaluate with custom thresholds
    result = gate.evaluate(
        herg_prob=0.42,  # Would fail standard, but passes custom
        ames_prob=0.10,
    )
    
    print(f"Custom Discovery Config:")
    print(f"  hERG Threshold: {gate.config.herg_threshold}")
    print(f"  Ames Threshold: {gate.config.ames_threshold}")
    print(f"  Evaluation Result: {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"  Active Thresholds: {result['active_thresholds']}")


# ============================================================================
# EXAMPLE 7: Integration with Existing Pipelines
# ============================================================================

def example_pipeline_integration():
    """Integrate strict compliance into existing drug discovery pipeline."""
    print("\n=== EXAMPLE 7: Pipeline Integration ===\n")
    
    # Simulated compound library
    library = [
        {"id": "COMP-001", "smiles": "CC(=O)O"},
        {"id": "COMP-002", "smiles": "CCO"},
        {"id": "COMP-003", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    ]
    
    # Setup strict compliance gate and audit logger
    gate = StrictComplianceGate(compliance_level=ComplianceLevel.STRICT)
    audit_logger = ComplianceAuditLogger()
    
    approved_compounds = []
    rejected_compounds = []
    
    print("Processing compound library:\n")
    
    for compound in library:
        # Log screening
        audit_logger.log_compound_screened(
            smiles=compound["smiles"],
            compound_id=compound["id"],
            user_id="pipeline_automation",
        )
        
        # Evaluate
        assessment = gate.evaluate(compound["smiles"])
        
        # Log assessment
        audit_logger.log_quality_assessment(
            compound_id=compound["id"],
            smiles=compound["smiles"],
            quality_tier=assessment.quality_tier.value,
            overall_passed=assessment.overall_passed,
            risk_factors=[rf.value for rf in assessment.risk_factors],
        )
        
        # Track decision
        if assessment.overall_passed:
            approved_compounds.append(compound["id"])
            decision = "approved"
        else:
            rejected_compounds.append(compound["id"])
            decision = "rejected"
        
        # Log decision
        audit_logger.log_approval_decision(
            compound_id=compound["id"],
            decision=decision,
            reason=assessment.recommendation,
            user_id="pipeline_automation",
        )
        
        print(f"  {compound['id']}: {decision.upper()} - {assessment.quality_tier.value}")
    
    print(f"\nResults:")
    print(f"  Approved: {len(approved_compounds)} compounds")
    print(f"  Rejected: {len(rejected_compounds)} compounds")
    print(f"\nAudit Trail Entries: {len(audit_logger.audit_trail.entries)}")
    print(f"Audit Trail Integrity: {audit_logger.verify_integrity()}")


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

if __name__ == "__main__":
    """Run all examples."""
    
    print("=" * 70)
    print("STRICT COMPLIANCE & QUALITY HARDENING - INTEGRATION GUIDE")
    print("=" * 70)
    
    example_basic_compliance()
    example_regulatory_tiers()
    example_parametrized_toxicity_gate()
    example_batch_evaluation()
    example_audit_trail()
    example_custom_configuration()
    example_pipeline_integration()
    
    print("\n" + "=" * 70)
    print("For more information, see:")
    print("  - drug_discovery/safety/strict_compliance_gate.py")
    print("  - drug_discovery/safety/parametrized_toxicity_gate.py")
    print("  - drug_discovery/compliance/audit_trail.py")
    print("  - tests/test_strict_compliance_gate.py")
    print("=" * 70 + "\n")
