"""Data Integrity and Audit Trail Module for Compliance.

Implements:
- Immutable audit logging with cryptographic verification
- Data provenance tracking
- Compliance event recording
- Regulatory audit trail requirements (FDA Part 11)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of compliance audit events."""

    COMPOUND_SCREENED = "compound_screened"
    TOXICITY_PREDICTION = "toxicity_prediction"
    QUALITY_ASSESSMENT = "quality_assessment"
    THRESHOLD_VIOLATION = "threshold_violation"
    APPROVAL_DECISION = "approval_decision"
    REJECTION_DECISION = "rejection_decision"
    CONFIG_CHANGE = "config_change"
    BATCH_EVALUATION = "batch_evaluation"
    AUDIT_TRAIL_INTEGRITY_CHECK = "audit_trail_integrity_check"


@dataclass
class ComplianceAuditEntry:
    """Single audit trail entry (immutable).
    
    Follows FDA Part 11 ALCOA+ principles:
    - Attributable: Who performed the action
    - Legible: Readable text
    - Contemporaneous: Recorded in real-time
    - Original: No modification possible
    - Accurate: Cryptographically verified
    - Complete: All context captured
    """

    event_type: AuditEventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str = "system"
    
    # Event details
    compound_id: Optional[str] = None
    smiles: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    
    # Integrity verification
    event_hash: str = ""  # SHA256 of event data
    previous_hash: Optional[str] = None  # Hash chaining for integrity
    
    # Regulatory metadata
    audit_id: str = ""  # Unique identifier
    
    def compute_hash(self, previous_hash: Optional[str] = None) -> str:
        """Compute cryptographic hash for immutability verification.
        
        Args:
            previous_hash: Hash of previous entry for chain verification
            
        Returns:
            SHA256 hash of this entry
        """
        # Create canonical JSON (deterministic)
        data = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "compound_id": self.compound_id,
            "smiles": self.smiles,
            "details": self.details,
            "previous_hash": previous_hash,
        }
        
        json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()


@dataclass
class AuditTrail:
    """Immutable audit trail for compliance.
    
    Maintains hash chain to detect any modification of entries.
    """

    entries: list[ComplianceAuditEntry] = field(default_factory=list)
    chain_verified: bool = True
    last_verification: Optional[datetime] = None

    def add_entry(
        self,
        event_type: AuditEventType,
        user_id: str = "system",
        compound_id: Optional[str] = None,
        smiles: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> ComplianceAuditEntry:
        """Add new entry to audit trail.
        
        Args:
            event_type: Type of event
            user_id: User performing action
            compound_id: Optional compound identifier
            smiles: Optional SMILES string
            details: Additional event details
            
        Returns:
            The created audit entry
        """
        previous_hash = self.entries[-1].event_hash if self.entries else None

        entry = ComplianceAuditEntry(
            event_type=event_type,
            user_id=user_id,
            compound_id=compound_id,
            smiles=smiles,
            details=details or {},
            previous_hash=previous_hash,
        )

        # Compute and set hash
        entry.event_hash = entry.compute_hash(previous_hash)
        entry.audit_id = self._generate_audit_id(entry)

        self.entries.append(entry)
        logger.info(
            f"Audit entry recorded: {event_type.value} "
            f"(audit_id={entry.audit_id})"
        )

        return entry

    def _generate_audit_id(self, entry: ComplianceAuditEntry) -> str:
        """Generate unique audit ID."""
        data = f"{entry.timestamp.isoformat()}|{entry.user_id}|{entry.event_type.value}"
        return hashlib.sha256(data.encode()).hexdigest()[:16].upper()

    def verify_chain_integrity(self) -> bool:
        """Verify entire audit chain hasn't been tampered with.
        
        Returns:
            True if chain is valid, False if tampering detected
        """
        if not self.entries:
            self.chain_verified = True
            self.last_verification = datetime.now(timezone.utc)
            return True

        # Verify first entry has no previous hash
        if self.entries[0].previous_hash is not None:
            logger.error("Chain integrity error: first entry has previous_hash")
            self.chain_verified = False
            return False

        # Verify hash chain
        previous_hash = None
        for entry in self.entries:
            expected_hash = entry.compute_hash(previous_hash)
            if entry.event_hash != expected_hash:
                logger.error(
                    f"Chain integrity error at entry {entry.audit_id}: "
                    f"expected {expected_hash}, got {entry.event_hash}"
                )
                self.chain_verified = False
                return False
            previous_hash = entry.event_hash

        self.chain_verified = True
        self.last_verification = datetime.now(timezone.utc)
        logger.info("Audit trail chain integrity verified")
        return True

    def get_entries_for_compound(self, compound_id: str) -> list[ComplianceAuditEntry]:
        """Get all audit entries for a specific compound."""
        return [e for e in self.entries if e.compound_id == compound_id]

    def get_entries_since(
        self,
        start_time: datetime,
        event_type: Optional[AuditEventType] = None,
    ) -> list[ComplianceAuditEntry]:
        """Get audit entries since a specific time."""
        entries = [e for e in self.entries if e.timestamp >= start_time]
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        return entries

    def export_audit_report(self) -> dict[str, Any]:
        """Export audit trail as regulatory report."""
        return {
            "total_entries": len(self.entries),
            "date_range": {
                "start": self.entries[0].timestamp.isoformat() if self.entries else None,
                "end": self.entries[-1].timestamp.isoformat() if self.entries else None,
            },
            "chain_integrity": self.chain_verified,
            "last_verification": self.last_verification.isoformat() if self.last_verification else None,
            "events_by_type": self._count_events_by_type(),
            "users_involved": self._get_users(),
            "entries": [self._serialize_entry(e) for e in self.entries],
        }

    def _count_events_by_type(self) -> dict[str, int]:
        """Count events by type."""
        counts = {}
        for entry in self.entries:
            key = entry.event_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _get_users(self) -> list[str]:
        """Get list of users in audit trail."""
        return sorted(set(e.user_id for e in self.entries))

    def _serialize_entry(self, entry: ComplianceAuditEntry) -> dict[str, Any]:
        """Serialize entry for export."""
        return {
            "audit_id": entry.audit_id,
            "event_type": entry.event_type.value,
            "timestamp": entry.timestamp.isoformat(),
            "user_id": entry.user_id,
            "compound_id": entry.compound_id,
            "details": entry.details,
            "event_hash": entry.event_hash,
        }


class ComplianceAuditLogger:
    """Convenience logger for recording compliance events."""

    def __init__(self, audit_trail: Optional[AuditTrail] = None):
        """Initialize audit logger."""
        self.audit_trail = audit_trail or AuditTrail()

    def log_compound_screened(
        self,
        smiles: str,
        compound_id: Optional[str] = None,
        user_id: str = "system",
    ) -> ComplianceAuditEntry:
        """Log compound screening event."""
        return self.audit_trail.add_entry(
            AuditEventType.COMPOUND_SCREENED,
            user_id=user_id,
            compound_id=compound_id,
            smiles=smiles,
        )

    def log_toxicity_prediction(
        self,
        compound_id: str,
        smiles: str,
        predictions: dict[str, float],
        user_id: str = "system",
    ) -> ComplianceAuditEntry:
        """Log toxicity prediction event."""
        return self.audit_trail.add_entry(
            AuditEventType.TOXICITY_PREDICTION,
            user_id=user_id,
            compound_id=compound_id,
            smiles=smiles,
            details={"predictions": predictions},
        )

    def log_quality_assessment(
        self,
        compound_id: str,
        smiles: str,
        quality_tier: str,
        overall_passed: bool,
        risk_factors: list[str],
        user_id: str = "system",
    ) -> ComplianceAuditEntry:
        """Log quality assessment event."""
        return self.audit_trail.add_entry(
            AuditEventType.QUALITY_ASSESSMENT,
            user_id=user_id,
            compound_id=compound_id,
            smiles=smiles,
            details={
                "quality_tier": quality_tier,
                "passed": overall_passed,
                "risk_factors": risk_factors,
            },
        )

    def log_approval_decision(
        self,
        compound_id: str,
        decision: str,
        reason: str,
        user_id: str = "system",
    ) -> ComplianceAuditEntry:
        """Log approval decision."""
        event_type = (
            AuditEventType.APPROVAL_DECISION
            if decision == "approved"
            else AuditEventType.REJECTION_DECISION
        )
        return self.audit_trail.add_entry(
            event_type,
            user_id=user_id,
            compound_id=compound_id,
            details={"decision": decision, "reason": reason},
        )

    def log_config_change(
        self,
        config_name: str,
        old_value: Any,
        new_value: Any,
        user_id: str = "system",
    ) -> ComplianceAuditEntry:
        """Log configuration change."""
        return self.audit_trail.add_entry(
            AuditEventType.CONFIG_CHANGE,
            user_id=user_id,
            details={
                "config_name": config_name,
                "old_value": str(old_value),
                "new_value": str(new_value),
            },
        )

    def verify_integrity(self) -> bool:
        """Verify audit trail integrity."""
        return self.audit_trail.verify_chain_integrity()

    def export_report(self) -> dict[str, Any]:
        """Export compliance audit report."""
        return self.audit_trail.export_audit_report()
