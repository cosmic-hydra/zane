"""Audit adapter exposing the project's audit trail from a top-level package."""
from __future__ import annotations

from typing import Any

from drug_discovery.compliance.audit_trail import (
    AuditTrail,
    ComplianceAuditEntry,
    ComplianceAuditLogger,
)


class ComplianceAuditAdapter:
    """Simple adapter to centralize audit usage across the repo."""

    def __init__(self, trail: AuditTrail | None = None):
        self.logger = ComplianceAuditLogger(trail)

    def log_screen(self, smiles: str, compound_id: str | None = None, user_id: str = "system") -> ComplianceAuditEntry:
        return self.logger.log_compound_screened(smiles=smiles, compound_id=compound_id, user_id=user_id)

    def log_prediction(self, compound_id: str, smiles: str, predictions: dict[str, float], user_id: str = "system") -> ComplianceAuditEntry:
        return self.logger.log_toxicity_prediction(compound_id=compound_id, smiles=smiles, predictions=predictions, user_id=user_id)

    def verify(self) -> bool:
        return self.logger.verify_integrity()

    def export(self) -> dict[str, Any]:
        return self.logger.export_report()
