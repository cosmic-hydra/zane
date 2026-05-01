"""Top-level audit adapter package to expose audit utilities.

This module provides a thin adapter around `drug_discovery.compliance.audit_trail`
so teams can import `audit` as a separate module.
"""
from .audit_adapter import ComplianceAuditAdapter

__all__ = ["ComplianceAuditAdapter"]
