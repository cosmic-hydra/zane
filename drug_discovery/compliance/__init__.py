"""FDA 21 CFR Part 11 Compliance Engine.

Provides immutable audit logging, electronic signatures, and RBAC
for GxP-regulated drug discovery workflows.
"""

__all__: list[str] = []

try:
    from drug_discovery.compliance.audit_ledger import (
        AuditEntry as AuditEntry,
    )
    from drug_discovery.compliance.audit_ledger import (
        AuditLedger as AuditLedger,
    )
    from drug_discovery.compliance.audit_ledger import (
        compliance_log as compliance_log,
    )

    __all__.extend(["AuditLedger", "AuditEntry", "compliance_log"])
except ImportError:
    pass

try:
    from drug_discovery.compliance.rbac import (
        Permission as Permission,
    )
    from drug_discovery.compliance.rbac import (
        RBACManager as RBACManager,
    )
    from drug_discovery.compliance.rbac import (
        Role as Role,
    )
    from drug_discovery.compliance.rbac import (
        User as User,
    )
    from drug_discovery.compliance.rbac import (
        require_permission as require_permission,
    )
    from drug_discovery.compliance.rbac import (
        require_signature as require_signature,
    )

    __all__.extend(["RBACManager", "User", "Role", "Permission", "require_permission", "require_signature"])
except ImportError:
    pass
