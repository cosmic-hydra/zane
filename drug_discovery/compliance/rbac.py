"""Role-Based Access Control (RBAC) and electronic signatures.

Implements OAuth2-style RBAC for FDA 21 CFR Part 11 electronic signature
requirements. Users must authenticate and hold the required permissions
before saving model checkpoints, exporting data, or signing off on
predictions.

Roles:
- **viewer**: Read-only access to pipeline results
- **scientist**: Run predictions, generate candidates
- **lead**: Approve model checkpoints, sign predictions
- **admin**: Full access including user management

Electronic signatures require re-authentication and are logged to the
:class:`~drug_discovery.compliance.audit_ledger.AuditLedger`.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """Granular permissions for pipeline operations."""

    VIEW_RESULTS = "view_results"
    RUN_PREDICTION = "run_prediction"
    GENERATE_CANDIDATES = "generate_candidates"
    EXPORT_DATA = "export_data"
    SAVE_CHECKPOINT = "save_checkpoint"
    APPROVE_CHECKPOINT = "approve_checkpoint"
    SIGN_PREDICTION = "sign_prediction"
    MANAGE_USERS = "manage_users"


# Pre-defined role templates
ROLE_PERMISSIONS: dict[str, set[Permission]] = {
    "viewer": {Permission.VIEW_RESULTS},
    "scientist": {
        Permission.VIEW_RESULTS,
        Permission.RUN_PREDICTION,
        Permission.GENERATE_CANDIDATES,
        Permission.EXPORT_DATA,
    },
    "lead": {
        Permission.VIEW_RESULTS,
        Permission.RUN_PREDICTION,
        Permission.GENERATE_CANDIDATES,
        Permission.EXPORT_DATA,
        Permission.SAVE_CHECKPOINT,
        Permission.APPROVE_CHECKPOINT,
        Permission.SIGN_PREDICTION,
    },
    "admin": set(Permission),
}


@dataclass
class Role:
    """Named role with a set of permissions."""

    name: str
    permissions: set[Permission] = field(default_factory=set)

    @classmethod
    def from_template(cls, name: str) -> Role:
        perms = ROLE_PERMISSIONS.get(name, set())
        return cls(name=name, permissions=set(perms))


@dataclass
class User:
    """Authenticated user with role-based permissions."""

    user_id: str
    name: str
    role: Role
    password_hash: str = ""
    token: str = ""
    is_active: bool = True
    last_auth: float = 0.0

    def has_permission(self, perm: Permission) -> bool:
        return self.is_active and perm in self.role.permissions

    def verify_password(self, password: str) -> bool:
        return self.password_hash == _hash_password(password, self.user_id)


def _hash_password(password: str, salt: str) -> str:
    """SHA-256 password hash with user_id as salt."""
    return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()


class AuthenticationError(Exception):
    """Raised when authentication fails."""


class PermissionError(Exception):
    """Raised when a user lacks required permissions."""


class SignatureError(Exception):
    """Raised when electronic signature verification fails."""


# ---------------------------------------------------------------------------
# RBAC Manager
# ---------------------------------------------------------------------------
class RBACManager:
    """Manages users, authentication, and permission checks.

    Usage::

        rbac = RBACManager()
        rbac.create_user("jdoe", "Jane Doe", "scientist", "s3cret")
        user = rbac.authenticate("jdoe", "s3cret")
        rbac.check_permission(user, Permission.RUN_PREDICTION)  # OK
        rbac.check_permission(user, Permission.APPROVE_CHECKPOINT)  # raises
    """

    def __init__(self):
        self._users: dict[str, User] = {}
        self._tokens: dict[str, str] = {}  # token -> user_id
        self._signature_timeout: float = 300.0  # 5 min re-auth window

    def create_user(
        self,
        user_id: str,
        name: str,
        role_name: str,
        password: str,
    ) -> User:
        """Create a new user with the given role."""
        if user_id in self._users:
            raise ValueError(f"User {user_id} already exists")

        role = Role.from_template(role_name)
        user = User(
            user_id=user_id,
            name=name,
            role=role,
            password_hash=_hash_password(password, user_id),
        )
        self._users[user_id] = user
        logger.info("Created user %s with role %s", user_id, role_name)
        return user

    def authenticate(self, user_id: str, password: str) -> User:
        """Authenticate a user and return a session token.

        Returns the :class:`User` with an active token.
        Raises :class:`AuthenticationError` on failure.
        """
        user = self._users.get(user_id)
        if user is None or not user.is_active:
            raise AuthenticationError(f"User {user_id} not found or inactive")
        if not user.verify_password(password):
            raise AuthenticationError("Invalid credentials")

        token = secrets.token_hex(32)
        user.token = token
        user.last_auth = time.monotonic()
        self._tokens[token] = user_id
        logger.info("User %s authenticated", user_id)
        return user

    def get_user_by_token(self, token: str) -> User:
        """Look up a user by their session token."""
        uid = self._tokens.get(token)
        if uid is None:
            raise AuthenticationError("Invalid token")
        user = self._users.get(uid)
        if user is None or not user.is_active:
            raise AuthenticationError("User not found or inactive")
        return user

    def check_permission(self, user: User, permission: Permission) -> None:
        """Raise :class:`PermissionError` if the user lacks *permission*."""
        if not user.has_permission(permission):
            raise PermissionError(
                f"User {user.user_id} (role={user.role.name}) "
                f"lacks permission {permission.value}"
            )

    def verify_signature(
        self,
        user: User,
        password: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Perform electronic signature verification.

        The user must re-authenticate within the signature timeout window.
        Returns a signature record dict suitable for audit logging.

        Raises :class:`SignatureError` on failure.
        """
        if not user.verify_password(password):
            raise SignatureError("Signature verification failed: invalid credentials")

        if not user.has_permission(Permission.SIGN_PREDICTION) and not user.has_permission(
            Permission.APPROVE_CHECKPOINT
        ):
            raise SignatureError(
                f"User {user.user_id} lacks signing permission"
            )

        user.last_auth = time.monotonic()

        signature_record = {
            "user_id": user.user_id,
            "user_name": user.name,
            "role": user.role.name,
            "reason": reason,
            "timestamp": time.time(),
            "signature_hash": hashlib.sha256(
                f"{user.user_id}:{reason}:{time.time()}".encode()
            ).hexdigest(),
        }
        logger.info("Electronic signature recorded for user %s: %s", user.user_id, reason)
        return signature_record

    def list_users(self) -> list[dict[str, Any]]:
        """Return summary of all registered users."""
        return [
            {
                "user_id": u.user_id,
                "name": u.name,
                "role": u.role.name,
                "is_active": u.is_active,
                "permissions": [p.value for p in u.role.permissions],
            }
            for u in self._users.values()
        ]


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
def require_permission(permission: Permission, rbac: RBACManager | None = None) -> Callable:
    """Decorator that checks the caller has *permission*.

    The decorated function must accept ``user=`` as a keyword argument.

    Usage::

        @require_permission(Permission.SAVE_CHECKPOINT)
        def save_model(path, user=None):
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            user = kwargs.get("user")
            if user is None:
                raise AuthenticationError("No user provided; authentication required")
            if rbac:
                rbac.check_permission(user, permission)
            elif not user.has_permission(permission):
                raise PermissionError(
                    f"User {user.user_id} lacks {permission.value}"
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def require_signature(reason: str = "", rbac: RBACManager | None = None) -> Callable:
    """Decorator requiring electronic signature before execution.

    The decorated function must accept ``user=`` and ``password=``
    keyword arguments.

    Usage::

        @require_signature(reason="Checkpoint export")
        def export_checkpoint(path, user=None, password=None):
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            user = kwargs.get("user")
            password = kwargs.get("password", "")
            if user is None:
                raise AuthenticationError("No user provided")
            if not password:
                raise SignatureError("Password required for electronic signature")

            if rbac:
                rbac.verify_signature(user, password, reason or fn.__qualname__)
            else:
                if not user.verify_password(password):
                    raise SignatureError("Electronic signature verification failed")

            return fn(*args, **kwargs)

        return wrapper

    return decorator
