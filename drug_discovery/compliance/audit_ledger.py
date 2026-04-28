"""Immutable audit ledger for FDA 21 CFR Part 11 compliance.

Every model prediction, dataset change, and user action is hashed with
SHA-256 and appended to an append-only ledger. Each entry is chained to
the previous via a hash link, making tampering detectable.

Supports two backends:
- **In-memory** (default, for testing and development)
- **PostgreSQL** (production, via SQLAlchemy when available)

The :func:`compliance_log` decorator wraps any function to automatically
record its inputs, outputs, and execution metadata.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
    from sqlalchemy.orm import Session, declarative_base, sessionmaker

    _SQLALCHEMY = True
    Base = declarative_base()
except ImportError:
    _SQLALCHEMY = False
    Base = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class AuditEntry:
    """Single immutable audit record."""

    entry_id: str = ""
    timestamp: str = ""
    user_id: str = "system"
    action: str = ""
    function_name: str = ""
    input_hash: str = ""
    output_hash: str = ""
    data_hash: str = ""
    previous_hash: str = ""
    entry_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "action": self.action,
            "function_name": self.function_name,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "data_hash": self.data_hash,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
            "metadata": self.metadata,
        }

    def verify(self) -> bool:
        """Re-compute the entry hash and verify it matches."""
        expected = _compute_entry_hash(self)
        return expected == self.entry_hash


# ---------------------------------------------------------------------------
# SQLAlchemy model (optional)
# ---------------------------------------------------------------------------
if _SQLALCHEMY and Base is not None:

    class AuditRecord(Base):  # type: ignore[misc]
        __tablename__ = "audit_ledger"
        id = Column(Integer, primary_key=True, autoincrement=True)
        entry_id = Column(String(64), unique=True, nullable=False)
        timestamp = Column(DateTime, nullable=False)
        user_id = Column(String(128), nullable=False)
        action = Column(String(256), nullable=False)
        function_name = Column(String(256), nullable=False)
        input_hash = Column(String(64), nullable=False)
        output_hash = Column(String(64), nullable=False)
        data_hash = Column(String(64), nullable=False)
        previous_hash = Column(String(64), nullable=False)
        entry_hash = Column(String(64), nullable=False)
        metadata_json = Column(Text, default="{}")


# ---------------------------------------------------------------------------
# Hash utilities
# ---------------------------------------------------------------------------
def sha256_hash(data: Any) -> str:
    """Compute SHA-256 hex digest for arbitrary data."""
    raw = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _compute_entry_hash(entry: AuditEntry) -> str:
    """Compute the chained hash for an audit entry."""
    payload = (
        f"{entry.entry_id}|{entry.timestamp}|{entry.user_id}|"
        f"{entry.action}|{entry.function_name}|"
        f"{entry.input_hash}|{entry.output_hash}|{entry.data_hash}|"
        f"{entry.previous_hash}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------
class AuditLedger:
    """Append-only audit ledger with hash-chained entries.

    Usage::

        ledger = AuditLedger()
        entry = ledger.log(
            action="model_prediction",
            function_name="predict_binding",
            input_data={"smiles": "CCO"},
            output_data={"delta_g": -8.3},
        )
        assert entry.verify()

    For PostgreSQL persistence::

        ledger = AuditLedger(database_url="postgresql://user:pass@host/db")
    """

    def __init__(self, database_url: str | None = None):
        self._entries: list[AuditEntry] = []
        self._db_session: Any = None

        if database_url and _SQLALCHEMY:
            try:
                engine = create_engine(database_url)
                Base.metadata.create_all(engine)  # type: ignore[union-attr]
                self._db_session = sessionmaker(bind=engine)()
                logger.info("AuditLedger connected to PostgreSQL")
            except Exception as exc:
                logger.warning("Could not connect to PostgreSQL: %s. Using in-memory ledger.", exc)

    @property
    def chain_length(self) -> int:
        return len(self._entries)

    @property
    def last_hash(self) -> str:
        if self._entries:
            return self._entries[-1].entry_hash
        return "0" * 64  # genesis hash

    def log(
        self,
        action: str,
        function_name: str = "",
        input_data: Any = None,
        output_data: Any = None,
        user_id: str = "system",
        extra_metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Append an entry to the ledger.

        Returns the new :class:`AuditEntry` with its computed hash.
        """
        entry = AuditEntry(
            entry_id=uuid.uuid4().hex[:16],
            timestamp=datetime.datetime.utcnow().isoformat(),
            user_id=user_id,
            action=action,
            function_name=function_name,
            input_hash=sha256_hash(input_data) if input_data is not None else "",
            output_hash=sha256_hash(output_data) if output_data is not None else "",
            data_hash=sha256_hash({"input": input_data, "output": output_data}),
            previous_hash=self.last_hash,
            metadata=extra_metadata or {},
        )
        entry.entry_hash = _compute_entry_hash(entry)
        self._entries.append(entry)

        # Persist to DB if available
        if self._db_session is not None and _SQLALCHEMY:
            try:
                record = AuditRecord(
                    entry_id=entry.entry_id,
                    timestamp=datetime.datetime.fromisoformat(entry.timestamp),
                    user_id=entry.user_id,
                    action=entry.action,
                    function_name=entry.function_name,
                    input_hash=entry.input_hash,
                    output_hash=entry.output_hash,
                    data_hash=entry.data_hash,
                    previous_hash=entry.previous_hash,
                    entry_hash=entry.entry_hash,
                    metadata_json=json.dumps(entry.metadata, default=str),
                )
                self._db_session.add(record)
                self._db_session.commit()
            except Exception as exc:
                logger.warning("Failed to persist audit entry: %s", exc)
                self._db_session.rollback()

        return entry

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire audit chain."""
        prev_hash = "0" * 64
        for entry in self._entries:
            if entry.previous_hash != prev_hash:
                logger.error("Chain broken at entry %s", entry.entry_id)
                return False
            if not entry.verify():
                logger.error("Hash mismatch at entry %s", entry.entry_id)
                return False
            prev_hash = entry.entry_hash
        return True

    def get_entries(
        self,
        action: str | None = None,
        user_id: str | None = None,
        since: str | None = None,
    ) -> list[AuditEntry]:
        """Query entries with optional filters."""
        results = self._entries
        if action:
            results = [e for e in results if e.action == action]
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if since:
            results = [e for e in results if e.timestamp >= since]
        return results

    def export_json(self) -> str:
        """Export the full ledger as JSON."""
        return json.dumps([e.as_dict() for e in self._entries], indent=2, default=str)


# ---------------------------------------------------------------------------
# Compliance logging decorator
# ---------------------------------------------------------------------------
# Module-level default ledger (created on first use)
_default_ledger: AuditLedger | None = None


def get_default_ledger() -> AuditLedger:
    """Return the module-level default audit ledger."""
    global _default_ledger
    if _default_ledger is None:
        _default_ledger = AuditLedger()
    return _default_ledger


def compliance_log(
    action: str = "function_execution",
    ledger: AuditLedger | None = None,
    capture_output: bool = True,
) -> Callable:
    """Decorator that logs function calls to the audit ledger.

    Usage::

        @compliance_log(action="model_prediction")
        def predict(smiles: str) -> float:
            return -8.3
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            target_ledger = ledger or get_default_ledger()
            input_data = {"args": str(args)[:500], "kwargs": str(kwargs)[:500]}

            t0 = time.monotonic()
            result = fn(*args, **kwargs)
            elapsed = time.monotonic() - t0

            output_data = str(result)[:500] if capture_output else None

            target_ledger.log(
                action=action,
                function_name=fn.__qualname__,
                input_data=input_data,
                output_data=output_data,
                extra_metadata={"elapsed_seconds": elapsed},
            )
            return result

        return wrapper

    return decorator
