"""SMILES validation and sanitization module.

Ensures that every candidate SMILES entering the pipeline is chemically
valid, properly canonicalized, and passes basic drug-likeness sanity
checks. Invalid molecules are caught early -- before expensive Oracle
or docking computations.

Targets 99.9 %+ generation success rate by:
1. Catching and repairing common SMILES syntax errors
2. Canonicalizing to a single representation
3. Rejecting molecules with impossible valences or disconnected fragments
4. Applying Lipinski-like size guards
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem  # type: ignore[import-untyped]
    from rdkit.Chem import Descriptors  # type: ignore[import-untyped]

    _RDKIT = True
except ImportError:
    Chem = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    _RDKIT = False


@dataclass
class ValidationResult:
    """Outcome of validating a single SMILES string."""

    original: str
    canonical: str | None = None
    is_valid: bool = False
    was_repaired: bool = False
    rejection_reason: str | None = None
    molecular_weight: float | None = None
    heavy_atom_count: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.is_valid and self.rejection_reason is None


class SmilesValidator:
    """Validate, canonicalize, and sanitize SMILES strings.

    Usage::

        validator = SmilesValidator()
        result = validator.validate("c1ccccc1")
        assert result.passed
        assert result.canonical == "c1ccccc1"

        batch = validator.validate_batch(["CCO", "INVALID", "c1ccccc1"])
        valid = [r for r in batch if r.passed]
    """

    def __init__(
        self,
        max_heavy_atoms: int = 100,
        max_molecular_weight: float = 1000.0,
        min_heavy_atoms: int = 2,
        allow_disconnected: bool = False,
        allow_charged: bool = True,
        repair_attempts: int = 3,
    ):
        self.max_heavy_atoms = max_heavy_atoms
        self.max_molecular_weight = max_molecular_weight
        self.min_heavy_atoms = min_heavy_atoms
        self.allow_disconnected = allow_disconnected
        self.allow_charged = allow_charged
        self.repair_attempts = repair_attempts

    def validate(self, smiles: str) -> ValidationResult:
        """Validate a single SMILES string."""
        if not smiles or not smiles.strip():
            return ValidationResult(original=smiles, rejection_reason="Empty SMILES")

        smiles = smiles.strip()

        # Try RDKit validation first
        if _RDKIT:
            return self._validate_rdkit(smiles)
        return self._validate_heuristic(smiles)

    def validate_batch(self, smiles_list: Sequence[str]) -> list[ValidationResult]:
        """Validate a batch of SMILES, returning results in the same order."""
        return [self.validate(s) for s in smiles_list]

    def filter_valid(self, smiles_list: Sequence[str]) -> list[str]:
        """Return only valid, canonical SMILES from a batch."""
        results = self.validate_batch(smiles_list)
        return [r.canonical for r in results if r.passed and r.canonical]

    def success_rate(self, smiles_list: Sequence[str]) -> float:
        """Compute fraction of valid molecules in a batch."""
        if not smiles_list:
            return 0.0
        results = self.validate_batch(smiles_list)
        valid = sum(1 for r in results if r.passed)
        return valid / len(smiles_list)

    # ------------------------------------------------------------------
    # RDKit path
    # ------------------------------------------------------------------
    def _validate_rdkit(self, smiles: str) -> ValidationResult:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)

        # Attempt repair if initial parse fails
        if mol is None:
            repaired = self._attempt_repair(smiles)
            if repaired:
                mol = Chem.MolFromSmiles(repaired, sanitize=False)
                if mol is not None:
                    smiles = repaired

        if mol is None:
            return ValidationResult(original=smiles, rejection_reason="RDKit cannot parse SMILES")

        # Sanitize
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            return ValidationResult(original=smiles, rejection_reason=f"Sanitization failed: {exc}")

        canonical = Chem.MolToSmiles(mol)
        was_repaired = canonical != smiles

        # Disconnected fragment check
        if not self.allow_disconnected and "." in canonical:
            return ValidationResult(
                original=smiles,
                canonical=canonical,
                rejection_reason="Disconnected fragments not allowed",
            )

        # Size checks
        mw = Descriptors.MolWt(mol)
        heavy = mol.GetNumHeavyAtoms()

        if heavy < self.min_heavy_atoms:
            return ValidationResult(
                original=smiles,
                canonical=canonical,
                molecular_weight=mw,
                heavy_atom_count=heavy,
                rejection_reason=f"Too few heavy atoms ({heavy} < {self.min_heavy_atoms})",
            )
        if heavy > self.max_heavy_atoms:
            return ValidationResult(
                original=smiles,
                canonical=canonical,
                molecular_weight=mw,
                heavy_atom_count=heavy,
                rejection_reason=f"Too many heavy atoms ({heavy} > {self.max_heavy_atoms})",
            )
        if mw > self.max_molecular_weight:
            return ValidationResult(
                original=smiles,
                canonical=canonical,
                molecular_weight=mw,
                heavy_atom_count=heavy,
                rejection_reason=f"Molecular weight too high ({mw:.1f} > {self.max_molecular_weight})",
            )

        return ValidationResult(
            original=smiles,
            canonical=canonical,
            is_valid=True,
            was_repaired=was_repaired,
            molecular_weight=mw,
            heavy_atom_count=heavy,
        )

    # ------------------------------------------------------------------
    # Repair heuristics
    # ------------------------------------------------------------------
    def _attempt_repair(self, smiles: str) -> str | None:
        """Try common fixes for malformed SMILES."""
        repairs = [
            # Remove trailing/leading whitespace and special chars
            lambda s: re.sub(r"[^\w\[\]()=#@+\-/\\.:]+", "", s),
            # Balance parentheses
            lambda s: self._balance_parens(s),
            # Remove charge annotations
            lambda s: re.sub(r"\[([A-Z][a-z]?)[+-]\d*\]", r"\1", s),
        ]

        for repair_fn in repairs[: self.repair_attempts]:
            try:
                fixed = repair_fn(smiles)
                if fixed and Chem.MolFromSmiles(fixed) is not None:
                    return fixed
            except Exception:
                continue
        return None

    @staticmethod
    def _balance_parens(s: str) -> str:
        depth = 0
        result = []
        for ch in s:
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth <= 0:
                    continue
                depth -= 1
            result.append(ch)
        result.extend(")" * depth)
        return "".join(result)

    # ------------------------------------------------------------------
    # Heuristic fallback (no RDKit)
    # ------------------------------------------------------------------
    def _validate_heuristic(self, smiles: str) -> ValidationResult:
        """Basic validation when RDKit is not installed."""
        # Check for obviously invalid patterns
        if not re.match(r"^[A-Za-z0-9\[\]()=#@+\-/\\.:]+$", smiles):
            return ValidationResult(original=smiles, rejection_reason="Invalid characters in SMILES")

        # Check balanced brackets
        if smiles.count("[") != smiles.count("]"):
            return ValidationResult(original=smiles, rejection_reason="Unbalanced brackets")

        # Check balanced parentheses
        depth = 0
        for ch in smiles:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth < 0:
                return ValidationResult(original=smiles, rejection_reason="Unbalanced parentheses")
        if depth != 0:
            return ValidationResult(original=smiles, rejection_reason="Unbalanced parentheses")

        # Estimate size from character count (rough)
        atom_chars = re.findall(r"[A-Z][a-z]?", smiles)
        if len(atom_chars) < self.min_heavy_atoms:
            return ValidationResult(
                original=smiles,
                rejection_reason=f"Too few atoms (estimated {len(atom_chars)})",
            )
        if len(atom_chars) > self.max_heavy_atoms:
            return ValidationResult(
                original=smiles,
                rejection_reason=f"Too many atoms (estimated {len(atom_chars)})",
            )

        return ValidationResult(
            original=smiles,
            canonical=smiles,
            is_valid=True,
            heavy_atom_count=len(atom_chars),
        )
