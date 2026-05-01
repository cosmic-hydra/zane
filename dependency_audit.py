"""Stdlib-only dependency audit helpers for local development workflows."""

from __future__ import annotations

import argparse
import json
from importlib.util import find_spec
from collections.abc import Iterable, Sequence

DEFAULT_MODULES = (
    "numpy",
    "pandas",
    "torch",
    "transformers",
    "datasets",
    "peft",
    "accelerate",
    "pubchempy",
    "chembl_webresource_client",
    "rdkit",
)


def audit_missing_modules(module_names: Iterable[str]) -> list[str]:
    """Return the subset of modules that are not importable in the current environment."""
    missing: list[str] = []
    for module_name in module_names:
        if find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def format_missing_modules(missing_modules: Sequence[str]) -> str:
    """Format a human-readable dependency summary."""
    return ", ".join(missing_modules) if missing_modules else "none"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Python dependencies for ZANE workflows")
    parser.add_argument(
        "--modules",
        nargs="+",
        default=list(DEFAULT_MODULES),
        help="Module names to check",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    missing = audit_missing_modules(args.modules)

    if args.json:
        print(json.dumps({"missing": missing, "missing_count": len(missing)}))
    else:
        print("Missing deps:", format_missing_modules(missing))

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
