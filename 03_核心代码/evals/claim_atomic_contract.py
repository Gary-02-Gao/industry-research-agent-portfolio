"""Frozen field names and structural quality rules shared by decomposition and verification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FIELD_NAMES = (
    "subject", "predicate", "object", "value", "unit", "currency",
    "time_scope", "comparison_scope",
)

FIELD_RESULT_STATUSES = ("matched", "missing", "conflict", "not_applicable")
GENERIC_PREDICATES = frozenset({"", "为", "陈述", "达到", "达", "实现", "披露", "待人工细化的关系"})


def validate_atomic_definition(atomic: Mapping[str, Any], *, label: str = "atomic") -> None:
    """Reject structures that cannot be independently reviewed before semantic approval."""

    fragment = atomic.get("claim_fragment")
    fields = atomic.get("fields")
    if not isinstance(fragment, str) or not fragment.strip():
        raise ValueError(f"empty claim_fragment: {label}")
    if not isinstance(fields, Mapping) or set(fields) != set(FIELD_NAMES):
        raise ValueError(f"fixed fields mismatch: {label}")
    for field, value in fields.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"field values must be non-empty strings or null: {label}/{field}")
    subject = fields["subject"]
    predicate = fields["predicate"]
    if not subject:
        raise ValueError(f"independent atomic requires subject: {label}")
    if predicate in GENERIC_PREDICATES:
        raise ValueError(f"independent atomic requires a specific predicate: {label}")
    if subject not in fragment:
        raise ValueError(f"claim_fragment must carry its subject: {label}")
    if fields["time_scope"] and fields["time_scope"] not in fragment:
        raise ValueError(f"claim_fragment must carry its time_scope: {label}")
    if "分别" in fragment:
        raise ValueError(f"respective/list values must be separate atomics: {label}")

__all__ = [
    "FIELD_NAMES", "FIELD_RESULT_STATUSES", "GENERIC_PREDICATES", "validate_atomic_definition",
]
