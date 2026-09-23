"""Small normalization helpers shared by the dependency-free metrics."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, Callable


def safe_divide(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    """Return a division result without leaking ZeroDivisionError into an evaluation run."""

    if denominator == 0:
        return default
    return numerator / denominator


def f1_score(precision: float, recall: float) -> float:
    """Return the harmonic mean of precision and recall."""

    return safe_divide(2.0 * precision * recall, precision + recall)


def as_list(value: Any) -> list[Any]:
    """Normalize common JSON collection shapes to a list.

    Strings are treated as one identifier rather than an iterable of characters.
    Wrapper objects produced by adapters (for example ``{"results": [...]}``) are
    unwrapped.  A non-wrapper mapping is treated as one record.
    """

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, Mapping):
        for key in (
            "results",
            "items",
            "records",
            "documents",
            "docs",
            "contexts",
            "citations",
            "tools",
            "calls",
            "values",
        ):
            nested = value.get(key)
            if isinstance(nested, (list, tuple, set)):
                return list(nested)
        return [value]
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def identifier(
    value: Any,
    *,
    keys: tuple[str, ...] = (
        "id",
        "doc_id",
        "document_id",
        "chunk_id",
        "source_id",
        "citation_id",
        "url",
        "uri",
        "name",
    ),
) -> str:
    """Extract a stable identifier from a scalar or JSON object.

    The JSON fallback deliberately sorts keys so equivalent mapping values compare
    equal even when their original key order differs.
    """

    if isinstance(value, Mapping):
        for key in keys:
            candidate = value.get(key)
            if candidate is not None and not isinstance(candidate, (dict, list)):
                return str(candidate)
        for container_key in ("document", "source", "citation", "tool"):
            nested = value.get(container_key)
            if isinstance(nested, Mapping):
                return identifier(nested, keys=keys)
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def unique_identifiers(
    values: Any,
    *,
    extractor: Callable[[Any], str] = identifier,
) -> list[str]:
    """Return identifiers in first-seen order with duplicates removed."""

    seen: set[str] = set()
    result: list[str] = []
    for value in as_list(values):
        item_id = extractor(value)
        if item_id not in seen:
            seen.add(item_id)
            result.append(item_id)
    return result
