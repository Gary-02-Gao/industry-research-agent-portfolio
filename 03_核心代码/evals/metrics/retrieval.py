"""Information-retrieval metrics implemented with the Python standard library."""

from __future__ import annotations

from typing import Any

from ._common import as_list, identifier, unique_identifiers


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def precision_at_k(retrieved: Any, relevant: Any, k: int) -> float:
    """Compute set-aware precision at *k*.

    Duplicate document identifiers receive credit only once but still occupy rank
    slots.  The denominator is always ``k`` (the conventional IR definition), so
    returning fewer than ``k`` results is penalized.  With no relevant documents
    the score is ``0.0``.
    """

    _validate_k(k)
    ranked = [identifier(item) for item in as_list(retrieved)[:k]]
    gold = set(unique_identifiers(relevant))
    if not gold:
        return 0.0
    hits = len(set(ranked) & gold)
    return hits / k


def recall_at_k(retrieved: Any, relevant: Any, k: int) -> float:
    """Compute set-aware recall at *k*, returning ``0.0`` for an empty gold set."""

    _validate_k(k)
    ranked = {identifier(item) for item in as_list(retrieved)[:k]}
    gold = set(unique_identifiers(relevant))
    if not gold:
        return 0.0
    return len(ranked & gold) / len(gold)


def retrieval_metrics_at_k(retrieved: Any, relevant: Any, k: int) -> dict[str, float]:
    """Return precision and recall using explicit ``@k`` metric names."""

    return {
        f"retrieval_precision@{k}": precision_at_k(retrieved, relevant, k),
        f"retrieval_recall@{k}": recall_at_k(retrieved, relevant, k),
    }


# Descriptive aliases make imports unambiguous in larger evaluation suites.
retrieval_precision_at_k = precision_at_k
retrieval_recall_at_k = recall_at_k
