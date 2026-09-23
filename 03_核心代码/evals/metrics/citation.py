"""Citation overlap metrics for reference-backed generation."""

from __future__ import annotations

from typing import Any

from ._common import f1_score, safe_divide, unique_identifiers


def citation_precision_recall_f1(predicted: Any, expected: Any) -> dict[str, float]:
    """Measure citation identifiers against a reference set.

    An empty prediction and empty reference is a correct no-citation case and
    therefore receives perfect scores.  When only the reference is empty, recall
    is vacuously one but precision and F1 are zero because every citation is extra.
    """

    predicted_ids = set(unique_identifiers(predicted))
    expected_ids = set(unique_identifiers(expected))

    if not predicted_ids and not expected_ids:
        return {
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "citation_f1": 1.0,
        }

    true_positives = len(predicted_ids & expected_ids)
    precision = safe_divide(true_positives, len(predicted_ids))
    recall = safe_divide(true_positives, len(expected_ids), default=1.0)
    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1_score(precision, recall),
    }


def citation_metrics(predicted: Any, expected: Any) -> dict[str, float]:
    """Alias for :func:`citation_precision_recall_f1`."""

    return citation_precision_recall_f1(predicted, expected)


def citation_precision(predicted: Any, expected: Any) -> float:
    """Return only citation precision."""

    return citation_precision_recall_f1(predicted, expected)["citation_precision"]


def citation_recall(predicted: Any, expected: Any) -> float:
    """Return only citation recall."""

    return citation_precision_recall_f1(predicted, expected)["citation_recall"]


def citation_f1(predicted: Any, expected: Any) -> float:
    """Return only citation F1."""

    return citation_precision_recall_f1(predicted, expected)["citation_f1"]
