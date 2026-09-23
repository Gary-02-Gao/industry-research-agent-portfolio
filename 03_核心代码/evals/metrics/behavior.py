"""Answerability, refusal, and tool-routing metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._common import as_list, f1_score, safe_divide, unique_identifiers


_REFUSAL_MARKERS = (
    "cannot answer",
    "can't answer",
    "unable to answer",
    "insufficient information",
    "insufficient evidence",
    "not enough information",
    "not enough evidence",
    "i don't know",
    "i do not know",
    "无法回答",
    "不能回答",
    "信息不足",
    "证据不足",
    "无法确定",
    "不知道",
)

# These phrases explicitly summarize that every requested field is unavailable.
# Unlike a bare ``无法确定`` later in a long answer, they are safe to classify as
# a whole-answer refusal even when they appear in a concluding sentence.
_COMPLETE_REFUSAL_MARKERS = (
    "均无法确定",
    "全部无法确定",
    "全都无法确定",
)


def infer_refusal(value: Any) -> bool:
    """Infer whether an adapter output represents a refusal.

    Explicit boolean fields take precedence.  String matching is intentionally
    conservative and is meant as a fallback for adapters without a refusal flag.
    """

    if isinstance(value, bool):
        return value
    if value is None:
        return True
    if isinstance(value, Mapping):
        for key in ("refused", "is_refusal", "abstained", "is_abstention"):
            if key in value:
                return bool(value[key])
        for key in ("answer", "response", "text", "output"):
            if key in value:
                return infer_refusal(value[key])
        return False
    text = str(value).strip().lower()
    if not text:
        return True
    if any(marker in text for marker in _COMPLETE_REFUSAL_MARKERS):
        return True
    marker_offsets = [text.find(marker) for marker in _REFUSAL_MARKERS]
    marker_offsets = [offset for offset in marker_offsets if offset >= 0]
    if not marker_offsets:
        return False

    # A long, otherwise substantive answer can legitimately state that one
    # requested sub-field is unavailable.  Treating any later caveat as a
    # whole-answer refusal over-counts abstentions in multi-part RAG questions.
    # Without an explicit adapter flag, only an early refusal phrase (or a
    # short refusal response) is a conservative whole-answer abstention.
    first_marker = min(marker_offsets)
    return first_marker <= 80 or len(text) <= 160


def answerable_refusal_metrics(
    answerable: Sequence[bool],
    refused: Sequence[bool],
) -> dict[str, float]:
    """Score whether the system answers answerable cases and refuses the rest.

    Refusal is the positive class for precision/recall/F1.  Additional rates make
    the two important error modes—over-refusal and answering unanswerable
    questions—visible rather than hiding them in one accuracy number.
    """

    if len(answerable) != len(refused):
        raise ValueError("answerable and refused must have equal length")

    total = len(answerable)
    answerable_count = sum(bool(item) for item in answerable)
    unanswerable_count = total - answerable_count
    true_refusals = sum((not bool(gold)) and bool(pred) for gold, pred in zip(answerable, refused))
    false_refusals = sum(bool(gold) and bool(pred) for gold, pred in zip(answerable, refused))
    missed_refusals = sum((not bool(gold)) and (not bool(pred)) for gold, pred in zip(answerable, refused))
    correct_answers = sum(bool(gold) and (not bool(pred)) for gold, pred in zip(answerable, refused))

    precision = safe_divide(true_refusals, true_refusals + false_refusals)
    recall = safe_divide(true_refusals, true_refusals + missed_refusals)
    return {
        "answerable_refusal_accuracy": safe_divide(true_refusals + correct_answers, total),
        "answer_rate_when_answerable": safe_divide(correct_answers, answerable_count),
        "correct_refusal_rate": safe_divide(true_refusals, unanswerable_count),
        "over_refusal_rate": safe_divide(false_refusals, answerable_count),
        "answer_when_unanswerable_rate": safe_divide(missed_refusals, unanswerable_count),
        "refusal_precision": precision,
        "refusal_recall": recall,
        "refusal_f1": f1_score(precision, recall),
    }


def answerable_refusal_accuracy(
    answerable: Sequence[bool],
    refused: Sequence[bool],
) -> float:
    """Return the fraction of correct answer-vs-refuse decisions."""

    return answerable_refusal_metrics(answerable, refused)["answerable_refusal_accuracy"]


def refusal_precision_recall_f1(
    answerable: Sequence[bool],
    refused: Sequence[bool],
) -> dict[str, float]:
    """Return refusal-class precision, recall, and F1 only."""

    metrics = answerable_refusal_metrics(answerable, refused)
    return {key: metrics[key] for key in ("refusal_precision", "refusal_recall", "refusal_f1")}


def _tool_identifier(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("tool", "tool_name", "name", "action", "type"):
            candidate = value.get(key)
            if candidate is not None and not isinstance(candidate, (dict, list)):
                return str(candidate)
    return str(value)


def _route_samples(value: Any) -> list[list[Any]]:
    """Treat a flat tool list as one case and a nested list as many cases."""

    items = as_list(value)
    if not items:
        return [[]]
    if all(not isinstance(item, (list, tuple, set)) for item in items):
        return [items]
    return [as_list(item) for item in items]


def tool_routing_precision_recall_f1(predicted: Any, expected: Any) -> dict[str, float]:
    """Compute micro tool-routing P/R/F1 plus exact match and sample macro F1."""

    predicted_samples = _route_samples(predicted)
    expected_samples = _route_samples(expected)
    if len(predicted_samples) != len(expected_samples):
        raise ValueError("predicted and expected routes must have equal sample counts")

    total_tp = total_fp = total_fn = exact = 0
    sample_f1: list[float] = []
    for predicted_sample, expected_sample in zip(predicted_samples, expected_samples):
        predicted_set = set(
            unique_identifiers(predicted_sample, extractor=_tool_identifier)
        )
        expected_set = set(
            unique_identifiers(expected_sample, extractor=_tool_identifier)
        )
        tp = len(predicted_set & expected_set)
        fp = len(predicted_set - expected_set)
        fn = len(expected_set - predicted_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        exact += predicted_set == expected_set

        if not predicted_set and not expected_set:
            sample_f1.append(1.0)
        else:
            p = safe_divide(tp, tp + fp)
            r = safe_divide(tp, tp + fn)
            sample_f1.append(f1_score(p, r))

    if total_tp == total_fp == total_fn == 0:
        precision = recall = f1 = 1.0
    else:
        precision = safe_divide(total_tp, total_tp + total_fp)
        recall = safe_divide(total_tp, total_tp + total_fn)
        f1 = f1_score(precision, recall)

    count = len(predicted_samples)
    return {
        "tool_routing_precision": precision,
        "tool_routing_recall": recall,
        "tool_routing_f1": f1,
        "tool_routing_exact_match": safe_divide(exact, count),
        "tool_routing_macro_f1": safe_divide(sum(sample_f1), count),
    }


def tool_routing_metrics(predicted: Any, expected: Any) -> dict[str, float]:
    """Alias for :func:`tool_routing_precision_recall_f1`."""

    return tool_routing_precision_recall_f1(predicted, expected)


def tool_routing_precision(predicted: Any, expected: Any) -> float:
    return tool_routing_precision_recall_f1(predicted, expected)["tool_routing_precision"]


def tool_routing_recall(predicted: Any, expected: Any) -> float:
    return tool_routing_precision_recall_f1(predicted, expected)["tool_routing_recall"]


def tool_routing_f1(predicted: Any, expected: Any) -> float:
    return tool_routing_precision_recall_f1(predicted, expected)["tool_routing_f1"]
