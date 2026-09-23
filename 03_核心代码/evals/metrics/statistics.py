"""Deterministic paired statistics using only the Python standard library."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def _paired_differences(candidate: Sequence[float], baseline: Sequence[float]) -> list[float]:
    if len(candidate) != len(baseline):
        raise ValueError("candidate and baseline must have equal length")
    if not candidate:
        raise ValueError("paired samples must not be empty")
    return [float(new) - float(old) for new, old in zip(candidate, baseline)]


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return math.fsum(float(value) for value in values) / len(values)


def paired_mean_difference(candidate: Sequence[float], baseline: Sequence[float]) -> float:
    """Return the paired mean difference in ``candidate - baseline`` direction."""

    return mean(_paired_differences(candidate, baseline))


def mean_difference(candidate: Sequence[float], baseline: Sequence[float]) -> float:
    """Alias for :func:`paired_mean_difference`."""

    return paired_mean_difference(candidate, baseline)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("values must not be empty")
    if probability <= 0:
        return float(sorted_values[0])
    if probability >= 1:
        return float(sorted_values[-1])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(sorted_values[upper]) * weight


def paired_bootstrap(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Return a deterministic percentile CI for a paired mean difference.

    Resampling is performed over paired differences, preserving the experimental
    unit.  A local ``random.Random`` instance avoids changing global RNG state.
    """

    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    differences = _paired_differences(candidate, baseline)
    rng = random.Random(seed)
    count = len(differences)
    bootstrap_means = []
    for _ in range(n_resamples):
        bootstrap_means.append(
            math.fsum(differences[rng.randrange(count)] for _ in range(count)) / count
        )
    bootstrap_means.sort()
    tail = (1.0 - confidence) / 2.0
    return {
        "mean_difference": mean(differences),
        "ci_low": _percentile(bootstrap_means, tail),
        "ci_high": _percentile(bootstrap_means, 1.0 - tail),
        "confidence": confidence,
        "n": count,
        "n_resamples": n_resamples,
        "seed": seed,
    }

