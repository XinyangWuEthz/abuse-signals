"""Binary account-level metrics shared by rule and classifier evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable


def binary_metrics(y_true: Iterable[int | bool], predicted: Iterable[int | bool]) -> dict:
    """Return counts and rates, retaining ``None`` for undefined quantities.

    The precision interval describes the observed accounts under an independence
    assumption. Accounts in a shared signup cohort can be dependent, so this is
    not a cohort-adjusted guarantee about deployment precision.
    """
    truth, guesses = list(y_true), list(predicted)
    if len(truth) != len(guesses):
        raise ValueError("truth and predictions must have the same length")
    if any(value not in (0, 1) for value in truth + guesses):
        raise ValueError("truth and predictions must contain only binary 0/1 values")

    tp = sum(bool(actual) and bool(guess) for actual, guess in zip(truth, guesses))
    fp = sum(not actual and bool(guess) for actual, guess in zip(truth, guesses))
    fn = sum(bool(actual) and not guess for actual, guess in zip(truth, guesses))
    tn = len(truth) - tp - fp - fn
    flagged, positives, negatives = tp + fp, tp + fn, tn + fp
    precision = tp / flagged if flagged else None
    lower, upper = None, None
    if flagged:
        z = 1.959963984540054
        denominator = 1 + z * z / flagged
        center = (precision + z * z / (2 * flagged)) / denominator
        margin = z * math.sqrt(
            precision * (1 - precision) / flagged + z * z / (4 * flagged * flagged)
        ) / denominator
        lower, upper = max(0.0, center - margin), min(1.0, center + margin)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "flagged": flagged, "positives": positives, "negatives": negatives,
        "precision": precision,
        "recall": tp / positives if positives else None,
        "fpr": fp / negatives if negatives else None,
        "false_positives_per_1000": 1000 * fp / negatives if negatives else None,
        "precision_ci_95": {
            "lower": lower, "upper": upper,
            "method": "Wilson score, 95%",
            "assumption": "Descriptive account-level interval; assumes independent accounts "
                          "and does not account for cohort dependence.",
        },
    }
