"""Group-disjoint offline comparison of rules and supervised account classifiers.

Models fit training accounts, thresholds and model selection use validation accounts,
and all policies are evaluated on the same untouched test accounts. These are
synthetic, retrospective account splits, not a chronological deployment simulation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, load_features
from .metrics import binary_metrics
from .rules import load_rules

TIER_PRECISION_FLOORS = {"throttle": 0.90, "suspend": 0.99}
PARTITION_FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}


def to_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[row[c] for c in FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.array([row["label"] != "normal" for row in rows], dtype=int)
    return x, y


def split_accounts(rows: list[dict], seed: int = 7) -> dict[str, list[dict]]:
    """Keep ASN/device groups together and stratify synthetic labels and behaviors.

    Behaviors with at least three groups are represented in every partition. Rare
    behaviors are pooled within their label. Seeded group order changes membership
    without searching for favorable model results. Three independent groups per
    label are required. Mixed-label groups fail explicitly.
    """
    if not rows or len({r["account_id"] for r in rows}) != len(rows):
        raise ValueError("evaluation requires nonempty rows with unique account_id values")
    labels = {row["label"] for row in rows}
    if "normal" not in labels or len(labels) < 2:
        raise ValueError("evaluation requires normal accounts and at least one abuse label")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("group_id"):
            raise ValueError("every evaluation account requires an ASN/device group_id")
        groups[row["group_id"]].append(row)
    by_label: dict[str, list[list[dict]]] = defaultdict(list)
    for group in groups.values():
        group_labels = {row["label"] for row in group}
        if len(group_labels) != 1:
            raise ValueError("mixed-label ASN/device groups are unsupported by this stratifier")
        by_label[group[0]["label"]].append(group)

    rng = random.Random(seed)
    partitions: dict[str, list[dict]] = {name: [] for name in PARTITION_FRACTIONS}
    for label in sorted(by_label):
        label_groups = sorted(by_label[label], key=lambda group: str(group[0]["group_id"]))
        if len(label_groups) < 3:
            raise ValueError(
                f"label {label!r} has {len(label_groups)} independent groups; "
                "at least 3 are required for train/validation/test"
            )
        by_behavior: dict[tuple[str, ...], list[list[dict]]] = defaultdict(list)
        for group in label_groups:
            behavior = tuple(sorted({row.get("behavior", label) for row in group}))
            by_behavior[behavior].append(group)
        strata = [groups for _, groups in sorted(by_behavior.items()) if len(groups) >= 3]
        rare = [group for _, groups in sorted(by_behavior.items()) if len(groups) < 3 for group in groups]
        if rare:
            strata.append(rare)
        for stratum in strata:
            rng.shuffle(stratum)
            total = sum(map(len, stratum))
            targets = {name: fraction * total for name, fraction in PARTITION_FRACTIONS.items()}
            counts = {name: 0 for name in partitions}
            for index, group in enumerate(stratum):
                empty = [name for name in partitions if counts[name] == 0]
                remaining = len(stratum) - index
                eligible = empty if len(stratum) >= 3 and remaining == len(empty) else list(partitions)
                destination = min(
                    eligible,
                    key=lambda name: ((counts[name] + len(group) - targets[name]) ** 2
                                      - (counts[name] - targets[name]) ** 2) / targets[name],
                )
                partitions[destination].extend(group)
                counts[destination] += len(group)
    return {name: sorted(accounts, key=lambda row: row["account_id"])
            for name, accounts in partitions.items()}


def apply_threshold(scores: np.ndarray, threshold: float | None) -> np.ndarray:
    """None disables a tier, including for accounts with score == 1."""
    scores = np.asarray(scores, dtype=float)
    return np.zeros(len(scores), dtype=bool) if threshold is None else scores >= threshold


def choose_threshold(
    y_validation: np.ndarray,
    scores_validation: np.ndarray,
    precision_floor: float,
    min_alerts: int = 5,
    minimum_threshold: float | None = None,
) -> float | None:
    """Maximize validation recall subject to empirical precision and alert support.

    Tied scores cannot be split. Recall ties prefer higher precision, then the higher
    threshold. No feasible supported prediction set returns None, never a score of 1.
    """
    y = np.asarray(y_validation)
    scores = np.asarray(scores_validation, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not len(y):
        raise ValueError("validation labels and scores must be matching nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("validation labels must be binary and scores must be finite")
    y = y.astype(int)
    if (not 0 < precision_floor <= 1 or not isinstance(min_alerts, int)
            or isinstance(min_alerts, bool) or min_alerts < 1):
        raise ValueError("precision_floor must be in (0, 1] and min_alerts must be positive")
    if not y.sum():
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    true_positives = np.cumsum(y[order])
    ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    candidates = []
    for end in ends:
        alerts = int(end) + 1
        threshold = float(sorted_scores[end])
        precision = float(true_positives[end]) / alerts
        if alerts < min_alerts or precision < precision_floor:
            continue
        if minimum_threshold is not None and threshold < minimum_threshold:
            continue
        candidates.append((int(true_positives[end]), precision, threshold))
    return max(candidates)[2] if candidates else None


def calibrate_tiers(
    y_validation: np.ndarray, scores_validation: np.ndarray, min_alerts: int = 5
) -> dict:
    calibrated = {}
    throttle_threshold = None
    for tier, floor in TIER_PRECISION_FLOORS.items():
        threshold = choose_threshold(
            y_validation, scores_validation, floor, min_alerts,
            minimum_threshold=throttle_threshold,
        )
        if tier == "throttle":
            throttle_threshold = threshold
        elif throttle_threshold is None:
            threshold = None
        calibrated[tier] = {
            "threshold": threshold,
            "precision_floor": floor,
            "calibration": {
                "metrics": binary_metrics(y_validation, apply_threshold(scores_validation, threshold)),
                "min_alerts": min_alerts,
                "feasible": threshold is not None,
                "reason": None if threshold is not None else "no_supported_validation_threshold",
            },
        }
    return calibrated


def select_model(validation_average_precision: dict[str, float]) -> str:
    """Lock the model using validation AP; ties use alphabetical model name."""
    if not validation_average_precision:
        raise ValueError("model selection requires validation scores")
    return min(validation_average_precision, key=lambda name: (-validation_average_precision[name], name))


def _partition_manifest(rows: list[dict]) -> dict:
    return {
        "account_ids": [row["account_id"] for row in rows],
        "group_ids": sorted({row["group_id"] for row in rows}),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "behavior_counts": dict(sorted(Counter(row.get("behavior", row["label"]) for row in rows).items())),
    }


def _slice_metrics(rows: list[dict], predictions: np.ndarray) -> dict:
    patterns = {}
    behaviors = {}
    for label in sorted({row["label"] for row in rows} - {"normal"}):
        indices = [i for i, row in enumerate(rows) if row["label"] == label]
        detected = sum(bool(predictions[i]) for i in indices)
        patterns[label] = {"positives": len(indices), "detected": detected, "recall": detected / len(indices)}
    for behavior in sorted({row.get("behavior", row["label"]) for row in rows if row["label"] == "normal"}):
        indices = [i for i, row in enumerate(rows)
                   if row["label"] == "normal" and row.get("behavior", row["label"]) == behavior]
        false_positives = sum(bool(predictions[i]) for i in indices)
        behaviors[behavior] = {
            "negatives": len(indices), "false_positives": false_positives,
            "false_positive_rate": false_positives / len(indices),
        }
    return {"per_pattern_recall": patterns, "behavior_false_positives": behaviors}


def _actions(scores: np.ndarray, tiers: dict) -> list[str]:
    throttle = apply_threshold(scores, tiers["throttle"]["threshold"])
    suspend = apply_threshold(scores, tiers["suspend"]["threshold"])
    return ["suspend" if s else "throttle" if t else "none" for t, s in zip(throttle, suspend)]


def train(db_path: str, seed: int = 7, min_alerts: int = 5) -> dict:
    if not isinstance(min_alerts, int) or isinstance(min_alerts, bool) or min_alerts < 1:
        raise ValueError("min_alerts must be a positive integer")
    partitions = split_accounts(load_features(db_path), seed)
    x_train, y_train = to_matrix(partitions["train"])
    x_validation, y_validation = to_matrix(partitions["validation"])
    x_test, y_test = to_matrix(partitions["test"])
    test_rows = partitions["test"]
    estimators = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }
    model_reports = {}
    for name, model in estimators.items():
        model.fit(x_train, y_train)
        validation_scores = model.predict_proba(x_validation)[:, 1]
        model_reports[name] = {
            "validation_average_precision": float(average_precision_score(y_validation, validation_scores)),
            "tiers": calibrate_tiers(y_validation, validation_scores, min_alerts),
        }
    validation_aps = {name: entry["validation_average_precision"] for name, entry in model_reports.items()}
    selected_model = select_model(validation_aps)

    ruleset = load_rules()
    verdicts = [ruleset.evaluate(row) for row in test_rows]
    rules_predictions = {
        tier: np.array([bool(v["action"]) and ruleset.tiers[v["action"]] >= ruleset.tiers[tier]
                        for v in verdicts], dtype=bool)
        for tier in TIER_PRECISION_FLOORS
    }
    rules_report = {"tiers": {
        tier: {"metrics": binary_metrics(y_test, predicted), **_slice_metrics(test_rows, predicted)}
        for tier, predicted in rules_predictions.items()
    }}
    test_scores = {}
    model_actions = {}
    policy_predictions = {"rules": rules_predictions}
    for name, model in estimators.items():
        scores = model.predict_proba(x_test)[:, 1]
        test_scores[name] = scores
        entry = model_reports[name]
        entry["average_precision"] = float(average_precision_score(y_test, scores))
        policy_predictions[name] = {}
        model_actions[name] = _actions(scores, entry["tiers"])
        for tier, tier_report in entry["tiers"].items():
            predicted = apply_threshold(scores, tier_report["threshold"])
            rule_predicted = rules_predictions[tier]
            policy_predictions[name][tier] = predicted
            tier_report.update({
                "metrics": binary_metrics(y_test, predicted),
                **_slice_metrics(test_rows, predicted),
                "incremental_vs_rules": {
                    "additional_true_positives": int(np.sum(predicted & ~rule_predicted & (y_test == 1))),
                    "additional_false_positives": int(np.sum(predicted & ~rule_predicted & (y_test == 0))),
                    "missed_rule_true_positives": int(np.sum(~predicted & rule_predicted & (y_test == 1))),
                    "avoided_rule_false_positives": int(np.sum(~predicted & rule_predicted & (y_test == 0))),
                },
            })

    predictions = []
    errors = []
    for index, row in enumerate(test_rows):
        common = {
            "account_id": row["account_id"], "group_id": row["group_id"],
            "label": row["label"], "behavior": row.get("behavior", row["label"]),
            "features": {column: row[column] for column in FEATURE_COLUMNS},
            "matched_rules": verdicts[index]["matched_rules"],
        }
        rules_action = verdicts[index]["action"] or "none"
        predictions.append({
            **common, "rules_action": rules_action,
            "model_scores": {name: float(scores[index]) for name, scores in test_scores.items()},
            "model_actions": {name: actions[index] for name, actions in model_actions.items()},
        })
        for policy, tiers in policy_predictions.items():
            for tier, predicted in tiers.items():
                if bool(predicted[index]) == bool(y_test[index]):
                    continue
                errors.append({
                    **common, "policy": policy, "tier": tier,
                    "score": None if policy == "rules" else float(test_scores[policy][index]),
                    "predicted_action": rules_action if policy == "rules" else model_actions[policy][index],
                    "error": "false_positive" if predicted[index] else "false_negative",
                })
    return {
        "schema_version": 2, "seed": seed, "min_alerts": min_alerts,
        "split_strategy": "synthetic label/behavior-stratified ASN/device groups, seeded group order, target 60/20/20; rare behaviors pooled within label",
        "evaluation_scope": "retrospective synthetic accounts; no chronological holdout",
        "n_train": len(y_train), "n_validation": len(y_validation), "n_test": len(y_test),
        "abuse_rate_test": float(y_test.mean()),
        "splits": {name: _partition_manifest(rows) for name, rows in partitions.items()},
        "selected_model": selected_model,
        "selection": {"metric": "validation_average_precision", "scores": validation_aps,
                      "tie_break": "alphabetical model name"},
        "rules": rules_report, "models": model_reports,
        "predictions": predictions, "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/abuse.db")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-alerts", type=int, default=5)
    parser.add_argument("--out", default="metrics.json")
    args = parser.parse_args()
    report = train(args.db, args.seed, args.min_alerts)
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"accounts: train {report['n_train']}, validation {report['n_validation']}, test {report['n_test']}")
    print(f"model selected on validation: {report['selected_model']}")
    for name, entry in report["models"].items():
        print(f"{name}: validation AP {entry['validation_average_precision']:.4f}, test AP {entry['average_precision']:.4f}")
        for tier, stats in entry["tiers"].items():
            print(f"  {tier}: threshold {stats['threshold']}, test P/R "
                  f"{stats['metrics']['precision']}/{stats['metrics']['recall']}")


if __name__ == "__main__":
    main()
