"""Classifier layer: scikit-learn models over the SQL features, evaluated at the
enforcement tiers' precision floors.

The interesting output is not accuracy - it is: at the precision each enforcement
action demands (suspend >= 0.99, throttle >= 0.90), how much recall can we buy?
That trade-off is what an abuse team actually tunes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, load_features

# Precision each enforcement action must clear before it may fire (see README).
TIER_PRECISION_FLOORS = {"throttle": 0.90, "suspend": 0.99}


def to_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[row[c] for c in FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.array([row["label"] != "normal" for row in rows], dtype=int)
    return x, y


def recall_at_precision(y_true: np.ndarray, scores: np.ndarray, floor: float) -> tuple[float, float]:
    """Best achievable recall subject to precision >= floor, and the threshold that buys it."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    best_recall, best_threshold = 0.0, 1.0
    # precision/recall have one more entry than thresholds; skip the sentinel point.
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= floor and r > best_recall:
            best_recall, best_threshold = r, t
    return best_recall, float(best_threshold)


def train(db_path: str, seed: int = 7) -> dict:
    rows = load_features(db_path)
    x, y = to_matrix(rows)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=seed
    )

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }

    report: dict = {"n_train": len(y_train), "n_test": len(y_test),
                    "abuse_rate_test": float(y_test.mean()), "models": {}}
    for name, model in models.items():
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_test)[:, 1]
        entry = {"pr_auc": float(average_precision_score(y_test, scores)), "tiers": {}}
        for tier, floor in TIER_PRECISION_FLOORS.items():
            recall, threshold = recall_at_precision(y_test, scores, floor)
            entry["tiers"][tier] = {"precision_floor": floor,
                                    "recall": float(recall),
                                    "threshold": threshold}
        report["models"][name] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/abuse.db")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="metrics.json")
    args = parser.parse_args()

    report = train(args.db, args.seed)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"test set: {report['n_test']} accounts, "
          f"abuse rate {report['abuse_rate_test']:.3f}")
    for name, entry in report["models"].items():
        print(f"\n{name}: PR-AUC {entry['pr_auc']:.4f}")
        for tier, stats in entry["tiers"].items():
            print(f"  {tier:>9} (precision >= {stats['precision_floor']:.2f}): "
                  f"recall {stats['recall']:.3f} at threshold {stats['threshold']:.4f}")


if __name__ == "__main__":
    main()
