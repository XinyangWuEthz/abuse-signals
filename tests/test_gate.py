"""End-to-end regression gate (the CI quality floor).

On the fixed-seed synthetic dataset, detection quality must not regress below:
  - rules layer: suspend-tier precision >= 0.90, overall abusive recall >= 0.50
  - classifier:  PR-AUC >= 0.85

If a change to the generator, features SQL, rules, or model breaks these floors,
CI fails - the same shape as a production release gate.
"""

from abuse_signals.features import load_features
from abuse_signals.rules import load_rules, precision_recall


def test_rules_layer_meets_quality_floors(db_path):
    ruleset = load_rules()
    rows = load_features(db_path)
    verdicts = [ruleset.evaluate(row) for row in rows]

    suspend = precision_recall(verdicts, rows, "suspend", ruleset.tiers)
    assert suspend["precision"] >= 0.90, f"suspend-tier precision degraded: {suspend}"

    monitor_or_above = precision_recall(verdicts, rows, "monitor", ruleset.tiers)
    assert monitor_or_above["recall"] >= 0.50, f"rules recall degraded: {monitor_or_above}"


def test_classifier_meets_quality_floor(db_path):
    from abuse_signals.train import train

    report = train(db_path, seed=7)
    best_pr_auc = max(entry["pr_auc"] for entry in report["models"].values())
    assert best_pr_auc >= 0.85, f"classifier PR-AUC degraded: {best_pr_auc:.4f}"
