"""Evaluation contracts: isolated groups, validation decisions, and audited test output."""

from copy import deepcopy
import json
import random

import numpy as np
import pytest

from abuse_signals.features import FEATURE_COLUMNS
from abuse_signals import train as training


def _row(account_id, label="normal", group_id=None, behavior=None, **features):
    row = {column: 0 for column in FEATURE_COLUMNS}
    row.update(total_events=account_id, signup_cohort_30min=1, distinct_payloads=20)
    row.update(features)
    return {**row, "account_id": account_id, "label": label,
            "group_id": group_id or f"device-{account_id}", "behavior": behavior or label}


def test_group_split_is_disjoint_stratified_and_deterministic():
    rows = []
    for label in ["normal", "farm", "quota", "spam_fanout", "ato"]:
        for group in range(20):
            for _ in range(1 + group % 4):
                rows.append(_row(len(rows), label, f"{label}-{group}"))
    rng_state = random.getstate()
    partitions = training.split_accounts(rows, seed=17)
    assert random.getstate() == rng_state
    assert partitions == training.split_accounts(list(reversed(rows)), seed=17)
    assert partitions != training.split_accounts(rows, seed=18)
    group_sets = [{row["group_id"] for row in accounts} for accounts in partitions.values()]
    assert all(not left & right for i, left in enumerate(group_sets) for right in group_sets[i + 1:])
    assert sorted(row["account_id"] for accounts in partitions.values() for row in accounts) == list(range(len(rows)))
    for partition, fraction in training.PARTITION_FRACTIONS.items():
        assert {row["label"] for row in partitions[partition]} == {row["label"] for row in rows}
        assert abs(len(partitions[partition]) - fraction * len(rows)) <= 20


def test_shared_device_challenge_behavior_occurs_in_every_partition():
    # Large cohorts are a minority among singleton normals. They must not all be
    # assigned to train just because its absolute target count is largest.
    rows = [_row(index) for index in range(300)]
    for group in range(6):
        for _ in range(8):
            rows.append(_row(len(rows), group_id=f"shared-{group}", behavior="legitimate_shared_device"))
    for group in range(3):
        for _ in range(5):
            rows.append(_row(len(rows), "farm", group_id=f"farm-{group}"))
    partitions = training.split_accounts(rows)
    for partition in partitions.values():
        assert any(row["behavior"] == "legitimate_shared_device" for row in partition)
        assert any(row["label"] == "farm" for row in partition)


def test_rare_behavior_preserves_groups_without_claiming_three_way_coverage():
    rows = [_row(index, "normal" if index < 10 else "quota") for index in range(20)]
    rows += [_row(20, group_id="one-household", behavior="rare_normal"),
             _row(21, group_id="one-household", behavior="rare_normal")]
    partitions = training.split_accounts(rows)
    assert sum(any(row["behavior"] == "rare_normal" for row in p) for p in partitions.values()) == 1
    assert sum(map(len, partitions.values())) == len(rows)


@pytest.mark.parametrize("rows, message", [
    ([], "nonempty"),
    ([_row(1), _row(1, "quota")], "unique account_id"),
    ([_row(i) for i in range(5)], "normal accounts and at least one abuse"),
    ([_row(i) for i in range(5)] + [_row(5, "farm"), _row(6, "farm")], "at least 3"),
    ([_row(1, group_id="shared"), _row(2, "farm", group_id="shared")], "mixed-label"),
])
def test_infeasible_splits_fail_explicitly(rows, message):
    with pytest.raises(ValueError, match=message):
        training.split_accounts(rows)


def test_disabled_threshold_cannot_flag_a_score_of_one():
    y = np.array([1, 0])
    scores = np.array([1.0, 0.5])
    tiers = training.calibrate_tiers(y, scores, min_alerts=5)
    for entry in tiers.values():
        assert entry["threshold"] is None
        assert not entry["calibration"]["feasible"]
        assert entry["calibration"]["metrics"]["precision"] is None
        assert not training.apply_threshold(np.array([1.0]), entry["threshold"]).any()


def test_thresholds_respect_score_ties_precision_support_and_nesting():
    assert training.choose_threshold(np.array([1, 0]), np.array([1.0, 1.0]), 0.99, 1) is None
    y = np.array([1] * 5 + [0] + [1] * 4 + [0, 0])
    scores = np.linspace(1, 0.1, len(y))
    tiers = training.calibrate_tiers(y, scores, min_alerts=5)
    assert tiers["suspend"]["threshold"] >= tiers["throttle"]["threshold"]
    throttle = training.apply_threshold(scores, tiers["throttle"]["threshold"])
    suspend = training.apply_threshold(scores, tiers["suspend"]["threshold"])
    assert np.all(~suspend | throttle)
    assert throttle.sum() == 10 and suspend.sum() == 5
    for entry in tiers.values():
        assert entry["calibration"]["metrics"]["precision"] >= entry["precision_floor"]
        assert entry["calibration"]["metrics"]["flagged"] >= 5


def test_threshold_selection_rejects_fractional_labels():
    with pytest.raises(ValueError, match="binary"):
        training.choose_threshold(np.array([0.5, 1]), np.array([0.2, 0.8]), 0.9, 1)


def _fixed_evaluation(monkeypatch):
    partitions = {
        "train": [_row(i, "normal" if i < 3 else "quota") for i in range(6)],
        "validation": [_row(i, "normal" if i < 15 else "quota") for i in range(10, 20)],
        "test": [_row(i, "normal" if i < 25 else "quota") for i in range(20, 30)],
    }
    partitions["test"][0]["signup_cohort_30min"] = 8  # One rule false positive.
    partitions["test"][5]["max_burst_5min"] = 200
    partitions["test"][6]["signup_cohort_30min"] = 8
    fit_ids = []

    class Estimator:
        def __init__(self, name):
            self.name = name

        def fit(self, x, y):
            fit_ids.append((self.name, x[:, 0].tolist(), y.tolist()))
            return self

        def predict_proba(self, x):
            ids = x[:, 0].astype(int)
            score = (ids % 10 + 1) / 11
            # LR wins validation; HGB wins test. The latter cannot select HGB.
            reverse = (ids >= 20) if self.name == "lr" else (ids < 20)
            score = np.where(reverse, 1 - score, score)
            return np.column_stack([1 - score, score])

    monkeypatch.setattr(training, "load_features", lambda _: [])
    monkeypatch.setattr(training, "split_accounts", lambda rows, seed: deepcopy(partitions))
    monkeypatch.setattr(training, "make_pipeline", lambda *steps: Estimator("lr"))
    monkeypatch.setattr(training, "HistGradientBoostingClassifier", lambda **kwargs: Estimator("hgb"))
    return partitions, fit_ids


def test_test_labels_cannot_change_thresholds_or_model_selection(monkeypatch):
    partitions, fit_ids = _fixed_evaluation(monkeypatch)
    original = training.train("unused.db", min_alerts=5)
    assert original["selected_model"] == "logistic_regression"
    assert (original["models"]["hist_gradient_boosting"]["average_precision"]
            > original["models"]["logistic_regression"]["average_precision"])
    for row in partitions["test"]:
        row["label"] = "quota" if row["label"] == "normal" else "normal"
    changed = training.train("unused.db", min_alerts=5)
    assert changed["selection"] == original["selection"]
    assert changed["selected_model"] == original["selected_model"]
    for name in original["models"]:
        for tier in training.TIER_PRECISION_FLOORS:
            before = original["models"][name]["tiers"][tier]
            after = changed["models"][name]["tiers"][tier]
            assert before["threshold"] == after["threshold"]
            assert before["calibration"] == after["calibration"]
    assert all(ids == list(range(6)) for _, ids, _ in fit_ids)


def test_report_uses_same_test_accounts_and_auditable_cumulative_metrics(monkeypatch):
    _fixed_evaluation(monkeypatch)
    report = training.train("unused.db", min_alerts=5)
    json.dumps(report, allow_nan=False)
    assert [row["account_id"] for row in report["predictions"]] == report["splits"]["test"]["account_ids"]
    assert report["rules"]["tiers"]["throttle"]["metrics"]["flagged"] == 3
    assert report["rules"]["tiers"]["suspend"]["metrics"]["flagged"] == 2
    for policy, policy_report in {"rules": report["rules"], **report["models"]}.items():
        for tier, entry in policy_report["tiers"].items():
            metrics = entry["metrics"]
            assert sum(metrics[key] for key in ["tp", "fp", "tn", "fn"]) == report["n_test"]
            errors = [row for row in report["errors"] if row["policy"] == policy and row["tier"] == tier]
            assert sum(row["error"] == "false_positive" for row in errors) == metrics["fp"]
            assert sum(row["error"] == "false_negative" for row in errors) == metrics["fn"]
            assert sum(b["false_positives"] for b in entry["behavior_false_positives"].values()) == metrics["fp"]
            assert sum(p["detected"] for p in entry["per_pattern_recall"].values()) == metrics["tp"]
            if policy != "rules":
                rule_metrics = report["rules"]["tiers"][tier]["metrics"]
                incremental = entry["incremental_vs_rules"]
                assert metrics["tp"] - rule_metrics["tp"] == (
                    incremental["additional_true_positives"] - incremental["missed_rule_true_positives"])
                assert metrics["fp"] - rule_metrics["fp"] == (
                    incremental["additional_false_positives"] - incremental["avoided_rule_false_positives"])


def test_model_selection_tie_break_is_explicit():
    assert training.select_model({"z_model": 1.0, "a_model": 1.0}) == "a_model"
