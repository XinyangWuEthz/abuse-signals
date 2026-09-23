import json

import pytest

from abuse_signals.benchmark import aggregate_runs, run_benchmark


def test_saved_evidence_reconstructs_report(tmp_path):
    summary = run_benchmark(tmp_path, accounts=3_000, seeds=(7,), scenarios=("baseline",))
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved == summary
    assert summary["aggregate"] == aggregate_runs(summary["runs"])
    run = summary["runs"][0]
    directory = tmp_path / "baseline" / "seed-7"
    splits = json.loads((directory / "splits.json").read_text())
    predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    errors = [json.loads(line) for line in (directory / "errors.jsonl").read_text().splitlines()]
    assert {row["account_id"] for row in predictions} == set(splits["test"]["account_ids"])
    assert len(predictions) == run["n_test"]
    assert sum(run[key] for key in ("n_train", "n_validation", "n_test")) == 3_000
    for policy, entry in {"rules": run["rules"], **run["models"]}.items():
        for tier, result in entry["tiers"].items():
            m = result["metrics"]
            actions = {"none": 0, "monitor": 1, "throttle": 2, "suspend": 3}
            actual_abuse = {row["account_id"] for row in predictions if row["label"] != "normal"}
            flagged = {
                row["account_id"] for row in predictions
                if actions[row["rules_action"] if policy == "rules" else row["model_actions"][policy]]
                >= actions[tier]
            }
            assert len(flagged) == m["flagged"]
            assert len(flagged & actual_abuse) == m["tp"]
            assert len(flagged - actual_abuse) == m["fp"]
            assert len(actual_abuse - flagged) == m["fn"]
            subset = [error for error in errors if error["policy"] == policy and error["tier"] == tier]
            assert sum(error["error"] == "false_positive" for error in subset) == m["fp"]
            assert sum(error["error"] == "false_negative" for error in subset) == m["fn"]
    text = (tmp_path / "summary.md").read_text()
    assert "validation data only" in text
    assert "does not measure transfer" in text
    assert summary["provenance"]["dependencies"]["scikit-learn"]


@pytest.mark.parametrize("kwargs", [
    {"seeds": ()}, {"seeds": (7, 7)}, {"scenarios": ()},
    {"scenarios": ("baseline", "baseline")}, {"min_alerts": 0},
    {"scenarios": ("unknown",)},
])
def test_invalid_benchmark_does_not_create_output(tmp_path, kwargs):
    out = tmp_path / "not-created"
    with pytest.raises(ValueError):
        run_benchmark(out, **kwargs)
    assert not out.exists()
