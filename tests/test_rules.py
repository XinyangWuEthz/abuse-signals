import pytest

from abuse_signals.rules import Condition, Rule, RuleSet, load_rules

TIERS = {"monitor": 1, "throttle": 2, "suspend": 3}


def _row(**overrides):
    row = {"account_id": 1, "signup_cohort_30min": 1, "max_burst_5min": 5,
           "max_fanout_per_payload": 2, "distinct_payloads": 40,
           "max_gap_hours": 10, "events_last_48h": 20}
    row.update(overrides)
    return row


def test_conditions_are_anded():
    rule = Rule("R1", "throttle", (
        Condition("max_gap_hours", ">=", 168),
        Condition("events_last_48h", ">=", 100),
    ))
    ruleset = RuleSet((rule,), TIERS)
    assert ruleset.evaluate(_row(max_gap_hours=200))["action"] is None
    assert ruleset.evaluate(_row(max_gap_hours=200, events_last_48h=150))["action"] == "throttle"


def test_highest_tier_wins():
    ruleset = RuleSet((
        Rule("R_throttle", "throttle", (Condition("max_burst_5min", ">=", 100),)),
        Rule("R_suspend", "suspend", (Condition("signup_cohort_30min", ">=", 8),)),
    ), TIERS)
    verdict = ruleset.evaluate(_row(max_burst_5min=500, signup_cohort_30min=10))
    assert verdict["action"] == "suspend"
    assert set(verdict["matched_rules"]) == {"R_throttle", "R_suspend"}


def test_no_match_returns_none():
    ruleset = load_rules()
    verdict = ruleset.evaluate(_row())
    assert verdict["action"] is None
    assert verdict["matched_rules"] == []


def test_shipped_rules_file_is_valid():
    ruleset = load_rules()
    assert len(ruleset.rules) >= 4
    assert {rule.action for rule in ruleset.rules} <= set(ruleset.tiers)


def test_unknown_action_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: 1\ntiers: {monitor: 1}\n"
        "rules:\n  - id: R9\n    action: obliterate\n"
        "    conditions: [{feature: api_calls, op: '>=', threshold: 1}]\n"
    )
    with pytest.raises(ValueError, match="unknown action"):
        load_rules(bad)
