from dataclasses import asdict
import json
import sqlite3

import pytest

from abuse_signals.config import GenConfig
from abuse_signals.generate import generate


def test_challenge_has_named_slices_and_preserves_labels(tmp_path):
    config = dict(n_accounts=240, frac_farm=0.08, frac_quota=0.06,
                  frac_spam_fanout=0.06, frac_ato=0.04)
    baseline = generate(tmp_path / "baseline.db", GenConfig(**config))
    cfg = GenConfig(**config, scenario="challenge")
    challenge = generate(tmp_path / "challenge.db", cfg)
    for key in ("accounts", "normal", "farm", "quota", "spam_fanout", "ato"):
        assert baseline[key] == challenge[key]
    with sqlite3.connect(tmp_path / "challenge.db") as conn:
        slices = dict(conn.execute("SELECT behavior, COUNT(*) FROM account_metadata GROUP BY behavior"))
        assert set(slices) == {
            "normal", "legitimate_shared_device", "legitimate_bursty_api",
            "legitimate_broadcast", "legitimate_returning", "farm", "farm_small",
            "quota", "quota_slow", "spam_fanout", "spam_varied", "ato", "ato_subtle",
        }
        assert sum(slices.values()) == cfg.n_accounts
        per_behavior = int(challenge["normal"] * cfg.challenge_legitimate_fraction)
        assert all(count == per_behavior for name, count in slices.items()
                   if name.startswith("legitimate_"))
        assert conn.execute(
            "SELECT COUNT(*) FROM account_metadata m JOIN labels l USING (account_id) "
            "WHERE m.behavior LIKE 'legitimate_%' AND l.label != 'normal'"
        ).fetchone()[0] == 0
        cohorts = conn.execute(
            "SELECT device_fingerprint, COUNT(DISTINCT asn), COUNT(*) "
            "FROM accounts JOIN account_metadata USING (account_id) "
            "WHERE behavior IN ('farm', 'farm_small', 'legitimate_shared_device') "
            "GROUP BY device_fingerprint"
        ).fetchall()
        assert all(asns == 1 for _, asns, _ in cohorts)
        assert any(count > 1 for _, _, count in cohorts)


@pytest.mark.parametrize("scenario", ["baseline", "challenge"])
@pytest.mark.parametrize("days", [1, 14])
def test_metadata_and_timestamps_define_a_complete_observation_window(tmp_path, scenario, days):
    cfg = GenConfig(n_accounts=120, days=days, scenario=scenario,
                    frac_farm=0.1, frac_quota=0.1, frac_spam_fanout=0.1, frac_ato=0.1)
    path = tmp_path / "data.db"
    generate(path, cfg)
    with sqlite3.connect(path) as conn:
        metadata = dict(conn.execute("SELECT key, value FROM dataset_metadata"))
        assert json.loads(metadata["config"]) == asdict(cfg)
        assert int(metadata["horizon_end"]) == cfg.horizon_end
        assert conn.execute("SELECT COUNT(*) FROM account_metadata").fetchone()[0] == cfg.n_accounts
        assert conn.execute(
            "SELECT COUNT(*) FROM events e JOIN accounts a USING (account_id) "
            "WHERE e.ts < a.signup_ts OR e.ts >= ? OR e.ts < ?",
            (cfg.horizon_end, cfg.start_ts),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM accounts WHERE signup_ts < ? OR signup_ts >= ?",
            (cfg.start_ts, cfg.horizon_end),
        ).fetchone()[0] == 0


@pytest.mark.parametrize("kwargs", [
    {"scenario": "unknown"}, {"n_accounts": 0}, {"n_accounts": 2.5},
    {"seed": -1}, {"seed": 2**32},
    {"days": 0}, {"n_asns": 0}, {"min_daily_events": -1},
    {"min_daily_events": 41}, {"frac_farm": -0.1}, {"frac_quota": float("nan")},
    {"frac_farm": "0.1"},
    {"frac_farm": 0.8, "frac_quota": 0.3},
    {"challenge_legitimate_fraction": 0.3}, {"challenge_subtle_fraction": 1.1},
])
def test_invalid_generation_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        GenConfig(**kwargs)


def test_small_population_supports_spam_sampling(tmp_path):
    cfg = GenConfig(n_accounts=1, frac_farm=0, frac_quota=0,
                    frac_spam_fanout=1, frac_ato=0)
    counts = generate(tmp_path / "tiny.db", cfg)
    assert counts["accounts"] == counts["spam_fanout"] == 1
