import sqlite3
from pathlib import Path
from statistics import mean

import pytest

from abuse_signals.features import FEATURE_COLUMNS, build_features, load_features

from conftest import TEST_CONFIG


def _by_label(rows):
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    return grouped


def test_one_feature_row_per_account(db_path):
    rows = load_features(db_path)
    assert len(rows) == TEST_CONFIG.n_accounts
    assert [row["account_id"] for row in rows] == sorted(row["account_id"] for row in rows)
    assert all(row["group_id"] and row["behavior"] for row in rows)
    assert "group_id" not in FEATURE_COLUMNS
    assert "behavior" not in FEATURE_COLUMNS
    for row in rows[:50]:
        for column in FEATURE_COLUMNS:
            assert row[column] is not None


def test_each_abuse_pattern_moves_its_signal(db_path):
    """Every injected pattern must separate from normal on the signal built to catch it."""
    grouped = _by_label(load_features(db_path))
    normal = grouped["normal"]

    signal_by_label = {
        "farm": "signup_cohort_30min",
        "quota": "max_burst_5min",
        "spam_fanout": "max_fanout_per_payload",
        "ato": "max_gap_hours",
    }
    for label, signal in signal_by_label.items():
        abusive_mean = mean(row[signal] for row in grouped[label])
        normal_mean = mean(row[signal] for row in normal)
        assert abusive_mean > 3 * normal_mean, (
            f"{label}: {signal} does not separate "
            f"(abusive {abusive_mean:.1f} vs normal {normal_mean:.1f})"
        )


def _snapshot_db(tmp_path, accounts, events):
    path = tmp_path / "snapshot.db"
    schema = Path(__file__).resolve().parents[1] / "abuse_signals" / "sql" / "schema.sql"
    with sqlite3.connect(path) as conn:
        conn.executescript(schema.read_text())
        conn.executemany("INSERT INTO accounts VALUES (?,?,?,?,?,?)", accounts)
        conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", events)
        conn.executemany("INSERT INTO labels VALUES (?, 'normal')", [(a[0],) for a in accounts])
    return path


def test_future_accounts_and_events_do_not_change_snapshot(tmp_path):
    cutoff = 1_000_000
    path = _snapshot_db(tmp_path, [
        (10, cutoff - 1000, "ip", 1, "shared", "example.org"),
        (20, cutoff - 100, "ip", 1, "shared", "example.org"),
    ], [(10, cutoff - 50, "login", None, None, "ip")])
    build_features(path, as_of=cutoff)
    before = load_features(path)
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO accounts VALUES (?, ?, 'ip', 1, 'shared', 'example.org')",
                     (30, cutoff + 1))
        conn.execute("INSERT INTO labels VALUES (30, 'normal')")
        conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", [
            (10, cutoff + 1, "message_send", "future", 99, "ip"),
            (10, cutoff + 200000, "api_call", None, None, "ip"),
            (20, cutoff + 1, "message_send", "future", 100, "ip"),
            (30, cutoff + 1, "login", None, None, "ip"),
        ])
    build_features(path, as_of=cutoff)
    assert load_features(path) == before
    assert before[0]["signup_cohort_30min"] == 2
    assert before[0]["group_id"] == before[1]["group_id"]
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM dataset_metadata WHERE key = 'feature_as_of'").fetchone() == (str(cutoff),)


def test_cutoff_boundaries_and_accounts_without_events(tmp_path):
    cutoff = 1_000_000
    path = _snapshot_db(tmp_path, [
        (1, cutoff - 300000, "ip", 1, "one", "example.org"),
        (2, cutoff, "ip", 1, "two", "example.org"),
    ], [(1, ts, "api_call", None, None, "ip")
        for ts in [cutoff - 172801, cutoff - 172800, cutoff, cutoff + 1]])
    build_features(path, as_of=cutoff)
    active, quiet = load_features(path)
    assert active["total_events"] == 3
    assert active["api_calls"] == 3
    assert active["events_last_48h"] == 2
    assert active["max_gap_hours"] == 48
    assert quiet["total_events"] == quiet["max_gap_hours"] == quiet["events_last_48h"] == 0
    assert all(quiet[column] is not None for column in FEATURE_COLUMNS)


def test_old_database_requires_an_explicit_cutoff(tmp_path):
    path = _snapshot_db(tmp_path, [(1, 1000, "ip", 1, "one", "example.org")], [])
    with pytest.raises(ValueError, match="explicit --as-of"):
        build_features(path)
    assert build_features(path, as_of=1000) == 1
    assert load_features(path)[0]["behavior"] == "normal"
