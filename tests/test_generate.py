import sqlite3

import pytest

from abuse_signals.config import GenConfig
from abuse_signals.generate import generate

from conftest import TEST_CONFIG


def test_all_labels_present(db_path):
    conn = sqlite3.connect(db_path)
    labels = dict(conn.execute("SELECT label, COUNT(*) FROM labels GROUP BY label"))
    conn.close()
    assert set(labels) == {"normal", "farm", "quota", "spam_fanout", "ato"}
    assert labels["normal"] > sum(v for k, v in labels.items() if k != "normal")


def test_accounts_and_events_written(db_path):
    conn = sqlite3.connect(db_path)
    (n_accounts,) = conn.execute("SELECT COUNT(*) FROM accounts").fetchone()
    (n_events,) = conn.execute("SELECT COUNT(*) FROM events").fetchone()
    (orphans,) = conn.execute(
        "SELECT COUNT(*) FROM events e LEFT JOIN accounts a USING (account_id) "
        "WHERE a.account_id IS NULL"
    ).fetchone()
    conn.close()
    assert n_accounts == TEST_CONFIG.n_accounts
    assert n_events > n_accounts  # events dominate accounts in any realistic log
    assert orphans == 0


@pytest.mark.parametrize("scenario", ["baseline", "challenge"])
def test_same_seed_is_deterministic(tmp_path, scenario):
    cfg = GenConfig(seed=99, n_accounts=300, scenario=scenario)
    counts_a = generate(tmp_path / "a.db", cfg)
    counts_b = generate(tmp_path / "b.db", cfg)
    assert counts_a == counts_b

    def logical_rows(path):
        with sqlite3.connect(path) as conn:
            return {
                table: conn.execute(f"SELECT * FROM {table} ORDER BY {ordering}").fetchall()
                for table, ordering in [
                    ("accounts", "account_id"),
                    ("events", "account_id, ts, action, payload_hash, target_id, ip"),
                    ("labels", "account_id"),
                    ("account_metadata", "account_id"),
                    ("dataset_metadata", "key"),
                ]
            }

    assert logical_rows(tmp_path / "a.db") == logical_rows(tmp_path / "b.db")
