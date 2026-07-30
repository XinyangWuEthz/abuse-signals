import sqlite3

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


def test_same_seed_is_deterministic(tmp_path):
    cfg = GenConfig(seed=99, n_accounts=300)
    counts_a = generate(tmp_path / "a.db", cfg)
    counts_b = generate(tmp_path / "b.db", cfg)
    assert counts_a == counts_b

    def totals(path):
        conn = sqlite3.connect(path)
        row = conn.execute("SELECT COUNT(*), SUM(ts) FROM events").fetchone()
        conn.close()
        return row

    assert totals(tmp_path / "a.db") == totals(tmp_path / "b.db")
