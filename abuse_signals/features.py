"""Build the account_features table by running sql/features.sql against the event DB."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

FEATURE_COLUMNS = [
    "total_events",
    "active_span_hours",
    "events_per_hour",
    "max_burst_5min",
    "msg_events",
    "distinct_targets",
    "distinct_payloads",
    "payload_reuse_ratio",
    "max_fanout_per_payload",
    "api_calls",
    "max_gap_hours",
    "events_last_48h",
    "signup_cohort_30min",
]


def build_features(db_path: str | Path) -> int:
    sql = (Path(__file__).parent / "sql" / "features.sql").read_text()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(sql)
        conn.commit()
        (count,) = conn.execute("SELECT COUNT(*) FROM account_features").fetchone()
    finally:
        conn.close()
    return count


def load_features(db_path: str | Path) -> list[dict]:
    """account_features joined with ground-truth labels, as a list of dicts."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT f.*, l.label FROM account_features f "
            "JOIN labels l ON l.account_id = f.account_id"
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/abuse.db")
    args = parser.parse_args()
    count = build_features(args.db)
    print(f"account_features rows: {count}")


if __name__ == "__main__":
    main()
