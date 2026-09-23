"""Build the account_features table by running sql/features.sql against the event DB."""

from __future__ import annotations

import argparse
import json
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


def build_features(db_path: str | Path, as_of: int | None = None) -> int:
    """Build a retrospective snapshot using only observations at or before as_of.

    Generated datasets supply a fixed horizon in their metadata. Older databases
    require an explicit cutoff rather than deriving recency from observed events.
    The trailing 48-hour window includes both endpoints.
    """
    sql = (Path(__file__).parent / "sql" / "features.sql").read_text()
    conn = sqlite3.connect(db_path)
    try:
        if as_of is None:
            metadata_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'dataset_metadata'"
            ).fetchone()
            horizon = conn.execute(
                "SELECT value FROM dataset_metadata WHERE key = 'horizon_end'"
            ).fetchone() if metadata_exists else None
            if horizon is None:
                raise ValueError("database has no horizon_end metadata; supply an explicit --as-of timestamp")
            as_of = int(horizon[0])
        if isinstance(as_of, bool) or not isinstance(as_of, int):
            raise ValueError("as_of must be an integer Unix timestamp")
        conn.execute("CREATE TEMP TABLE _feature_cutoff (as_of INTEGER NOT NULL)")
        conn.execute("INSERT INTO _feature_cutoff VALUES (?)", (as_of,))
        conn.executescript("BEGIN IMMEDIATE;\n" + sql)
        conn.execute("CREATE TABLE IF NOT EXISTS dataset_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT OR REPLACE INTO dataset_metadata VALUES ('feature_as_of', ?)", (str(as_of),))
        conn.commit()
        (count,) = conn.execute("SELECT COUNT(*) FROM account_features").fetchone()
    finally:
        conn.close()
    return count


def load_features(db_path: str | Path) -> list[dict]:
    """Load sorted feature rows with labels, evaluation groups, and behaviors.

    Evaluation metadata never enters FEATURE_COLUMNS. ASN/device groups keep
    related signup cohorts in one split. Old datasets use labels as behaviors.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        has_behaviors = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'account_metadata'"
        ).fetchone()
        behavior_sql = "COALESCE(m.behavior, l.label)" if has_behaviors else "l.label"
        metadata_join = "LEFT JOIN account_metadata m USING (account_id) " if has_behaviors else ""
        rows = conn.execute(
            f"SELECT f.*, l.label, a.asn, a.device_fingerprint, {behavior_sql} AS behavior "
            "FROM account_features f JOIN labels l USING (account_id) "
            "JOIN accounts a USING (account_id) " + metadata_join + "ORDER BY f.account_id"
        ).fetchall()
    finally:
        conn.close()
    result = []
    for row in rows:
        item = dict(row)
        item["group_id"] = json.dumps([item.pop("asn"), item.pop("device_fingerprint")],
                                      separators=(",", ":"))
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/abuse.db")
    parser.add_argument("--as-of", type=int, help="Unix timestamp cutoff; defaults to the recorded generation horizon")
    args = parser.parse_args()
    count = build_features(args.db, as_of=args.as_of)
    print(f"account_features rows: {count}")


if __name__ == "__main__":
    main()
