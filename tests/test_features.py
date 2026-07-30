from statistics import mean

from abuse_signals.features import FEATURE_COLUMNS, load_features

from conftest import TEST_CONFIG


def _by_label(rows):
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    return grouped


def test_one_feature_row_per_account(db_path):
    rows = load_features(db_path)
    assert len(rows) == TEST_CONFIG.n_accounts
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
