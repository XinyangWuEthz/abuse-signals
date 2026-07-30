"""End-to-end run: generate -> features -> rules evaluation -> classifier report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abuse_signals.config import GenConfig
from abuse_signals.features import build_features, load_features
from abuse_signals.generate import generate
from abuse_signals.rules import load_rules, precision_recall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/abuse.db")
    parser.add_argument("--accounts", type=int, default=GenConfig.n_accounts)
    parser.add_argument("--seed", type=int, default=GenConfig.seed)
    args = parser.parse_args()

    print("== generate ==")
    counts = generate(args.db, GenConfig(seed=args.seed, n_accounts=args.accounts))
    print(", ".join(f"{k}={v}" for k, v in counts.items()))

    print("\n== features ==")
    print(f"account_features rows: {build_features(args.db)}")

    print("\n== rules layer ==")
    ruleset = load_rules()
    rows = load_features(args.db)
    verdicts = [ruleset.evaluate(row) for row in rows]
    for action in sorted(ruleset.tiers, key=ruleset.tiers.get):
        stats = precision_recall(verdicts, rows, action, ruleset.tiers)
        print(f"{action:>9}-or-above: flagged {stats['flagged']:>5}  "
              f"precision {stats['precision']:.3f}  recall {stats['recall']:.3f}")

    print("\n== classifiers ==")
    try:
        from abuse_signals.train import train
    except ImportError:
        print("scikit-learn not installed; skipping classifier stage")
        return
    report = train(args.db, args.seed)
    for name, entry in report["models"].items():
        print(f"{name}: PR-AUC {entry['pr_auc']:.4f}  "
              + "  ".join(f"{tier} recall@P>={s['precision_floor']:.2f}: {s['recall']:.3f}"
                          for tier, s in entry["tiers"].items()))


if __name__ == "__main__":
    main()
