"""Run one dataset through the shared held-out evaluation. Use benchmark.py for full reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abuse_signals.config import GenConfig
from abuse_signals.features import build_features
from abuse_signals.generate import generate
from abuse_signals.benchmark import fmt, write_json
from abuse_signals.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/abuse.db")
    parser.add_argument("--accounts", type=int, default=GenConfig.n_accounts)
    parser.add_argument("--seed", type=int, default=GenConfig.seed)
    parser.add_argument("--scenario", choices=("baseline", "challenge"), default="baseline")
    parser.add_argument("--out", type=Path, default=Path("reports/pipeline.json"))
    args = parser.parse_args()

    print("== generate ==")
    cfg = GenConfig(seed=args.seed, n_accounts=args.accounts, scenario=args.scenario)
    counts = generate(args.db, cfg)
    print(", ".join(f"{k}={v}" for k, v in counts.items()))

    print("\n== features ==")
    print(f"account_features rows: {build_features(args.db, as_of=cfg.horizon_end)}")

    print("\n== held-out evaluation ==")
    report = train(args.db, args.seed)
    write_json(args.out, report)
    print(f"train / validation / test: {report['n_train']} / "
          f"{report['n_validation']} / {report['n_test']}")
    print(f"model selected on validation: {report['selected_model']}")
    for name, entry in {"rules": report["rules"], **report["models"]}.items():
        print(f"{name}: average precision {fmt(entry.get('average_precision'))}")
        for tier, result in entry["tiers"].items():
            metrics = result["metrics"]
            print(f"  {tier}-or-higher: precision {fmt(metrics['precision'])}, "
                  f"recall {fmt(metrics['recall'])}, false positives {metrics['fp']}")
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
