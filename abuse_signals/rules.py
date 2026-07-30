"""YAML-configured deterministic rules layer.

Rules are the precision-first fast path of the detector: interpretable, versioned,
instantly deployable, and safe to attach severe actions to. Model-based coverage
lives in train.py; both meet in the enforcement tiers.
"""

from __future__ import annotations

import argparse
import operator
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .features import load_features

_OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq}

DEFAULT_RULES_PATH = Path(__file__).parent / "rules.yaml"


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    threshold: float

    def matches(self, row: dict) -> bool:
        return _OPS[self.op](row[self.feature], self.threshold)


@dataclass(frozen=True)
class Rule:
    id: str
    action: str
    conditions: tuple[Condition, ...]
    description: str = ""


@dataclass(frozen=True)
class RuleSet:
    rules: tuple[Rule, ...]
    tiers: dict[str, int] = field(default_factory=dict)

    def evaluate(self, row: dict) -> dict:
        """Verdict for one account's feature row: matched rule ids + highest-tier action."""
        matched = [r for r in self.rules if all(c.matches(row) for c in r.conditions)]
        action = None
        if matched:
            action = max((r.action for r in matched), key=lambda a: self.tiers[a])
        return {
            "account_id": row.get("account_id"),
            "matched_rules": [r.id for r in matched],
            "action": action,
        }


def load_rules(path: str | Path = DEFAULT_RULES_PATH) -> RuleSet:
    raw = yaml.safe_load(Path(path).read_text())
    tiers = raw["tiers"]
    rules = []
    for r in raw["rules"]:
        if r["action"] not in tiers:
            raise ValueError(f"rule {r['id']}: unknown action {r['action']!r}")
        conditions = tuple(
            Condition(c["feature"], c["op"], c["threshold"]) for c in r["conditions"]
        )
        for c in conditions:
            if c.op not in _OPS:
                raise ValueError(f"rule {r['id']}: unknown op {c.op!r}")
        rules.append(Rule(r["id"], r["action"], conditions, r.get("description", "")))
    return RuleSet(tuple(rules), tiers)


def precision_recall(verdicts: list[dict], rows: list[dict], action: str, tiers: dict) -> dict:
    """P/R of 'action-or-above' verdicts against ground truth (label != normal)."""
    floor = tiers[action]
    flagged = {v["account_id"] for v in verdicts
               if v["action"] and tiers[v["action"]] >= floor}
    abusive = {r["account_id"] for r in rows if r["label"] != "normal"}
    true_pos = len(flagged & abusive)
    precision = true_pos / len(flagged) if flagged else 1.0
    recall = true_pos / len(abusive) if abusive else 0.0
    return {"action": action, "flagged": len(flagged),
            "precision": precision, "recall": recall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/abuse.db")
    parser.add_argument("--rules", default=str(DEFAULT_RULES_PATH))
    args = parser.parse_args()

    ruleset = load_rules(args.rules)
    rows = load_features(args.db)
    verdicts = [ruleset.evaluate(row) for row in rows]

    print(f"{'tier':>10} {'flagged':>8} {'precision':>10} {'recall':>8}")
    for action in sorted(ruleset.tiers, key=ruleset.tiers.get):
        stats = precision_recall(verdicts, rows, action, ruleset.tiers)
        print(f"{action:>10} {stats['flagged']:>8} "
              f"{stats['precision']:>10.3f} {stats['recall']:>8.3f}")


if __name__ == "__main__":
    main()
