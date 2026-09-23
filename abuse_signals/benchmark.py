"""Generate repeatable synthetic benchmarks and save reports plus account-level evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from statistics import mean
import subprocess

from .config import GenConfig
from .features import build_features
from .generate import generate
from .train import train

DEFAULT_SEEDS = (7, 17, 29)
DEFAULT_SCENARIOS = ("baseline", "challenge")
TIERS = ("throttle", "suspend")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = sorted(path for path in (root / "abuse_signals").rglob("*")
                   if path.suffix in {".py", ".sql", ".yaml"})
    paths += [root / "requirements.txt", root / "requirements-benchmark.txt"]
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in paths}
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    packages = ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "pyyaml")
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {package: version(package) for package in packages},
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")},
        "git_head_at_run": commit, "working_tree_dirty_at_run": dirty,
        "source_sha256": fingerprint, "source_files_sha256": files,
    }


def distribution(values: list) -> dict:
    defined = [value for value in values if value is not None]
    return {"mean": mean(defined) if defined else None,
            "min": min(defined) if defined else None,
            "max": max(defined) if defined else None,
            "defined_runs": len(defined), "total_runs": len(values)}


def aggregate_runs(runs: list[dict]) -> dict:
    aggregate = {}
    for scenario in sorted({run["scenario"] for run in runs}):
        scenario_runs = [run for run in runs if run["scenario"] == scenario]
        policies = {}
        names = ["rules", *scenario_runs[0]["models"]]
        for name in names:
            entries = [run["rules"] if name == "rules" else run["models"][name]
                       for run in scenario_runs]
            policies[name] = {
                "average_precision": distribution([entry.get("average_precision")
                                                   for entry in entries]),
                "tiers": {tier: {
                    metric: distribution([entry["tiers"][tier]["metrics"][metric]
                                          for entry in entries])
                    for metric in ("precision", "recall", "false_positives_per_1000", "flagged")
                } for tier in TIERS},
            }
        aggregate[scenario] = policies
    return aggregate


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def spread(value: dict) -> str:
    if value["mean"] is None:
        return "n/a"
    return f"{fmt(value['mean'])} [{fmt(value['min'])}, {fmt(value['max'])}]"


def render_markdown(summary: dict) -> str:
    lines = [
        "# Synthetic evaluation report", "",
        f"Each run generates {summary['accounts_per_run']:,} accounts. "
        f"Seeds: {', '.join(map(str, summary['seeds']))}. "
        f"Scenarios: {', '.join(summary['scenarios'])}.", "",
        "Each scenario is trained and calibrated separately. Its test accounts are held out. "
        "This compares detection within each synthetic scenario; it does not measure transfer "
        "to a new traffic distribution or production effectiveness.", "",
        "## How to read the results", "",
        "- Rules and both classifiers use the same test accounts in each run.",
        "- Model thresholds and the model selection decision use validation data only.",
        "- Throttle means throttle-or-higher, including suspension. Suspensions are counted once.",
        "- Precision is n/a when no accounts are flagged. An unavailable model tier is disabled.",
        "- The 0.90 throttle and 0.99 suspend targets apply to validation. Test precision can be lower.",
        "- Average precision measures model ranking; it is separate from action-level precision.",
        "- Mean [min, max] summarizes variation across seeds, not a confidence interval.",
        "- Per-run JSON contains 95% Wilson precision intervals. These describe account-level "
        "sampling uncertainty under an independence assumption; related accounts can violate "
        "that assumption. They are not production precision guarantees.", "",
        "## Results across seeds", "",
        "| Scenario | Policy | Average precision, mean [min, max] |",
        "|---|---|---|",
    ]
    for scenario, policies in summary["aggregate"].items():
        for name, result in policies.items():
            lines.append(f"| {scenario} | {name} | {spread(result['average_precision'])} |")
    lines += ["", "| Scenario | Policy | Tier | Precision, mean [min, max] | "
              "Recall, mean [min, max] | FP per 1,000 legitimate accounts, mean [min, max] | "
              "Runs with defined precision |", "|---|---|---|---|---|---|---|"]
    for scenario, policies in summary["aggregate"].items():
        for name, result in policies.items():
            for tier, metrics in result["tiers"].items():
                precision = metrics["precision"]
                lines.append(f"| {scenario} | {name} | {tier} | {spread(precision)} | "
                             f"{spread(metrics['recall'])} | "
                             f"{spread(metrics['false_positives_per_1000'])} | "
                             f"{precision['defined_runs']}/{precision['total_runs']} |")
    lines += ["", "## Individual runs", "",
              "| Scenario | Seed | Train / validation / test accounts | Model chosen on validation |",
              "|---|---|---|---|"]
    for run in summary["runs"]:
        lines.append(f"| {run['scenario']} | {run['seed']} | {run['n_train']} / "
                     f"{run['n_validation']} / {run['n_test']} | {run['selected_model']} |")
    lines += ["", "| Scenario / seed | Policy | Tier | Flagged | TP / FP / FN | "
              "Precision | Recall | FP per 1,000 legitimate accounts |", "|---|---|---|---|---|---|---|---|"]
    for run in summary["runs"]:
        for name, entry in {"rules": run["rules"], **run["models"]}.items():
            for tier, result in entry["tiers"].items():
                m = result["metrics"]
                lines.append(f"| {run['scenario']} / {run['seed']} | {name} | {tier} | "
                             f"{m['flagged']} | {m['tp']} / {m['fp']} / {m['fn']} | "
                             f"{fmt(m['precision'])} | {fmt(m['recall'])} | "
                             f"{fmt(m['false_positives_per_1000'])} |")
    lines += ["", "## Reproduce and inspect", "", "```bash", summary["reproduce_command"],
              "```", "", "The output directory contains:", "",
              "- `summary.json` and `summary.md`: comparisons, environment and source fingerprint.",
              "- `<scenario>/seed-<seed>/report.json`: thresholds, calibration and test metrics, "
              "recall by abuse pattern, errors by legitimate behavior, and incremental model catches.",
              "- `<scenario>/seed-<seed>/splits.json`: exact account and group assignments.",
              "- `<scenario>/seed-<seed>/predictions.jsonl`: every test account, its features and decisions.",
              "- `<scenario>/seed-<seed>/errors.jsonl`: false positives and false negatives by policy and tier.",
              "- `<scenario>/seed-<seed>/accounts.db`: generated events and cutoff-bounded features.",
              "", "Generated databases and detailed account reports are ignored by Git. "
              "A compact reference snapshot can be committed separately.", "",
              f"Source SHA-256: `{summary['provenance']['source_sha256']}`.", "",
              "The source fingerprint includes the evaluator, generator, feature SQL, rules, "
              "and dependency files. The recorded Git HEAD can precede uncommitted changes; "
              "the file hashes identify the code actually evaluated.", ""]
    return "\n".join(lines)


def run_benchmark(out: Path, accounts: int = 20_000, seeds: tuple[int, ...] = DEFAULT_SEEDS,
                  scenarios: tuple[str, ...] = DEFAULT_SCENARIOS, min_alerts: int = 5) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one seed, without duplicates")
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("Provide at least one scenario, without duplicates")
    if not isinstance(min_alerts, int) or isinstance(min_alerts, bool) or min_alerts < 1:
        raise ValueError("min_alerts must be a positive integer")
    # Validate all configurations before creating or replacing any output.
    configs = [GenConfig(n_accounts=accounts, seed=seed, scenario=scenario)
               for scenario in scenarios for seed in seeds]
    out.mkdir(parents=True, exist_ok=True)
    source = provenance()
    runs = []
    for cfg in configs:
        directory = out / cfg.scenario / f"seed-{cfg.seed}"
        directory.mkdir(parents=True, exist_ok=True)
        database = directory / "accounts.db"
        print(f"{cfg.scenario}, seed {cfg.seed}: generating {accounts:,} accounts", flush=True)
        counts = generate(database, cfg)
        build_features(database, as_of=cfg.horizon_end)
        report = train(str(database), seed=cfg.seed, min_alerts=min_alerts)
        report.update({"scenario": cfg.scenario, "config": asdict(cfg), "counts": counts,
                       "as_of": cfg.horizon_end, "source_sha256": source["source_sha256"]})
        write_json(directory / "splits.json", report.pop("splits"))
        write_jsonl(directory / "predictions.jsonl", report.pop("predictions"))
        write_jsonl(directory / "errors.jsonl", report.pop("errors"))
        write_json(directory / "report.json", report)
        runs.append(report)
        print(f"  test accounts: {report['n_test']}; selected on validation: "
              f"{report['selected_model']}", flush=True)
    command = (f"python -m abuse_signals.benchmark --accounts {accounts} --seeds "
               f"{' '.join(map(str, seeds))} --scenarios {' '.join(scenarios)} "
               f"--min-alerts {min_alerts} --out reports/evaluation")
    summary = {"schema_version": 1, "accounts_per_run": accounts, "seeds": list(seeds),
               "scenarios": list(scenarios), "min_alerts": min_alerts,
               "provenance": source, "runs": runs, "aggregate": aggregate_runs(runs),
               "reproduce_command": command}
    write_json(out / "summary.json", summary)
    (out / "summary.md").write_text(render_markdown(summary))
    print(f"Report: {out / 'summary.md'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accounts", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--scenarios", nargs="+", choices=DEFAULT_SCENARIOS,
                        default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--min-alerts", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("reports/evaluation"))
    args = parser.parse_args()
    try:
        run_benchmark(args.out, args.accounts, tuple(args.seeds), tuple(args.scenarios), args.min_alerts)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
