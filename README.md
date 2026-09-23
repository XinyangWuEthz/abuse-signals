# abuse-signals

[![CI](https://github.com/XinyangWuEthz/abuse-signals/actions/workflows/ci.yml/badge.svg)](https://github.com/XinyangWuEthz/abuse-signals/actions/workflows/ci.yml)

An account-level abuse-detection prototype with a reproducible offline evaluation.
It generates synthetic activity, builds SQL behavioral features, and compares rules with
supervised classifiers on the same held-out accounts.

**All accounts, events, and labels are synthetic.** The results describe these experiments,
not real traffic or a deployed enforcement system.

## What is implemented

| Part | What it does |
|---|---|
| Data generator | Creates normal users and four abuse types: signup farms, quota farming, spam fan-out, and account takeover. |
| Two scenarios | `baseline` contains clear injected patterns. `challenge` adds legitimate lookalikes and subtler abuse. |
| SQL features | Builds 13 account-level signals with a fixed observation cutoff, using indexed intermediate tables. |
| Rules | Four YAML rules return matched rule IDs and suggested actions. |
| Classifiers | Logistic regression and histogram gradient boosting learn from the SQL features. |
| Evaluation | Keeps related accounts together, separates training/calibration/testing, and compares all detectors on identical test accounts. |
| Reports | Saves metrics, split assignments, predictions, and inspectable mistakes. |
| API | FastAPI serves the rule verdict for a stored account. It does not serve classifiers or execute enforcement actions. |
| CI | Runs tests and quality regression gates; main-branch and manual runs also publish the full benchmark as an artifact. |

```text
synthetic accounts + events
          |
          v
SQL features at a fixed cutoff
          |
          v
grouped train / validation / test split
          |
          +--> train: fit models and preprocessing
          +--> validation: choose thresholds and select a model
          +--> test: compare rules and both fixed models
                          |
                          v
               metrics + predictions + mistakes
```

## What the benchmark found

The reference run uses 20,000 accounts per experiment, two scenarios, and three seeds.
These are mean held-out results across the three seeds. P/R means precision / recall;
throttle includes suspension. The full report includes the range and account counts.

| Scenario | Detector | Average precision | Throttle P/R | Suspend P/R |
|---|---|---|---|---|
| Baseline | Rules | n/a | 1.000 / 1.000 | 1.000 / 0.666 |
| Baseline | Logistic regression | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Baseline | Gradient boosting | 1.000 | 1.000 / 0.993 | 1.000 / 0.993 |
| Challenge | Rules | n/a | 0.195 / 0.508 | 0.378 / 0.341 |
| Challenge | Logistic regression | 0.883 | 0.920 / 0.641 | 1.000 / 0.173 |
| Challenge | Gradient boosting | 0.988 | 0.938 / 0.958 | 0.996 / 0.905 |

The clear baseline patterns remain easy to detect. Adding legitimate lookalikes exposes
many false positives in the fixed rules. Gradient boosting handles this constructed
challenge better, but it was trained on the challenge distribution too. This experiment
does not demonstrate adaptation to unseen adversarial behavior.

See the [reference report](benchmarks/reference/summary.md) for details. These results
supersede the original single-seed measurements, which used a different split and a
recency window tied to the latest observed event.

## Run it

Use Python 3.12 for the reference benchmark. The numerical dependency versions are pinned.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-benchmark.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2

# One smaller end-to-end run, also suitable for the rule API
python scripts/run_pipeline.py --accounts 5000 --db data/abuse.db

# Full benchmark: 20K accounts per run, 3 seeds, 2 scenarios
python -m abuse_signals.benchmark --out reports/evaluation

# Tests
pytest -q
```

The full command uses seeds `7 17 29`. Override them explicitly if needed:

```bash
python -m abuse_signals.benchmark --accounts 20000 --seeds 7 17 29 \
  --scenarios baseline challenge --min-alerts 5 --out reports/evaluation
```

To inspect the rule API after the smaller run:

```bash
uvicorn abuse_signals.serve:app
curl -X POST localhost:8000/score -H 'Content-Type: application/json' \
  -d '{"account_id": 42}'
```

## How evaluation works

1. **Fix the observation time.** All features use events and signups at or before the recorded
   generation horizon. The recent-activity window ends at that timestamp. Future events cannot
   change an earlier snapshot. A custom cutoff is available through `features --as-of`.
2. **Keep related accounts together.** Accounts sharing an ASN and device stay in one partition.
   The split targets roughly 60% training, 20% validation, and 20% testing while preserving
   abuse-label and sufficiently represented behavior slices. Exact sizes vary with cohort sizes. Too few independent groups
   produce a clear error.
3. **Fit and select before testing.** Preprocessing and classifiers fit only training data.
   Validation selects the model by average precision and selects action thresholds.
   The test set never chooses models or thresholds.
4. **Evaluate fixed decisions.** Rules and both classifiers use the same test accounts.
   Throttle calibration targets precision of 0.90; suspend targets 0.99.
   Each threshold needs at least five validation alerts by default. An infeasible tier is disabled.
   These are empirical validation targets; test precision may be lower.
5. **Repeat the experiment.** Each scenario is trained and calibrated separately for each seed.
   Report the mean and range across seeds, keeping baseline and challenge results separate.

The challenge scenario assigns 4% of normal accounts to each of four behaviors: shared-device
signups, bursty API applications, authorized broadcasts, and returning dormant users.
Half of each abuse category uses a subtler variant, such as smaller signup groups, slower
requests, or more diverse message payloads. These distributions are deliberately constructed,
not estimates of production traffic.

Group and behavior metadata are used for splitting and reporting only. They are excluded
from the 13 classifier inputs. The shared observation cutoff defines retrospective scoring;
this is not a simulation of detection as each event arrives.

## Read the results

Start with [`benchmarks/reference/summary.md`](benchmarks/reference/summary.md) for the checked-in
reference run, or open `reports/evaluation/summary.md` after running the benchmark.

| Measurement | Meaning |
|---|---|
| Average precision | Classifier ranking quality on held-out accounts. Computed with `average_precision_score`, not trapezoidal PR-AUC. |
| Precision and recall by tier | Detection at thresholds already chosen on validation data. `throttle` means throttle-or-higher, including suspended accounts. |
| TP / FP / FN and flagged count | The account counts behind each percentage. No alerts means undefined precision, shown as `n/a`. |
| False positives per 1,000 legitimate accounts | How often a policy flags normal users. |
| Recall by abuse pattern | Which kinds of abuse are missed. |
| False positives by normal behavior | Which legitimate users are affected. |
| Incremental model detections | Additional abusive and normal accounts flagged beyond the rules at the same tier, plus rule detections the model misses. |
| Precision interval | A descriptive 95% Wilson interval. It assumes independent accounts and does not correct for cohort dependence. |

Repeated-seed ranges are not confidence intervals. Neither an interval nor a perfect synthetic
score establishes a production guarantee. Account-level abuse labels also do not establish
that suspension is an appropriate response to every detected account.

Each run writes the following files under `<scenario>/seed-<seed>/`:

- `report.json`: generation settings, cutoff, selected thresholds, validation results, test
  metrics, behavior slices, and incremental detections.
- `splits.json`: exact account IDs and group IDs in every partition.
- `predictions.jsonl`: all test accounts with their features, rule matches, model scores, and actions.
- `errors.jsonl`: false positives and false negatives by detector and tier.
- `accounts.db`: synthetic source records and aggregated features.

`summary.json` records the Python/package versions and hashes of the source files.
The recorded Git HEAD may precede uncommitted changes; the source hashes identify the code
actually evaluated. Generated databases and detailed reports are ignored by Git.
The compact reference snapshot contains the summary and metrics, without databases or
per-account records.

## Tests and CI

The 63 tests cover deterministic generation, scenario composition, future-data exclusion,
group isolation, validation-only threshold selection, disabled tiers, cumulative actions,
undefined metrics, and saved evidence matching the report.

A fixed-seed 5,000-account baseline fixture checks held-out rule suspension precision
of at least 0.90 with nonzero support, rule throttle recall of at least 0.50, and
average precision of at least 0.85 for **each** classifier. Each classifier also needs
throttle precision of at least 0.85, recall of at least 0.50, and at least five alerts.
These floors detect regressions on the simple fixture. They are separate from the
validation calibration targets and from the published 20K-account measurements.

Pull requests run the tests. Main-branch pushes and manual CI runs also execute the six-run
benchmark and upload its JSON, Markdown, and per-account evidence as a 30-day artifact.

## Remaining work

- Evaluate transfer across changing traffic distributions with frozen models and thresholds.
- Add action-specific cost scenarios after choosing explicit cost assumptions.
- Validate on data beyond hand-designed synthetic patterns.
- Persist models and add classifier inference if the API is extended.

The current project uses batch SQLite. It has no streaming ingestion, real enforcement,
appeals workflow, or production deployment.

Built with AI-assisted development tooling. This repository is a personal learning project.
