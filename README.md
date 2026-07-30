# abuse-signals

[![CI](https://github.com/XinyangWuEthz/abuse-signals/actions/workflows/ci.yml/badge.svg)](https://github.com/XinyangWuEthz/abuse-signals/actions/workflows/ci.yml)

An **account-centric abuse-detection sandbox**: synthetic account event logs → SQL feature
aggregation of account-level signals → a deterministic **rules layer** plus **scikit-learn
classifiers** → precision/recall evaluated at **graduated enforcement thresholds**
(monitor → throttle → suspend).

Built to explore how anti-abuse systems are actually shaped: the unit of detection is the
**account**, not the individual request. One prompt or API call rarely proves abuse; an
account's *history* — signup context, velocity, fan-out, dormancy patterns — usually does.

> Personal learning project. All data is synthetic and generated locally; no real user data
> is involved anywhere.

## Why account-centric?

Content-level filters answer "is this output bad?". They miss the actor: bulk-registered
accounts farming free-tier quota, spam rings fanning one payload out to hundreds of targets,
dormant accounts that suddenly burst after a takeover. Catching the *actor* requires
aggregating signals per account over time — which is what this sandbox does end to end.

## Architecture

```
generate.py                    features.sql                 rules.py / train.py
┌──────────────────┐   SQLite  ┌──────────────────┐  SQLite ┌─────────────────────────┐
│ synthetic events │ ────────► │ SQL window-fn    │ ───────►│ rules layer (YAML)      │
│ + labeled abuse  │  events   │ feature          │ account │   precision-first       │
│   patterns       │  accounts │ aggregation      │ features│ + sklearn classifiers   │
│                  │  labels   │                  │         │   coverage              │
└──────────────────┘           └──────────────────┘         └───────────┬─────────────┘
                                                                        │ verdicts
                                                            ┌───────────▼─────────────┐
                                                            │ enforcement tiers       │
                                                            │ monitor→throttle→suspend│
                                                            │ (serve.py FastAPI API)  │
                                                            └─────────────────────────┘
```

## Abuse patterns and the signals that catch them

| Injected pattern | What it models | Primary signal (SQL) |
|---|---|---|
| `farm` | Bulk account creation from one ASN + device | `signup_cohort_30min` (windowed count over signups) |
| `quota` | Free-tier API farming in high-velocity bursts | `max_burst_5min` (max events per 5-min bucket) |
| `spam_fanout` | One payload fanned out to many targets | `max_fanout_per_payload` (distinct targets per payload) |
| `ato` | Account takeover: long dormancy, then burst | `max_gap_hours` AND `events_last_48h` |

All features are computed in a single SQL pass (window functions, grouped aggregates) —
see [`abuse_signals/sql/features.sql`](abuse_signals/sql/features.sql).

## Why rules AND models

- **Rules** (`rules.yaml`) are the precision-first fast path: interpretable, instantly
  deployable, safe to attach severe actions to. A signup cohort of 10 accounts sharing one
  device fingerprint is abuse with near-certainty — no model needed.
- **Classifiers** add coverage: they catch accounts that stay under every individual
  threshold but look wrong in combination. Severe actions still route through
  precision-calibrated thresholds.

## Graduated enforcement: the false-positive cost asymmetry

Suspending a legitimate account costs far more than missing one abusive account — so severe
actions demand high precision, while cheap reversible actions can lean toward recall:

| Tier | Action | Calibration target |
|---|---|---|
| monitor | log only | maximize recall |
| throttle | rate-limit, reversible | precision ≥ 0.90 |
| suspend | account disabled, appeal path | precision ≥ 0.99 |

`train.py` sweeps the precision/recall curve and reports the achievable recall at each
tier's precision floor — the actual trade-off an enforcement system has to make.

## Quickstart

```bash
pip install -r requirements.txt
python -m abuse_signals.generate --db data/abuse.db          # synthetic logs (~20k accounts)
python -m abuse_signals.features --db data/abuse.db          # SQL feature aggregation
python -m abuse_signals.rules    --db data/abuse.db          # rules layer + per-tier P/R
python -m abuse_signals.train    --db data/abuse.db          # classifiers + threshold sweep
uvicorn abuse_signals.serve:app                              # POST /score {"account_id": 42}
```

Or the whole pipeline at once: `python scripts/run_pipeline.py`

## Testing & CI

`pytest` runs unit tests plus an end-to-end **regression gate**: on a fixed-seed synthetic
dataset, the rules layer must hold suspend-tier precision ≥ 0.90 and the classifier must
hold PR-AUC ≥ 0.85 — the same shape as a production release gate, so a change that degrades
detection fails CI.

## Honest scope & limitations

- **Synthetic data.** Patterns are injected, hence separable; real abuse is noisier, and
  real systems fight label delay and adversarial drift (yesterday's thresholds decay).
- No streaming ingestion, feature store, or appeals/feedback loop — batch SQLite only.
- Thresholds are tuned on the generator's distributions, not on real traffic.

## Roadmap

- [x] Phase 0-1 — synthetic generator, SQL feature aggregation, rules layer, tests, CI gate
- [x] Phase 2 — scikit-learn classifiers + graduated-threshold evaluation
- [x] Phase 3 — FastAPI scoring endpoint
- [ ] Phase 4 — cost-weighted evaluation (explicit FP/FN costs per tier)
- [ ] Phase 5 — adversarial round: mutate abuse generators until detection decays, re-tune

## Development notes

Built with AI-assisted development tooling; problem framing, design decisions, code review,
and validation are my own. Python 3.11+, stdlib `sqlite3`; `scikit-learn`, `fastapi`,
`pyyaml` as the only substantive dependencies.
