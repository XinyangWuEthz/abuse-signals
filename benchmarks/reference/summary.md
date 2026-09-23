# Synthetic evaluation report

Each run generates 20,000 accounts. Seeds: 7, 17, 29. Scenarios: baseline, challenge.

Each scenario is trained and calibrated separately. Its test accounts are held out. This compares detection within each synthetic scenario; it does not measure transfer to a new traffic distribution or production effectiveness.

## How to read the results

- Rules and both classifiers use the same test accounts in each run.
- Model thresholds and the model selection decision use validation data only.
- Throttle means throttle-or-higher, including suspension. Suspensions are counted once.
- Precision is n/a when no accounts are flagged. An unavailable model tier is disabled.
- The 0.90 throttle and 0.99 suspend targets apply to validation. Test precision can be lower.
- Average precision measures model ranking; it is separate from action-level precision.
- Mean [min, max] summarizes variation across seeds, not a confidence interval.
- Per-run JSON contains 95% Wilson precision intervals. These describe account-level sampling uncertainty under an independence assumption; related accounts can violate that assumption. They are not production precision guarantees.

## Results across seeds

| Scenario | Policy | Average precision, mean [min, max] |
|---|---|---|
| baseline | rules | n/a |
| baseline | logistic_regression | 1.000 [1.000, 1.000] |
| baseline | hist_gradient_boosting | 1.000 [1.000, 1.000] |
| challenge | rules | n/a |
| challenge | logistic_regression | 0.883 [0.867, 0.906] |
| challenge | hist_gradient_boosting | 0.988 [0.981, 0.993] |

| Scenario | Policy | Tier | Precision, mean [min, max] | Recall, mean [min, max] | FP per 1,000 legitimate accounts, mean [min, max] | Runs with defined precision |
|---|---|---|---|---|---|---|
| baseline | rules | throttle | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 3/3 |
| baseline | rules | suspend | 1.000 [1.000, 1.000] | 0.666 [0.655, 0.674] | 0.000 [0.000, 0.000] | 3/3 |
| baseline | logistic_regression | throttle | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 3/3 |
| baseline | logistic_regression | suspend | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 3/3 |
| baseline | hist_gradient_boosting | throttle | 1.000 [1.000, 1.000] | 0.993 [0.989, 1.000] | 0.000 [0.000, 0.000] | 3/3 |
| baseline | hist_gradient_boosting | suspend | 1.000 [1.000, 1.000] | 0.993 [0.989, 1.000] | 0.000 [0.000, 0.000] | 3/3 |
| challenge | rules | throttle | 0.195 [0.185, 0.210] | 0.508 [0.495, 0.522] | 100.575 [93.717, 104.919] | 3/3 |
| challenge | rules | suspend | 0.378 [0.351, 0.426] | 0.341 [0.332, 0.346] | 27.128 [22.251, 29.566] | 3/3 |
| challenge | logistic_regression | throttle | 0.920 [0.891, 0.967] | 0.641 [0.628, 0.648] | 2.704 [1.047, 3.663] | 3/3 |
| challenge | logistic_regression | suspend | 1.000 [1.000, 1.000] | 0.173 [0.110, 0.228] | 0.000 [0.000, 0.000] | 3/3 |
| challenge | hist_gradient_boosting | throttle | 0.938 [0.921, 0.957] | 0.958 [0.951, 0.967] | 3.053 [2.093, 3.925] | 3/3 |
| challenge | hist_gradient_boosting | suspend | 0.996 [0.988, 1.000] | 0.905 [0.886, 0.929] | 0.174 [0.000, 0.523] | 3/3 |

## Individual runs

| Scenario | Seed | Train / validation / test accounts | Model chosen on validation |
|---|---|---|---|
| baseline | 7 | 11994 / 4002 / 4004 | logistic_regression |
| baseline | 17 | 12011 / 3995 / 3994 | hist_gradient_boosting |
| baseline | 29 | 12004 / 3995 / 4001 | logistic_regression |
| challenge | 7 | 11996 / 4002 / 4002 | hist_gradient_boosting |
| challenge | 17 | 12001 / 3994 / 4005 | hist_gradient_boosting |
| challenge | 29 | 11988 / 4006 / 4006 | hist_gradient_boosting |

| Scenario / seed | Policy | Tier | Flagged | TP / FP / FN | Precision | Recall | FP per 1,000 legitimate accounts |
|---|---|---|---|---|---|---|---|
| baseline / 7 | rules | throttle | 184 | 184 / 0 / 0 | 1.000 | 1.000 | 0.000 |
| baseline / 7 | rules | suspend | 124 | 124 / 0 / 60 | 1.000 | 0.674 | 0.000 |
| baseline / 7 | logistic_regression | throttle | 184 | 184 / 0 / 0 | 1.000 | 1.000 | 0.000 |
| baseline / 7 | logistic_regression | suspend | 184 | 184 / 0 / 0 | 1.000 | 1.000 | 0.000 |
| baseline / 7 | hist_gradient_boosting | throttle | 182 | 182 / 0 / 2 | 1.000 | 0.989 | 0.000 |
| baseline / 7 | hist_gradient_boosting | suspend | 182 | 182 / 0 / 2 | 1.000 | 0.989 | 0.000 |
| baseline / 17 | rules | throttle | 174 | 174 / 0 / 0 | 1.000 | 1.000 | 0.000 |
| baseline / 17 | rules | suspend | 114 | 114 / 0 / 60 | 1.000 | 0.655 | 0.000 |
| baseline / 17 | logistic_regression | throttle | 174 | 174 / 0 / 0 | 1.000 | 1.000 | 0.000 |
| baseline / 17 | logistic_regression | suspend | 174 | 174 / 0 / 0 | 1.000 | 1.000 | 0.000 |
| baseline / 17 | hist_gradient_boosting | throttle | 174 | 174 / 0 / 0 | 1.000 | 1.000 | 0.000 |
| baseline / 17 | hist_gradient_boosting | suspend | 174 | 174 / 0 / 0 | 1.000 | 1.000 | 0.000 |
| baseline / 29 | rules | throttle | 181 | 181 / 0 / 0 | 1.000 | 1.000 | 0.000 |
| baseline / 29 | rules | suspend | 121 | 121 / 0 / 60 | 1.000 | 0.669 | 0.000 |
| baseline / 29 | logistic_regression | throttle | 181 | 181 / 0 / 0 | 1.000 | 1.000 | 0.000 |
| baseline / 29 | logistic_regression | suspend | 181 | 181 / 0 / 0 | 1.000 | 1.000 | 0.000 |
| baseline / 29 | hist_gradient_boosting | throttle | 179 | 179 / 0 / 2 | 1.000 | 0.989 | 0.000 |
| baseline / 29 | hist_gradient_boosting | suspend | 179 | 179 / 0 / 2 | 1.000 | 0.989 | 0.000 |
| challenge / 7 | rules | throttle | 453 | 95 / 358 / 87 | 0.210 | 0.522 | 93.717 |
| challenge / 7 | rules | suspend | 148 | 63 / 85 / 119 | 0.426 | 0.346 | 22.251 |
| challenge / 7 | logistic_regression | throttle | 122 | 118 / 4 / 64 | 0.967 | 0.648 | 1.047 |
| challenge / 7 | logistic_regression | suspend | 20 | 20 / 0 / 162 | 1.000 | 0.110 | 0.000 |
| challenge / 7 | hist_gradient_boosting | throttle | 186 | 174 / 12 / 8 | 0.935 | 0.956 | 3.141 |
| challenge / 7 | hist_gradient_boosting | suspend | 164 | 164 / 0 / 18 | 1.000 | 0.901 | 0.000 |
| challenge / 17 | rules | throttle | 487 | 93 / 394 / 90 | 0.191 | 0.508 | 103.087 |
| challenge / 17 | rules | suspend | 176 | 63 / 113 / 120 | 0.358 | 0.344 | 29.566 |
| challenge / 17 | logistic_regression | throttle | 129 | 115 / 14 / 68 | 0.891 | 0.628 | 3.663 |
| challenge / 17 | logistic_regression | suspend | 33 | 33 / 0 / 150 | 1.000 | 0.180 | 0.000 |
| challenge / 17 | hist_gradient_boosting | throttle | 189 | 174 / 15 / 9 | 0.921 | 0.951 | 3.925 |
| challenge / 17 | hist_gradient_boosting | suspend | 172 | 170 / 2 / 13 | 0.988 | 0.929 | 0.523 |
| challenge / 29 | rules | throttle | 492 | 91 / 401 / 93 | 0.185 | 0.495 | 104.919 |
| challenge / 29 | rules | suspend | 174 | 61 / 113 / 123 | 0.351 | 0.332 | 29.566 |
| challenge / 29 | logistic_regression | throttle | 132 | 119 / 13 / 65 | 0.902 | 0.647 | 3.401 |
| challenge / 29 | logistic_regression | suspend | 42 | 42 / 0 / 142 | 1.000 | 0.228 | 0.000 |
| challenge / 29 | hist_gradient_boosting | throttle | 186 | 178 / 8 / 6 | 0.957 | 0.967 | 2.093 |
| challenge / 29 | hist_gradient_boosting | suspend | 163 | 163 / 0 / 21 | 1.000 | 0.886 | 0.000 |

## Reproduce and inspect

```bash
python -m abuse_signals.benchmark --accounts 20000 --seeds 7 17 29 --scenarios baseline challenge --min-alerts 5 --out reports/evaluation
```

The output directory contains:

- `summary.json` and `summary.md`: comparisons, environment and source fingerprint.
- `<scenario>/seed-<seed>/report.json`: thresholds, calibration and test metrics, recall by abuse pattern, errors by legitimate behavior, and incremental model catches.
- `<scenario>/seed-<seed>/splits.json`: exact account and group assignments.
- `<scenario>/seed-<seed>/predictions.jsonl`: every test account, its features and decisions.
- `<scenario>/seed-<seed>/errors.jsonl`: false positives and false negatives by policy and tier.
- `<scenario>/seed-<seed>/accounts.db`: generated events and cutoff-bounded features.

Generated databases and detailed account reports are ignored by Git. A compact reference snapshot can be committed separately.

Source SHA-256: `26e0b1376dd847d71880168739c27fdc4afe2d350d5310651da6204d724d11db`.

The source fingerprint includes the evaluator, generator, feature SQL, rules, and dependency files. The recorded Git HEAD can precede uncommitted changes; the file hashes identify the code actually evaluated.
