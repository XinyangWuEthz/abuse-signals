# Reference benchmark

Start with [summary.md](summary.md). [summary.json](summary.json) contains the
underlying metrics, configuration, dependency versions, and source-file hashes.

This snapshot evaluates 20,000 synthetic accounts per run across seeds 7, 17, and 29
in both baseline and challenge scenarios. Each scenario fits its own models and
thresholds before scoring its held-out accounts.

The snapshot omits generated databases and per-account records. Recreate those
with the command in the report using the setup in the root README. GitHub CI also
uploads the full reports on main-branch pushes and manual runs.

The source hashes identify the evaluated implementation even when the recorded
Git HEAD is the commit preceding local changes. The numerical stack is pinned in
requirements-benchmark.txt; Python, platform, and thread settings are recorded in
summary.json.
