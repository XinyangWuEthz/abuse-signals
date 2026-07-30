-- Account-level feature aggregation, built in stages.
--
-- Performance note (found by profiling, kept as a warning to future me): the first
-- version expressed every signal family as a CTE in one WITH...SELECT. SQLite inlines
-- single-reference CTEs into the outer join and re-evaluates the grouped subqueries
-- per outer row - the build took ~20 minutes on 228k events. Materializing each family
-- into an indexed intermediate table makes every join index-driven: same result in
-- seconds. Regression-guarded by the test-suite runtime itself.

DROP TABLE IF EXISTS account_features;
DROP TABLE IF EXISTS _f_base;
DROP TABLE IF EXISTS _f_bursts;
DROP TABLE IF EXISTS _f_msg;
DROP TABLE IF EXISTS _f_fanout;
DROP TABLE IF EXISTS _f_api;
DROP TABLE IF EXISTS _f_gaps;
DROP TABLE IF EXISTS _f_recent;
DROP TABLE IF EXISTS _f_cohort;

-- Volume and lifetime span.
CREATE TABLE _f_base AS
SELECT a.account_id,
       COUNT(e.ts)                                   AS total_events,
       COALESCE((MAX(e.ts) - MIN(e.ts)) / 3600.0, 0) AS active_span_hours
FROM accounts a
LEFT JOIN events e ON e.account_id = a.account_id
GROUP BY a.account_id;
CREATE UNIQUE INDEX _ix_base ON _f_base(account_id);

-- Velocity: the busiest 5-minute bucket an account ever produced.
CREATE TABLE _f_bursts AS
SELECT account_id, MAX(cnt) AS max_burst_5min
FROM (
    SELECT account_id, ts / 300 AS bucket, COUNT(*) AS cnt
    FROM events
    GROUP BY account_id, bucket
)
GROUP BY account_id;
CREATE UNIQUE INDEX _ix_bursts ON _f_bursts(account_id);

-- Messaging shape: content reuse and spread.
CREATE TABLE _f_msg AS
SELECT account_id,
       COUNT(*)                     AS msg_events,
       COUNT(DISTINCT target_id)    AS distinct_targets,
       COUNT(DISTINCT payload_hash) AS distinct_payloads
FROM events
WHERE action = 'message_send'
GROUP BY account_id;
CREATE UNIQUE INDEX _ix_msg ON _f_msg(account_id);

-- Fan-out: distinct recipients of the single most-spread payload.
CREATE TABLE _f_fanout AS
SELECT account_id, MAX(t) AS max_fanout_per_payload
FROM (
    SELECT account_id, payload_hash, COUNT(DISTINCT target_id) AS t
    FROM events
    WHERE action = 'message_send'
    GROUP BY account_id, payload_hash
)
GROUP BY account_id;
CREATE UNIQUE INDEX _ix_fanout ON _f_fanout(account_id);

CREATE TABLE _f_api AS
SELECT account_id, COUNT(*) AS api_calls
FROM events
WHERE action = 'api_call'
GROUP BY account_id;
CREATE UNIQUE INDEX _ix_api ON _f_api(account_id);

-- Dormancy: the longest silence between two consecutive events.
CREATE TABLE _f_gaps AS
SELECT account_id, MAX(gap) / 3600.0 AS max_gap_hours
FROM (
    SELECT account_id,
           ts - LAG(ts) OVER (PARTITION BY account_id ORDER BY ts) AS gap
    FROM events
)
WHERE gap IS NOT NULL
GROUP BY account_id;
CREATE UNIQUE INDEX _ix_gaps ON _f_gaps(account_id);

-- Recency: activity inside the trailing 48h of the horizon.
-- (Uncorrelated scalar subquery: evaluated once, not per row.)
CREATE TABLE _f_recent AS
SELECT account_id, COUNT(*) AS events_last_48h
FROM events
WHERE ts > (SELECT MAX(ts) FROM events) - 172800
GROUP BY account_id;
CREATE UNIQUE INDEX _ix_recent ON _f_recent(account_id);

-- Signup clustering: accounts sharing ASN + device fingerprint that signed up
-- within +/-30 minutes of each other (windowed count, includes self).
CREATE TABLE _f_cohort AS
SELECT account_id,
       COUNT(*) OVER (
           PARTITION BY asn, device_fingerprint
           ORDER BY signup_ts
           RANGE BETWEEN 1800 PRECEDING AND 1800 FOLLOWING
       ) AS signup_cohort_30min
FROM accounts;
CREATE UNIQUE INDEX _ix_cohort ON _f_cohort(account_id);

-- Assemble: one indexed join per signal family.
CREATE TABLE account_features AS
SELECT b.account_id,
       b.total_events,
       b.active_span_hours,
       CASE WHEN b.active_span_hours > 0
            THEN b.total_events / b.active_span_hours
            ELSE b.total_events END                  AS events_per_hour,
       COALESCE(bu.max_burst_5min, 0)               AS max_burst_5min,
       COALESCE(m.msg_events, 0)                    AS msg_events,
       COALESCE(m.distinct_targets, 0)              AS distinct_targets,
       COALESCE(m.distinct_payloads, 0)             AS distinct_payloads,
       CASE WHEN COALESCE(m.distinct_payloads, 0) > 0
            THEN m.msg_events * 1.0 / m.distinct_payloads
            ELSE 0 END                              AS payload_reuse_ratio,
       COALESCE(f.max_fanout_per_payload, 0)        AS max_fanout_per_payload,
       COALESCE(ap.api_calls, 0)                    AS api_calls,
       COALESCE(g.max_gap_hours, 0)                 AS max_gap_hours,
       COALESCE(r.events_last_48h, 0)               AS events_last_48h,
       c.signup_cohort_30min                        AS signup_cohort_30min
FROM _f_base b
LEFT JOIN _f_bursts bu USING (account_id)
LEFT JOIN _f_msg    m  USING (account_id)
LEFT JOIN _f_fanout f  USING (account_id)
LEFT JOIN _f_api    ap USING (account_id)
LEFT JOIN _f_gaps   g  USING (account_id)
LEFT JOIN _f_recent r  USING (account_id)
JOIN _f_cohort      c  USING (account_id);

CREATE UNIQUE INDEX idx_features_account ON account_features(account_id);

DROP TABLE _f_base;
DROP TABLE _f_bursts;
DROP TABLE _f_msg;
DROP TABLE _f_fanout;
DROP TABLE _f_api;
DROP TABLE _f_gaps;
DROP TABLE _f_recent;
DROP TABLE _f_cohort;
