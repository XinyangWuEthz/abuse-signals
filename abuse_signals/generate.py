"""Synthetic account/event-log generator with labeled abuse patterns.

Everything is drawn from a seeded RNG: the same config gives identical logical
rows. Baseline retains the original patterns; challenge adds legitimate overlap
and less conspicuous abuse. Behavior metadata is evaluation-only ground truth.

Injected patterns (see config.py for fractions):
  farm         bulk signups sharing one ASN + device fingerprint inside a 30-min window
  quota        high-velocity api_call bursts (free-tier farming)
  spam_fanout  one payload_hash fanned out to hundreds of distinct targets
  ato          account takeover: long dormancy, then a burst from a new IP
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
import sqlite3
from pathlib import Path

from .config import GenConfig

# Waking-hours weighting for normal users; abuse generators ignore it on purpose.
DIURNAL_WEIGHTS = [1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 7, 7, 8, 8, 8, 7, 6, 5, 4, 3, 2, 1]

NORMAL_EMAIL_DOMAINS = ["gmail.com", "outlook.com", "proton.me", "yahoo.com", "gmx.ch"]


def _diurnal_ts(rng: random.Random, day_start: int) -> int:
    hour = rng.choices(range(24), weights=DIURNAL_WEIGHTS)[0]
    return day_start + hour * 3600 + rng.randrange(3600)


def _ip(rng: random.Random, asn: int) -> str:
    return f"{10 + asn % 200}.{asn % 250}.{rng.randrange(256)}.{rng.randrange(256)}"


class _Dataset:
    def __init__(self) -> None:
        self.accounts: list[tuple] = []  # (id, signup_ts, ip, asn, device, email_domain)
        self.events: list[tuple] = []    # (account_id, ts, action, payload_hash, target_id, ip)
        self.labels: list[tuple] = []    # (account_id, label)
        self.behaviors: list[tuple] = [] # (account_id, evaluation-only behavior)


def _add_normal(ds: _Dataset, rng: random.Random, cfg: GenConfig, account_id: int,
                behavior: str = "normal") -> None:
    asn = rng.randrange(cfg.n_asns)
    signup = cfg.start_ts + rng.randrange(max(1, cfg.days * 86_400 - 86_400))
    ip = _ip(rng, asn)
    ds.accounts.append((account_id, signup, ip, asn, f"dev-{account_id}",
                        rng.choice(NORMAL_EMAIL_DOMAINS)))
    ds.labels.append((account_id, "normal"))
    ds.behaviors.append((account_id, behavior))

    daily = rng.randint(cfg.min_daily_events, cfg.max_daily_events)
    active_days = max(1, (cfg.horizon_end - signup) // 86_400)
    personal_payloads = [f"p{account_id}-{i}" for i in range(rng.randint(20, 120))]
    for _ in range(int(daily * active_days * rng.uniform(0.5, 1.2))):
        day = rng.randrange(active_days)
        ts = _diurnal_ts(rng, signup + day * 86_400)
        action = rng.choices(["login", "api_call", "message_send"], weights=[1, 6, 3])[0]
        payload = rng.choice(personal_payloads) if action == "message_send" else None
        target = rng.randrange(cfg.n_accounts * 5) if action == "message_send" else None
        ds.events.append((account_id, ts, action, payload, target, ip))


def _add_farm(ds: _Dataset, rng: random.Random, cfg: GenConfig, ids: list[int],
              subtle: bool = False) -> None:
    """Cohorts of 8-25 accounts signing up from one ASN+device within 30 minutes."""
    i = 0
    cohort_no = 0
    while i < len(ids):
        size = min(rng.randint(2, 6) if subtle else rng.randint(8, 25), len(ids) - i)
        asn = rng.randrange(cfg.n_asns)
        device = f"dev-farm-{'small-' if subtle else ''}{cohort_no}"
        domain = rng.choice(NORMAL_EMAIL_DOMAINS) if subtle else f"{rng.randrange(10**8):08x}.top"
        base = cfg.start_ts + rng.randrange(cfg.days * 86_400 - 7_200)
        for account_id in ids[i:i + size]:
            signup = base + rng.randrange(min(7_200 if subtle else 1_800, cfg.horizon_end - base))
            ip = _ip(rng, asn)
            ds.accounts.append((account_id, signup, ip, asn, device, domain))
            ds.labels.append((account_id, "farm"))
            ds.behaviors.append((account_id, "farm_small" if subtle else "farm"))
            for _ in range(rng.randint(0, 5)):  # farms stay quiet after creation
                ds.events.append((account_id, signup + rng.randrange(min(86_400, cfg.horizon_end - signup)),
                                  "login", None, None, ip))
        i += size
        cohort_no += 1


def _add_quota(ds: _Dataset, rng: random.Random, cfg: GenConfig, ids: list[int],
               subtle: bool = False) -> None:
    """Normal-looking signup, then 300-700 api_calls compressed into 1-5 minute bursts."""
    for account_id in ids:
        asn = rng.randrange(cfg.n_asns)
        signup = cfg.start_ts + rng.randrange(cfg.days * 86_400 // 2)
        ip = _ip(rng, asn)
        ds.accounts.append((account_id, signup, ip, asn, f"dev-{account_id}",
                            rng.choice(NORMAL_EMAIL_DOMAINS)))
        ds.labels.append((account_id, "quota"))
        ds.behaviors.append((account_id, "quota_slow" if subtle else "quota"))
        for _ in range(rng.randint(2, 5)):
            start = signup + rng.randrange(max(1, cfg.horizon_end - signup - 3_600))
            window = rng.randint(900, 3_600) if subtle else rng.randint(60, 300)
            for _ in range(rng.randint(120, 350) if subtle else rng.randint(300, 700)):
                ds.events.append((account_id, start + rng.randrange(window),
                                  "api_call", None, None, ip))


def _add_spam_fanout(ds: _Dataset, rng: random.Random, cfg: GenConfig, ids: list[int],
                     subtle: bool = False) -> None:
    """A handful of payloads, one of them sent to 180-500 distinct targets."""
    for account_id in ids:
        asn = rng.randrange(cfg.n_asns)
        signup = cfg.start_ts + rng.randrange(max(1, cfg.days * 86_400 - 86_400))
        ip = _ip(rng, asn)
        ds.accounts.append((account_id, signup, ip, asn, f"dev-{account_id}",
                            rng.choice(NORMAL_EMAIL_DOMAINS)))
        ds.labels.append((account_id, "spam_fanout"))
        ds.behaviors.append((account_id, "spam_varied" if subtle else "spam_fanout"))
        main_payload = f"spam-{account_id}"
        targets = rng.sample(range(cfg.n_accounts * 5), min(cfg.n_accounts * 5, rng.randint(180, 500)))
        n_payloads = rng.randint(5, 20) if subtle else 1
        span = max(3_600, cfg.horizon_end - signup)
        for target in targets:
            ts = signup + rng.randrange(span)
            payload = f"{main_payload}-{rng.randrange(n_payloads)}" if subtle else main_payload
            ds.events.append((account_id, ts, "message_send", payload, target, ip))
        for _ in range(rng.randint(5, 30)):  # a little cover traffic
            ds.events.append((account_id, signup + rng.randrange(span), "login", None, None, ip))


def _add_ato(ds: _Dataset, rng: random.Random, cfg: GenConfig, ids: list[int],
             subtle: bool = False) -> None:
    """Early signup, light activity, >=7 days of silence, then a takeover burst."""
    for account_id in ids:
        asn = rng.randrange(cfg.n_asns)
        signup = cfg.start_ts + rng.randrange(min(2 * 86_400, cfg.days * 86_400 // 7))
        home_ip = _ip(rng, asn)
        ds.accounts.append((account_id, signup, home_ip, asn, f"dev-{account_id}",
                            rng.choice(NORMAL_EMAIL_DOMAINS)))
        ds.labels.append((account_id, "ato"))
        ds.behaviors.append((account_id, "ato_subtle" if subtle else "ato"))
        for _ in range(rng.randint(5, 20)):  # benign early activity, then silence
            ds.events.append((account_id, signup + rng.randrange(min(3 * 86_400, cfg.days * 86_400 // 4)),
                              "login" if rng.random() < 0.5 else "api_call",
                              None, None, home_ip))
        attacker_ip = _ip(rng, (asn + 137) % cfg.n_asns)
        if subtle and rng.random() < 0.5:
            attacker_ip = home_ip
        burst_span = min((4 if subtle else 2) * 86_400,
                         cfg.days * 86_400 // (3 if subtle else 7))
        burst_start = cfg.horizon_end - burst_span
        n_payloads = rng.randint(4, 12) if subtle else 1
        for _ in range(rng.randint(70, 180) if subtle else rng.randint(120, 300)):
            ts = burst_start + rng.randrange(max(1, burst_span - 3_600))
            action = rng.choices(["api_call", "message_send"], weights=[3, 2])[0]
            payload = f"ato-{account_id}" if action == "message_send" else None
            if subtle and payload is not None:
                payload += f"-{rng.randrange(n_payloads)}"
            target = rng.randrange(cfg.n_accounts * 5) if action == "message_send" else None
            ds.events.append((account_id, ts, action, payload, target, attacker_ip))


def _add_shared_devices(ds: _Dataset, rng: random.Random, cfg: GenConfig,
                        ids: list[int]) -> None:
    """Households or managed-device onboarding can resemble signup farms."""
    offset = 0
    while offset < len(ids):
        size = min(rng.randint(3, 12), len(ids) - offset)
        asn = rng.randrange(cfg.n_asns)
        base = cfg.start_ts + rng.randrange(max(1, cfg.days * 86_400 - 86_400))
        device = f"dev-shared-{offset}"
        for account_id in ids[offset:offset + size]:
            signup = base + rng.randrange(1_800)
            ip = _ip(rng, asn)
            ds.accounts.append((account_id, signup, ip, asn, device,
                                rng.choice(NORMAL_EMAIL_DOMAINS)))
            ds.labels.append((account_id, "normal"))
            ds.behaviors.append((account_id, "legitimate_shared_device"))
            for _ in range(rng.randint(2, 35)):
                action = rng.choice(["login", "api_call"])
                ds.events.append((account_id, signup + rng.randrange(cfg.horizon_end - signup),
                                  action, None, None, ip))
        offset += size


def _add_legitimate_activity(ds: _Dataset, rng: random.Random, cfg: GenConfig,
                             ids: list[int], behavior: str) -> None:
    for account_id in ids:
        _add_normal(ds, rng, cfg, account_id, behavior=behavior)
        _, signup, ip, _, _, _ = ds.accounts[-1]
        if behavior == "legitimate_bursty_api":
            for _ in range(rng.randint(2, 5)):
                start = signup + rng.randrange(max(1, cfg.horizon_end - signup - 900))
                window = rng.randint(60, 900)
                for _ in range(rng.randint(120, 500)):
                    ds.events.append((account_id, start + rng.randrange(window),
                                      "api_call", None, None, ip))
        else:
            # Authorization is deliberately absent from event features. A
            # legitimate newsletter may look like unsolicited bulk messages.
            targets = rng.sample(range(cfg.n_accounts * 5),
                                 min(cfg.n_accounts * 5, rng.randint(120, 450)))
            for target in targets:
                ds.events.append((account_id, signup + rng.randrange(cfg.horizon_end - signup),
                                  "message_send", f"newsletter-{account_id}", target, ip))


def _add_returning_users(ds: _Dataset, rng: random.Random, cfg: GenConfig,
                         ids: list[int]) -> None:
    for account_id in ids:
        asn = rng.randrange(cfg.n_asns)
        signup = cfg.start_ts + rng.randrange(min(2 * 86_400, cfg.days * 86_400 // 7))
        home_ip = _ip(rng, asn)
        ds.accounts.append((account_id, signup, home_ip, asn, f"dev-{account_id}",
                            rng.choice(NORMAL_EMAIL_DOMAINS)))
        ds.labels.append((account_id, "normal"))
        ds.behaviors.append((account_id, "legitimate_returning"))
        for _ in range(rng.randint(5, 20)):
            ds.events.append((account_id, signup + rng.randrange(min(3 * 86_400, cfg.days * 86_400 // 4)),
                              rng.choice(["login", "api_call"]), None, None, home_ip))
        return_span = min(2 * 86_400, cfg.days * 86_400 // 7)
        return_ip = _ip(rng, (asn + 137) % cfg.n_asns)
        for _ in range(rng.randint(80, 250)):
            action = rng.choices(["api_call", "message_send"], weights=[3, 2])[0]
            payload = f"return-{account_id}" if action == "message_send" else None
            target = rng.randrange(cfg.n_accounts * 5) if action == "message_send" else None
            ds.events.append((account_id, cfg.horizon_end - return_span + rng.randrange(return_span),
                              action, payload, target, return_ip))


def _add_challenge_normal(ds: _Dataset, rng: random.Random, cfg: GenConfig,
                          ids: list[int]) -> None:
    per_behavior = int(len(ids) * cfg.challenge_legitimate_fraction)
    _add_shared_devices(ds, rng, cfg, ids[:per_behavior])
    _add_legitimate_activity(ds, rng, cfg, ids[per_behavior:2 * per_behavior],
                             "legitimate_bursty_api")
    _add_legitimate_activity(ds, rng, cfg, ids[2 * per_behavior:3 * per_behavior],
                             "legitimate_broadcast")
    _add_returning_users(ds, rng, cfg, ids[3 * per_behavior:4 * per_behavior])
    for account_id in ids[4 * per_behavior:]:
        _add_normal(ds, rng, cfg, account_id)


def generate(db_path: str | Path, cfg: GenConfig = GenConfig()) -> dict[str, int]:
    rng = random.Random(cfg.seed)

    n_farm = int(cfg.n_accounts * cfg.frac_farm)
    n_quota = int(cfg.n_accounts * cfg.frac_quota)
    n_spam = int(cfg.n_accounts * cfg.frac_spam_fanout)
    n_ato = int(cfg.n_accounts * cfg.frac_ato)

    ids = list(range(cfg.n_accounts))
    rng.shuffle(ids)
    farm_ids = ids[:n_farm]
    quota_ids = ids[n_farm:n_farm + n_quota]
    spam_ids = ids[n_farm + n_quota:n_farm + n_quota + n_spam]
    ato_ids = ids[n_farm + n_quota + n_spam:n_farm + n_quota + n_spam + n_ato]
    normal_ids = ids[n_farm + n_quota + n_spam + n_ato:]

    ds = _Dataset()
    if cfg.scenario == "challenge":
        _add_challenge_normal(ds, rng, cfg, normal_ids)
    else:
        for account_id in normal_ids:
            _add_normal(ds, rng, cfg, account_id)
    for add_pattern, pattern_ids in ((_add_farm, farm_ids), (_add_quota, quota_ids),
                                    (_add_spam_fanout, spam_ids), (_add_ato, ato_ids)):
        subtle_count = (int(len(pattern_ids) * cfg.challenge_subtle_fraction)
                        if cfg.scenario == "challenge" else 0)
        add_pattern(ds, rng, cfg, pattern_ids[subtle_count:])
        if subtle_count:
            add_pattern(ds, rng, cfg, pattern_ids[:subtle_count], subtle=True)

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript((Path(__file__).parent / "sql" / "schema.sql").read_text())
        conn.executemany("INSERT INTO accounts VALUES (?,?,?,?,?,?)", ds.accounts)
        conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", ds.events)
        conn.executemany("INSERT INTO labels VALUES (?,?)", ds.labels)
        conn.executemany("INSERT INTO account_metadata VALUES (?,?)", ds.behaviors)
        conn.executemany("INSERT INTO dataset_metadata VALUES (?,?)", [
            ("config", json.dumps(asdict(cfg), sort_keys=True)),
            ("horizon_end", str(cfg.horizon_end)),
        ])
        conn.commit()
    finally:
        conn.close()

    return {
        "accounts": len(ds.accounts),
        "events": len(ds.events),
        "normal": len(normal_ids),
        "farm": n_farm,
        "quota": n_quota,
        "spam_fanout": n_spam,
        "ato": n_ato,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/abuse.db")
    parser.add_argument("--accounts", type=int, default=GenConfig.n_accounts)
    parser.add_argument("--seed", type=int, default=GenConfig.seed)
    parser.add_argument("--scenario", choices=["baseline", "challenge"], default=GenConfig.scenario)
    args = parser.parse_args()
    counts = generate(args.db, GenConfig(seed=args.seed, n_accounts=args.accounts,
                                        scenario=args.scenario))
    for key, value in counts.items():
        print(f"{key:>12}: {value}")


if __name__ == "__main__":
    main()
