"""Synthetic account/event-log generator with labeled abuse patterns.

Everything is drawn from a seeded RNG: same (seed, config) -> byte-identical dataset,
which is what lets the CI regression gate assert quality floors on fixed numbers.

Injected patterns (see config.py for fractions):
  farm         bulk signups sharing one ASN + device fingerprint inside a 30-min window
  quota        high-velocity api_call bursts (free-tier farming)
  spam_fanout  one payload_hash fanned out to hundreds of distinct targets
  ato          account takeover: long dormancy, then a burst from a new IP
"""

from __future__ import annotations

import argparse
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


def _add_normal(ds: _Dataset, rng: random.Random, cfg: GenConfig, account_id: int) -> None:
    asn = rng.randrange(cfg.n_asns)
    signup = cfg.start_ts + rng.randrange(cfg.days * 86_400 - 86_400)
    ip = _ip(rng, asn)
    ds.accounts.append((account_id, signup, ip, asn, f"dev-{account_id}",
                        rng.choice(NORMAL_EMAIL_DOMAINS)))
    ds.labels.append((account_id, "normal"))

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


def _add_farm(ds: _Dataset, rng: random.Random, cfg: GenConfig, ids: list[int]) -> None:
    """Cohorts of 8-25 accounts signing up from one ASN+device within 30 minutes."""
    i = 0
    cohort_no = 0
    while i < len(ids):
        size = min(rng.randint(8, 25), len(ids) - i)
        asn = rng.randrange(cfg.n_asns)
        device = f"dev-farm-{cohort_no}"
        domain = f"{rng.randrange(10**8):08x}.top"
        base = cfg.start_ts + rng.randrange(cfg.days * 86_400 - 7_200)
        for account_id in ids[i:i + size]:
            signup = base + rng.randrange(1_800)
            ip = _ip(rng, asn)
            ds.accounts.append((account_id, signup, ip, asn, device, domain))
            ds.labels.append((account_id, "farm"))
            for _ in range(rng.randint(0, 5)):  # farms stay quiet after creation
                ds.events.append((account_id, signup + rng.randrange(86_400),
                                  "login", None, None, ip))
        i += size
        cohort_no += 1


def _add_quota(ds: _Dataset, rng: random.Random, cfg: GenConfig, ids: list[int]) -> None:
    """Normal-looking signup, then 300-700 api_calls compressed into 1-5 minute bursts."""
    for account_id in ids:
        asn = rng.randrange(cfg.n_asns)
        signup = cfg.start_ts + rng.randrange(cfg.days * 86_400 // 2)
        ip = _ip(rng, asn)
        ds.accounts.append((account_id, signup, ip, asn, f"dev-{account_id}",
                            rng.choice(NORMAL_EMAIL_DOMAINS)))
        ds.labels.append((account_id, "quota"))
        for _ in range(rng.randint(2, 5)):
            start = signup + rng.randrange(max(1, cfg.horizon_end - signup - 3_600))
            window = rng.randint(60, 300)
            for _ in range(rng.randint(300, 700)):
                ds.events.append((account_id, start + rng.randrange(window),
                                  "api_call", None, None, ip))


def _add_spam_fanout(ds: _Dataset, rng: random.Random, cfg: GenConfig, ids: list[int]) -> None:
    """A handful of payloads, one of them sent to 180-500 distinct targets."""
    for account_id in ids:
        asn = rng.randrange(cfg.n_asns)
        signup = cfg.start_ts + rng.randrange(cfg.days * 86_400 - 86_400)
        ip = _ip(rng, asn)
        ds.accounts.append((account_id, signup, ip, asn, f"dev-{account_id}",
                            rng.choice(NORMAL_EMAIL_DOMAINS)))
        ds.labels.append((account_id, "spam_fanout"))
        main_payload = f"spam-{account_id}"
        targets = rng.sample(range(cfg.n_accounts * 5), rng.randint(180, 500))
        span = max(3_600, cfg.horizon_end - signup)
        for target in targets:
            ts = signup + rng.randrange(span)
            ds.events.append((account_id, ts, "message_send", main_payload, target, ip))
        for _ in range(rng.randint(5, 30)):  # a little cover traffic
            ds.events.append((account_id, signup + rng.randrange(span), "login", None, None, ip))


def _add_ato(ds: _Dataset, rng: random.Random, cfg: GenConfig, ids: list[int]) -> None:
    """Early signup, light activity, >=7 days of silence, then a takeover burst."""
    for account_id in ids:
        asn = rng.randrange(cfg.n_asns)
        signup = cfg.start_ts + rng.randrange(2 * 86_400)
        home_ip = _ip(rng, asn)
        ds.accounts.append((account_id, signup, home_ip, asn, f"dev-{account_id}",
                            rng.choice(NORMAL_EMAIL_DOMAINS)))
        ds.labels.append((account_id, "ato"))
        for _ in range(rng.randint(5, 20)):  # benign early activity, then silence
            ds.events.append((account_id, signup + rng.randrange(3 * 86_400),
                              "login" if rng.random() < 0.5 else "api_call",
                              None, None, home_ip))
        attacker_ip = _ip(rng, (asn + 137) % cfg.n_asns)
        burst_start = cfg.horizon_end - 2 * 86_400
        for _ in range(rng.randint(120, 300)):
            ts = burst_start + rng.randrange(2 * 86_400 - 3_600)
            action = rng.choices(["api_call", "message_send"], weights=[3, 2])[0]
            payload = f"ato-{account_id}" if action == "message_send" else None
            target = rng.randrange(cfg.n_accounts * 5) if action == "message_send" else None
            ds.events.append((account_id, ts, action, payload, target, attacker_ip))


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
    for account_id in normal_ids:
        _add_normal(ds, rng, cfg, account_id)
    _add_farm(ds, rng, cfg, farm_ids)
    _add_quota(ds, rng, cfg, quota_ids)
    _add_spam_fanout(ds, rng, cfg, spam_ids)
    _add_ato(ds, rng, cfg, ato_ids)

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
    args = parser.parse_args()
    counts = generate(args.db, GenConfig(seed=args.seed, n_accounts=args.accounts))
    for key, value in counts.items():
        print(f"{key:>12}: {value}")


if __name__ == "__main__":
    main()
