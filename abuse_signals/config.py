"""Experiment settings saved with each dataset.

The same config reproduces logical rows with the same generator version.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GenConfig:
    seed: int = 7
    n_accounts: int = 20_000
    days: int = 14
    start_ts: int = 1_780_000_000  # fixed epoch anchor; horizon = start_ts + days
    scenario: str = "baseline"

    # Fraction of accounts assigned to each injected abuse pattern.
    frac_farm: float = 0.020        # bulk signups sharing ASN + device
    frac_quota: float = 0.010       # high-velocity API bursts
    frac_spam_fanout: float = 0.010 # one payload -> many targets
    frac_ato: float = 0.005         # dormancy then takeover burst

    # Normal-account behavior.
    n_asns: int = 800
    min_daily_events: int = 5
    max_daily_events: int = 40

    # Each legitimate challenge behavior gets this share of normal accounts.
    challenge_legitimate_fraction: float = 0.04
    challenge_subtle_fraction: float = 0.50

    def __post_init__(self) -> None:
        if self.scenario not in {"baseline", "challenge"}:
            raise ValueError("scenario must be 'baseline' or 'challenge'")
        for name in ("seed", "n_accounts", "days", "start_ts", "n_asns",
                     "min_daily_events", "max_daily_events"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be between 0 and 2**32 - 1 for classifier reproducibility")
        if self.n_accounts <= 0 or self.days <= 0 or self.n_asns <= 0:
            raise ValueError("n_accounts, days, and n_asns must be positive")
        if not 0 <= self.min_daily_events <= self.max_daily_events:
            raise ValueError("daily event bounds must satisfy 0 <= min <= max")
        fractions = (self.frac_farm, self.frac_quota, self.frac_spam_fanout, self.frac_ato)
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               or not 0 <= value <= 1 for value in fractions):
            raise ValueError("abuse fractions must be finite values between 0 and 1")
        if sum(fractions) > 1:
            raise ValueError("abuse fractions must sum to at most 1")
        for name, upper in (("challenge_legitimate_fraction", 0.25),
                            ("challenge_subtle_fraction", 1)):
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not 0 <= value <= upper):
                raise ValueError(f"{name} must be between 0 and {upper}")

    @property
    def horizon_end(self) -> int:
        return self.start_ts + self.days * 86_400
