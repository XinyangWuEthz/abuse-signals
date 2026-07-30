"""Generation parameters. Every distribution the generator draws from lives here,
so an experiment is reproducible from (seed, config) alone."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GenConfig:
    seed: int = 7
    n_accounts: int = 20_000
    days: int = 14
    start_ts: int = 1_780_000_000  # fixed epoch anchor; horizon = start_ts + days

    # Fraction of accounts assigned to each injected abuse pattern.
    frac_farm: float = 0.020        # bulk signups sharing ASN + device
    frac_quota: float = 0.010       # high-velocity API bursts
    frac_spam_fanout: float = 0.010 # one payload -> many targets
    frac_ato: float = 0.005         # dormancy then takeover burst

    # Normal-account behavior.
    n_asns: int = 800
    min_daily_events: int = 5
    max_daily_events: int = 40

    @property
    def horizon_end(self) -> int:
        return self.start_ts + self.days * 86_400
