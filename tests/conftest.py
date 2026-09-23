import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abuse_signals.config import GenConfig
from abuse_signals.features import build_features
from abuse_signals.generate import generate

TEST_CONFIG = GenConfig(seed=11, n_accounts=5_000)


@pytest.fixture(scope="session")
def db_path(tmp_path_factory) -> str:
    """One small fixed-seed dataset shared by the whole test session."""
    path = tmp_path_factory.mktemp("data") / "abuse-test.db"
    generate(path, TEST_CONFIG)
    build_features(path)
    return str(path)


@pytest.fixture(scope="session")
def evaluation(db_path):
    from abuse_signals.train import train

    return train(db_path, seed=7)
