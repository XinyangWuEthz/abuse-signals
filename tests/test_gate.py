"""Baseline quality floors on shared held-out accounts, not production guarantees."""

import pytest


def test_rules_layer_meets_quality_floors(evaluation):
    throttle = evaluation["rules"]["tiers"]["throttle"]["metrics"]
    suspend = evaluation["rules"]["tiers"]["suspend"]["metrics"]
    assert suspend["flagged"] >= 5, suspend
    assert suspend["precision"] is not None and suspend["precision"] >= 0.90, suspend
    assert throttle["recall"] >= 0.50, throttle


@pytest.mark.parametrize("name", ["logistic_regression", "hist_gradient_boosting"])
def test_each_classifier_meets_quality_floor(evaluation, name):
    model = evaluation["models"][name]
    assert model["average_precision"] >= 0.85, model["average_precision"]
    throttle = model["tiers"]["throttle"]["metrics"]
    assert throttle["flagged"] >= 5, throttle
    assert throttle["precision"] is not None and throttle["precision"] >= 0.85, throttle
    assert throttle["recall"] >= 0.50, throttle
