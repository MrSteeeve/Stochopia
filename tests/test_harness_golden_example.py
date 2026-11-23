"""Test-only reference for the hand-calculated Tier 0 example in Draft_2 §5."""

from __future__ import annotations

from itertools import product
from math import inf, sqrt
from statistics import mean

import pytest


# The columns hold (peak, end) on the same frozen day, in ticks.
_PATHS = {
    "A": {"truth": (4, 2), "base": (8, 2), "flow": (8, 2), "price": (6, 3)},
    "B": {"truth": (6, 3), "base": (8, 3), "flow": (8, 3), "price": (7, 4)},
}

# Pre-result choices and predictions are sealed independently of trial outcomes.
_SELECTIONS = {
    "evidence_llm": (("e-r1", "price"),),
    "metrics_llm": (("m-r1", "flow"), ("m-r2", "price")),
    "no_llm": (("n-r1", "flow"), ("n-r2", "price")),
}
_DEVELOPMENT_TRIALS = {
    "e-r1": (1, True),
    "m-r1": (1, False),
    "m-r2": (1, True),
    "n-r1": (1, False),
    "n-r2": (1, True),
}
_PRICE_PREDICTIONS = {
    "evidence_llm": ("increase", (0.5, 1.5)),
    "metrics_llm": ("decrease", (-1.5, -0.5)),
}

_MAX_SEARCH_COST = 2
_PASSING_RMSE_DELTA = -0.4
_MIN_EFFECT = 0.2
_EQUIVALENCE_BAND = (-0.2, 0.2)


# These helpers are an executable oracle for the documented fixture. They do
# not implement the harness runner, scoring, or Verifier.
def _rmse_of_mean_path(candidate, days):
    errors = [
        mean(_PATHS[day][candidate][time] - _PATHS[day]["truth"][time] for day in days)
        for time in range(2)
    ]
    return sqrt(mean(error**2 for error in errors))


def _paired_draws(candidate):
    return {
        "".join(days): _rmse_of_mean_path(candidate, days) - _rmse_of_mean_path("base", days)
        for days in product(_PATHS, repeat=2)
    }


def _linear_quantile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _interval(values):
    return _linear_quantile(values, 0.025), _linear_quantile(values, 0.975)


def _end_interval(candidate):
    draws = [
        mean(_PATHS[day][candidate][1] - _PATHS[day]["base"][1] for day in days)
        for days in product(_PATHS, repeat=2)
    ]
    return _interval(draws)


def _verdict(measured, predicted):
    low, high = measured
    target_low, target_high = predicted
    if target_low <= low and high <= target_high:
        return "SUPPORTED"
    if high < target_low or target_high < low:
        return "REFUTED"
    return "INCONCLUSIVE"


def _directional_verdict(measured, direction, predicted):
    direction_band = (_MIN_EFFECT, inf) if direction == "increase" else (-inf, -_MIN_EFFECT)
    direction_verdict = _verdict(measured, direction_band)
    magnitude_verdict = _verdict(measured, predicted)
    parts = (direction_verdict, magnitude_verdict)
    if "REFUTED" in parts:
        return direction_verdict, magnitude_verdict, "REFUTED"
    if parts == ("SUPPORTED", "SUPPORTED"):
        return direction_verdict, magnitude_verdict, "SUPPORTED"
    return direction_verdict, magnitude_verdict, "INCONCLUSIVE"


def _frozen_report():
    model_scores = {candidate: -_rmse_of_mean_path(candidate, _PATHS) for candidate in ("base", "flow", "price")}
    frozen_qualified = {
        candidate: _interval(_paired_draws(candidate).values())[1] <= _PASSING_RMSE_DELTA
        for candidate in ("flow", "price")
    }
    report = {}
    for group, choices in _SELECTIONS.items():
        cost = 0
        final_id = None
        final_candidate = "base"
        for selection_id, candidate in choices:
            trial_cost, development_qualified = _DEVELOPMENT_TRIALS[selection_id]
            cost += trial_cost
            if development_qualified:
                final_id = selection_id
                final_candidate = candidate
                break
        assert cost <= _MAX_SEARCH_COST
        prediction = _PRICE_PREDICTIONS.get(group)
        report[group] = {
            "first_selection_id": choices[0][0],
            "first_candidate": choices[0][1],
            "first_frozen_qualified": frozen_qualified[choices[0][1]],
            "first_development_pass_cost": cost if final_id is not None else None,
            "final_selection_id": final_id,
            "final_candidate": final_candidate,
            "final_model_gain": model_scores[final_candidate] - model_scores["base"],
            "price_prediction": None if prediction is None else _directional_verdict(
                _end_interval("price"), *prediction
            ),
            "flow_prediction": None if prediction is None else _verdict(
                _end_interval("flow"), _EQUIVALENCE_BAND
            ),
        }
    return report


def test_golden_paired_rmse_enumerates_all_four_ordered_draws():
    base_rmse = 3 / sqrt(2)
    price_rmse = sqrt(13 / 8)
    assert _rmse_of_mean_path("base", _PATHS) == pytest.approx(base_rmse, abs=1e-12)
    assert _rmse_of_mean_path("flow", _PATHS) == pytest.approx(base_rmse, abs=1e-12)
    assert _rmse_of_mean_path("price", _PATHS) == pytest.approx(price_rmse, abs=1e-12)

    draws = _paired_draws("price")
    assert tuple(draws) == ("AA", "AB", "BA", "BB")
    assert draws == pytest.approx(
        {
            "AA": sqrt(5 / 2) - sqrt(8),
            "AB": price_rmse - base_rmse,
            "BA": price_rmse - base_rmse,
            "BB": 1 - sqrt(2),
        },
        abs=1e-12,
    )
    assert _interval(draws.values()) == pytest.approx(
        (-1.2172340824494587, -0.4466399550822213), abs=1e-12
    )
    assert tuple(_paired_draws("flow").values()) == (0, 0, 0, 0)
    assert _end_interval("price") == (1, 1)
    assert _end_interval("flow") == (0, 0)

    mean_of_daily_rmse_deltas = mean((draws["AA"], draws["BB"]))
    assert mean_of_daily_rmse_deltas == pytest.approx(-0.8307509285175476, abs=1e-12)
    assert mean_of_daily_rmse_deltas != pytest.approx(draws["AB"], abs=1e-12)


def test_golden_frozen_report_keeps_choices_costs_and_forecasts_separate():
    expected_model_gain = pytest.approx(0.8465654651614463, abs=1e-12)
    report = _frozen_report()
    assert report == {
        "evidence_llm": {
            "first_selection_id": "e-r1",
            "first_candidate": "price",
            "first_frozen_qualified": True,
            "first_development_pass_cost": 1,
            "final_selection_id": "e-r1",
            "final_candidate": "price",
            "final_model_gain": expected_model_gain,
            "price_prediction": ("SUPPORTED", "SUPPORTED", "SUPPORTED"),
            "flow_prediction": "SUPPORTED",
        },
        "metrics_llm": {
            "first_selection_id": "m-r1",
            "first_candidate": "flow",
            "first_frozen_qualified": False,
            "first_development_pass_cost": 2,
            "final_selection_id": "m-r2",
            "final_candidate": "price",
            "final_model_gain": expected_model_gain,
            "price_prediction": ("REFUTED", "REFUTED", "REFUTED"),
            "flow_prediction": "SUPPORTED",
        },
        "no_llm": {
            "first_selection_id": "n-r1",
            "first_candidate": "flow",
            "first_frozen_qualified": False,
            "first_development_pass_cost": 2,
            "final_selection_id": "n-r2",
            "final_candidate": "price",
            "final_model_gain": expected_model_gain,
            "price_prediction": None,
            "flow_prediction": None,
        },
    }


def test_golden_direction_can_be_supported_while_magnitude_is_refuted():
    assert _directional_verdict(_end_interval("price"), "increase", (2, 3)) == (
        "SUPPORTED",
        "REFUTED",
        "REFUTED",
    )


def test_zero_crossing_interval_does_not_establish_equivalence():
    assert _verdict((-0.3, 0.3), _EQUIVALENCE_BAND) == "INCONCLUSIVE"
