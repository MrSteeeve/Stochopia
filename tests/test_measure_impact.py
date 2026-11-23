"""Impact-law estimators: planted laws are recovered, nulls are not identified."""

from __future__ import annotations

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from stochopia.measure.impact import (  # noqa: E402
    ImpactClaimStatus,
    ImpactMetrics,
    ImpactReference,
    antisymmetrise,
    child_impact,
    delta_bootstrap_ci,
    fit_power_law,
    identify_law,
    interval_distance,
    relaxation,
    score_impact,
    score_impact_panel,
)
from stochopia.sim.impact import kernel_values  # noqa: E402
from stochopia.sim.world import (  # noqa: E402
    WorldSpec,
    injection_schedule,
    simulate_day,
    true_response,
)


def _planted(delta, amplitude, noise, n_units=40, n_children=50, seed=0):
    rng = np.random.default_rng(seed)
    q = np.arange(1, n_children + 1)
    return amplitude * q**delta + rng.normal(0.0, noise, size=(n_units, n_children))


@pytest.mark.parametrize("delta", [0.5, 0.75, 1.0])
def test_planted_exponent_is_recovered(delta):
    law = identify_law(_planted(delta, 1.0, noise=1.0))
    assert law.identified
    assert law.delta == pytest.approx(delta, abs=0.03)
    assert law.amplitude == pytest.approx(1.0, rel=0.1)


def test_fit_power_law_is_exact_without_noise():
    q = np.arange(1, 60)
    assert fit_power_law(q, 2.5 * q**0.62) == pytest.approx((0.62, 2.5))


def test_null_response_is_rarely_identified():
    """The gate is a one-sided 3-sigma test, so the false-positive rate is small."""
    rng = np.random.default_rng(0)
    trials = 3000
    iid = sum(identify_law(rng.normal(size=(20, 50))).identified for _ in range(trials))
    walk = sum(
        identify_law(rng.normal(size=(20, 50)).cumsum(axis=1)).identified for _ in range(trials)
    )
    assert iid / trials < 0.01 and walk / trials < 0.01


def test_negative_response_is_not_identified():
    law = identify_law(_planted(0.5, -1.0, noise=0.1))
    assert not law.identified and law.reason == "peak not resolved"


def test_flat_response_hits_the_grid_bound():
    law = identify_law(np.ones((20, 30)) + np.random.default_rng(0).normal(0, 0.01, (20, 30)))
    assert not law.identified and law.reason == "exponent on grid bound"


def test_identify_law_rejects_malformed_input():
    with pytest.raises(ValueError):
        identify_law(np.ones((1, 30)))
    with pytest.raises(ValueError):
        identify_law(np.ones(30))


def test_bootstrap_interval_covers_planted_exponent():
    lo, hi = delta_bootstrap_ci(_planted(0.6, 1.0, noise=1.0), seed=1, n_resamples=300)
    assert lo <= 0.6 <= hi and hi - lo < 0.2


def test_antisymmetrise_cancels_common_drift():
    drift = np.linspace(0.0, 5.0, 10)
    signal = np.arange(10.0)
    assert np.allclose(antisymmetrise(drift + signal, drift - signal), signal)


def test_relaxation_reports_decay_and_permanence():
    schedule = injection_schedule(n_children=10, spacing=9, n_relax=400)
    last = int(np.flatnonzero(schedule)[-1])
    rise = np.linspace(0.1, 1.0, last + 1)
    permanent = np.concatenate([rise, np.ones(schedule.size - last - 1)])
    decaying = np.concatenate([rise, np.linspace(1.0, 0.5, schedule.size - last - 1)])
    flat, down = relaxation(permanent, schedule), relaxation(decaying, schedule)
    assert flat.end_over_peak == pytest.approx(1.0) and flat.slope == pytest.approx(0.0, abs=1e-9)
    assert down.end_over_peak == pytest.approx(0.5, abs=0.02) and down.slope < 0.0


def test_estimator_reads_the_simulator_truth_end_to_end():
    """The measured law equals the fit of the closed-form kernel sum."""
    spec = WorldSpec(n_trades=4000, burn_in=500)
    schedule = injection_schedule(n_children=50, spacing=9, n_relax=500)
    paths = np.stack(
        [
            antisymmetrise(
                true_response(spec, day, 1000, schedule, +1),
                true_response(spec, day, 1000, schedule, -1),
            )
            for day in (simulate_day(spec, "t", i) for i in range(4))
        ]
    )
    # exogenous flow and a linear kernel: the truth is the same on every day
    law = identify_law(child_impact(paths, schedule))
    # children sit 10 events apart, so child k sees lags 0, 10, ..., 10k
    kernel_sum = np.cumsum(kernel_values(spec.kernel, 10.0 * np.arange(50)))
    assert law.identified
    assert (law.delta, law.amplitude) == pytest.approx(
        fit_power_law(np.arange(1, 51), kernel_sum), abs=1e-3
    )
    assert law.peak == pytest.approx(kernel_sum[-1], abs=1e-9)
    # no permanent impact in this world: the response comes back down
    assert relaxation(paths.mean(axis=0), schedule).end_over_peak < 0.5


def test_composite_errors_and_inclusive_drift_cutoff() -> None:
    assert [interval_distance(x, 0.5, 1.0) for x in (0.25, 0.5, 0.75, 1.0, 1.5)] == [0.25, 0, 0, 0, 0.5]
    metrics = ImpactMetrics(y=0.25, delta=0.7, end_over_peak=0.5, idle_drift_ticks=-20.0, law_identified=True)
    corrected = score_impact(metrics)
    assert (corrected.amplitude_error, corrected.exponent_error, corrected.relaxation_error) == pytest.approx((1, 1, 1))
    assert corrected.scored_end_over_peak == 1.0
    assert corrected.score == pytest.approx(50.0)

    raw = score_impact(metrics, drift_correction=False)
    assert raw.relaxation_error == 0.0
    assert raw.score == pytest.approx(100.0 / 1.65)
    assert score_impact(replace(metrics, idle_drift_ticks=20.0)) == corrected
    assert score_impact(replace(metrics, idle_drift_ticks=19.99)).score == raw.score
    assert score_impact(replace(metrics, idle_drift_ticks=None)).score is None
    assert score_impact(replace(metrics, idle_drift_ticks=None), drift_correction=False).score == raw.score


@pytest.mark.parametrize("bad_amplitude", [None, 0.0, -0.1, float("nan")])
def test_composite_requires_positive_amplitude(bad_amplitude: float | None) -> None:
    metrics = ImpactMetrics(
        y=bad_amplitude, delta=0.5, end_over_peak=0.6, idle_drift_ticks=0.0, law_identified=True,
    )
    result = score_impact(metrics)
    assert result.score is None
    assert result.amplitude_error is None
    assert result.exponent_error == result.relaxation_error == 0.0


def test_claims_use_replay_reference_raw_relaxation_and_blind_peak() -> None:
    metrics = ImpactMetrics(
        y=1.0, delta=0.6, end_over_peak=0.7, idle_drift_ticks=5, law_identified=True,
        gamma_flow=1.1, hurst=0.52, visible_peak_ticks=6.0, blind_peak_ticks=2.0, relaxation_slope=0.0,
    )
    reference = ImpactReference(gamma_flow=1.0, hurst=0.5, idle_drift_ticks=0.0, ow_end_over_peak=0.7)
    result = score_impact(metrics, reference)
    assert result.visible_to_blind_ratio == 3.0
    assert all(claim.status is ImpactClaimStatus.PASS for claim in result.claims.values())
    assert result.relaxation_error > 0  # C5 uses the same-stock kernel, the score uses the empirical band.

    at_limit = replace(metrics, idle_drift_ticks=-20)
    corrected = score_impact(at_limit, reference)
    raw = score_impact(at_limit, reference, drift_correction=False)
    assert corrected.claims == raw.claims
    assert corrected.claims["C5"].status is ImpactClaimStatus.FAIL

    no_blind = score_impact(replace(metrics, blind_peak_ticks=None), reference)
    assert no_blind.visible_to_blind_ratio is None
    assert no_blind.claims["C3"].status is ImpactClaimStatus.INCONCLUSIVE
    zero_blind = score_impact(replace(metrics, blind_peak_ticks=0.0), reference)
    assert zero_blind.visible_to_blind_ratio is None
    assert zero_blind.claims["C3"].status is ImpactClaimStatus.INCONCLUSIVE
    ow = score_impact(replace(metrics, is_ow_baseline=True), reference)
    assert ow.claims["C5"].status is ImpactClaimStatus.INCONCLUSIVE


def test_near_zero_requires_contained_interval_and_registered_band() -> None:
    metrics = ImpactMetrics(law_identified=False, peak_interval_ticks=(-0.2, 0.2), visible_peak_ticks=0.0)
    reference = ImpactReference(near_zero_peak_band_ticks=(-0.3, 0.3))
    near_zero = score_impact(metrics, reference)
    assert near_zero.score is None
    assert near_zero.claims["C2"].status is ImpactClaimStatus.FAIL
    assert "near zero" in near_zero.claims["C2"].reason
    assert near_zero.claims["C4"].status is near_zero.claims["C5"].status is ImpactClaimStatus.INCONCLUSIVE
    for unresolved in (replace(metrics, peak_interval_ticks=(-0.4, 0.2)), replace(metrics, peak_interval_ticks=None)):
        claim = score_impact(unresolved, reference).claims["C2"]
        assert claim.status is ImpactClaimStatus.INCONCLUSIVE
    assert score_impact(metrics).claims["C2"].status is ImpactClaimStatus.INCONCLUSIVE


def test_panel_means_only_use_common_stocks_and_keep_missing_in_coverage() -> None:
    ideal = ImpactMetrics(y=0.75, delta=0.5, end_over_peak=0.6, idle_drift_ticks=0.0, law_identified=True)
    worse = replace(ideal, y=2.0)
    models = {
        "a": {"AMD": worse, "NVDA": ideal, "EA": replace(ideal, law_identified=False)},
        "b": {"AMD": ideal, "EA": worse},
    }
    panel = score_impact_panel(models, ("AMD", "NVDA", "EA"))
    assert panel.common_symbols == ("AMD",)
    assert panel.common_coverage == pytest.approx(1 / 3)
    assert panel.models["a"].mean_score == score_impact(worse).score
    assert panel.models["b"].mean_score == 100.0
    assert panel.models["a"].coverage == panel.models["b"].coverage == pytest.approx(2 / 3)
    assert panel.models["a"].per_symbol["EA"].score is None
    assert panel.models["b"].per_symbol["NVDA"].score is None

    no_overlap = score_impact_panel({"a": {"AMD": ideal}, "b": {"EA": ideal}}, ("AMD", "EA"))
    assert no_overlap.common_symbols == ()
    assert no_overlap.common_coverage == 0.0
    assert all(model.mean_score is None for model in no_overlap.models.values())


# Nigmatulin et al. (2026), LOB-Impact, Table 3, arXiv:2610.27100v1, CC BY 4.0.
# https://arxiv.org/html/2610.27100v1#S3
# Each row contains Y, delta, end/peak, idle drift, gamma_flow, H, visible/blind, and slope in percent.
_SYMBOLS = ("AMD", "NVDA", "GOOG", "MSFT", "EA")
_NEURAL_ROWS = {
    "S5-120M": (
        (2.24, .66, 1.19, -18, 1.90, .55, 21, .6),
        (7.36, .98, .99, -7, 2.32, .55, 23, -.7),
        (1.70, .73, 1.14, -18, 1.66, .56, 26, .4),
        (3.07, .63, 1.14, -30, 1.96, .55, 11, .3),
        (12.23, .84, 1.18, -3, 1.68, .57, 29, None),
    ),
    "S5-4k": (
        (4.00, .79, 1.15, 9, 1.83, .54, 26, 0.0),
        (7.49, .97, 1.11, -4, 1.94, .54, 23, -.6),
        (1.83, .75, 1.16, 1, 1.92, .55, 23, .9),
        (3.48, .69, 1.21, 18, 1.39, .59, 18, .5),
        (19.18, 1.05, 1.24, 4, 1.62, .56, 38, .6),
    ),
    "GDN": (
        (8.57, .90, .88, -127, .79, .49, 12, -2.4),
        (15.43, 1.12, .89, -29, 1.99, .65, 13, -1.5),
        (8.42, 1.11, .18, -85, .79, .54, 9.6, -8.5),
        (8.28, .79, .95, -58, 1.03, .52, 11, -2.1),
        (43.54, 1.06, 1.34, -6, 2.11, .57, 13, 1.6),
    ),
    "Mamba3": (
        (7.65, .87, 1.16, -69, 1.20, .58, 9.5, .4),
        (19.05, 1.15, 1.67, -24, 1.08, .59, 7.7, 5.3),
        (6.24, .97, 1.67, -42, 1.95, .59, 6.9, 2.6),
        (7.60, .81, 1.19, -39, 1.47, .62, 8.8, .2),
        (49.52, 1.10, 1.37, -4, 2.03, .59, 25, 2.2),
    ),
    "Mamba3-4k": (
        (8.38, .93, 1.36, -251, 1.06, .58, 5.0, 3.7),
        (18.98, 1.11, 1.03, -27, 1.39, .62, 9.4, -.5),
        (12.56, 1.21, .24, -879, .97, .59, 3.4, -13.8),
        (8.79, .86, .72, -62, 1.82, .60, 9.6, -3.9),
        (41.66, 1.12, 1.14, -13, 1.82, .62, 20, .1),
    ),
}
# Same-stock Historic replay and the printed OW end/peak, including marked reference estimates.
_TABLE_REFERENCES = {
    "AMD": (1.66, .54, 1, .81),
    "NVDA": (1.31, .49, -1, .92),
    "GOOG": (1.77, .54, 0, .92),
    "MSFT": (1.51, .55, 12, .71),
    "EA": (1.31, .47, 0, .82),
}


def test_paper_table_3_panel_scores_claims_and_ranking_sensitivity() -> None:
    models = {}
    for model, rows in _NEURAL_ROWS.items():
        measurements = {}
        for symbol, (y, delta, ratio, drift, gamma, hurst, blind_ratio, slope_pct) in zip(_SYMBOLS, rows, strict=True):
            measurements[symbol] = ImpactMetrics(
                y=y, delta=delta, end_over_peak=ratio, idle_drift_ticks=drift, law_identified=True,
                gamma_flow=gamma, hurst=hurst, visible_to_blind_ratio=blind_ratio,
                relaxation_slope=None if slope_pct is None else slope_pct / 100,
            )
        models[model] = measurements
    references = {}
    for symbol, (gamma, hurst, drift, ow) in _TABLE_REFERENCES.items():
        references[symbol] = ImpactReference(gamma_flow=gamma, hurst=hurst, idle_drift_ticks=drift, ow_end_over_peak=ow)
    expected = {
        "S5-120M": ("39.0", "46.2", "29.1", "47.0", "45.1", "27.9"),
        "S5-4k": ("34.2", "35.9", "28.1", "44.6", "39.8", "22.6"),
        "GDN": ("26.4", "29.9", "23.6", "25.9", "32.8", "19.9"),
        "Mamba3": ("24.8", "29.7", "19.5", "24.8", "30.8", "19.1"),
        "Mamba3-4k": ("24.7", "26.5", "23.0", "23.1", "30.7", "20.2"),
    }
    panel = score_impact_panel(models, _SYMBOLS, references=references)
    assert panel.common_symbols == _SYMBOLS and panel.common_coverage == 1.0
    for model, detail in panel.models.items():
        assert detail.coverage == 1.0
        scores = tuple(f"{detail.per_symbol[symbol].score:.1f}" for symbol in _SYMBOLS)
        assert (f"{detail.mean_score:.1f}", *scores) == expected[model]
        for symbol, stock in detail.per_symbol.items():
            assert stock.visible_to_blind_ratio == models[model][symbol].visible_to_blind_ratio
            assert tuple(claim.status for claim in stock.claims.values()) == (
                ImpactClaimStatus.FAIL, ImpactClaimStatus.PASS, ImpactClaimStatus.PASS,
                ImpactClaimStatus.FAIL, ImpactClaimStatus.FAIL,
            )

    equal = score_impact_panel(models, _SYMBOLS, weights=(1 / 3, 1 / 3, 1 / 3))
    no_drift = score_impact_panel(models, _SYMBOLS, drift_correction=False)
    assert sorted(panel.models, key=lambda model: panel.models[model].mean_score, reverse=True) == list(expected)
    assert sorted(equal.models, key=lambda model: equal.models[model].mean_score, reverse=True) == list(expected)
    assert sorted(no_drift.models, key=lambda model: no_drift.models[model].mean_score, reverse=True) == [
        "S5-120M", "S5-4k", "GDN", "Mamba3-4k", "Mamba3",
    ]
