"""Impact-law estimators: planted laws are recovered, nulls are not identified."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from stochopia.measure.impact import (  # noqa: E402
    antisymmetrise,
    child_impact,
    delta_bootstrap_ci,
    fit_power_law,
    identify_law,
    relaxation,
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
