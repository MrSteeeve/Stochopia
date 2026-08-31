"""Fast, offline contract tests for the synthetic regime-switching market API."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import replace
from datetime import date, timedelta
from statistics import pstdev

import pytest

from stochopia.benchmark import ProductDomainSpec, RiskBudget
from stochopia.environment import EpisodeTask
from stochopia.market_generator import (
    SCHEMA_VERSION,
    GeneratedMarketPath,
    RegimeParameters,
    RegimeSwitchingGBMSpecV0,
    ScenarioProvenance,
    generate_regime_switching_gbm,
)
from stochopia.pricing import QuotePolicy
from stochopia.products import ClientProfile


def _regimes(
    *,
    drift_shift: float = 0.0,
    volatility_scale: float = 1.0,
    iv_shift: float = 0.0,
) -> tuple[RegimeParameters, RegimeParameters]:
    return (
        RegimeParameters(
            name="calm",
            physical_annual_return=0.06 + drift_shift,
            physical_annual_volatility=0.16 * volatility_scale,
            pricing_iv_1m=0.18 + iv_shift,
            pricing_iv_3m=0.20 + iv_shift,
            pricing_iv_6m=0.22 + iv_shift,
        ),
        RegimeParameters(
            name="stress",
            physical_annual_return=-0.08 + drift_shift,
            physical_annual_volatility=0.34 * volatility_scale,
            pricing_iv_1m=0.38 + iv_shift,
            pricing_iv_3m=0.36 + iv_shift,
            pricing_iv_6m=0.34 + iv_shift,
        ),
    )


def _spec(
    *,
    drift_shift: float = 0.0,
    volatility_scale: float = 1.0,
    iv_shift: float = 0.0,
    risk_free_rate: float = 0.02,
    carry_rate: float = 0.01,
) -> RegimeSwitchingGBMSpecV0:
    return RegimeSwitchingGBMSpecV0(
        regimes=_regimes(
            drift_shift=drift_shift,
            volatility_scale=volatility_scale,
            iv_shift=iv_shift,
        ),
        transition_matrix=((0.94, 0.06), (0.18, 0.82)),
        initial_distribution=(0.8, 0.2),
        risk_free_rate=risk_free_rate,
        carry_rate=carry_rate,
        parameter_source="offline-test-fixture-v1",
        calibration_cutoff=date(2024, 1, 31),
        calibration_data_hash="a" * 64,
    )


def _path(
    *,
    spec: RegimeSwitchingGBMSpecV0 | None = None,
    seed: int = 20260831,
    decision_rounds: int = 7,
) -> GeneratedMarketPath:
    return generate_regime_switching_gbm(
        spec or _spec(),
        episode_id="synthetic-test-episode",
        underlying="CSI500",
        start_date=date(2024, 1, 31),
        start_spot=5000.0,
        decision_rounds=decision_rounds,
        seed=seed,
    )


def _assert_hex_digest(value: str) -> None:
    assert isinstance(value, str)
    assert len(value) == 64
    int(value, 16)


def _is_last_weekday_of_month(value: date) -> bool:
    if value.weekday() >= 5:
        return False
    next_weekday = value + timedelta(days=1)
    while next_weekday.weekday() >= 5:
        next_weekday += timedelta(days=1)
    return next_weekday.month != value.month


def test_same_spec_and_seed_are_byte_identical_but_a_new_seed_changes_path():
    first = _path()
    second = _path()
    changed_seed = _path(seed=20260832)

    assert first.canonical_json == second.canonical_json
    assert first.root_hash == second.root_hash
    assert first.daily_spots != changed_seed.daily_spots
    assert first.root_hash != changed_seed.root_hash
    assert first.root_hash == hashlib.sha256(first.canonical_json.encode("utf-8")).hexdigest()
    assert json.loads(first.canonical_json) == first.manifest


def test_clean_subprocesses_produce_byte_identical_artifacts():
    program = '''
import json
from datetime import date
from stochopia.market_generator import RegimeParameters, RegimeSwitchingGBMSpecV0, generate_regime_switching_gbm
spec = RegimeSwitchingGBMSpecV0(
    regimes=(
        RegimeParameters("calm", 0.06, 0.16, 0.18, 0.20, 0.22),
        RegimeParameters("stress", -0.08, 0.34, 0.38, 0.36, 0.34),
    ),
    transition_matrix=((0.94, 0.06), (0.18, 0.82)),
    initial_distribution=(0.8, 0.2), risk_free_rate=0.02, carry_rate=0.01,
    parameter_source="offline-test-fixture-v1", calibration_cutoff=date(2024, 1, 31),
    calibration_data_hash="a" * 64,
)
path = generate_regime_switching_gbm(spec, episode_id="synthetic-test-episode", underlying="CSI500", start_date=date(2024, 1, 31), start_spot=5000.0, decision_rounds=7, seed=20260831)
print(json.dumps({"canonical_json": path.canonical_json, "root_hash": path.root_hash}, sort_keys=True, separators=(",", ":")))
'''
    outputs = [
        subprocess.check_output([sys.executable, "-c", program], text=True).strip()
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]


def test_snapshots_are_month_end_market_inputs_and_materialize_an_episode_task():
    path = _path(decision_rounds=3)
    snapshots = path.snapshots

    assert len(snapshots) == 4
    assert [item.round_num for item in snapshots] == [1, 2, 3, 4]
    assert [item.as_of for item in snapshots] == sorted(item.as_of for item in snapshots)
    assert len({item.as_of for item in snapshots}) == len(snapshots)
    assert all(_is_last_weekday_of_month(item.as_of) for item in snapshots)
    assert all(item.episode_id == "synthetic-test-episode" for item in snapshots)
    assert all(item.underlying == "CSI500" for item in snapshots)
    for item in snapshots:
        assert all(
            math.isfinite(value) and value > 0
            for value in (item.spot, item.atm_iv_1m, item.atm_iv_3m, item.atm_iv_6m)
        )

    task = EpisodeTask(
        snapshots=snapshots,
        client=ClientProfile(
            id="test-client", name="Test Client", capital=1_000_000.0,
            max_loss_pct=1.0, min_return_pct=0.01, risk_appetite="moderate",
            max_maturity_months=12, min_hit_prob=0.2,
        ),
        risk_budget=RiskBudget(1_000_000.0, 1_000_000.0, 2_000_000.0, 1_000_000.0, 1_000_000.0),
        domain=ProductDomainSpec(),
        quote_policy=QuotePolicy(diagnostic_paths=16),
        task_seed=20260831,
    )
    assert task.snapshots == snapshots


def test_latent_regime_is_not_exposed_through_market_snapshots():
    path = _path()

    assert len(path.latent_regime_path) == len(path.daily_dates)
    assert set(path.latent_regime_path) <= {"calm", "stress"}
    assert all(snapshot.regime == "synthetic" for snapshot in path.snapshots)
    for snapshot in path.snapshots:
        assert snapshot.source.startswith("synthetic;")
        assert f"schema={SCHEMA_VERSION}" in snapshot.source
        assert f"spec_hash={_spec().spec_hash}" in snapshot.source
        assert "seed=20260831" in snapshot.source
        public_brief = snapshot.public_brief()
        assert public_brief["market_regime"] == "synthetic"
        assert "source" not in public_brief
        assert "trend_alpha" not in public_brief
        assert "calm" not in json.dumps(public_brief)
        assert "stress" not in json.dumps(public_brief)


def test_q_inputs_do_not_change_physical_path_but_do_change_pricing_fields():
    base = _path()
    q_changed = _path(spec=_spec(iv_shift=0.05, risk_free_rate=0.07, carry_rate=0.015))

    assert q_changed.daily_dates == base.daily_dates
    assert q_changed.daily_spots == base.daily_spots
    assert q_changed.latent_regime_path == base.latent_regime_path
    assert [item.risk_free_rate for item in q_changed.snapshots] != [item.risk_free_rate for item in base.snapshots]
    assert [item.carry_rate for item in q_changed.snapshots] != [item.carry_rate for item in base.snapshots]
    assert [item.atm_iv_3m for item in q_changed.snapshots] != [item.atm_iv_3m for item in base.snapshots]


def test_p_inputs_change_spots_without_changing_latent_states_or_q_fields():
    base = _path()
    p_changed_paths = (
        _path(spec=_spec(drift_shift=0.10)),
        _path(spec=_spec(volatility_scale=1.25)),
    )

    for p_changed in p_changed_paths:
        assert p_changed.latent_regime_path == base.latent_regime_path
        assert p_changed.daily_spots != base.daily_spots
        for name in ("risk_free_rate", "carry_rate", "atm_iv_1m", "atm_iv_3m", "atm_iv_6m"):
            assert [getattr(item, name) for item in p_changed.snapshots] == [getattr(item, name) for item in base.snapshots]


def test_trend_alpha_round_trips_each_regimes_physical_spot_drift():
    spec = _spec()
    path = _path(spec=spec)
    regimes = {regime.name: regime for regime in spec.regimes}
    date_to_index = {value: index for index, value in enumerate(path.daily_dates)}

    for snapshot in path.snapshots:
        latent_name = path.latent_regime_path[date_to_index[snapshot.as_of]]
        reconstructed = (
            snapshot.risk_free_rate + snapshot.trend_alpha - snapshot.carry_rate
        )
        assert reconstructed == pytest.approx(
            regimes[latent_name].physical_annual_return,
            rel=1e-12,
            abs=1e-12,
        )


def test_manifest_and_provenance_include_versioned_hashes_and_rng_metadata():
    path = _path()
    provenance = path.provenance

    assert isinstance(provenance, ScenarioProvenance)
    assert SCHEMA_VERSION in path.canonical_json
    assert "weekday-only-v0" in path.canonical_json
    assert '"transition_interval":"each-weekday-v0"' in path.canonical_json
    assert "offline-test-fixture-v1" in path.canonical_json
    assert "rng" in path.canonical_json.lower()
    for value in (
        path.root_hash,
        provenance.spec_hash,
        provenance.generator_source_hash,
        provenance.daily_path_hash,
        provenance.snapshot_hash,
    ):
        _assert_hex_digest(value)
    assert provenance.root_seed == 20260831
    assert provenance.calendar_id == "weekday-only-v0"
    assert provenance.parameter_source == "offline-test-fixture-v1"
    assert provenance.spec_canonical_json == _spec().canonical_json
    assert provenance.rng_algorithm
    assert provenance.regime_seed != provenance.innovation_seed
    assert all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (provenance.regime_seed, provenance.innovation_seed)
    )


def test_history_features_are_none_until_available_then_finite_after_warmup():
    path = _path(decision_rounds=7)
    first, last = path.snapshots[0], path.snapshots[-1]

    assert first.return_20d is None
    assert first.realized_vol_20d is None
    assert first.realized_vol_60d is None
    assert first.drawdown_6m is None
    assert all(
        value is not None and math.isfinite(value)
        for value in (last.return_20d, last.realized_vol_20d, last.realized_vol_60d, last.drawdown_6m)
    )


def test_history_features_match_the_exact_documented_windows():
    spec = _spec()
    path = _path(spec=spec, decision_rounds=7)
    date_to_index = {value: index for index, value in enumerate(path.daily_dates)}

    for snapshot in path.snapshots:
        end = date_to_index[snapshot.as_of]
        history = path.daily_spots[: end + 1]
        if len(history) >= 21:
            assert snapshot.return_20d == pytest.approx(history[-1] / history[-21] - 1.0)
            log_returns_20 = [
                math.log(history[index] / history[index - 1])
                for index in range(len(history) - 20, len(history))
            ]
            assert snapshot.realized_vol_20d == pytest.approx(
                pstdev(log_returns_20) * math.sqrt(spec.trading_days_per_year)
            )
        else:
            assert snapshot.return_20d is None
            assert snapshot.realized_vol_20d is None

        if len(history) >= 61:
            log_returns_60 = [
                math.log(history[index] / history[index - 1])
                for index in range(len(history) - 60, len(history))
            ]
            assert snapshot.realized_vol_60d == pytest.approx(
                pstdev(log_returns_60) * math.sqrt(spec.trading_days_per_year)
            )
        else:
            assert snapshot.realized_vol_60d is None

        drawdown_window = max(1, spec.trading_days_per_year // 2)
        if len(history) >= drawdown_window:
            window = history[-drawdown_window:]
            assert snapshot.drawdown_6m == pytest.approx(window[-1] / max(window) - 1.0)
        else:
            assert snapshot.drawdown_6m is None


def test_provenance_rejects_metadata_that_disagrees_with_the_frozen_algorithm():
    provenance = _path().provenance

    with pytest.raises(ValueError, match="regime_seed disagrees"):
        replace(provenance, regime_seed=provenance.regime_seed + 1)
    with pytest.raises(ValueError, match="return lookback"):
        replace(provenance, return_lookback_days=21)
    with pytest.raises(ValueError, match="trading_days_per_year"):
        replace(provenance, trading_days_per_year=365)


def test_generated_path_rejects_snapshots_inconsistent_with_spec_or_daily_history():
    path = _path()
    q_tampered = replace(
        path.snapshots[0],
        risk_free_rate=path.snapshots[0].risk_free_rate + 0.01,
    )
    with pytest.raises(ValueError, match="snapshot Q inputs"):
        replace(path, snapshots=(q_tampered, *path.snapshots[1:]))

    history_tampered = replace(path.snapshots[0], return_20d=0.0)
    with pytest.raises(ValueError, match="snapshot history metrics"):
        replace(path, snapshots=(history_tampered, *path.snapshots[1:]))


@pytest.mark.parametrize(
    ("replacement_factory", "error"),
    [
        (lambda: {"regimes": _regimes()[:1]}, ValueError),
        (lambda: {"regimes": (_regimes()[0], _regimes()[0])}, ValueError),
        (lambda: {"regimes": (RegimeParameters("bad", float("nan"), 0.2, 0.2, 0.2, 0.2), _regimes()[1])}, ValueError),
        (lambda: {"regimes": (RegimeParameters("bad", 0.01, 0.0, 0.2, 0.2, 0.2), _regimes()[1])}, ValueError),
        (lambda: {"regimes": (RegimeParameters("bad", 0.01, float("inf"), 0.2, 0.2, 0.2), _regimes()[1])}, ValueError),
        (lambda: {"regimes": (RegimeParameters("bad", 0.01, 1e308, 0.2, 0.2, 0.2), _regimes()[1])}, ValueError),
        (lambda: {"regimes": (RegimeParameters("bad", 0.01, 0.2, 0.0, 0.2, 0.2), _regimes()[1])}, ValueError),
        (lambda: {"regimes": (RegimeParameters("bad", 0.01, 0.2, 0.2, float("inf"), 0.2), _regimes()[1])}, ValueError),
        (lambda: {"regimes": (RegimeParameters("bad", 0.01, 0.2, 0.80, 0.10, 0.10), _regimes()[1])}, ValueError),
        (lambda: {"risk_free_rate": float("nan")}, ValueError),
        (lambda: {"carry_rate": float("inf")}, ValueError),
        (lambda: {"transition_matrix": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))}, ValueError),
        (lambda: {"transition_matrix": ((1.1, -0.1), (0.2, 0.8))}, ValueError),
        (lambda: {"transition_matrix": ((0.9, 0.05), (0.2, 0.8))}, ValueError),
        (lambda: {"initial_distribution": (0.8, 0.1)}, ValueError),
        (lambda: {"initial_distribution": (1.1, -0.1)}, ValueError),
        (lambda: {"parameter_source": ""}, ValueError),
        (lambda: {"calibration_cutoff": date(2024, 1, 31), "calibration_data_hash": None}, ValueError),
        (lambda: {"calibration_cutoff": None, "calibration_data_hash": "a" * 64}, ValueError),
        (lambda: {"calibration_data_hash": "not-a-sha256"}, ValueError),
        (lambda: {"risk_free_rate": 1e308, "carry_rate": 1e308}, ValueError),
    ],
)
def test_spec_validation_fails_closed(replacement_factory, error):
    values = {
        "regimes": _regimes(), "transition_matrix": ((0.94, 0.06), (0.18, 0.82)),
        "initial_distribution": (0.8, 0.2), "risk_free_rate": 0.02, "carry_rate": 0.01,
        "parameter_source": "offline-test-fixture-v1", "calibration_cutoff": date(2024, 1, 31),
        "calibration_data_hash": "a" * 64,
    }
    with pytest.raises(error):
        values.update(replacement_factory())
        RegimeSwitchingGBMSpecV0(**values)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"seed": True}, TypeError),
        ({"decision_rounds": True}, TypeError),
        ({"decision_rounds": 0}, ValueError),
        ({"start_date": date(2024, 1, 30)}, ValueError),
        ({"start_spot": 0.0}, ValueError),
        ({"start_spot": float("inf")}, ValueError),
    ],
)
def test_generator_input_validation_fails_closed(kwargs, error):
    values = {
        "episode_id": "synthetic-test-episode", "underlying": "CSI500",
        "start_date": date(2024, 1, 31), "start_spot": 5000.0,
        "decision_rounds": 2, "seed": 20260831,
    }
    values.update(kwargs)
    with pytest.raises(error):
        generate_regime_switching_gbm(_spec(), **values)
