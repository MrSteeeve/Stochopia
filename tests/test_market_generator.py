"""Offline contract tests for exchange-clocked, P/Q-separated generation."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from statistics import pstdev

import pytest

from stochopia.market_generator import (
    SCHEMA_VERSION,
    GeneratedMarketPath,
    RegimeParameters,
    RegimeSwitchingGBMSpecV1,
    ScenarioProvenance,
    _spec_from_manifest,
    generate_regime_switching_gbm,
)
from stochopia.q_side_bundle import QFieldObservation, QMarketSnapshot, QScenarioInput
from stochopia.trading_calendar import CalendarSourceFile, FrozenTradingCalendar


def _calendar(*, source_version: str = "test-v1", omit: date | None = None) -> FrozenTradingCalendar:
    start, end = date(2024, 1, 1), date(2025, 1, 31)
    closures = {
        date(2024, 1, 1),
        *(date(2024, 2, 9) + timedelta(days=offset) for offset in range(10)),
        *(date(2024, 4, 4) + timedelta(days=offset) for offset in range(3)),
        *(date(2024, 5, 1) + timedelta(days=offset) for offset in range(5)),
        date(2024, 6, 10),
        date(2024, 9, 16),
        date(2024, 9, 17),
        *(date(2024, 10, 1) + timedelta(days=offset) for offset in range(7)),
        date(2025, 1, 1),
        *(date(2025, 1, 28) + timedelta(days=offset) for offset in range(4)),
    }
    values: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in closures and current != omit:
            values.append(current)
        current += timedelta(days=1)
    return FrozenTradingCalendar(
        calendar_id="sse-test-2024-v1",
        exchange="SSE",
        timezone="Asia/Shanghai",
        source="Tushare Pro trade_cal test fixture",
        source_version=source_version,
        coverage_start=start,
        coverage_end=end,
        open_dates=tuple(values),
        source_files=(
            CalendarSourceFile(
                csv_path="trade_cal/test.csv",
                sidecar_path="trade_cal/test.json",
                request_start=start,
                request_end=end,
                retrieved_at="2026-08-31T00:00:00+00:00",
                sdk_version="test",
                bridge_host="offline-test",
            ),
        ),
    )


def _regimes(
    *, drift_shift: float = 0.0, volatility_scale: float = 1.0
) -> tuple[RegimeParameters, RegimeParameters]:
    return (
        RegimeParameters("calm", 0.06 + drift_shift, 0.16 * volatility_scale),
        RegimeParameters("stress", -0.08 + drift_shift, 0.34 * volatility_scale),
    )


def _spec(
    *,
    drift_shift: float = 0.0,
    volatility_scale: float = 1.0,
    calendar: FrozenTradingCalendar | None = None,
) -> RegimeSwitchingGBMSpecV1:
    return RegimeSwitchingGBMSpecV1(
        regimes=_regimes(
            drift_shift=drift_shift,
            volatility_scale=volatility_scale,
        ),
        transition_matrix=((0.94, 0.06), (0.18, 0.82)),
        initial_distribution=(0.8, 0.2),
        calendar=calendar or _calendar(),
        p_parameter_source="offline-test-P-fixture-v1",
        p_calibration_cutoff=date(2023, 12, 31),
    )


def _q_input(
    *,
    risk_free_rate: float = 0.02,
    carry_rate: float = 0.01,
    iv_shift: float = 0.0,
    as_of: date = date(2024, 1, 31),
) -> QScenarioInput:
    def observed(value: float, source: str, target_days: int) -> QFieldObservation:
        return QFieldObservation(
            value=value,
            unit="decimal_annualized",
            source=source,
            observation_date=as_of,
            target_days=target_days,
            method="offline-test-observed",
        )

    snapshot = QMarketSnapshot(
        underlying="CSI500",
        as_of=as_of,
        partition="dev",
        fields={
            "risk_free_rate": observed(risk_free_rate, "SHIBOR", 91),
            "carry_rate": observed(carry_rate, "CFFEX IC", 91),
            "atm_iv_1m": observed(0.18 + iv_shift, "510500 ETF options", 30),
            "atm_iv_3m": observed(0.19 + iv_shift, "510500 ETF options", 91),
            "atm_iv_6m": observed(0.20 + iv_shift, "510500 ETF options", 182),
        },
    )
    return QScenarioInput(snapshot=snapshot, q_cutoff=date(2025, 12, 31))


def _path(
    *,
    spec: RegimeSwitchingGBMSpecV1 | None = None,
    q_input: QScenarioInput | None = None,
    seed: int = 20260831,
    decision_rounds: int = 7,
    start_date: date = date(2024, 1, 31),
) -> GeneratedMarketPath:
    return generate_regime_switching_gbm(
        spec or _spec(),
        q_input=q_input or _q_input(as_of=start_date),
        episode_id="synthetic-test-episode",
        underlying="CSI500",
        start_date=start_date,
        start_spot=5000.0,
        decision_rounds=decision_rounds,
        seed=seed,
    )


def test_same_inputs_and_seed_are_identical_but_new_seed_changes_path():
    first = _path()
    second = _path()
    changed_seed = _path(seed=20260832)

    assert first.manifest == second.manifest
    assert first.daily_spots != changed_seed.daily_spots
    assert first.manifest != changed_seed.manifest
    assert json.loads(json.dumps(first.manifest)) == first.manifest


def test_exchange_holidays_and_actual_month_ends_drive_the_path():
    path = _path(decision_rounds=9)
    dates = set(path.daily_dates)
    snapshot_dates = [item.as_of for item in path.snapshots]

    assert date(2024, 2, 8) in dates
    assert date(2024, 2, 19) in dates
    assert not dates.intersection(
        date(2024, 2, 9) + timedelta(days=offset) for offset in range(10)
    )
    assert date(2024, 9, 30) in dates
    assert date(2024, 10, 8) in dates
    assert not dates.intersection(
        date(2024, 10, 1) + timedelta(days=offset) for offset in range(7)
    )
    assert date(2024, 2, 29) in snapshot_dates
    assert date(2024, 9, 30) in snapshot_dates
    assert date(2024, 10, 31) in snapshot_dates


def test_terminal_month_uses_exchange_month_end_across_year_boundary():
    path = _path(
        start_date=date(2024, 12, 31),
        q_input=_q_input(as_of=date(2024, 12, 31)),
        decision_rounds=1,
    )
    assert [item.as_of for item in path.snapshots] == [
        date(2024, 12, 31),
        date(2025, 1, 27),
    ]
    assert path.daily_dates[-1] == date(2025, 1, 27)


def test_q_input_changes_snapshots_and_root_but_not_physical_path():
    base = _path()
    changed = _path(q_input=_q_input(risk_free_rate=0.07, carry_rate=0.015, iv_shift=0.05))

    assert changed.daily_dates == base.daily_dates
    assert changed.daily_spots == base.daily_spots
    assert changed.latent_regime_path == base.latent_regime_path
    assert changed.provenance.q_selection != base.provenance.q_selection
    assert changed.manifest != base.manifest
    assert changed.snapshots[0].atm_iv_3m != base.snapshots[0].atm_iv_3m


def test_p_input_changes_spots_without_changing_latent_states_or_q_fields():
    base = _path()
    for changed_spec in (_spec(drift_shift=0.10), _spec(volatility_scale=1.25)):
        changed = _path(spec=changed_spec)
        assert changed.latent_regime_path == base.latent_regime_path
        assert changed.daily_spots != base.daily_spots
        for name in ("risk_free_rate", "carry_rate", "atm_iv_1m", "atm_iv_3m", "atm_iv_6m"):
            assert [getattr(item, name) for item in changed.snapshots] == [
                getattr(item, name) for item in base.snapshots
            ]


def test_calendar_metadata_and_open_sessions_are_recorded_in_provenance():
    base = _path()
    metadata_changed = _path(spec=_spec(calendar=_calendar(source_version="test-v2")))
    session_changed = _path(spec=_spec(calendar=_calendar(omit=date(2024, 3, 15))))

    assert metadata_changed.daily_spots == base.daily_spots
    assert metadata_changed.provenance.spec != base.provenance.spec
    assert date(2024, 3, 15) not in session_changed.daily_dates
    assert session_changed.daily_dates != base.daily_dates


def test_q_is_frozen_at_start_and_public_source_omits_episode_and_seed():
    q_input = _q_input()
    path = _path(q_input=q_input, decision_rounds=3)
    for snapshot in path.snapshots:
        assert snapshot.risk_free_rate == q_input.snapshot.fields["risk_free_rate"].value
        assert snapshot.carry_rate == q_input.snapshot.fields["carry_rate"].value
        assert snapshot.atm_iv_6m == q_input.snapshot.fields["atm_iv_6m"].value
        assert snapshot.source == f"synthetic;schema={SCHEMA_VERSION}"
        assert snapshot.regime == "synthetic"
    assert path.provenance.q_dynamics == "frozen_at_scenario_start"
    assert path.provenance.q_selection == q_input.manifest
    assert path.provenance.q_observation_date == "2024-01-31"


def test_public_trend_alpha_is_the_regime_independent_stationary_drift():
    spec = _spec()
    path = _path(spec=spec)
    # pi for ((0.94, 0.06), (0.18, 0.82)) is (0.75, 0.25).
    expected = sum(
        weight * regime.physical_annual_return
        for weight, regime in zip((0.75, 0.25), spec.regimes)
    )
    assert len(set(path.latent_regime_path)) > 1
    for snapshot in path.snapshots:
        reconstructed = snapshot.risk_free_rate + snapshot.trend_alpha - snapshot.carry_rate
        assert reconstructed == pytest.approx(expected)
    assert len({snapshot.trend_alpha for snapshot in path.snapshots}) == 1


def test_history_features_match_exact_exchange_session_windows():
    spec = _spec()
    path = _path(spec=spec, decision_rounds=7)
    date_to_index = {value: index for index, value in enumerate(path.daily_dates)}
    for snapshot in path.snapshots:
        history = path.daily_spots[: date_to_index[snapshot.as_of] + 1]
        if len(history) >= 21:
            assert snapshot.return_20d == pytest.approx(history[-1] / history[-21] - 1.0)
            log_returns = [
                math.log(history[index] / history[index - 1])
                for index in range(len(history) - 20, len(history))
            ]
            assert snapshot.realized_vol_20d == pytest.approx(
                pstdev(log_returns) * math.sqrt(spec.trading_days_per_year)
            )
        else:
            assert snapshot.return_20d is None
            assert snapshot.realized_vol_20d is None
        if len(history) >= 61:
            log_returns = [
                math.log(history[index] / history[index - 1])
                for index in range(len(history) - 60, len(history))
            ]
            assert snapshot.realized_vol_60d == pytest.approx(
                pstdev(log_returns) * math.sqrt(spec.trading_days_per_year)
            )
        else:
            assert snapshot.realized_vol_60d is None


def test_provenance_embeds_spec_calendar_and_q_selection():
    path = _path()
    provenance = path.provenance
    assert isinstance(provenance, ScenarioProvenance)
    text = json.dumps(path.manifest)
    assert SCHEMA_VERSION in text
    assert "weekday-only-v0" not in text
    assert "each-exchange-open-day-v1" in text
    assert provenance.calendar_id == "sse-test-2024-v1"
    assert provenance.p_parameter_source == "offline-test-P-fixture-v1"
    assert provenance.spec == _spec().manifest
    assert provenance.normalized_spec == _spec()
    assert provenance.q_input == _q_input()


def test_old_weekday_schema_is_rejected_instead_of_reinterpreted():
    with pytest.raises(ValueError, match="explicit migration"):
        _spec_from_manifest(
            {
                "schema": "stochopia.synthetic-regime-scenario.v1",
                "calendar_id": "weekday-only-v0",
            }
        )


def test_generator_rejects_non_month_end_or_misaligned_q_input():
    with pytest.raises(ValueError, match="final exchange open session"):
        _path(start_date=date(2024, 1, 30), q_input=_q_input(as_of=date(2024, 1, 30)))
    with pytest.raises(ValueError, match="scenario start_date"):
        _path(q_input=_q_input(as_of=date(2024, 2, 29)))
    with pytest.raises(TypeError, match="complete QScenarioInput"):
        generate_regime_switching_gbm(
            _spec(),
            q_input=None,  # type: ignore[arg-type]
            episode_id="bad",
            underlying="CSI500",
            start_date=date(2024, 1, 31),
            start_spot=5000.0,
            decision_rounds=1,
            seed=1,
        )



@pytest.mark.parametrize(
    "kwargs",
    [
        {"regimes": _regimes()[:1]},
        {"regimes": (_regimes()[0], _regimes()[0])},
        {"transition_matrix": ((1.0, 0.0),)},
        {"transition_matrix": ((0.8, 0.1), (0.2, 0.8))},
        {"initial_distribution": (1.0,)},
    ],
)
def test_invalid_p_specs_fail_closed(kwargs):
    defaults = {
        "regimes": _regimes(),
        "transition_matrix": ((0.94, 0.06), (0.18, 0.82)),
        "initial_distribution": (0.8, 0.2),
        "calendar": _calendar(),
        "p_parameter_source": "fixture",
        "p_calibration_cutoff": date(2023, 12, 31),
    }
    defaults.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        RegimeSwitchingGBMSpecV1(**defaults)
