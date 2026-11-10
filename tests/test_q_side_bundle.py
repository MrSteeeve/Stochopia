"""Tests for immutable Q-side records and strict scenario selection."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from stochopia.q_side_bundle import (
    QFieldObservation,
    QMarketSnapshot,
    QSideBundle,
    load_q_side_bundle,
    write_q_side_bundle,
)


def _observed(value: float, *, days: int, source: str) -> QFieldObservation:
    return QFieldObservation(
        value=value,
        unit="decimal_annualized",
        source=source,
        observation_date=date(2023, 1, 31),
        target_days=days,
        method="formal-observed-test",
        details={"raw_input_sha256": "d" * 64},
    )


def _unavailable(*, days: int, source: str, reason: str) -> QFieldObservation:
    return QFieldObservation(
        value=None,
        unit="decimal_annualized",
        source=source,
        observation_date=None,
        target_days=days,
        unavailable_reason=reason,
    )


def _snapshot(*, complete: bool = True) -> QMarketSnapshot:
    fields = {
        "risk_free_rate": _observed(0.02, days=91, source="Tushare SHIBOR"),
        "carry_rate": _observed(0.01, days=91, source="CFFEX IC settlement"),
        "atm_iv_1m": _observed(0.18, days=30, source="510500 ETF option proxy"),
        "atm_iv_3m": _observed(0.19, days=91, source="510500 ETF option proxy"),
        "atm_iv_6m": _observed(0.20, days=182, source="510500 ETF option proxy"),
    }
    if not complete:
        fields["atm_iv_6m"] = _unavailable(
            days=182,
            source="510500 ETF option proxy",
            reason="quality_rejected:no formal-chain tenor passed",
        )
    return QMarketSnapshot(
        underlying="CSI500",
        as_of=date(2023, 1, 31),
        partition="dev",
        fields=fields,
    )


def _bundle(*, complete: bool = True) -> QSideBundle:
    return QSideBundle(
        p_history={"cutoff": "2022-12-31", "usage": "P_calibration_only"},
        q_data={
            "start": "2022-09-19",
            "cutoff": "2025-12-31",
            "usage": "point_in_time_scenario_input_or_sensitivity",
        },
        split_policy={
            "train": {"end": "2022-12-31"},
            "dev": {"end": "2023-12-31"},
            "evaluation_reused_non_private": {"end": "2025-12-31"},
            "private_evaluation": "unavailable",
        },
        records=(_snapshot(complete=complete),),
    )


def test_bundle_round_trips_and_selects_complete_record(tmp_path):
    bundle = _bundle()
    selection = bundle.select("CSI500", date(2023, 1, 31))

    assert selection.snapshot == bundle.records[0]
    assert selection.q_cutoff == date(2025, 12, 31)
    assert bundle.p_history["cutoff"] == "2022-12-31"
    target = write_q_side_bundle(bundle, tmp_path)
    assert target.name == "q_side_bundle.json"
    assert load_q_side_bundle(target) == bundle


def test_incomplete_observed_q_cannot_drive_generator():
    bundle = _bundle(complete=False)
    record = bundle.records[0]
    assert record.fields["atm_iv_6m"].value is None
    assert record.fields["atm_iv_6m"].availability_status == "quality_rejected"
    assert record.fields["atm_iv_6m"].unavailable_reason.startswith("quality_rejected")
    with pytest.raises(ValueError, match="incomplete"):
        bundle.select("CSI500", date(2023, 1, 31))


def test_total_variance_must_not_decrease_with_tenor():
    fields = dict(_snapshot().fields)
    fields["atm_iv_6m"] = _observed(0.05, days=182, source="bad chain")
    with pytest.raises(ValueError, match="total variance decreases"):
        replace(_snapshot(), fields=fields)


def test_future_dated_observation_is_rejected():
    fields = dict(_snapshot().fields)
    fields["carry_rate"] = replace(
        fields["carry_rate"], observation_date=date(2023, 2, 1)
    )
    with pytest.raises(ValueError, match="future-dated"):
        replace(_snapshot(), fields=fields)


def test_q_split_cannot_claim_private_evaluation():
    bundle = _bundle()
    split = dict(bundle.split_policy)
    split["private_evaluation"] = {"start": "2024-01-01"}
    with pytest.raises(ValueError, match="must not claim"):
        replace(bundle, split_policy=split)
