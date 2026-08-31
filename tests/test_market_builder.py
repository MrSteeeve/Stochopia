"""Past-only historical feature construction."""

from __future__ import annotations

from datetime import date, timedelta

from stochopia.market_builder import DailyClose, build_monthly_snapshots


def test_month_end_builder_uses_warmup_and_creates_half_year_rounds():
    start = date(2022, 9, 1)
    rows = []
    current = start
    value = 5000.0
    while current <= date(2023, 7, 31):
        if current.weekday() < 5:
            value *= 1.0005 if current.toordinal() % 2 else 0.9999
            rows.append(DailyClose(current, "CSI500", value, "fixture"))
        current += timedelta(days=1)
    snapshots = build_monthly_snapshots(rows, evaluation_end=date(2023, 6, 30))
    assert len(snapshots) == 6
    assert {item.episode_id for item in snapshots} == {"CSI500_2023H1"}
    assert [item.round_num for item in snapshots] == [1, 2, 3, 4, 5, 6]
    assert snapshots[0].realized_vol_60d is not None
    assert snapshots[0].source == "fixture"


def test_builder_uses_only_information_available_as_of_month_end():
    start = date(2022, 10, 1)
    rows = []
    current = start
    value = 5000.0
    while current <= date(2023, 2, 28):
        if current.weekday() < 5:
            value += 1.0
            rows.append(DailyClose(current, "CSI500", value, "fixture"))
        current += timedelta(days=1)
    original = build_monthly_snapshots(rows)
    # A future February price mutation cannot alter the January feature row.
    mutated = [
        DailyClose(row.as_of, row.underlying, row.close * (2 if row.as_of.month == 2 else 1), row.source)
        for row in rows
    ]
    changed = build_monthly_snapshots(mutated)
    january_original = next(item for item in original if item.as_of.month == 1)
    january_changed = next(item for item in changed if item.as_of.month == 1)
    assert january_original.public_brief() == january_changed.public_brief()
