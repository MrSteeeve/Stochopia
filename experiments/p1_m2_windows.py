"""Registered P1/P2 market windows, training calibration and replay references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from experiments.p1_m2_metrics import TapeStats, tape_statistics
from stochopia.data import load_day_slice
from stochopia.neural.lob_interface import ACTION_BINS, DT_BINS, LOT_BINS, REL_TICK_BINS
from stochopia.neural.lob_training import EncodedDay, encode_day


SYMBOLS = ("000001.SZ", "300001.SZ")
CONTEXT = 128
CHILDREN = 8
PARTICIPATION = 0.10
WINDOW_MIDPOINTS = (37_800_000, 50_310_000)
SESSION_DURATION_MS = 7_200_000 + 7_020_000
HEADS = ("dt", "action", "rel_ticks", "volume")


@dataclass(frozen=True)
class Window:
    data: EncodedDay
    session: int
    start: int
    end: int

    @property
    def max_events(self) -> int:
        return max(6_000, 8 * (self.end - self.start))


def _load_days(root: Path, symbol: str, dates: list[int]) -> list[EncodedDay]:
    loaded = []
    for date in dates:
        day = load_day_slice(root / "cache" / str(date) / f"{symbol}.npz")
        if day.trading_date != date or day.wind_code != symbol:
            raise ValueError(f"Cache metadata does not match {symbol} on {date}")
        loaded.append(encode_day(day))
    return loaded


def _marginal(days: list[EncodedDay]) -> list[np.ndarray]:
    observed = torch.cat([day.tokens for day in days]).numpy()
    marginals = []
    for axis, size in enumerate((DT_BINS, ACTION_BINS, REL_TICK_BINS, LOT_BINS)):
        counts = np.bincount(observed[:, axis], minlength=size) + 0.5
        marginals.append(-np.log(counts / counts.sum()))
    return marginals


def _calibrate(root: Path, train: list[EncodedDay], symbol: str) -> dict[str, int | float]:
    size_arrays = []
    touch_arrays = []
    daily_shares = []
    daily_price_changes = []
    for data in train:
        day = data.day
        aggressive = (day.action_type == 2) | (day.action_type == 3)
        size_arrays.append(day.volume_shares[aggressive])
        touch_arrays.append(day.pre_book_shares[:, :, 0].reshape(-1))
        quotes = pq.read_table(root / "raw" / str(day.trading_date) / "行情.parquet", columns=["cum_volume"],
                               filters=[("wind_code", "==", symbol)])
        daily_shares.append(max(v for v in quotes["cum_volume"].to_pylist() if v is not None))
        daily_price_changes.append(day.mid_ticks_x2[-1] / 2 - day.prev_close_ticks)
    sizes = np.concatenate(size_arrays)
    frequency = len(sizes) / (len(train) * SESSION_DURATION_MS)
    spacing = 10 * max(1, round(1.0 / (10 * frequency * PARTICIPATION)))
    median_touch = float(np.median(np.concatenate(touch_arrays)))
    median_order = int(round(float(np.median(sizes))))
    queue_scaled_size = 100 * round(0.08 * median_touch / 100)
    return {"child_shares": max(median_order, queue_scaled_size), "spacing_ms": spacing,
            "median_touch_shares": median_touch, "median_aggressive_shares": median_order,
            "queue_scaled_child_shares": queue_scaled_size,
            "train_aggressive_per_second": 1000 * frequency, "daily_shares": float(np.mean(daily_shares)),
            "sigma_ticks": float(np.std(daily_price_changes, ddof=1))}


def _windows(days: list[EncodedDay], horizon: int) -> list[Window]:
    windows = []
    for data in days:
        times = data.day.time_ms
        for session, (start, end) in enumerate(data.day.session_bounds):
            position = max(start + CONTEXT, int(np.searchsorted(times, WINDOW_MIDPOINTS[session])))
            cutoff = int(np.searchsorted(times, times[position] + horizon, side="right")) if position < end else end
            if not position < cutoff < end:
                raise ValueError(f"Incomplete dev window for {data.day.wind_code} {data.day.trading_date} session {session}")
            windows.append(Window(data, session, position, cutoff))
    return windows


def _replay_marks(window: Window, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    day = window.data.day
    times = day.time_ms[window.start:window.end] - day.time_ms[window.start]
    marks = np.searchsorted(times, grid, side="right") + window.start
    marks[0] = window.start  # both the model and replay start at the pre-event book
    return times, marks


def _replay_stats(window: Window, grid: np.ndarray) -> TapeStats:
    day = window.data.day
    times, marks = _replay_marks(window, grid)
    return tape_statistics(np.column_stack((times, day.action_type[window.start:window.end])),
                           day.mid_ticks_x2[marks] / 2, day.spread_ticks[marks], day.pre_book_shares[marks, :, 0])
