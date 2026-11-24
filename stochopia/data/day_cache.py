"""Portable NPZ caches for reconstructed Shenzhen stock-days."""

from io import BytesIO
from pathlib import Path

import numpy as np

from stochopia.common import atomic_write_bytes
from .szse_l2 import SZSEDaySlice


def save_day_slice(path: Path, day: SZSEDaySlice) -> None:
    """Atomically store one reconstructed stock-day without pickling."""
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        wind_code=day.wind_code,
        trading_date=day.trading_date,
        prev_close_ticks=day.prev_close_ticks,
        session_bounds=day.session_bounds,
        time_ms=day.time_ms,
        seq_id=day.seq_id,
        dt_ms=day.dt_ms,
        action_type=day.action_type,
        rel_ticks=day.rel_ticks,
        volume_shares=day.volume_shares,
        cancel_age_ms=day.cancel_age_ms,
        pre_book_ticks=day.pre_book_ticks,
        pre_book_shares=day.pre_book_shares,
        mid_ticks_x2=day.mid_ticks_x2,
        spread_ticks=day.spread_ticks,
    )
    atomic_write_bytes(path, buffer.getvalue())


def load_day_slice(path: Path) -> SZSEDaySlice:
    """Load a cached day with its original dtypes and read-only event arrays."""
    metadata = {"wind_code", "trading_date", "prev_close_ticks", "session_bounds"}
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files if name not in metadata}
        for array in arrays.values():
            array.setflags(write=False)
        return SZSEDaySlice(
            wind_code=str(data["wind_code"].item()),
            trading_date=int(data["trading_date"].item()),
            prev_close_ticks=int(data["prev_close_ticks"].item()),
            session_bounds=tuple((int(start), int(end)) for start, end in data["session_bounds"]),
            **arrays,
        )
