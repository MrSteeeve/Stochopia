"""Replay one Shenzhen stock's Wind orders and trade-stream messages."""

from bisect import bisect_left, insort
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from stochopia.measure.data_properties import measure_reconstruction_quality

ACTION_TYPES = (
    "passive_buy", "passive_sell", "aggressive_buy", "aggressive_sell", "cancel_buy", "cancel_sell",
)
_SESSIONS = ((34_200_000, 41_400_000), (46_800_000, 53_820_000))
_ORDER_COLUMNS = ("ex_order_id", "order_type", "order_code", "price", "volume")
_TRADE_COLUMNS = ("trade_id", "trade_code", "bid_order_id", "ask_order_id", "price", "volume")
_QUOTE_COLUMNS = tuple(
    f"{side}_{field}{i}" for side in ("bid", "ask") for field in ("px", "vol") for i in range(1, 11)
)
_MESSAGE = np.dtype([
    ("ms", "i8"), ("seq", "u8"), ("kind", "U1"), ("side", "i1"),
    ("price", "i8"), ("shares", "i8"), ("bid", "u8"), ("ask", "u8"),
])
_EVENT = np.dtype([
    ("time_ms", "i8"), ("seq_id", "u8"), ("dt_ms", "i8"), ("action_type", "i1"),
    ("rel_ticks", "i4"), ("volume_shares", "i8"), ("cancel_age_ms", "i8"),
])


@dataclass(frozen=True)
class SZSEDaySlice:
    """Aligned exogenous events and their pre-event bid/ask book, padded to ten levels."""

    wind_code: str
    trading_date: int
    prev_close_ticks: int
    session_bounds: tuple[tuple[int, int], ...]
    time_ms: np.ndarray
    seq_id: np.ndarray
    dt_ms: np.ndarray
    action_type: np.ndarray
    rel_ticks: np.ndarray
    volume_shares: np.ndarray
    cancel_age_ms: np.ndarray
    pre_book_ticks: np.ndarray
    pre_book_shares: np.ndarray
    mid_ticks_x2: np.ndarray
    spread_ticks: np.ndarray

    @property
    def volume_lots(self) -> np.ndarray:
        """Event sizes in lots of 100 shares, including fractional lots."""
        return self.volume_shares / 100.0


@dataclass(frozen=True)
class BookReconciliation:
    """Continuous-session quote agreement. Empty comparisons have NaN rates."""

    n_snapshots: int
    touch_match_rate: float
    top5_log_vol_mae: float
    top10_log_vol_mae: float
    max_deviation_time_ms: int | None = None
    vol_aligned_touch_match_rate: float = np.nan
    n_volume_aligned_snapshots: int = 0
    top1_quote_match_rate: float = np.nan
    top10_depth_mae: float = np.nan
    invalid_spread_rate: float = np.nan


def parse_wind_time_ms(hhmmssmmm: int | np.ndarray) -> int | np.ndarray:
    """Convert scalar or array HHMMSSmmm to milliseconds since local midnight."""
    raw = np.asarray(hhmmssmmm)
    if raw.dtype.kind not in "iu":
        raise ValueError("HHMMSSmmm must contain integers")
    raw = raw.astype(np.int64, copy=False)
    hour, minute, second = raw // 10_000_000, raw // 100_000 % 100, raw // 1_000 % 100
    if np.any((raw < 0) | (hour > 23) | (minute > 59) | (second > 59)):
        raise ValueError("Invalid HHMMSSmmm clock value")
    result = ((hour * 60 + minute) * 60 + second) * 1_000 + raw % 1_000
    return int(result) if raw.ndim == 0 else result


def validate_chronological_splits(
    train_dates: Sequence[int], dev_dates: Sequence[int], test_dates: Sequence[int],
) -> None:
    """Require nonempty, strictly increasing and disjoint train, dev and test dates."""
    for dates in (train_dates, dev_dates, test_dates):
        if len(dates) == 0 or any(a >= b for a, b in zip(dates, dates[1:])):
            raise ValueError("Each split requires nonempty, strictly increasing trading dates")
    if train_dates[-1] >= dev_dates[0] or dev_dates[-1] >= test_dates[0]:
        raise ValueError("Trading-date splits must be chronological and disjoint")


# --- Parquet input and message ordering ---

def _read(path: Path, wind_code: str, fields: tuple[str, ...], *, waterline: bool = False) -> pa.Table:
    columns = ["wind_code", "date", "time", *fields]
    names = set(pq.read_schema(path).names)
    missing = set(columns) - names
    if missing:
        raise ValueError(f"{path.name} missing required columns: {', '.join(sorted(missing))}")
    if waterline and "cum_volume" in names:
        columns.append("cum_volume")
    return pq.read_table(path, columns=columns, filters=[("wind_code", "==", wind_code)])


def _numbers(table: pa.Table, name: str) -> np.ndarray:
    return table[name].fill_null(0).to_numpy()


def _ticks(raw: np.ndarray) -> np.ndarray:
    return np.rint(raw / 100.0).astype(np.int64)


def _messages(orders: pa.Table, trades: pa.Table) -> np.ndarray:
    packed = np.zeros(len(orders) + len(trades), dtype=_MESSAGE)
    adds, outcomes = packed[:len(orders)], packed[len(orders):]
    for block, table, sequence in ((adds, orders, "ex_order_id"), (outcomes, trades, "trade_id")):
        block["ms"] = parse_wind_time_ms(table["time"].to_numpy())
        block["seq"] = _numbers(table, sequence)
        block["price"] = _ticks(_numbers(table, "price"))
        block["shares"] = _numbers(table, "volume").astype(np.int64)
    order_type = orders["order_type"].to_numpy()
    # Wind exports Shenzhen limit orders as either '0' or '2'.
    adds["kind"] = np.where(np.isin(order_type, ("0", "2")), "L", order_type)
    adds["side"] = orders["order_code"].to_numpy() == "S"
    trade_code = trades["trade_code"].to_numpy()
    outcomes["kind"] = np.where(np.isin(trade_code, ("0", "C")), trade_code, "?")
    outcomes["bid"] = _numbers(trades, "bid_order_id")
    outcomes["ask"] = _numbers(trades, "ask_order_id")
    return packed[np.lexsort((packed["seq"], packed["ms"]))]


def _quote_books(quotes: pa.Table, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    keep = np.r_[sorted_times[:-1] != sorted_times[1:], True]
    session = np.zeros(len(times), dtype=bool)
    for start, end in _SESSIONS:
        session |= (sorted_times >= start) & (sorted_times < end)
    order = order[keep & session]
    prices = _ticks(np.column_stack([_numbers(quotes, col)[order] for col in _QUOTE_COLUMNS if "_px" in col]))
    shares = np.column_stack([_numbers(quotes, col)[order] for col in _QUOTE_COLUMNS if "_vol" in col])
    prices = prices.reshape(-1, 2, 10)
    shares = shares.reshape(-1, 2, 10).astype(np.int64)
    valid = (prices[:, :, 0] > 0).all(axis=1)
    shares[prices <= 0] = 0
    selected = order[valid]
    volumes = np.zeros(len(selected), dtype=np.int64)
    if "cum_volume" in quotes.column_names:
        volumes = _numbers(quotes, "cum_volume")[selected].astype(np.int64)
    return times[selected], prices[valid], shares[valid], volumes


# --- Book replay ---

class _Book:
    def __init__(self) -> None:
        self.orders: dict[int, tuple[int, int, int, int]] = {}
        self.levels: tuple[dict[int, int], dict[int, int]] = ({}, {})
        self.keys: tuple[list[int], list[int]] = ([], [])

    def _change(self, side: int, price: int, size: int) -> None:
        if price <= 0 or size == 0:
            return
        key = price if side else -price
        levels, keys = self.levels[side], self.keys[side]
        previous = levels.get(key, 0)
        depth = previous + size
        if depth:
            levels[key] = depth
            if previous == 0:
                insort(keys, key)
        else:
            del levels[key]
            keys.pop(bisect_left(keys, key))

    def add(self, seq: int, side: int, price: int, size: int, ms: int) -> None:
        self.orders[seq] = (side, price, size, ms)
        self._change(side, price, size)

    def reduce(self, seq: int, size: int) -> None:
        order = self.orders.get(seq)
        if order is not None:
            side, price, remaining, submit = order
            removed = min(size, remaining)
            self._change(side, price, -removed)
            if remaining > removed:
                self.orders[seq] = (side, price, remaining - removed, submit)
            else:
                del self.orders[seq]

    def touch(self) -> tuple[int, int]:
        bids, asks = self.keys
        return (-bids[0] if bids else 0, asks[0] if asks else 0)

    def write_top(self, prices: np.ndarray, shares: np.ndarray) -> None:
        for side, keys in enumerate(self.keys):
            for i, key in enumerate(keys[:10]):
                prices[side, i] = key if side else -key
                shares[side, i] = self.levels[side][key]


def _arrival(
    row: np.void, following: np.ndarray, touch: tuple[int, int], auto_cancels: set[int],
) -> tuple[int, int, int]:
    """Resolve effective price, action and relative ticks from immediate outcomes."""
    ms, seq, kind, side, price, size, _, _ = row
    direction = 1 if side == 0 else -1
    opposite = touch[1 - side]
    filled = 0
    canceled = 0
    furthest = 0

    # Attribute only adjacent same-time outcomes that actually reference this arrival.
    for time, outcome, code, _, execution_price, qty, bid, ask in following:
        if time != ms or code not in ("0", "C"):
            break
        if (bid if side == 0 else ask) != seq:
            continue
        if code == "0":
            filled += int(qty)
            if furthest == 0 or (execution_price - furthest) * direction > 0:
                furthest = int(execution_price)
        elif kind not in ("L", "U"):
            canceled += int(qty)
            auto_cancels.add(int(outcome))

    resting = int(size) - filled - canceled
    if kind == "U":
        # Shenzhen own-best orders join the current touch; Wind may report a price limit.
        price = touch[side]
    elif kind == "1":
        # A market residual can become a limit at the deepest immediately executed price.
        price = furthest or opposite
    elif price <= 0:
        price = furthest
    if filled or (kind == "L" and price > 0 and opposite > 0 and (price - opposite) * direction >= 0):
        target = price if kind == "L" or not furthest else furthest
        return int(price), 2 + int(side), max(0, int((target - opposite) * direction))
    action = int(side) if price > 0 and resting > 0 else -1
    return int(price), action, int((touch[side] - price) * direction)


def _reconciliation(
    times: np.ndarray, actual_prices: np.ndarray, actual_depth: np.ndarray,
    replay_prices: np.ndarray, replay_depth: np.ndarray,
    volumes: np.ndarray, waterline_books: dict[int, tuple[int, int]],
) -> BookReconciliation:
    """For [snapshot, side, level], mean absolute log(1+shares) difference and its worst row."""
    if not len(times):
        return BookReconciliation(0, np.nan, np.nan, np.nan)
    quality = measure_reconstruction_quality([{
        "observed_ticks": actual_prices,
        "reconstructed_ticks": replay_prices,
        "observed_shares": actual_depth,
        "reconstructed_shares": replay_depth,
    }])
    error = np.abs(np.log1p(actual_depth) - np.log1p(replay_depth))
    worst = int(times[error.mean(axis=(1, 2)).argmax()])
    # A plateau may contain quote changes between fills. Compare to the first
    # post-fill touch reaching the exact positive volume, without fitting a lag.
    matches = [tuple(px) == waterline_books[v] for v, px in zip(volumes, actual_prices[:, :, 0])
               if v in waterline_books]
    return BookReconciliation(
        len(times),
        quality["top1_quote_match_rate"],
        float(error[:, :, :5].mean()),
        float(error.mean()),
        worst,
        float(np.mean(matches)) if matches else np.nan,
        len(matches),
        quality["top1_quote_match_rate"],
        quality["top10_depth_mae"],
        quality["invalid_spread_rate"],
    )


def reconstruct_szse_day(
    day_dir: Path, wind_code: str, *, reconcile: bool = True,
) -> tuple[SZSEDaySlice, BookReconciliation | None]:
    """Replay a Shenzhen stock-day and optionally compare its observed ten-level quotes."""
    if not wind_code.endswith(".SZ"):
        raise ValueError("Expected a Shenzhen stock code ending in .SZ")
    quote_fields = ("prev_close", *_QUOTE_COLUMNS) if reconcile else ("prev_close",)
    quotes = _read(day_dir / "行情.parquet", wind_code, quote_fields, waterline=reconcile)
    orders = _read(day_dir / "逐笔委托.parquet", wind_code, _ORDER_COLUMNS)
    trades = _read(day_dir / "逐笔成交.parquet", wind_code, _TRADE_COLUMNS)
    if not len(quotes):
        raise ValueError(f"No quotes for {wind_code} in {day_dir}")
    date = quotes["date"][0].as_py()
    if any(np.any(table["date"].to_numpy() != date) for table in (quotes, orders, trades)):
        raise ValueError(f"Tables in {day_dir} must describe the same trading date")

    quote_times = np.asarray(parse_wind_time_ms(quotes["time"].to_numpy()))
    prev_close = int(_ticks(_numbers(quotes, "prev_close"))[len(quotes) - 1 - quote_times[::-1].argmax()])
    if reconcile:
        times, actual_px, actual_qty, volumes = _quote_books(quotes, quote_times)
    else:
        times = np.empty(0, dtype=np.int64)
        volumes = np.empty(0, dtype=np.int64)
        actual_px = np.empty((0, 2, 10), dtype=np.int64)
        actual_qty = np.empty((0, 2, 10), dtype=np.int64)
    replay_px = np.zeros_like(actual_px)
    replay_qty = np.zeros_like(actual_qty)
    requested_volumes = {int(v) for v in volumes if v > 0}
    waterline_books: dict[int, tuple[int, int]] = {}

    messages = _messages(orders, trades)
    events = np.empty(np.count_nonzero(messages["kind"] != "0"), dtype=_EVENT)
    prices = np.zeros((len(events), 2, 10), dtype=np.int32)
    depth = np.zeros_like(prices, dtype=np.int64)
    book = _Book()
    auto_cancels: set[int] = set()
    counts = [0, 0]
    previous = [0, 0]
    cumulative_volume = 0
    n = 0
    q = 0
    for index, row in enumerate(messages):
        ms, seq, kind, side, _, size, bid, ask = row
        while q < len(times) and times[q] < ms:
            book.write_top(replay_px[q], replay_qty[q])
            q += 1

        if kind == "0":
            book.reduce(int(bid), int(size))
            book.reduce(int(ask), int(size))
            cumulative_volume += abs(int(size))
            if cumulative_volume in requested_volumes:
                waterline_books[cumulative_volume] = book.touch()
                requested_volumes.remove(cumulative_volume)
            continue
        if kind == "?":
            continue

        touch = book.touch()
        age = 0
        action = -1
        if kind == "C":
            order_id = int(bid or ask)
            order = book.orders.get(order_id)
            if order is not None and order[1] > 0 and seq not in auto_cancels:
                side, price, remaining, submit = order
                size = min(int(size), remaining)
                offset = max(0, (touch[side] - price) * (1 if side == 0 else -1))
                age = int(ms) - submit
                action = 4 + side
        else:
            price, action, offset = _arrival(row, messages[index + 1:], touch, auto_cancels)

        for session, (start, end) in enumerate(_SESSIONS):
            if start <= ms < end and min(touch) > 0 and action >= 0 and size > 0:
                dt = int(ms) - previous[session] if counts[session] else 0
                events[n] = (ms, seq, dt, action, offset, size, age)
                book.write_top(prices[n], depth[n])
                previous[session] = int(ms)
                counts[session] += 1
                n += 1
        if kind == "C":
            book.reduce(order_id, int(size))
        else:
            book.add(int(seq), int(side), price, int(size), int(ms))
    while q < len(times):
        book.write_top(replay_px[q], replay_qty[q])
        q += 1
    report = None
    if reconcile:
        report = _reconciliation(times, actual_px, actual_qty, replay_px, replay_qty, volumes, waterline_books)

    prices = prices[:n].copy()
    depth = depth[:n].copy()
    arrays = {name: events[name][:n].copy() for name in events.dtype.names}
    arrays.update(
        pre_book_ticks=prices, pre_book_shares=depth,
        mid_ticks_x2=prices[:, :, 0].sum(axis=1, dtype=np.int64),
        spread_ticks=prices[:, 1, 0] - prices[:, 0, 0],
    )
    for array in arrays.values():
        array.setflags(write=False)
    return SZSEDaySlice(wind_code, date, prev_close, ((0, counts[0]), (counts[0], n)), **arrays), report
