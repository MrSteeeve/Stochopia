"""Short, paired JAX-LOB replay and aligned LOBSTER-format export."""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module as _import_module
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from stochopia.common import atomic_write_text
from stochopia.harness.contract import ReportStatus


# --- Input and trace records ---


_NS_PER_SECOND = 10**9
_INT32_MAX = 2**31 - 1
# JAX-LOB 0.19 masks non-priority time entries with this value in both columns.
_TIME_MASK = 999_999_999
_MAX_REPLAY_SECONDS = min(_INT32_MAX, _TIME_MASK - 1)


@dataclass(frozen=True)
class OrderEvent:
    time_ns: int  # nanoseconds on the input clock
    kind: Literal["limit", "cancel", "market"]
    order_id: int
    size: int
    price: int  # LOBSTER integer units, zero for a market instruction
    direction: Literal[-1, 1]


@dataclass(frozen=True)
class Fill:
    order_id: int
    size: int
    price: int
    direction: int  # resting order's side
    aggressor_id: int


@dataclass(frozen=True)
class LobTrace:
    message_rows: tuple[tuple[str, int, int, int, int, int], ...]
    orderbook_rows: tuple[tuple[int, ...], ...]
    fills: tuple[Fill, ...]


@dataclass(frozen=True)
class LobComparison:
    status: ReportStatus
    with_order: LobTrace | None = None
    without_order: LobTrace | None = None
    fills: tuple[Fill, ...] = ()
    matching_version: str | None = None
    reason: str | None = None


# --- Physical clock and price-time priority ---


def _clock(time_ns: int) -> str:
    seconds, fractional = divmod(time_ns, _NS_PER_SECOND)
    return f"{seconds}.{fractional:09d}"


def _relative_clocks(events: Sequence[OrderEvent]) -> list[tuple[int, int]]:
    """Check the entire tape before matching and split physical elapsed nanoseconds."""
    origin = events[0].time_ns if events else 0
    previous = 0
    clocks = []
    for event in events:
        stamp = event.time_ns
        if not isinstance(stamp, Integral) or isinstance(stamp, bool) or stamp < previous:
            raise ValueError("order time must be nonnegative integer nanoseconds in chronological order")
        previous = int(stamp)
        seconds, nanos = divmod(previous - int(origin), _NS_PER_SECOND)
        if seconds > _MAX_REPLAY_SECONDS:
            raise ValueError("relative replay timestamp exceeds JAX-LOB clock capacity")
        clocks.append((seconds, nanos))
    return clocks


def _book_row(engine: ModuleType, asks: Any, bids: Any, depth: int) -> tuple[int, ...]:
    row = [int(value) for value in engine.get_L2_state(asks, bids, depth).tolist()]
    for index in range(0, len(row), 4):
        if row[index] == -1:
            row[index] = 9999999999
        if row[index + 2] == -1:
            row[index + 2] = -9999999999
    return tuple(row)


def _best_resting_order(
    engine: ModuleType, book: Any, aggressor_side: int, arrivals: Mapping[int, int],
) -> tuple[int, int]:
    """Find price-time priority, with exchange order breaking exact clock ties."""
    if bool((book[:, 5] == _TIME_MASK).any()):
        # A real fractional nanosecond can equal the mask used by the engine's selector.
        def priority(row: list[int]) -> tuple[int, int, int, int]:
            return aggressor_side * row[0], row[4], row[5], arrivals[row[2]]

        chosen = min((row for row in book.tolist() if row[1] > 0), key=priority)
        return chosen[0], chosen[2]

    top = engine.get_next_executable_order(0 if aggressor_side == 1 else 1, book)
    same_priority = (book[:, 0] == top[0]) & (book[:, 4] == top[4]) & (book[:, 5] == top[5])
    order_ids = book[same_priority, 2].tolist()
    # The engine chooses the first array slot on ties. Cancelled slots may be reused.
    return int(top[0]), min(order_ids, key=arrivals.__getitem__)


# --- Matching, short fixture and export ---


def _replay(engine: ModuleType, jnp: ModuleType, events: Sequence[OrderEvent], depth: int) -> LobTrace:
    clocks = _relative_clocks(events)
    capacity = max(len(events) + 1, depth + 1)
    asks = engine.init_orderside(capacity)
    bids = engine.init_orderside(capacity)
    emitted = []
    snapshots = []
    fills = []
    arrivals: dict[int, int] = {}

    def emit(event_time: str, event_type: int, order_id: int, size: int, price: int, direction: int) -> None:
        emitted.append((event_time, event_type, order_id, size, price, direction))
        snapshots.append(_book_row(engine, asks, bids, depth))

    for position, (event, (sec, ns)) in enumerate(zip(events, clocks, strict=True)):
        if (not 0 < event.order_id <= _INT32_MAX or not 0 < event.size <= _INT32_MAX
                or event.direction not in (-1, 1)):
            raise ValueError("invalid order identity, size or side")
        clock = _clock(int(event.time_ns))
        remaining = event.size
        if event.kind == "market":
            if event.price != 0:
                raise ValueError("market orders require zero input price")
        elif not 0 < event.price < 999999999:
            raise ValueError("resting prices must use positive LOBSTER integer units")

        if event.kind in ("limit", "market"):
            if event.order_id in arrivals:
                raise ValueError("a new order ID must be unique within the replay")
            arrivals[event.order_id] = position
            opposite = asks if event.direction == 1 else bids
            while remaining and bool((opposite[:, 1] > 0).any()):
                # Execute one resting fill through JAX-LOB and snapshot immediately.
                # This also handles sweeps without writing the terminal state on every row.
                top_price, top_id = _best_resting_order(engine, opposite, event.direction, arrivals)
                if event.kind == "limit" and (event.price - top_price) * event.direction < 0:
                    break
                index = jnp.where(opposite[:, 2] == top_id, size=1)[0]
                trades = jnp.full((1, 6), -1, dtype=jnp.int32)
                opposite, unfilled, _, trades, *_ = engine.match_order(
                    (index, opposite, remaining, event.price, trades, event.order_id, sec, ns)
                )
                price, filled, order_id, aggressor, _, _ = map(int, trades[0].tolist())
                if not 0 < filled <= remaining or max(int(unfilled), 0) != remaining - filled:
                    raise RuntimeError("JAX-LOB returned an inconsistent fill quantity")
                remaining -= filled
                side = -event.direction
                if side == -1:
                    asks = opposite
                else:
                    bids = opposite
                fills.append(Fill(order_id, filled, price, side, aggressor))
                emit(clock, 4, order_id, filled, price, side)
            if event.kind == "market" or remaining == 0:
                continue

        resting = bids if event.direction == 1 else asks
        payload = {
            "price": event.price, "quantity": remaining, "orderid": event.order_id,
            "traderid": event.order_id, "time": sec, "time_ns": ns,
        }
        if event.kind == "limit":
            changed = engine.add_order(resting, payload)
            event_type = 1
        elif event.kind == "cancel":
            matching = resting[resting[:, 2] == event.order_id].tolist()
            if not matching or matching[0][0] != event.price or matching[0][1] < event.size:
                raise ValueError("cancellation must reference enough live quantity at the registered price")
            changed = engine.cancel_order(resting, payload)
            event_type = 3 if matching[0][1] == event.size else 2
        else:
            raise ValueError(f"unsupported short-replay event: {event.kind}")
        if event.direction == 1:
            bids = changed
        else:
            asks = changed
        emit(clock, event_type, event.order_id, remaining, event.price, event.direction)

    return LobTrace(tuple(emitted), tuple(snapshots), tuple(fills))


def run_short_lob(*, depth: int = 2, import_module: Callable[[str], ModuleType] = _import_module) -> LobComparison:
    """Replay a deterministic injection and its no-injection control using JAX-LOB."""
    if not isinstance(depth, Integral) or isinstance(depth, bool) or depth < 1:
        raise ValueError("depth must be a positive integer")
    depth = int(depth)
    try:
        jnp = import_module("jax.numpy")
        engine = import_module("jaxlob.gymnax_exchange.jaxob.JaxOrderBookArrays")
        engine_version = f"jaxlob=={version('jaxlob')}"
    except (ImportError, PackageNotFoundError) as exc:
        return LobComparison(ReportStatus.UNAVAILABLE, reason=f"JAX-LOB dependency unavailable: {exc}")

    start = 34200 * 10**9
    initial = (
        OrderEvent(start, "limit", 201, 80, 100000, 1),
        OrderEvent(start + 10**6, "limit", 102, 70, 100200, -1),
        OrderEvent(start + 2 * 10**6, "limit", 202, 50, 99900, 1),
        OrderEvent(start + 3 * 10**6, "limit", 101, 100, 100100, -1),
    )
    child = OrderEvent(start + 4 * 10**6, "market", 301, 40, 0, 1)
    cancel = OrderEvent(start + 5 * 10**6, "cancel", 101, 15, 100100, -1)
    with_order = _replay(engine, jnp, (*initial, child, cancel), depth)
    without_order = _replay(engine, jnp, (*initial, cancel), depth)
    return LobComparison(
        ReportStatus.SUPPORTED, with_order, without_order, with_order.fills, engine_version,
    )


def export_lob_bench(trace: LobTrace, message_path: str | Path, orderbook_path: str | Path) -> None:
    """Write headerless LOBSTER rows in original event order with post-event states."""
    if Path(message_path).resolve() == Path(orderbook_path).resolve():
        raise ValueError("message and orderbook outputs must be distinct files")
    if len(trace.message_rows) != len(trace.orderbook_rows):
        raise ValueError("message rows and post-event book rows must align")
    for rows, path in ((trace.message_rows, message_path), (trace.orderbook_rows, orderbook_path)):
        with io.StringIO(newline="") as buffer:
            csv.writer(buffer, lineterminator="\n").writerows(rows)
            atomic_write_text(path, buffer.getvalue())
