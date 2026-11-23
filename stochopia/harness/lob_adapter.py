"""Short, paired JAX-LOB replay and aligned LOBSTER-format export."""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module as _import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from stochopia.common import atomic_write_text
from stochopia.harness.contract import ReportStatus


@dataclass(frozen=True)
class OrderEvent:
    time_ns: int  # nanoseconds since local midnight
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


def _clock(time_ns: int) -> tuple[int, int, str]:
    seconds, fractional = divmod(time_ns, 10**9)
    return seconds, fractional, f"{seconds}.{fractional:09d}"


def _book_row(engine: ModuleType, asks: Any, bids: Any, depth: int) -> tuple[int, ...]:
    row = [int(value) for value in engine.get_L2_state(asks, bids, depth).tolist()]
    for index in range(0, len(row), 4):
        if row[index] == -1:
            row[index] = 9999999999
        if row[index + 2] == -1:
            row[index + 2] = -9999999999
    return tuple(row)


def _replay(engine: ModuleType, jnp: ModuleType, events: Sequence[OrderEvent], depth: int) -> LobTrace:
    capacity = max(len(events) + 1, depth + 1)
    asks = engine.init_orderside(capacity)
    bids = engine.init_orderside(capacity)
    emitted = []
    snapshots = []
    fills = []
    identifiers = set()

    def emit(event_time: str, event_type: int, order_id: int, size: int, price: int, direction: int) -> None:
        emitted.append((event_time, event_type, order_id, size, price, direction))
        snapshots.append(_book_row(engine, asks, bids, depth))

    previous_time = -1
    for event in events:
        if (not previous_time <= event.time_ns < 86400 * 10**9 or event.time_ns < 0
                or not 0 < event.order_id < 2**31 or not 0 < event.size < 2**31
                or event.direction not in (-1, 1)):
            raise ValueError("invalid order identity, size, side or chronological time")
        previous_time = event.time_ns
        sec, ns, clock = _clock(event.time_ns)
        if event.kind == "market":
            if event.price != 0 or event.order_id in identifiers:
                raise ValueError("market orders require an unused ID and zero input price")
            identifiers.add(event.order_id)
            opposite = asks if event.direction == 1 else bids
            remaining = event.size
            while remaining and bool((opposite[:, 1] > 0).any()):
                # Execute one resting fill through JAX-LOB and snapshot immediately.
                # This also handles sweeps without writing the terminal state on every row.
                top = engine.get_next_executable_order(0 if event.direction == 1 else 1, opposite)
                index = jnp.where(opposite[:, 2] == top[2], size=1)[0]
                trades = jnp.full((1, 6), -1, dtype=jnp.int32)
                opposite, unfilled, _, trades, *_ = engine.match_order(
                    (index, opposite, remaining, 0, trades, event.order_id, sec, ns)
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
            continue

        if not 0 < event.price < 999999999:
            raise ValueError("resting prices must use positive LOBSTER integer units")
        resting = bids if event.direction == 1 else asks
        payload = {
            "price": event.price, "quantity": event.size, "orderid": event.order_id,
            "traderid": event.order_id, "time": sec, "time_ns": ns,
        }
        if event.kind == "limit":
            if event.order_id in identifiers:
                raise ValueError("a new order ID must be unique within the replay")
            best_ask, _, best_bid, _ = _book_row(engine, asks, bids, 1)
            if ((event.direction == 1 and event.price >= best_ask)
                    or (event.direction == -1 and event.price <= best_bid)):
                raise ValueError("short replay expects passive limits and explicit market orders")
            identifiers.add(event.order_id)
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
        emit(clock, event_type, event.order_id, event.size, event.price, event.direction)

    return LobTrace(tuple(emitted), tuple(snapshots), tuple(fills))


def run_short_lob(*, depth: int = 2, import_module: Callable[[str], ModuleType] = _import_module) -> LobComparison:
    """Replay a deterministic injection and its no-injection control using JAX-LOB."""
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be a positive integer")
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
