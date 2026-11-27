"""Short exchange replay and LOBSTER export regression tests."""

import csv
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

import pytest

from stochopia.harness.lob_adapter import LobReplayStatus, OrderEvent, _replay, export_lob_bench, run_short_lob


def test_lob_matches_and_exports_the_post_message_book(tmp_path: Path) -> None:
    pytest.importorskip("jaxlob", reason="actual matching requires the harness-lob extra on Python 3.10–3.12")
    result = run_short_lob(depth=2)
    assert result.status is LobReplayStatus.SUPPORTED, result.reason
    assert [(fill.order_id, fill.size, fill.price, fill.direction) for fill in result.fills] == [
        (101, 40, 100100, -1),
    ]
    assert result.matching_version == "jaxlob==0.19"
    assert result.with_order is not None and result.without_order is not None
    assert not result.without_order.fills
    messages, book = tmp_path / "messages.csv", tmp_path / "orderbook.csv"
    export_lob_bench(result.with_order, messages, book)
    message_rows = list(csv.reader(messages.read_text().splitlines()))
    book_rows = list(csv.reader(book.read_text().splitlines()))
    assert len(message_rows) == len(book_rows) == len(result.with_order.message_rows)
    execution = next(index for index, row in enumerate(message_rows) if row[1] == "4")
    assert message_rows[execution][2:] == ["101", "40", "100100", "-1"]
    assert book_rows[execution - 1][:2] == ["100100", "100"]
    assert book_rows[execution][:4] == ["100100", "60", "100000", "80"]
    assert execution == 4
    assert book_rows[-1][:4] == ["100100", "45", "100000", "80"]
    assert tuple(result.without_order.orderbook_rows[-1][:4]) == (100100, 85, 100000, 80)


def test_lob_missing_import_returns_unavailable_and_no_trace() -> None:
    def no_module(name: str) -> NoReturn:
        raise ModuleNotFoundError(name, name=name)

    result = run_short_lob(import_module=no_module)
    assert result.status is LobReplayStatus.UNAVAILABLE
    assert result.reason and "jax" in result.reason.lower()
    assert result.with_order is result.without_order is result.matching_version is None
    assert not result.fills


# --- LOB matching and replay clock ---


@pytest.mark.parametrize("side", [1, -1])
def test_crossing_limit_fills_then_rests_the_remainder(side: int) -> None:
    pytest.importorskip("jaxlob", reason="actual matching requires the harness-lob extra on Python 3.10–3.12")
    jnp = pytest.importorskip("jax.numpy")
    engine = pytest.importorskip("jaxlob.gymnax_exchange.jaxob.JaxOrderBookArrays")
    epoch = 1_700_000_000 * 10**9
    events = (
        OrderEvent(epoch, "limit", 11, 8, 100 - 3 * side, side),
        OrderEvent(epoch, "limit", 21, 3, 100, -side),
        OrderEvent(epoch, "limit", 22, 2, 100 + side, -side),
        OrderEvent(epoch, "limit", 23, 4, 100 + 2 * side, -side),
        OrderEvent(epoch + 1, "limit", 31, 9, 100 + side, side),
    )
    trace = _replay(engine, jnp, events, depth=2)
    clock = "1700000000.000000001"
    assert trace.message_rows[-3:] == (
        (clock, 4, 21, 3, 100, -side),
        (clock, 4, 22, 2, 100 + side, -side),
        (clock, 1, 31, 4, 100 + side, side),
    )
    assert [(fill.order_id, fill.aggressor_id) for fill in trace.fills] == [(21, 31), (22, 31)]
    rest_before = 100 - 3 * side
    if side == 1:
        assert [row[:4] for row in trace.orderbook_rows[-3:]] == [
            (101, 2, rest_before, 8), (102, 4, rest_before, 8), (102, 4, 101, 4),
        ]
    else:
        assert [row[:4] for row in trace.orderbook_rows[-3:]] == [
            (rest_before, 8, 99, 2), (rest_before, 8, 98, 4), (99, 4, 98, 4),
        ]


def test_replay_preserves_physical_time_in_resting_and_trade_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    jnp = pytest.importorskip("jax.numpy")
    engine = pytest.importorskip("jaxlob.gymnax_exchange.jaxob.JaxOrderBookArrays")
    add_order, match_order = engine.add_order, engine.match_order
    resting_times = []
    trades = []

    def record_add(book: Any, order: dict[str, int]) -> Any:
        updated = add_order(book, order)
        row = updated[updated[:, 2] == order["orderid"]][0].tolist()
        resting_times.append(tuple(row[4:6]))
        return updated

    def record_trade(args: Any) -> Any:
        updated = match_order(args)
        trade, = updated[3].tolist()
        trades.append(tuple(trade))
        return updated

    monkeypatch.setattr(engine, "add_order", record_add)
    monkeypatch.setattr(engine, "match_order", record_trade)
    first_time = 1_700_000_000_900_000_000
    trace = _replay(engine, jnp, (
        OrderEvent(first_time, "limit", 10, 20, 101, -1),
        OrderEvent(first_time + 1_205_000_002, "limit", 20, 30, 101, -1),
        OrderEvent(first_time + 3_001_000_000, "market", 30, 25, 0, 1),
    ), depth=1)

    assert resting_times == [(0, 0), (1, 205_000_002)]
    assert trades == [(101, 20, 10, 30, 3, 1_000_000), (101, 5, 20, 30, 3, 1_000_000)]
    seconds, nanos = trades[-1][-2:]
    rest_seconds, rest_nanos = resting_times[-1]
    assert (seconds - rest_seconds) * 10**9 + nanos - rest_nanos == 1_795_999_998
    assert [row[0] for row in trace.message_rows] == [
        "1700000000.900000000", "1700000002.105000002", "1700000003.901000000", "1700000003.901000000",
    ]


@pytest.mark.parametrize("side", [1, -1])
def test_equal_time_fills_use_fifo_after_a_book_slot_is_reused(side: int) -> None:
    jnp = pytest.importorskip("jax.numpy")
    engine = pytest.importorskip("jaxlob.gymnax_exchange.jaxob.JaxOrderBookArrays")
    first_time = 30 * 10**9
    worse = 101 + side
    trace = _replay(engine, jnp, (
        OrderEvent(first_time, "limit", 10, 1, worse, -side),
        OrderEvent(first_time + 9, "limit", 21, 1, 101, -side),
        OrderEvent(first_time + 9, "cancel", 10, 1, worse, -side),
        OrderEvent(first_time + 9, "limit", 13, 1, 101, -side),
        OrderEvent(first_time + 11, "market", 40, 2, 0, side),
    ), depth=1)
    assert [fill.order_id for fill in trace.fills] == [21, 13]


def test_last_clock_tick_can_match_the_best_price_at_the_nanosecond_sentinel() -> None:
    jnp = pytest.importorskip("jax.numpy")
    engine = pytest.importorskip("jaxlob.gymnax_exchange.jaxob.JaxOrderBookArrays")
    first_time = 30 * 10**9
    last_tick = first_time + 999_999_999 * 10**9 - 1
    trace = _replay(engine, jnp, (
        OrderEvent(first_time, "limit", 10, 1, 102, -1),
        OrderEvent(last_tick, "limit", 21, 1, 101, -1),
        OrderEvent(last_tick, "market", 30, 1, 0, 1),
    ), depth=1)
    assert [(fill.order_id, fill.price) for fill in trace.fills] == [(21, 101)]
    assert trace.message_rows[-1][0] == "1000000028.999999999"


def test_replay_clock_overflow_fails_before_calling_the_matching_engine() -> None:
    engine = ModuleType("unused_engine")
    jnp = ModuleType("unused_jax_numpy")
    first_time = 30 * 10**9
    with pytest.raises(ValueError, match="relative replay timestamp exceeds JAX-LOB clock capacity"):
        _replay(engine, jnp, (
            OrderEvent(first_time, "limit", 10, 1, 102, -1),
            OrderEvent(first_time + 999_999_999 * 10**9, "market", 30, 1, 0, 1),
        ), depth=1)
