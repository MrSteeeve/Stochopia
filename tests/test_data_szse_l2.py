"""Wind three-table fixtures with hand-calculated book and event expectations."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from stochopia.data import (
    ACTION_TYPES,
    parse_wind_time_ms,
    reconstruct_szse_day,
    validate_chronological_splits,
)


DATE = 20170103
CODE = "000001.SZ"
SEQ_BASE = (1 << 63) - 12
QUOTE_TABLE = "行情.parquet"
ORDER_TABLE = "逐笔委托.parquet"
TRADE_TABLE = "逐笔成交.parquet"
Row = dict[str, str | int | float]


def _seq(number: int) -> int:
    return SEQ_BASE + number if number else 0


def _order(seq: int, at: int, side: str, ticks: int, shares: int, kind: str = "0") -> Row:
    return {
        "wind_code": CODE, "date": DATE, "time": at, "order_id": seq + 10_000,
        "ex_order_id": _seq(seq), "order_type": kind, "order_code": side,
        "price": float(ticks * 100), "volume": shares,
    }


def _trade(
    seq: int, at: int, bid: int, ask: int, ticks: int, shares: int, kind: str = "0",
) -> Row:
    return {
        "wind_code": CODE, "date": DATE, "time": at, "trade_id": _seq(seq),
        "trade_code": kind, "bs_flag": "B", "bid_order_id": _seq(bid),
        "ask_order_id": _seq(ask), "price": float(ticks * 100), "volume": shares,
    }


def _quote(at: int, bids: Sequence[tuple[int, int]], asks: Sequence[tuple[int, int]]) -> Row:
    row: Row = {
        "wind_code": CODE, "date": DATE, "time": at, "prev_close": 99_500., "nun_trades": 5,
    }
    for side, levels in (("bid", bids), ("ask", asks)):
        for level, (ticks, shares) in enumerate([*levels, *[(0, 0)] * (10 - len(levels))], 1):
            row[f"{side}_px{level}"] = float(ticks * 100)
            row[f"{side}_vol{level}"] = shares
    return row


def _write(path: Path, rows: list[Row]) -> None:
    types = {"date": pa.uint32(), "time": pa.uint32()}
    for key in ("order_id", "ex_order_id", "trade_id", "bid_order_id", "ask_order_id"):
        types[key] = pa.uint64()
    columns = {}
    for key in rows[0]:
        columns[key] = pa.array([row[key] for row in rows], type=types.get(key))
    pq.write_table(pa.table(columns), path, row_group_size=4)


@pytest.fixture
def one_day(tmp_path: Path) -> Path:
    orders = [
        _order(1, 91_500_000, "B", 998, 600),
        _order(2, 91_500_000, "B", 997, 600),
        _order(3, 91_500_000, "S", 1000, 300),
        _order(4, 91_500_000, "S", 1001, 400),
        _order(5, 91_500_000, "S", 1002, 900),
        _order(6, 91_500_000, "B", 1000, 100),
        _order(9, 93_000_010, "B", 999, 300),
        _order(10, 93_000_020, "B", 998, 200, "2"),
        _order(11, 93_000_020, "B", 1001, 750),
        _order(15, 93_000_020, "S", 1004, 100),
        _order(17, 93_100_010, "S", 1003, 125),
        _order(18, 113_000_000, "B", 1001, 130),
        _order(19, 130_000_000, "S", 1003, 175),
        _order(20, 145_659_999, "S", 1002, 10),
        _order(21, 145_700_000, "S", 1001, 100),
    ]
    trades = [
        _trade(7, 92_000_000, 2, 0, 0, 100, "C"),
        _trade(8, 92_500_000, 6, 3, 1000, 100),
        _trade(12, 93_000_020, 11, 3, 1000, 200),
        _trade(13, 93_000_020, 11, 4, 1001, 400),
        _trade(14, 93_000_020, 9, 0, 0, 100, "C"),
        _trade(16, 93_059_990, 0, 5, 0, 200, "C"),
        _trade(22, 150_000_000, 11, 21, 1001, 100),
    ]
    wrong = _quote(93_000_020, [(900, 1)], [(1100, 1)])
    quotes = [
        wrong,
        _quote(130_000_000, [(1001, 280), (999, 200), (998, 800), (997, 500)],
               [(1002, 700), (1003, 300), (1004, 100)]),
        _quote(93_000_010, [(999, 300), (998, 600), (997, 500)],
               [(1000, 200), (1001, 400), (1002, 900)]),
        _quote(93_000_020, [(1001, 150), (999, 200), (998, 800), (997, 500)], [(1002, 900), (1004, 100)]),
        _quote(92_500_000, [], []),
        _quote(113_000_000, [], []),
        _quote(145_700_000, [], []),
        {**wrong, "wind_code": "300001.SZ"},
    ]
    # Unsorted individual tables and a second stock make projection and merge observable.
    _write(tmp_path / ORDER_TABLE, [*reversed(orders), {**orders[0], "wind_code": "300001.SZ", "volume": 1}])
    _write(tmp_path / TRADE_TABLE, [*reversed(trades), {**trades[0], "wind_code": "300001.SZ", "volume": 1}])
    _write(tmp_path / QUOTE_TABLE, quotes)
    return tmp_path


def test_wind_time_parser_handles_calendar_boundaries() -> None:
    assert parse_wind_time_ms(93_000_190) == 34_200_190
    clock = np.array([93_059_990, 93_100_010, 95_959_999, 100_000_001], dtype=np.uint32)
    parsed = parse_wind_time_ms(clock)
    np.testing.assert_array_equal(parsed, [34_259_990, 34_260_010, 35_999_999, 36_000_001])
    assert parsed[1] - parsed[0] == 20
    assert parsed[3] - parsed[2] == 2


@pytest.mark.parametrize("bad_clock", [-1, 126_000_000, 123_060_000, 240_000_000])
def test_wind_time_parser_rejects_invalid_clock(bad_clock: int) -> None:
    with pytest.raises(ValueError):
        parse_wind_time_ms(bad_clock)
    with pytest.raises(ValueError):
        parse_wind_time_ms(np.array([93_000_190, bad_clock], dtype=np.int64))


def test_replays_an_entire_day_without_counting_fills_as_events(one_day: Path) -> None:
    day, checks = reconstruct_szse_day(one_day, CODE)
    assert ACTION_TYPES == (
        "passive_buy", "passive_sell", "aggressive_buy", "aggressive_sell", "cancel_buy", "cancel_sell",
    )
    assert (day.wind_code, day.trading_date, day.prev_close_ticks) == (CODE, DATE, 995)
    assert day.session_bounds == ((0, 7), (7, 9))
    expected = {
        "seq_id": (np.uint64, [_seq(i) for i in (9, 10, 11, 14, 15, 16, 17, 19, 20)]),
        "time_ms": (np.int64, [34_200_010, *[34_200_020] * 4, 34_259_990, 34_260_010, 46_800_000, 53_819_999]),
        "dt_ms": (np.int64, [0, 10, 0, 0, 0, 59_970, 20, 0, 7_019_999]),
        "action_type": (np.int8, [0, 0, 2, 4, 1, 5, 1, 1, 1]),
        "rel_ticks": (np.int32, [-1, 1, 1, 2, 2, 0, 1, 1, 0]),
        "volume_shares": (np.int64, [300, 200, 750, 100, 100, 200, 125, 175, 10]),
        "cancel_age_ms": (np.int64, [0, 0, 0, 10, 0, 959_990, 0, 0, 0]),
        "mid_ticks_x2": (np.int64, [1998, 1999, 1999, 2003, 2003, 2003, 2003, 2003, 2003]),
        "spread_ticks": (np.int32, [2, 1, 1, 1, 1, 1, 1, 1, 1]),
    }
    for name, (dtype, values) in expected.items():
        observed = getattr(day, name)
        assert observed.dtype == dtype
        assert not observed.flags.writeable
        np.testing.assert_array_equal(observed, np.asarray(values, dtype=dtype))

    assert day.pre_book_ticks.shape == day.pre_book_shares.shape == (9, 2, 10)
    assert day.pre_book_ticks.dtype == np.int32
    assert day.pre_book_shares.dtype == np.int64
    np.testing.assert_array_equal(day.pre_book_ticks[0, 0], [998, 997, *[0] * 8])
    np.testing.assert_array_equal(day.pre_book_shares[0, 0, :2], [600, 500])
    np.testing.assert_array_equal(day.pre_book_shares[0, 1, :3], [200, 400, 900])
    np.testing.assert_array_equal(day.pre_book_shares[2, 0, :3], [300, 800, 500])
    np.testing.assert_array_equal(day.pre_book_ticks[3, 1, :2], [1002, 0])
    assert day.pre_book_shares[3, 0, 0] == 150
    assert day.pre_book_shares[4, 0, 1] == 200
    assert day.pre_book_shares[7, 0, 0] == 280
    assert day.pre_book_shares[7, 1, 1] == 125
    assert day.volume_lots[6] == 1.25
    assert checks is not None
    assert checks.n_snapshots == 3
    assert checks.touch_match_rate == 1.0
    assert checks.top5_log_vol_mae == checks.top10_log_vol_mae == 0.0


def test_special_orders_keep_automatic_and_external_cancels_distinct(tmp_path: Path) -> None:
    orders = [
        _order(1, 91_500_000, "B", 998, 500),
        _order(2, 91_500_000, "S", 1000, 200),
        _order(3, 91_500_000, "S", 1002, 250),
        _order(4, 91_500_000, "S", 1004, 250),
        _order(5, 91_500_000, "S", 1006, 400),
        _order(6, 93_000_010, "B", 0, 300, "1"),
        _order(8, 93_000_020, "S", 0, 50, "U"),
        _order(9, 93_000_030, "B", 0, 600, "3"),
        _order(16, 93_000_050, "B", 999, 15),
        _order(17, 93_000_060, "S", 1003, 250),
        _order(18, 93_000_070, "B", 999, 15),
        _order(19, 93_000_080, "S", 999, 200),
    ]
    trades = [
        _trade(7, 93_000_010, 6, 2, 1000, 200),
        _trade(10, 93_000_030, 9, 3, 1002, 250),
        _trade(11, 93_000_030, 9, 8, 1002, 50),
        _trade(12, 93_000_030, 9, 4, 1004, 250),
        _trade(13, 93_000_030, 9, 0, 0, 50, "C"),
        _trade(14, 93_000_030, 6, 0, 0, 25, "C"),
        _trade(15, 93_000_040, 0, 5, 0, 400, "C"),
        _trade(20, 93_000_080, 6, 19, 1000, 75),
        _trade(21, 93_000_080, 16, 19, 999, 15),
        _trade(22, 93_000_080, 18, 19, 999, 15),
    ]
    _write(tmp_path / ORDER_TABLE, orders)
    _write(tmp_path / TRADE_TABLE, trades)
    _write(tmp_path / QUOTE_TABLE, [_quote(93_000_080, [(998, 500)], [(999, 95), (1003, 250)])])

    day, checks = reconstruct_szse_day(tmp_path, CODE)
    seq_ids = np.asarray([_seq(i) for i in (6, 8, 9, 14, 15, 18, 19)], dtype=np.uint64)
    np.testing.assert_array_equal(day.seq_id, seq_ids)
    assert day.session_bounds == ((0, 7), (7, 7))
    np.testing.assert_array_equal(day.action_type, [2, 1, 2, 4, 5, 0, 3])
    np.testing.assert_array_equal(day.rel_ticks, [0, 0, 2, 0, 0, 1, 1])
    np.testing.assert_array_equal(day.dt_ms, [0, 10, 10, 0, 10, 30, 10])
    np.testing.assert_array_equal(day.cancel_age_ms, [0, 0, 0, 20, 900_040, 0, 0])
    assert day.volume_shares[2] == 600
    assert day.pre_book_shares[1, 0, 0] == 100
    assert day.pre_book_shares[2, 1, 0] == 300
    assert day.pre_book_ticks[3, 0, 0] == 1000
    assert day.pre_book_shares[3, 0, 0] == 100
    assert day.pre_book_shares[5, 0, 1] == 15
    assert checks is not None
    assert checks.touch_match_rate == 1.0
    assert checks.top10_log_vol_mae == 0.0


def test_reconciliation_reports_volume_error_by_book_depth(one_day: Path) -> None:
    snapshots = pq.read_table(one_day / QUOTE_TABLE).to_pylist()
    correct_later_quote = snapshots[3]
    correct_later_quote["ask_px1"] = 100_300.
    correct_later_quote["bid_px6"] = 99_000.
    correct_later_quote["bid_vol6"] = 3
    _write(one_day / QUOTE_TABLE, snapshots)

    _, checks = reconstruct_szse_day(one_day, CODE)
    assert checks is not None
    assert checks.n_snapshots == 3
    assert checks.touch_match_rate == pytest.approx(2 / 3)
    assert checks.top5_log_vol_mae == 0.0
    assert checks.top10_log_vol_mae == pytest.approx(np.log1p(3) / (3 * 2 * 10))
    assert checks.max_deviation_time_ms == 34_200_020


def test_book_keeps_deep_levels_and_promotes_the_eleventh_level(tmp_path: Path) -> None:
    bids = [(1000 - i, 100 + i) for i in range(11)]
    asks = [(1010 + i, 200 + i) for i in range(11)]
    orders = [_order(i, 91_500_000, "B", ticks, shares) for i, (ticks, shares) in enumerate(bids, 1)]
    orders += [_order(i, 91_500_000, "S", ticks, shares) for i, (ticks, shares) in enumerate(asks, 12)]
    _write(tmp_path / ORDER_TABLE, orders)
    _write(tmp_path / TRADE_TABLE, [_trade(23, 93_000_020, 1, 0, 0, 100, "C")])
    _write(tmp_path / QUOTE_TABLE, [
        _quote(93_000_010, bids[:10], asks[:10]),
        _quote(93_000_020, bids[1:], asks[:10]),
    ])

    day, checks = reconstruct_szse_day(tmp_path, CODE)
    np.testing.assert_array_equal(day.pre_book_ticks[0], [np.arange(1000, 990, -1), np.arange(1010, 1020)])
    np.testing.assert_array_equal(day.pre_book_shares[0], [np.arange(100, 110), np.arange(200, 210)])
    assert checks is not None
    assert checks.n_snapshots == 2
    assert checks.touch_match_rate == 1.0
    assert checks.top10_log_vol_mae == 0.0


def test_loader_pushes_stock_predicates_into_all_three_reads(one_day: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_read = pq.read_table
    reads: set[str] = set()

    def record_read(source: Path, *, columns: list[str], filters: list[tuple[str, str, str]]) -> pa.Table:
        assert columns
        assert filters == [("wind_code", "==", CODE)]
        reads.add(Path(source).name)
        return real_read(source, columns=columns, filters=filters)

    monkeypatch.setattr(pq, "read_table", record_read)
    day, _ = reconstruct_szse_day(one_day, CODE)
    assert reads == {QUOTE_TABLE, ORDER_TABLE, TRADE_TABLE}
    _, checks = reconstruct_szse_day(one_day, CODE, reconcile=False)
    assert checks is None
    assert len(day.seq_id) == 9


def test_loader_rejects_shanghai_before_reading_a_table(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.SZ|深市|深圳|SZSE"):
        reconstruct_szse_day(tmp_path, "600000.SH")


def test_loader_reports_a_missing_required_parquet_column(one_day: Path) -> None:
    path = one_day / ORDER_TABLE
    table = pq.read_table(path).drop(["order_code"])
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="order_code"):
        reconstruct_szse_day(one_day, CODE)


def test_valid_chronological_splits() -> None:
    validate_chronological_splits([20170103, 20170104], [20170109, 20170110], [20170112])


@pytest.mark.parametrize(
    ("train", "dev", "test"),
    [
        ([], [20170105], [20170106]),
        ([20170103], [], [20170106]),
        ([20170103], [20170105], []),
        ([20170104, 20170103], [20170105], [20170106]),
        ([20170103], [20170104, 20170104], [20170106]),
        ([20170103], [20170104], [20170106, 20170105]),
        ([20170103, 20170105], [20170105], [20170106]),
        ([20170103], [20170105, 20170109], [20170106]),
    ],
)
def test_rejects_non_disjoint_chronological_splits(train: list[int], dev: list[int], test: list[int]) -> None:
    with pytest.raises(ValueError):
        validate_chronological_splits(train, dev, test)
