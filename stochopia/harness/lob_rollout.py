"""Paired physical-clock rollouts through the JAX-LOB matching engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from heapq import merge
from typing import Any

import numpy as np
import torch

from stochopia.data import SZSEDaySlice
from stochopia.neural.lob_interface import LOBWorldModel, State, StepObservation, decode_events, encode_events


class RolloutArm(str, Enum):
    NO_INJECTION = "no_injection"
    BUY_VISIBLE = "buy_visible"
    SELL_VISIBLE = "sell_visible"
    BUY_BLIND = "buy_blind"
    SELL_BLIND = "sell_blind"


@dataclass(frozen=True)
class RolloutTrace:
    """Initial book at grid zero, then inclusive grid states and exact child states.

    Events are [N, 4]: elapsed ms, action, relative ticks or cancel rank, and
    requested shares. Child submissions are separate. Mid and spread are NaN
    when either side is empty. Touch shares [G, 2] are in bid, ask order.
    """

    grid_ms: np.ndarray
    mid_ticks: np.ndarray
    spread_ticks: np.ndarray
    touch_shares: np.ndarray
    events: np.ndarray
    after_child_mid_ticks: np.ndarray
    child_fill_shares: np.ndarray


@dataclass(frozen=True)
class ResidualProfile:
    """Likelihood, tape summaries and paired impact observations for one seed fit."""

    model_nll: Mapping[str, float]
    empirical_nll: Mapping[str, float]
    generated: Mapping[str, Any]
    historical: Mapping[str, Any]
    impact: Mapping[str, Any]


@dataclass(frozen=True)
class _Order:
    time_ms: int
    action: int
    relative: int
    shares: int
    price: int
    is_own: bool = False


# --- Single-order matching with a fixed-size book ---


@lru_cache(maxsize=1)
def _matching_functions() -> tuple[Any, Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
        from jaxlob.gymnax_exchange.jaxob import JaxOrderBookArrays as engine
    except ImportError as exc:
        raise ImportError("Closed-loop rollout requires the harness-lob extra") from exc

    @jax.jit
    def apply(own: Any, other: Any, args: Any) -> tuple[Any, Any, Any, Any]:
        direction, price, shares, order_id, seconds, nanos, cancel, ioc = args
        initial = jax.lax.select(cancel != 0, own, other)
        empty_trades = jnp.full((1, 6), -1, dtype=jnp.int32)

        def priority(rows: Any) -> tuple[Any, Any]:
            active = rows[:, 1] > 0
            ranked_price = rows[:, 0] * direction
            best = jnp.min(jnp.where(active, ranked_price, engine.MAX_INT))
            at_price = jnp.where(cancel != 0, rows[:, 0] == price, ranked_price == best)
            ids = jnp.where(active & at_price, rows[:, 2], engine.MAX_INT)
            index = jnp.argmin(ids).astype(jnp.int32)
            executable = (ids[index] != engine.MAX_INT) & ((cancel != 0) | (best <= price * direction))
            return index, executable

        def remaining(state: tuple[Any, Any, Any]) -> Any:
            rows, size, _ = state
            return (size > 0) & priority(rows)[1]

        def consume(state: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
            rows, size, _ = state
            index, _ = priority(rows)
            quantity = jnp.minimum(size, rows[index, 1])
            last_price = rows[index, 0]

            def cancel_at_price(_: None) -> Any:
                return engine.cancel_order(rows, {
                    "price": price, "orderid": rows[index, 2], "quantity": quantity,
                })

            def match_next(_: None) -> Any:
                return engine.match_order(
                    (index[None], rows, size, price, empty_trades, order_id, seconds, nanos),
                )[0]

            updated = jax.lax.cond(cancel != 0, cancel_at_price, match_next, None)
            return updated, size - quantity, last_price

        updated, unfilled, last_price = jax.lax.while_loop(remaining, consume, (initial, shares, jnp.int32(0)))
        resting = {
            "price": price, "quantity": unfilled, "orderid": order_id,
            "traderid": order_id, "time": seconds, "time_ns": nanos,
        }
        rests = (cancel == 0) & (ioc == 0) & (unfilled > 0)
        own = jax.lax.cond(rests, lambda rows: engine.add_order(rows, resting), lambda rows: rows, own)
        return (jax.lax.select(cancel != 0, updated, own),
                jax.lax.select(cancel != 0, other, updated), shares - unfilled, last_price)

    return jnp.asarray, apply, engine.get_L2_state


class _Path:
    def __init__(self, ticks: np.ndarray, shares: np.ndarray, capacity: int) -> None:
        as_array, self._match, self._top = _matching_functions()
        self._sides = []
        self._id = 0
        for side in range(2):
            rows = np.full((capacity, 6), -1, dtype=np.int32)
            for level in range(10):
                if ticks[side, level] > 0 and shares[side, level] > 0:
                    self._id += 1
                    rows[level] = (ticks[side, level], shares[side, level], self._id, self._id, 0, 0)
            self._sides.append(as_array(rows))
        self.tape: list[_Order] = []
        self._times: list[int] = []
        self._touch: list[tuple[float, float, int, int]] = []
        self._child_mids: list[float] = []
        self._child_fills: list[int] = []
        self._record(0)

    def _record(self, time_ms: int) -> None:
        rows = np.asarray(self._top(self._sides[1], self._sides[0], 11)).reshape(11, 2, 2)
        book = np.maximum(rows[:, ::-1].transpose(1, 0, 2), 0)
        self.prices = np.where(book[:, :, 1] > 0, book[:, :, 0], 0)
        self.shares = book[:, :, 1]
        bid, ask = map(int, self.prices[:, 0])
        mid = (bid + ask) / 2 if bid and ask else np.nan
        spread = ask - bid if bid and ask else np.nan
        best_bid, best_ask = map(int, self.shares[:, 0])
        self._times.append(time_ms)
        self._touch.append((mid, spread, best_bid, best_ask))

    def instruction(self, time_ms: int, tokens: list[int]) -> _Order:
        _, action, relative, shares = tokens
        side = action % 2
        direction = 1 - 2 * side
        if action >= 4:
            price = int(self.prices[side, relative])
        elif action >= 2:
            price = max(1, int(self.prices[1 - side, 0]) + direction * relative)
        else:
            best = int(self.prices[side, 0])
            opposite = int(self.prices[1 - side, 0])
            anchor = best or (max(1, opposite - direction) if opposite else 1)
            price = anchor - direction * relative
        return _Order(time_ms, action, relative, shares, price)

    def apply(self, order: _Order) -> int:
        """Apply one request and return the deepest execution price, or zero."""
        side = order.action % 2
        self._id += 1
        seconds, millis = divmod(order.time_ms, 1_000)
        args = np.array([1 - 2 * side, order.price, order.shares, self._id,
                         seconds, millis * 1_000_000, order.action >= 4, order.is_own], dtype=np.int32)
        own, other, filled, last = self._match(self._sides[side], self._sides[1 - side], args)
        self._sides[side] = own
        self._sides[1 - side] = other
        self._record(order.time_ms)
        if order.is_own:
            self._child_mids.append(self._touch[-1][0])
            self._child_fills.append(int(filled))
        else:
            self.tape.append(order)
        return int(last)

    def trace(self, grid: np.ndarray) -> RolloutTrace:
        last = np.searchsorted(self._times, grid, side="right") - 1
        if grid[0] == 0:
            last[0] = 0  # Preserve the pre-event book even when the tape starts with a burst.
        touch = np.array(self._touch)[last]
        events = [(order.time_ms, order.action, order.relative, order.shares) for order in self.tape]
        return RolloutTrace(
            grid, touch[:, 0], touch[:, 1], touch[:, 2:].astype(np.int64),
            np.array(events, dtype=np.int64).reshape(-1, 4),
            np.array(self._child_mids), np.array(self._child_fills, dtype=np.int64),
        )


# --- Paired tapes and recurrent warmup ---


def _children(times: Sequence[int], shares: int, side: int) -> list[_Order]:
    market_price = 999_999_999 if side == 0 else 0
    return [_Order(t, 2 + side, 0, shares, market_price, True) for t in times]


def _warm(
    model: LOBWorldModel, day: SZSEDaySlice, start: int, count: int, device: torch.device,
) -> tuple[State, torch.Tensor]:
    history = slice(start - count, start)
    ticks = torch.tensor(day.pre_book_ticks[history], device=device)
    shares = torch.tensor(day.pre_book_shares[history], device=device)
    targets = encode_events(
        torch.tensor(day.dt_ms[history], device=device), torch.tensor(day.action_type[history], device=device),
        torch.tensor(day.rel_ticks[history], device=device), torch.tensor(day.volume_shares[history], device=device), ticks,
    )
    previous = torch.zeros((1, 4), dtype=torch.long, device=device)
    is_own = torch.zeros((1,), dtype=torch.bool, device=device)
    hidden = None
    for i in range(count):
        _, hidden = model.forward_step(StepObservation(ticks[i:i + 1], shares[i:i + 1], previous, is_own), hidden)
        previous = targets[i:i + 1]
    return hidden, previous


def _simulate(
    model: LOBWorldModel, path: _Path, hidden: State, previous: torch.Tensor, orders: Sequence[_Order],
    horizon: int, seed: int, max_events: int,
) -> None:
    hidden = deepcopy(hidden)
    generator = torch.Generator(device=previous.device).manual_seed(seed)
    child = 0
    now = 0
    own = False
    for _ in range(max_events + len(orders) + 1):
        ticks = torch.tensor(path.prices[None, :, :10], device=previous.device)
        shares = torch.tensor(path.shares[None, :, :10], device=previous.device)
        is_own = torch.tensor([own], device=previous.device)
        deep = torch.tensor(path.prices[None, :, 10] > 0, device=previous.device)
        sampler, hidden = model.forward_step(StepObservation(ticks, shares, previous, is_own, deep), hidden)
        old_rng = generator.get_state() if child < len(orders) else None
        tokens = sampler.sample(generator)
        decoded = decode_events(tokens)[0].tolist()
        proposal_time = now + decoded[0]

        # Child submissions have priority at equal times. Reuse the unconsumed
        # background draw under the child's new visible book and own-event token.
        if child < len(orders) and orders[child].time_ms <= proposal_time:
            order = orders[child]
            touch = path.prices[1 - order.action % 2, 0]
            last = path.apply(order)
            direction = 1 - 2 * (order.action % 2)
            depth = max(0, (last - int(touch)) * direction) if last and touch else 0
            previous = encode_events(
                previous.new_tensor([order.time_ms - now]), previous.new_tensor([order.action]),
                previous.new_tensor([depth]), previous.new_tensor([order.shares]), ticks,
            )
            if old_rng is not None:
                generator.set_state(old_rng)
            child += 1
        elif proposal_time <= horizon:
            if len(path.tape) == max_events:
                raise RuntimeError("Background event budget exhausted before the rollout horizon")
            order = path.instruction(proposal_time, decoded)
            path.apply(order)
            previous = tokens
        else:
            break
        now = order.time_ms
        own = order.is_own


@torch.inference_mode()
def rollout_paired_window(
    model: LOBWorldModel, day: SZSEDaySlice, start_index: int, *,
    child_times_ms: Sequence[int], child_shares: int, grid_ms: Sequence[int], seed: int,
    context_len: int = 128, max_events: int = 1024,
) -> dict[RolloutArm, RolloutTrace]:
    """Warm on same-session history and replay five arms from one ten-level book.

    Grid and child times are elapsed milliseconds from the start index. The
    blind books receive the exact unperturbed absolute background instructions.
    Visible and unperturbed generations share draws by background event ordinal.
    """
    grid = np.asarray(grid_ms, dtype=np.int64)
    children = list(map(int, child_times_ms))
    if not len(grid) or grid[0] < 0 or (np.diff(grid) <= 0).any():
        raise ValueError("Rollout requires a nonnegative, strictly increasing measurement grid")
    if children and (child_shares <= 0 or children[0] < 0 or children[-1] > grid[-1] or
                     any(a >= b for a, b in zip(children, children[1:]))):
        raise ValueError("Child orders require positive size and increasing times within the horizon")
    if context_len <= 0 or not any(a + context_len <= start_index < b for a, b in day.session_bounds):
        raise ValueError("Rollout needs a complete same-session history before its start")
    if max_events <= 0:
        raise ValueError("Rollout needs a positive background event budget")

    ticks = day.pre_book_ticks[start_index]
    shares = day.pre_book_shares[start_index]
    if not 0 < ticks[0, 0] < ticks[1, 0]:
        raise ValueError("Initial replay book must have two uncrossed sides")
    device = next(model.parameters(), torch.empty(0)).device
    hidden, previous = _warm(model, day, start_index, context_len, device)
    capacity = 11 + max_events
    baseline = _Path(ticks, shares, capacity)
    _simulate(model, baseline, hidden, previous, [], int(grid[-1]), seed, max_events)
    traces = {RolloutArm.NO_INJECTION: baseline.trace(grid)}
    pairs = ((RolloutArm.BUY_VISIBLE, RolloutArm.BUY_BLIND), (RolloutArm.SELL_VISIBLE, RolloutArm.SELL_BLIND))
    for side, (visible_arm, blind_arm) in enumerate(pairs):
        orders = _children(children, child_shares, side)
        visible = _Path(ticks, shares, capacity)
        _simulate(model, visible, hidden, previous, orders, int(grid[-1]), seed, max_events)
        blind = _Path(ticks, shares, capacity)
        for order in merge(orders, baseline.tape, key=lambda order: order.time_ms):
            blind.apply(order)
        traces[visible_arm] = visible.trace(grid)
        traces[blind_arm] = blind.trace(grid)
    return traces
