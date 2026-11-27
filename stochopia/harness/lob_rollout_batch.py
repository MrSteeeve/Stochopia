"""V2 batched physical-clock rollouts with replay-lane common random numbers."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import blake2b
from heapq import merge
from itertools import zip_longest
from operator import index

import numpy as np
import torch

from stochopia.harness.lob_batch_matching import BatchedMatcher
from stochopia.harness.lob_rollout import RolloutArm, RolloutTrace, RolloutWindow, _Order, _children
from stochopia.neural.lob_interface import (
    LOBWorldModel, State, StepObservation, StepSampler, decode_events, encode_events, index_select_state,
)


ROLLOUT_PROTOCOL_V2 = "rollout_protocol_v2"


# --- Independent random tapes and recurrent warmup ---


def background_uniforms(base_seed: int, date: int, window_id: int, replicate: int, count: int) -> np.ndarray:
    """Return [background ordinal, conditional head] deterministic U[0,1) draws.

    Each addressed replicate has its own PCG64 key. Neither changing the
    batch width nor another lane's stopping time changes these uniforms.
    """
    key = f"{ROLLOUT_PROTOCOL_V2}/{base_seed}/{date}/{window_id}/{replicate}".encode()
    seed = int.from_bytes(blake2b(key, digest_size=16).digest(), byteorder="big")
    return np.random.default_rng(seed).random((count, 4))


class NonfiniteRolloutProbability(ValueError):
    """A conditional categorical head has no finite positive probability mass."""


def _sample_uniforms(sampler: StepSampler, values: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF sample each row at fixed uniforms using the existing masks."""
    tokens: list[torch.Tensor] = []
    for stage in range(4):
        logits = sampler.logits(stage, tuple(tokens))
        cdf = torch.cumsum(logits.softmax(dim=-1), dim=-1, dtype=torch.float64)
        mass = cdf[:, -1:]
        if not bool((torch.isfinite(mass) & (mass > 0)).all()):
            raise NonfiniteRolloutProbability(f"conditional head {stage} has zero or nonfinite probability mass")
        cdf = cdf / mass
        chosen = torch.searchsorted(cdf, values[:, stage:stage + 1].contiguous(), right=True)
        if torch.any(chosen >= logits.shape[-1]):
            raise NonfiniteRolloutProbability(f"conditional head {stage} did not define a categorical distribution")
        tokens.append(chosen[:, 0])
    return torch.stack(tokens, dim=-1)


def _warm_windows(
    model: LOBWorldModel, windows: Sequence[RolloutWindow], count: int, device: torch.device,
) -> tuple[State, torch.Tensor]:
    """Warm each physical window once; the warm state is shared by its replicas."""
    ticks = []
    shares = []
    targets = []
    for window in windows:
        day = window.day
        history = slice(window.start_index - count, window.start_index)
        book = torch.tensor(day.pre_book_ticks[history], device=device)
        quantity = torch.tensor(day.pre_book_shares[history], device=device)
        target = encode_events(
            torch.tensor(day.dt_ms[history], device=device), torch.tensor(day.action_type[history], device=device),
            torch.tensor(day.rel_ticks[history], device=device), torch.tensor(day.volume_shares[history], device=device),
            book,
        )
        ticks.append(book)
        shares.append(quantity)
        targets.append(target)
    books = torch.stack(ticks)
    quantities = torch.stack(shares)
    tokens = torch.stack(targets)
    previous = torch.zeros((len(windows), 4), dtype=torch.long, device=device)
    own = torch.zeros(len(windows), dtype=torch.bool, device=device)
    hidden: State = None
    for t in range(count):
        obs = StepObservation(books[:, t], quantities[:, t], previous, own)
        _, hidden = model.forward_step(obs, hidden)
        previous = tokens[:, t]
    return hidden, previous


# --- Visible and free paths, advancing at one proposed event per active lane ---


def _visible_batch(
    model: LOBWorldModel, windows: Sequence[RolloutWindow], *, child_times: list[int], child_shares: int,
    grid: np.ndarray, seed: int, context_len: int, replicas: int,
) -> tuple[BatchedMatcher, np.ndarray]:
    """Generate all free/buy-visible/sell-visible paths before blind replay."""
    device = next(model.parameters(), torch.empty(0)).device
    warmed, previous_by_window = _warm_windows(model, windows, context_len, device)
    ticks = []
    shares = []
    windows_by_lane = []
    scheduled: list[list[_Order]] = []
    random_tapes: list[np.ndarray] = []
    for window_index, window in enumerate(windows):
        for replicate in range(replicas):
            values = background_uniforms(seed, window.day.trading_date, window.window_id, replicate,
                                         window.max_events + 1)
            for children in ([], _children(child_times, child_shares, 0), _children(child_times, child_shares, 1)):
                ticks.append(window.day.pre_book_ticks[window.start_index])
                shares.append(window.day.pre_book_shares[window.start_index])
                windows_by_lane.append(window_index)
                scheduled.append(children)
                random_tapes.append(values)
    matcher = BatchedMatcher(ticks, shares)
    n_lanes = len(ticks)
    active = list(range(n_lanes))
    lane_windows = torch.tensor(windows_by_lane, device=device)
    state = index_select_state(warmed, lane_windows)
    previous = previous_by_window.index_select(0, lane_windows)
    background = np.zeros(n_lanes, dtype=np.int64)
    child = np.zeros(n_lanes, dtype=np.int64)
    now = np.zeros(n_lanes, dtype=np.int64)
    own = np.zeros(n_lanes, dtype=bool)
    capped = np.zeros(n_lanes, dtype=bool)
    horizon = int(grid[-1])

    while active:
        prices = np.stack([matcher.paths[lane].prices for lane in active])
        quantities = np.stack([matcher.paths[lane].shares for lane in active])
        book = torch.tensor(prices[:, :, :10], device=device)
        obs = StepObservation(
            book, torch.tensor(quantities[:, :, :10], device=device), previous,
            torch.tensor(own[active], device=device), torch.tensor(prices[:, :, 10] > 0, device=device),
        )
        sampler, next_state = model.forward_step(obs, state)
        uniforms = np.stack([random_tapes[lane][background[lane]] for lane in active])
        tokens = _sample_uniforms(sampler, torch.tensor(uniforms, device=device))
        decoded = decode_events(tokens).cpu().tolist()

        orders: list[_Order | None] = [None] * n_lanes
        pending_children: list[tuple[int, int]] = []
        for row, lane in enumerate(active):
            proposal = int(now[lane]) + decoded[row][0]
            if child[lane] < len(scheduled[lane]):
                order = scheduled[lane][child[lane]]
                if order.time_ms <= proposal:
                    orders[lane] = order
                    pending_children.append((row, lane))
                    continue
            if proposal <= horizon:
                orders[lane] = matcher.paths[lane].instruction(proposal, decoded[row])

        last_prices = matcher.apply(orders)
        if pending_children:
            _store_child_tokens(tokens, book, prices, pending_children, orders, last_prices, now, device)
        keep = []
        for row, lane in enumerate(active):
            order = orders[lane]
            if order is None:
                continue
            now[lane] = order.time_ms
            own[lane] = order.is_own
            if order.is_own:
                child[lane] += 1
            else:
                background[lane] += 1
            maximum = windows[windows_by_lane[lane]].max_events
            if background[lane] == maximum:
                capped[lane] = now[lane] < horizon
            else:
                keep.append(row)

        active = [active[row] for row in keep]
        if active:
            indices = torch.tensor(keep, device=device)
            state = index_select_state(next_state, indices)
            previous = tokens.index_select(0, indices)

    # The event cap limits background generation, not scheduled child orders.
    remainder = [scheduled[lane][child[lane]:] for lane in range(n_lanes)]
    for orders in zip_longest(*remainder):
        matcher.apply(orders)
    return matcher, capped


def _store_child_tokens(
    tokens: torch.Tensor, book: torch.Tensor, prices: np.ndarray, rows: list[tuple[int, int]],
    orders: Sequence[_Order | None], last_prices: np.ndarray, now: np.ndarray, device: torch.device,
) -> None:
    """Replace preempted proposal tokens; keep its random draw for the next step."""
    dt = []
    action = []
    depth = []
    shares = []
    indices = []
    for row, lane in rows:
        order = orders[lane]
        assert order is not None
        touch = prices[row, 1 - order.action % 2, 0]
        last = int(last_prices[lane])
        direction = 1 - 2 * (order.action % 2)
        depth.append(max(0, (last - int(touch)) * direction) if last and touch else 0)
        dt.append(order.time_ms - int(now[lane]))
        action.append(order.action)
        shares.append(order.shares)
        indices.append(row)
    selected = torch.tensor(indices, device=device)
    own_tokens = encode_events(tokens.new_tensor(dt), tokens.new_tensor(action), tokens.new_tensor(depth),
                               tokens.new_tensor(shares), book.index_select(0, selected))
    tokens.index_copy_(0, selected, own_tokens)


# --- Blind replays reuse each exact free tape ---


def _blind_batch(
    matcher: BatchedMatcher, windows: Sequence[RolloutWindow], child_times: list[int], child_shares: int,
    replicas: int,
) -> BatchedMatcher:
    ticks = []
    shares = []
    tapes = []
    for number, window in enumerate(windows):
        for replicate in range(replicas):
            free_index = 3 * (replicas * number + replicate)
            baseline = matcher.paths[free_index].tape
            for side in range(2):
                ticks.append(window.day.pre_book_ticks[window.start_index])
                shares.append(window.day.pre_book_shares[window.start_index])
                children = _children(child_times, child_shares, side)
                # Child messages have priority at equal timestamps, as in P1.
                tapes.append(list(merge(children, baseline, key=lambda order: order.time_ms)))
    result = BatchedMatcher(ticks, shares)
    for orders in zip_longest(*tapes):
        result.apply(orders)
    return result


@torch.inference_mode()
def rollout_batch(
    model: LOBWorldModel, windows: Sequence[RolloutWindow], *,
    child_times_ms: Sequence[int], child_shares: int, grid_ms: Sequence[int],
    seed: int, context_len: int = 128, num_replicates: int = 1,
) -> list[tuple[dict[RolloutArm, RolloutTrace], ...]]:
    """Run v2 in one active model batch and a vectorized matcher per step.

    The return layout is [physical window][replicate][arm]; aggregate replicas
    before computing any day-cluster uncertainty. The matcher and RNG keys
    are independent of the active batch size.
    """
    replicas = index(num_replicates)
    if replicas < 1 or not len(windows):
        raise ValueError("Rollout requires at least one window and replicate")
    grid = np.asarray(grid_ms, dtype=np.int64)
    children = list(map(int, child_times_ms))
    if not len(grid) or grid[0] < 0 or (np.diff(grid) <= 0).any():
        raise ValueError("Rollout requires a nonnegative, strictly increasing measurement grid")
    if children and (child_shares <= 0 or children[0] < 0 or children[-1] > grid[-1] or
                     any(a >= b for a, b in zip(children, children[1:]))):
        raise ValueError("Child orders require positive size and increasing times within the horizon")
    keys = set()
    for window in windows:
        day = window.day
        if (day.trading_date, window.window_id) in keys:
            raise ValueError("Duplicate deterministic rollout window key")
        keys.add((day.trading_date, window.window_id))
        if context_len <= 0 or not any(a + context_len <= window.start_index < b for a, b in day.session_bounds):
            raise ValueError("Rollout needs a complete same-session history before its start")
        if index(window.max_events) <= 0:
            raise ValueError("Rollout needs a positive background event budget")
        ticks = day.pre_book_ticks[window.start_index]
        if not 0 < ticks[0, 0] < ticks[1, 0]:
            raise ValueError("Initial replay book must have two uncrossed sides")
    visible, capped = _visible_batch(model, windows, child_times=children, child_shares=child_shares,
                                    grid=grid, seed=seed, context_len=context_len, replicas=replicas)
    blind = _blind_batch(visible, windows, children, child_shares, replicas)
    result = []
    for number in range(len(windows)):
        window_paths = []
        for replicate in range(replicas):
            flat = number * replicas + replicate
            free = 3 * flat
            window_paths.append({
                RolloutArm.NO_INJECTION: visible.paths[free].trace(grid, budget_exhausted=bool(capped[free])),
                RolloutArm.BUY_VISIBLE: visible.paths[free + 1].trace(grid, budget_exhausted=bool(capped[free + 1])),
                RolloutArm.SELL_VISIBLE: visible.paths[free + 2].trace(grid, budget_exhausted=bool(capped[free + 2])),
                RolloutArm.BUY_BLIND: blind.paths[2 * flat].trace(grid, budget_exhausted=bool(capped[free])),
                RolloutArm.SELL_BLIND: blind.paths[2 * flat + 1].trace(grid, budget_exhausted=bool(capped[free])),
            })
        result.append(tuple(window_paths))
    return result
