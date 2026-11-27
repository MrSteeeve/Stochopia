"""Standard model input adapter and CRN truth for the small queue world."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

import numpy as np

from stochopia.data import SZSEDaySlice
from stochopia.harness.lob_batch_matching import BatchedMatcher
from stochopia.harness.lob_rollout import RolloutWindow, _children
from stochopia.measure.data_properties import _sign_acf
from stochopia.stats import derive_seed

from .queue_reactive_world import (
    CHILD_SHARES, CHILD_TIMES_MS, CONTEXT_LEN, EVENTS_PER_SESSION, GRID_MS, INITIAL_TICK,
    MAX_EVENTS, QueueWorldSpec, clock_tape, initial_book, reactive_order,
)


@dataclass(frozen=True)
class QueueDataset:
    spec: QueueWorldSpec
    train: tuple[SZSEDaySlice, ...]
    evaluation: tuple[SZSEDaySlice, ...]
    windows: tuple[RolloutWindow, ...]
    # Hidden *generator* state before the next background draw at each anchor:
    # two trade-clock signs, Hawkes excitation, sub-millisecond clock phase.
    truth_initial: np.ndarray


def _rng(seed: int, purpose: str, day: int, session: int, replicate: int = 0) -> np.random.Generator:
    # No recipe name in the key: do(lambda) shares the random coordinates.
    return np.random.default_rng(derive_seed("p2-m2-queue", seed, purpose, day, session, replicate))


def _days(
    spec: QueueWorldSpec, seed: int, first: int, count: int, length: int, context: int,
) -> tuple[tuple[SZSEDaySlice, ...], np.ndarray]:
    """Generate fresh independent day/session lanes, then pack SZSE's pre-event format."""
    if not count:
        return (), np.empty((0, 4), dtype=float)
    randoms = []
    signs = []
    for day in range(first, first + count):
        for session in range(2):
            rng = _rng(seed, "history", day, session)
            signs.append(2 * rng.integers(0, 2, size=2) - 1)
            randoms.append(rng.random((length, 7)))
    marks = np.stack(randoms)
    sign_state = np.array(signs, dtype=np.int64)
    lanes = 2 * count
    times, excitation, phase = clock_tape(spec, marks[:, :, 0], np.zeros(lanes), np.zeros(lanes))
    px, qty = initial_book(spec)
    matcher = BatchedMatcher([px] * lanes, [qty] * lanes)
    book_ticks = np.empty((lanes, length, 2, 10), dtype=np.int64)
    book_shares = np.empty_like(book_ticks)
    actions = np.empty((lanes, length), dtype=np.int8)
    relative = np.empty((lanes, length), dtype=np.int32)
    shares = np.empty((lanes, length), dtype=np.int64)
    truth = np.empty((lanes, 4))
    for step in range(length):
        if step == context:
            truth[:, :2] = sign_state
            truth[:, 2] = excitation[:, step]
            truth[:, 3] = phase[:, step]
        orders = []
        for lane, path in enumerate(matcher.paths):
            book_ticks[lane, step] = path.prices[:, :10]
            book_shares[lane, step] = path.shares[:, :10]
            order = reactive_order(path, spec, sign_state[lane], int(times[lane, step]), marks[lane, step, 1:])
            actions[lane, step] = order.action
            relative[lane, step] = order.relative
            if order.action >= 4:
                # SZSEDaySlice stores tick distance, not model cancellation rank.
                relative[lane, step] = abs(order.price - int(path.prices[order.action % 2, 0]))
            shares[lane, step] = order.shares
            orders.append(order)
        matcher.apply(orders)
    results = []
    for d in range(count):
        pair = slice(2 * d, 2 * d + 2)
        day_label = int((date(2030, 1, 1) + timedelta(days=first + d)).strftime("%Y%m%d"))
        ticks = book_ticks[pair].reshape(2 * length, 2, 10)
        levels = book_shares[pair].reshape(2 * length, 2, 10)
        local = times[pair].copy()
        dt = np.diff(local, axis=1, prepend=0)
        local += np.array([34_200_000, 46_800_000])[:, None]
        results.append(SZSEDaySlice(
            wind_code="SYNTHETIC.SZ", trading_date=day_label, prev_close_ticks=INITIAL_TICK,
            session_bounds=((0, length), (length, 2 * length)), time_ms=local.reshape(-1),
            seq_id=np.arange(2 * length, dtype=np.uint64), dt_ms=dt.reshape(-1),
            action_type=actions[pair].reshape(-1), rel_ticks=relative[pair].reshape(-1),
            volume_shares=shares[pair].reshape(-1), cancel_age_ms=np.zeros(2 * length, dtype=np.int64),
            pre_book_ticks=ticks, pre_book_shares=levels, mid_ticks_x2=ticks[:, :, 0].sum(axis=1),
            spread_ticks=ticks[:, 1, 0] - ticks[:, 0, 0],
        ))
    return tuple(results), truth


def generate_dataset(
    spec: QueueWorldSpec, *, seed: int = 20261127, train_days: int = 16, eval_days: int = 12,
    events_per_session: int = EVENTS_PER_SESSION, context_len: int = CONTEXT_LEN,
) -> QueueDataset:
    """Chronologically disjoint synthetic histories; no fitted model or holdout acquisition."""
    if not 0 < context_len < events_per_session or train_days < 1 or eval_days < 0:
        raise ValueError("nonempty train and a complete context before each anchor required")
    train, _ = _days(spec, seed, 0, train_days, events_per_session, context_len)
    evaluation, state = _days(spec, seed, train_days, eval_days, events_per_session, context_len)
    windows = tuple(
        RolloutWindow(day, start + context_len, session, MAX_EVENTS)
        for day in evaluation
        for session, (start, _) in enumerate(day.session_bounds)
    )
    return QueueDataset(spec, train, evaluation, windows, state)


def _truth_chunk(
    dataset: QueueDataset, indices: Sequence[int], grid: np.ndarray,
    child_times: Sequence[int], child_shares: int, replicas: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return [window,grid] mean signed matched CRN response and support counts."""
    ticks = []
    volumes = []
    marks = []
    initial = []
    for i in indices:
        window = dataset.windows[i]
        book = window.day
        for replica in range(replicas):
            rng = _rng(seed, "truth", book.trading_date, window.window_id, replica)
            marks.append(rng.random((MAX_EVENTS, 7)))
            initial.append(dataset.truth_initial[i])
            ticks.extend([book.pre_book_ticks[window.start_index]] * 3)
            volumes.extend([book.pre_book_shares[window.start_index]] * 3)
    samples = np.stack(marks)
    state = np.stack(initial)
    clock, _, _ = clock_tape(dataset.spec, samples[:, :, 0], state[:, 2], state[:, 3])
    if (clock[:, -1] <= grid[-1]).any():
        raise RuntimeError("truth clock event budget exhausted")
    sign_state = np.repeat(state[:, :2].astype(np.int64), 3, axis=0)
    matcher = BatchedMatcher(ticks, volumes, capacity=32)
    programs = []
    for times in clock:
        background = [(int(t), 1, int(j)) for j, t in enumerate(times) if t <= grid[-1]]
        for arm in range(3):
            children = [] if arm == 0 else [(int(t), 0, j) for j, t in enumerate(child_times)]
            programs.append(sorted(background + children))
    child_orders = ((), _children(child_times, child_shares, 0), _children(child_times, child_shares, 1))
    for step in range(max(map(len, programs))):
        orders = []
        for lane, program in enumerate(programs):
            if step >= len(program):
                orders.append(None)
                continue
            when, kind, ordinal = program[step]
            if kind == 0:
                order = child_orders[lane % 3][ordinal]
            else:
                order = reactive_order(matcher.paths[lane], dataset.spec, sign_state[lane],
                                       when, samples[lane // 3, ordinal, 1:])
            orders.append(order)
        matcher.apply(orders)
    curves = np.array([path.trace(grid).mid_ticks for path in matcher.paths])
    curves = curves.reshape(len(indices), replicas, 3, len(grid))
    free, buy, sell = (curves[:, :, i] for i in range(3))
    signed = ((buy - free) - (sell - free)) / 2
    valid = np.isfinite(signed)
    counts = valid.sum(axis=1)
    sums = np.where(valid, signed, 0.0).sum(axis=1)
    means = np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0)
    return means, counts


def counterfactual_truth(
    dataset: QueueDataset, *, grid_ms: Sequence[int] = GRID_MS,
    child_times_ms: Sequence[int] = CHILD_TIMES_MS, child_shares: int = CHILD_SHARES,
    num_replicates: int = 16, seed: int = 9127, return_support: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Known-generator Monte Carlo truth on the same intervention and clock grid.

    The free/visible buy/visible sell paths have identical background random
    coordinates, and all three require finite two-sided quotes. This is the
    prescribed K=16 truth estimate, not a model-trained surrogate.
    """
    if num_replicates < 1 or not dataset.windows:
        raise ValueError("nonempty windows and positive truth replicate budget required")
    grid = np.asarray(grid_ms, dtype=np.int64)
    if not len(grid) or grid[0] != 0 or np.any(np.diff(grid) <= 0):
        raise ValueError("strictly increasing grid starting at zero required")
    children = tuple(map(int, child_times_ms))
    if any(t < 0 or t > grid[-1] for t in children) or child_shares <= 0:
        raise ValueError("child orders must lie on the evaluation horizon")
    parts = []
    support = []
    # Two same-day windows at once: fixed-width matcher, no per-path device sync.
    for offset in range(0, len(dataset.windows), 2):
        means, counts = _truth_chunk(dataset, range(offset, min(offset + 2, len(dataset.windows))),
                                     grid, children, child_shares, num_replicates, seed)
        parts.append(means)
        support.append(counts)
    truth, counts = np.concatenate(parts), np.concatenate(support)
    return (truth, counts) if return_support else truth


def tier0_manipulation_check(*, n_days: int = 16) -> dict:
    """Trade-level control: isolate flow persistence, and vary known kernel independently."""
    from .flow import FlowSpec
    from .impact import KernelSpec
    from .world import WorldSpec, injection_schedule, simulate_day, true_response

    schedule = injection_schedule(8, 4, 40)
    output = {}
    for label in ("short", "long"):
        flow = FlowSpec(n_slots=1, max_len=1) if label == "short" else FlowSpec(n_slots=1, alpha=1.1, max_len=256)
        model = WorldSpec(name="tier0-m2", flow=flow, n_trades=512, burn_in=128)
        kernel = KernelSpec(tau_max=16.0 if label == "short" else 1000.0)
        kernel_model = WorldSpec(name="tier0-m2", flow=flow, kernel=kernel, n_trades=512, burn_in=128)
        acfs = []
        responses = []
        kernel_response = []
        for d in range(n_days):
            day = simulate_day(model, "m2-train", d)
            acfs.append(float(_sign_acf(day.signs).sum()))
            for spec, results in ((model, responses), (kernel_model, kernel_response)):
                results.append((true_response(spec, day, 64, schedule, 1) -
                                true_response(spec, day, 64, schedule, -1)) / 2)
        q25, median, q75 = np.quantile(acfs, (0.25, 0.5, 0.75))
        output[label] = {
            "flow": {"n_slots": flow.n_slots, "alpha": flow.alpha, "max_len": flow.max_len},
            "sign_acf_sum_1_32": {"median": float(median), "intraday_iqr": float(q75 - q25)},
            "fixed_kernel_truth": np.mean(responses, axis=0).tolist(),
            "independent_kernel_tau_max": kernel.tau_max,
            "independent_kernel_truth": np.mean(kernel_response, axis=0).tolist(),
        }
    return output
