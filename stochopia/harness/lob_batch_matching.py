"""Vectorized JAX-LOB matching for independent, same-clock replay lanes."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from stochopia.harness.lob_rollout import _BookTape, _Order, _matching_functions


@lru_cache(maxsize=1)
def _batch_functions() -> tuple[Any, Any, Any, Any]:
    """Apply the exact scalar engine to each lane, with one host L2 readback."""
    as_array, match, top = _matching_functions()

    import jax
    import jax.numpy as jnp

    def l2(book: Any) -> Any:
        return top(book[1], book[0], 11)

    def apply_lane(book: Any, side: Any, args: Any) -> tuple[Any, Any, Any]:
        own = jax.lax.select(side == 0, book[0], book[1])
        other = jax.lax.select(side == 0, book[1], book[0])
        own, other, filled, last = match(own, other, args)
        bids = jax.lax.select(side == 0, own, other)
        asks = jax.lax.select(side == 0, other, own)
        return jnp.stack((bids, asks)), filled, last

    def summarize(books: Any) -> tuple[Any, Any]:
        return jax.vmap(l2)(books), (books[:, :, :, 1] > 0).sum(axis=2)

    @jax.jit
    def advance(books: Any, sides: Any, args: Any) -> tuple[Any, Any, Any, Any, Any]:
        books, filled, last = jax.vmap(apply_lane)(books, sides, args)
        rows, counts = summarize(books)
        return books, filled, last, rows, counts

    return as_array, jax.jit(summarize), advance, jnp.pad


class BatchedMatcher:
    """Hold fixed-width [lane, bid/ask, resting order, field] device books.

    Empty lanes submit a zero-share no-op, preserving the compiled batch width
    as other lanes finish. Books start with 128 rows/side and double only
    before a side runs out of rows. Appending empty rows preserves the scalar
    engine's first-empty insertion order and FIFO order IDs exactly.
    """

    def __init__(
        self, ticks: Sequence[np.ndarray], shares: Sequence[np.ndarray], *,
        capacity: int = 128,
    ) -> None:
        if len(ticks) != len(shares) or not len(ticks):
            raise ValueError("initial books must have one bid/ask pair per lane")
        if capacity < 10:
            raise ValueError("eleven-level mark extraction needs at least ten initial rows")
        as_array, summarize, self._advance, self._pad = _batch_functions()
        self._capacity = capacity
        labels = len(ticks)
        data = np.full((labels, 2, capacity, 6), -1, dtype=np.int32)
        self._ids = np.zeros(labels, dtype=np.int32)
        for lane, (prices, volumes) in enumerate(zip(ticks, shares, strict=True)):
            for side in range(2):
                for level in range(10):
                    price = prices[side, level]
                    quantity = volumes[side, level]
                    if price > 0 and quantity > 0:
                        self._ids[lane] += 1
                        order_id = self._ids[lane]
                        data[lane, side, level] = (price, quantity, order_id, order_id, 0, 0)

        self.books = as_array(data)
        rows, counts = summarize(self.books)
        self._counts = np.asarray(counts)
        initial_marks = np.asarray(rows)
        self.paths = [_BookTape() for _ in range(labels)]
        for path, mark in zip(self.paths, initial_marks, strict=True):
            path.record(0, mark)

    def apply(self, orders: Sequence[_Order | None]) -> np.ndarray:
        """Synchronously advance one order per lane and return deepest fills.

        JAX arrays stay on device, while touch marks and fill metadata cross
        the host boundary once per synchronized advance, not once per path.
        """
        if len(orders) != len(self.paths):
            raise ValueError("every matching lane must have one order or no-op")
        n_lanes = len(orders)
        sides = np.zeros(n_lanes, dtype=np.int32)
        args = np.zeros((n_lanes, 8), dtype=np.int32)
        args[:, 0] = 1
        active = []
        for lane, order in enumerate(orders):
            if order is None:
                continue
            active.append(lane)
            side = order.action % 2
            sides[lane] = side
            self._ids[lane] += 1
            seconds, millis = divmod(order.time_ms, 1_000)
            args[lane] = (
                1 - 2 * side, order.price, order.shares, self._ids[lane],
                seconds, millis * 1_000_000, order.action >= 4, order.is_own,
            )
        if not active:
            return np.zeros(n_lanes, dtype=np.int32)

        # Only a resting insertion could overflow; reserve room for it before
        # calling the engine's first-empty insertion primitive.
        if any(self._counts[lane, sides[lane]] >= self._capacity for lane in active):
            self.books = self._pad(self.books, ((0, 0), (0, 0), (0, self._capacity), (0, 0)),
                                   constant_values=-1)
            self._capacity *= 2

        self.books, filled, last, rows, counts = self._advance(self.books, sides, args)
        fills = np.asarray(filled)
        last_prices = np.asarray(last)
        marks = np.asarray(rows)
        self._counts = np.asarray(counts)
        for lane in active:
            order = orders[lane]
            assert order is not None
            path = self.paths[lane]
            path.record(order.time_ms, marks[lane])
            path.store_order(order, int(fills[lane]))
        return last_prices
