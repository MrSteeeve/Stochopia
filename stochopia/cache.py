"""Small bounded caches used by deterministic pricing paths.

The benchmark previously used unbounded process-global dictionaries.  That is
acceptable for a tiny frozen evaluation set, but randomised training tasks can
otherwise make a long-lived worker grow without limit.  ``BoundedLRUCache``
keeps the existing ``dict``-like surface (including ``clear``) while applying a
deterministic least-recently-used eviction policy.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from threading import RLock
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class BoundedLRUCache(MutableMapping[K, V], Generic[K, V]):
    """A minimal thread-safe bounded LRU mapping."""

    def __init__(self, maxsize: int) -> None:
        if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1:
            raise ValueError("maxsize must be a positive integer")
        self.maxsize = maxsize
        self._data: OrderedDict[K, V] = OrderedDict()
        self._lock = RLock()

    def __getitem__(self, key: K) -> V:
        with self._lock:
            value = self._data[key]
            self._data.move_to_end(key)
            return value

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def __delitem__(self, key: K) -> None:
        with self._lock:
            del self._data[key]

    def __iter__(self) -> Iterator[K]:
        with self._lock:
            return iter(tuple(self._data))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def get(self, key: K, default: V | None = None) -> V | None:
        with self._lock:
            if key not in self._data:
                return default
            value = self._data[key]
            self._data.move_to_end(key)
            return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
