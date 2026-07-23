"""Bounded deterministic cache primitives used by pricing and domain enumeration."""

from mirage.cache import BoundedLRUCache


def test_bounded_lru_cache_evicts_oldest_and_refreshes_hits():
    cache: BoundedLRUCache[str, int] = BoundedLRUCache(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    assert cache["a"] == 1  # refresh a, so b becomes least recently used

    cache["c"] = 3

    assert len(cache) == 2
    assert "a" in cache
    assert "b" not in cache
    assert "c" in cache

