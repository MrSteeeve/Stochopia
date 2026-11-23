"""Own-flag recipe of the tier-0 experiment, including label errors."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from experiments.tier0_feasibility import NO_SELF_PROB, make_own_flags  # noqa: E402
from stochopia.sim.world import WorldSpec, simulate_day  # noqa: E402

WINDOW = 257


@pytest.fixture(scope="module")
def days():
    world = WorldSpec(name="flags-test", n_trades=4000, burn_in=500)
    return [simulate_day(world, "flags", i) for i in range(3)]


def _draw(days, n=600, **kwargs):
    """Flags of ``n`` windows next to the flags of the same windows without
    label errors (same generator state at the start of every call)."""
    noisy, clean = make_own_flags(days, False, True, **kwargs), make_own_flags(days, False, True)
    picker = np.random.default_rng(0)
    out = []
    for k in range(n):
        day = int(picker.integers(0, len(days)))
        start = int(picker.integers(0, days[day].signs.size - WINDOW))
        a = noisy(day, start, WINDOW, np.random.default_rng(k))
        b = clean(day, start, WINDOW, np.random.default_rng(k))
        out.append((days[day].meta_id[start : start + WINDOW], a, b))
    return out


def test_clean_flags_mark_one_whole_metaorder_or_nothing(days):
    windows = _draw(days)
    empty = 0
    for ids, flags, clean in windows:
        assert np.array_equal(flags, clean)
        if not flags.any():
            empty += 1
            continue
        own = np.unique(ids[flags > 0])
        assert own.size == 1 and np.array_equal(flags > 0, ids == own[0])
    assert empty / len(windows) == pytest.approx(NO_SELF_PROB, abs=0.08)


def test_miss_only_removes_own_flags_at_the_set_rate(days):
    lost = kept = 0
    for _, flags, clean in _draw(days, miss=0.3):
        assert not np.any((flags > 0) & (clean == 0))
        lost += int(((clean > 0) & (flags == 0)).sum())
        kept += int((flags > 0).sum())
    assert lost / (lost + kept) == pytest.approx(0.3, abs=0.04)


def test_false_only_adds_flags_to_other_metaorders_at_the_set_share(days):
    wrong = right = 0
    for _, flags, clean in _draw(days, false_share=0.3):
        assert not np.any((clean > 0) & (flags == 0))
        wrong += int(((flags > 0) & (clean == 0)).sum())
        right += int((clean > 0).sum())
    assert wrong / (wrong + right) == pytest.approx(0.3, abs=0.04)
