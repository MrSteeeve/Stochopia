import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

import numpy as np
import pytest

from experiments.tier0_feasibility import WORLDS, evaluate_model, initialize_model
from stochopia.harness import runner
from stochopia.harness.contract import CandidateRef, ExperimentManifest, FitRecord, ReportStatus
from stochopia.harness.lob_adapter import OrderEvent, _replay, export_lob_bench, run_short_lob
from stochopia.harness.runner import Runner, _fit_args, classify_diagnosis
from stochopia.sim.world import simulate_day
from stochopia.stats import derive_seed

from test_harness_contract import manifest  # noqa: F401


# --- Shared Tier 0 fixture ---


@pytest.fixture
def small_manifest(manifest: ExperimentManifest) -> ExperimentManifest:
    scene = replace(
        manifest.scenarios[0],
        environment={"world": "informed", "world_overrides": {"n_trades": 48, "burn_in": 8}},
        action_plan={"children": 2, "spacing": 1, "relax": 2, "context": 6, "windows_per_day": 2},
    )
    splits = {}
    for name, split in manifest.splits.items():
        namespace = split.source["namespace"]
        world = split.source["world"]
        seeds = {}
        for index, unit in enumerate(split.unit_ids):
            seeds[unit] = derive_seed("stochopia.sim.world", world, namespace, index)
        splits[name] = replace(split, unit_seeds=seeds)
    base, flow, price = manifest.candidates
    candidates = (
        replace(base, recipe="anonymous"),
        replace(flow, recipe="separate_flow", implementation={}),
        replace(price, recipe="self"),
    )
    return replace(
        manifest, scenarios=(scene,), splits=splits, candidates=candidates,
        train_plan={"hidden": 4, "layers": 1, "steps": 2, "batch": 2, "window": 8},
    )


# --- Existing raw observations, diagnosis and short LOB replay ---


def test_tier0_opt_in_retains_raw_arms_without_changing_summary() -> None:
    args = argparse.Namespace(
        hidden=4, layers=1, recipe="anonymous", cell="gru", device="cpu",
        window=8, context=6, windows_per_day=2, children=2, spacing=2, relax=3,
    )
    world = replace(WORLDS["plain"], n_trades=40, burn_in=8)
    days = [simulate_day(world, "optional-observations", index) for index in range(2)]
    model = initialize_model(args, days, 49)
    previous = evaluate_model(model, args, world, days, rollout_seed=9, seed_ci=8)
    measured = evaluate_model(
        model, args, world, days, rollout_seed=9, seed_ci=8,
        return_observations=True,
    )
    assert "day_observations" not in previous
    assert measured["mean_paths"] == previous["mean_paths"]
    assert measured["heldout_loss"] == previous["heldout_loss"]
    rows = measured["day_observations"]
    assert len(rows) == 2
    assert tuple(rows[0]["paths"]) == ("free", "clamped", "truth")
    assert sum(row["heldout_loss"] for row in rows) / 2 == pytest.approx(previous["heldout_loss"])


@pytest.mark.parametrize(
    ("free", "clamped", "expected"),
    [((3, 3), (1, 1), "flow"), ((3, 3), (3, 3), "price"), ((1.1, 1.1), (1, 1), None)],
)
def test_tier0_diagnosis_retains_channel_error_scores(
    free: tuple[float, float], clamped: tuple[float, float], expected: str | None,
) -> None:
    outcome = classify_diagnosis(free, clamped, (1, 1), min_effect=0.2, dominance_margin=0.1)
    assert outcome["dominant_channel"] == expected
    assert outcome["status"] is (ReportStatus.SUPPORTED if expected else ReportStatus.INCONCLUSIVE)
    assert outcome["flow_rmse"] >= 0
    assert outcome["price_rmse"] >= 0


def test_lob_matches_and_exports_the_post_message_book(tmp_path: Path) -> None:
    pytest.importorskip("jaxlob", reason="actual matching requires the harness-lob extra on Python 3.10–3.12")
    result = run_short_lob(depth=2)
    assert result.status is ReportStatus.SUPPORTED, result.reason
    assert [(fill.order_id, fill.size, fill.price, fill.direction) for fill in result.fills] == [
        (101, 40, 100100, -1),
    ]
    assert not result.without_order.fills
    messages, book = tmp_path / "messages.csv", tmp_path / "orderbook.csv"
    export_lob_bench(result.with_order, messages, book)
    message_rows = list(csv.reader(messages.read_text().splitlines()))
    book_rows = list(csv.reader(book.read_text().splitlines()))
    assert len(message_rows) == len(book_rows)
    execution = next(index for index, row in enumerate(message_rows) if row[1] == "4")
    assert message_rows[execution][2:] == ["101", "40", "100100", "-1"]
    assert book_rows[execution - 1][:2] == ["100100", "100"]
    assert book_rows[execution][:4] == ["100100", "60", "100000", "80"]


def test_lob_missing_import_returns_unavailable_and_no_trace() -> None:
    def no_module(name: str) -> NoReturn:
        raise ModuleNotFoundError(name, name=name)

    result = run_short_lob(import_module=no_module)
    assert result.status is ReportStatus.UNAVAILABLE
    assert result.reason and "jax" in result.reason.lower()
    assert result.with_order is None and result.without_order is None


# --- Training-plan, scene-binding and fit-cache regressions ---


@pytest.mark.parametrize(
    ("plan", "field"),
    [
        ({"windwo": 8}, "windwo"),
        ({"lr": 0.01, "learning_rate": 0.01}, "learning_rate"),
        ({"lr": True}, "lr"),
        ({"lr": "0.01"}, "lr"),
        ({"label_miss": False}, "label_miss"),
        ({"label_false": float("inf")}, "label_false"),
    ],
)
def test_fit_args_reject_invalid_training_plan(plan: dict[str, Any], field: str) -> None:
    base = CandidateRef(candidate_id="base", backbone="gru", recipe="anonymous")
    with pytest.raises(ValueError, match=field):
        _fit_args(plan, base, "anonymous")


def test_fit_args_allow_world_settings_and_normalize_numpy_scalars() -> None:
    base = CandidateRef(candidate_id="base", backbone="gru", recipe="anonymous")
    plan = {"world": "informed", "world_overrides": {"n_trades": 48},
            "steps": np.int64(2), "learning_rate": np.float32(0.01)}
    args = _fit_args(plan, base, "anonymous")
    assert type(args.steps) is int and args.steps == 2
    assert type(args.lr) is float and args.lr == pytest.approx(0.01)


def test_native_price_only_recipe_fits_without_background(small_manifest: ExperimentManifest, tmp_path: Path) -> None:
    fit = Runner(small_manifest).fit_candidate("price", "repeat-1", tmp_path)
    assert fit.config["recipe"] == "self" and fit.price_weights_ref is not None
    assert fit.parent_id == "base" and fit.background_weights_ref is None


def test_price_only_recipe_must_preserve_explicit_separated_background(
    small_manifest: ExperimentManifest, tmp_path: Path,
) -> None:
    base, flow, price = small_manifest.candidates
    # Native self is allowed, but the explicit background override makes this a split recipe.
    price = replace(price, implementation={"background_candidate_id": "base"})
    changed = replace(small_manifest, candidates=(base, flow, price))
    with pytest.raises(ValueError, match="same separated background"):
        Runner(changed).fit_candidate("price", "repeat-1", tmp_path)
    price = replace(price, parent_id="flow")
    changed = replace(small_manifest, candidates=(base, flow, price))
    fit_runner = Runner(changed)
    parent = fit_runner.fit_candidate("flow", "repeat-1", tmp_path)
    child = fit_runner.fit_candidate("price", "repeat-1", tmp_path)
    assert child.background_weights_ref == parent.background_weights_ref
    assert child.background_weights_ref is not None


def test_fit_binds_ancestors_but_rejects_orphans(small_manifest: ExperimentManifest, tmp_path: Path) -> None:
    base, flow, price = small_manifest.candidates
    hidden = replace(base, candidate_id="hidden", backbone="ssm", parent_id="ancestor")
    ancestor = replace(base, candidate_id="ancestor")
    flow = replace(flow, implementation={"background_candidate_id": hidden.candidate_id})
    orphan = replace(base, candidate_id="orphan")
    manifest = replace(small_manifest, candidates=(base, flow, price, hidden, ancestor, orphan))
    fit_runner = Runner(manifest)
    with pytest.raises(ValueError, match="not bound to a scenario"):
        fit_runner.fit_candidate("orphan", "repeat-1", tmp_path)
    for candidate_id in ("flow", "hidden", "ancestor"):
        result = fit_runner.fit_candidate(candidate_id, "repeat-1", tmp_path)
        registration = json.loads(result.config["registered"])
        assert registration["training_world"]["name"] == WORLDS["informed"].name


def test_fit_rejects_candidate_shared_across_lob_and_tier0(
    small_manifest: ExperimentManifest, tmp_path: Path,
) -> None:
    scene = replace(small_manifest.scenarios[0], scenario_id="lob", environment_kind="lob", matching_version="jaxlob==0.19")
    analysis = replace(small_manifest.analysis[0], analysis_id="lob-review", scenario_id="lob")
    mixed = replace(
        small_manifest, scenarios=(*small_manifest.scenarios, scene), analysis=(*small_manifest.analysis, analysis),
        prediction_candidate_set={**small_manifest.prediction_candidate_set, "lob": ("flow", "price")},
    )
    with pytest.raises(ValueError, match="incompatible environment"):
        Runner(mixed).fit_candidate("base", "repeat-1", tmp_path)


def test_cached_fit_requires_distinct_price_weights(small_manifest: ExperimentManifest, tmp_path: Path) -> None:
    fit_runner = Runner(small_manifest)
    fit = fit_runner.fit_candidate("base", "repeat-1", tmp_path)
    cache = Path(fit.model_ref).with_name("fit.json")
    cache.write_text(json.dumps(replace(fit, price_weights_ref=str(tmp_path / "missing-price.pt")).to_dict()))
    assert Path(fit.model_ref).is_file()
    with pytest.raises(FileNotFoundError, match="missing-price"):
        fit_runner.fit_candidate("base", "repeat-1", tmp_path)


@pytest.mark.parametrize(("offset", "accepted"), [(5e-10, True), (2e-9, False)])
def test_measurement_tolerates_roundoff_in_background_scale(
    small_manifest: ExperimentManifest, monkeypatch: pytest.MonkeyPatch, offset: float, accepted: bool,
) -> None:
    def load(path: str, device: str) -> argparse.Namespace:
        scale = 1.0 if path == "price" else 1.0 + offset
        return argparse.Namespace(spec=argparse.Namespace(own_flag=False, return_scale=scale))

    def measure(*args: Any, **kwargs: Any) -> dict[str, bool]:
        return {"measured": True}

    monkeypatch.setattr(runner, "_load_model", load)
    monkeypatch.setattr(runner, "evaluate_model", measure)
    fit = FitRecord(
        fit_id="flow:repeat-1", candidate_id="flow", parent_id="base", train_repeat_id="repeat-1",
        train_data_ref="train", config={"window": 8}, seed=11, model_ref="price", background_weights_ref="background",
    )
    scenario = small_manifest.scenarios[0]
    fit_runner = Runner(small_manifest)
    if accepted:
        assert fit_runner._measure_tier0(scenario, WORLDS["informed"], [], fit, ()) == {"measured": True}
    else:
        with pytest.raises(ValueError, match="same training scale"):
            fit_runner._measure_tier0(scenario, WORLDS["informed"], [], fit, ())


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
