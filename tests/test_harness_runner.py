import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest

from experiments.tier0_feasibility import WORLDS, evaluate_model, initialize_model
from stochopia.harness.contract import ReportStatus
from stochopia.harness.lob_adapter import export_lob_bench, run_short_lob
from stochopia.harness.runner import classify_diagnosis
from stochopia.sim.world import simulate_day


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
