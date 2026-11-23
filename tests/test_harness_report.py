from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from stochopia.harness.contract import FrozenManifest
from stochopia.harness.report import build_report

from test_harness_contract import frozen, manifest, prediction  # noqa: F401
from test_harness_p0_acceptance import _eval_golden, golden  # noqa: F401


@pytest.fixture
def golden_report() -> Callable[[FrozenManifest], dict[str, Any]]:
    def evaluate(batch: FrozenManifest) -> dict[str, Any]:
        fits = {fit.fit_id: fit for fit in batch.fits}
        results = []
        for task in batch.tasks:
            parent, candidate = fits[task.parent_fit_id], fits[task.candidate_fit_id]
            results.append(_eval_golden(task, parent, candidate, batch))
        return build_report(batch, results)
    return evaluate


@pytest.mark.parametrize("llm_count", [1, 3])
def test_all_registered_llm_groups_share_marked_predictions(
    golden: FrozenManifest, golden_report: Callable[[FrozenManifest], dict[str, Any]], llm_count: int,
) -> None:
    if llm_count == 1:
        experiment = replace(golden.experiment, groups=tuple(group for group in golden.experiment.groups
                                                             if group.group_id != "metrics"))
        batch = replace(golden, experiment=experiment,
                        selections=tuple(row for row in golden.selections if row.group_id != "metrics"),
                        predictions=tuple(row for row in golden.predictions if row.group_id != "metrics"),
                        search_trials=tuple(row for row in golden.search_trials if row.group_id != "metrics"),
                        final_choices=tuple(row for row in golden.final_choices if row.group_id != "metrics"))
        expected_groups = {"evidence"}
    else:
        metrics = next(group for group in golden.experiment.groups if group.group_id == "metrics")
        experiment = replace(golden.experiment, groups=(*golden.experiment.groups, replace(metrics, group_id="extra")))
        first_choice = next(row for row in golden.selections if row.group_id == "metrics" and row.round_index == 1)
        choice = replace(first_choice, group_id="extra", selection_id="extra-r1")
        metrics_predictions = [row for row in golden.predictions if row.group_id == "metrics"]
        forecasts = tuple(replace(row, group_id="extra", prediction_id=f"extra-{row.prediction_id}")
                          for row in metrics_predictions)
        final = next(row for row in golden.final_choices if row.group_id == "metrics")
        final = replace(final, group_id="extra", selection_id=choice.selection_id, candidate_id=choice.candidate_id)
        batch = replace(golden, experiment=experiment, selections=(*golden.selections, choice),
                        predictions=(*golden.predictions, *forecasts), final_choices=(*golden.final_choices, final))
        expected_groups = {"evidence", "metrics", "extra"}

    result = golden_report(batch)
    assert set(result["x_to_delta_y"]) == expected_groups
    assert {group["n_common_forecasts"] for group in result["x_to_delta_y"].values()} == {2}
    assert len(result["predictions"]) == 2 * llm_count
    assert all(row["common_comparison"] for row in result["predictions"])


@pytest.mark.parametrize("score_direction,best_group", [("decrease", "evidence"), ("increase", "metrics")])
def test_common_menu_gap_is_oriented_for_higher_and_lower_better(
    golden: FrozenManifest, golden_report: Callable[[FrozenManifest], dict[str, Any]],
    score_direction: str, best_group: str,
) -> None:
    primary, forecast = golden.experiment.analysis
    if score_direction == "increase":
        primary = replace(primary, acceptance={"min_delta": 0.0})
    selection = next(row for row in golden.selections if row.group_id == "metrics" and row.round_index == 1)
    choices = tuple(replace(row, selection_id=selection.selection_id, candidate_id=selection.candidate_id)
                    if row.group_id == "metrics" else row for row in golden.final_choices)
    batch = replace(golden, experiment=replace(golden.experiment, analysis=(primary, forecast)), final_choices=choices)
    result = golden_report(batch)
    gaps = {row["group_id"]: row["common_menu_effect_gap"] for row in result["choices"]}
    other_group = "metrics" if best_group == "evidence" else "evidence"
    assert gaps[best_group] == 0.0
    assert gaps[other_group] == pytest.approx(0.8465654651614463)


@pytest.mark.parametrize("action", ["select", "keep_parent"])
def test_outperforming_common_menu_never_yields_negative_shortfall(
    golden: FrozenManifest, golden_report: Callable[[FrozenManifest], dict[str, Any]], action: str,
) -> None:
    scene = golden.experiment.scenarios[0]
    chosen_candidate = "price"
    common_candidate = "flow"
    if action == "keep_parent":
        first, forecast = golden.experiment.analysis
        primary = replace(first, acceptance={**first.acceptance, "score_direction": "increase"})
        golden = replace(golden, experiment=replace(golden.experiment, analysis=(primary, forecast)))
        chosen_candidate = scene.parent_id
        common_candidate = "price"
        initial = next(row for row in golden.selections if row.group_id == "evidence")
        choices = tuple(replace(row, action="keep_parent", candidate_id=None)
                        if row.selection_id == initial.selection_id else row for row in golden.selections)
        finals = tuple(replace(row, candidate_id=chosen_candidate) if row.group_id == "evidence" else row
                       for row in golden.final_choices)
        trials = tuple(row for row in golden.search_trials if row.selection_id != initial.selection_id)
        golden = replace(golden, selections=choices, final_choices=finals, search_trials=trials)
    experiment = replace(golden.experiment, prediction_candidate_set={scene.scenario_id: (common_candidate,)})
    result = golden_report(replace(golden, experiment=experiment))
    row = next(row for row in result["choices"] if row["group_id"] == "evidence")
    assert row["final_candidate"] == chosen_candidate
    assert row["final_model_gain"] > row["common_menu_best_gain"]
    assert row["common_menu_effect_gap"] == 0.0
