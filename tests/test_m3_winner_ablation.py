"""Zero-training regression tests for the exact M3 winner ablation."""

from pathlib import Path

import pytest
import torch

from experiments.synthetic_tau_z.ablate_m3_winner import (
    B0_SHA256, METRICS, SEEDS, VARIANTS, WINNER_SHA256,
    build_variants, data_protocol, digest, load_factory, paired_contrast,
    pool_training_seeds, run,
)
from stochopia.neural.lob_interface import verify_world_model_contract

FIXTURES = Path(__file__).parent / "fixtures" / "m3_ablation"


def _sources() -> tuple[bytes, bytes]:
    return (FIXTURES / "B0.txt").read_bytes(), (FIXTURES / "c06.txt").read_bytes()


def test_exact_factor_edits_and_capacity() -> None:
    b0, winner = _sources()
    variants = build_variants(b0, winner)
    assert tuple(variants) == VARIANTS
    assert digest(variants["A00"]) == B0_SHA256
    assert digest(variants["A11"]) == WINNER_SHA256
    assert variants["A10"] == winner.replace(
        b"self.measure(x).tanh() - self.predict(prior).tanh()", b"self.measure(x).tanh()"
    )
    assert b"self.clock" not in variants["A01"]
    assert b"torch.full((8,), 0.0707)" in variants["A01"]
    assert b"pace = 1.0" in variants["A01"]
    assert b"self.measure(x).tanh() - self.predict(prior).tanh()" in variants["A01"]
    expected = {"A00": 24_244, "A10": 23_029, "A01": 22_996, "A11": 23_029}
    for name, source in variants.items():
        torch.manual_seed(SEEDS[0])
        model = load_factory(source)()
        assert verify_world_model_contract(model, max_state_dim=60) == (expected[name], 60)


def test_unverified_source_is_not_used() -> None:
    b0, winner = _sources()
    with pytest.raises(ValueError, match="fingerprint"):
        build_variants(b0, winner + b"\n")


def _panel(values: list[float | None]) -> list[dict]:
    return [
        {"trading_date": day + 20300129, **{metric: value for metric in METRICS}}
        for day, value in enumerate(values)
    ]


def test_four_way_interaction_is_paired_not_difference_of_unpaired_means() -> None:
    # The same day effect cancels within the contrast; one missing arm removes
    # only that physical day from the four-way interaction.
    panels = {
        "A00": _panel([float(day) + 10 for day in range(12)]),
        "A10": _panel([float(day) + 8 for day in range(12)]),
        "A01": _panel([float(day) + 9 for day in range(12)]),
        "A11": _panel([None] + [float(day) + 6 for day in range(1, 12)]),
    }
    synergy = paired_contrast(panels, {"A11": 1, "A10": -1, "A01": -1, "A00": 1})
    for metric in METRICS:
        assert synergy[metric]["estimate"] == -1.0
        assert synergy[metric]["interval"] == [-1.0, -1.0]
        assert synergy[metric]["paired_days"] == 11
        assert synergy[metric]["daily"] == [None] + [-1.0] * 11
    panels["A11"] = _panel([6.0] * 5 + [None] * 7)
    metric = paired_contrast(panels, {"A11": 1, "A10": -1, "A01": -1, "A00": 1})["val_nll_total"]
    assert metric["paired_days"] == 5
    assert metric["interval"] is None


def test_seed_pooling_and_calendar_protection() -> None:
    seed_daily = {
        str(SEEDS[0]): _panel([1.] * 12),
        str(SEEDS[1]): _panel([2.] * 12),
        str(SEEDS[2]): _panel([None] + [3.] * 11),
    }
    pooled = pool_training_seeds(seed_daily)
    assert pooled[0]["val_nll_total"] is None
    assert pooled[1]["val_nll_total"] == 2.
    seed_daily[str(SEEDS[-1])][0]["trading_date"] += 1
    with pytest.raises(ValueError, match="calendar"):
        pool_training_seeds(seed_daily)
    with pytest.raises(ValueError, match="calendar"):
        paired_contrast({"A00": pooled, "A01": seed_daily[str(SEEDS[-1])]}, {"A01": 1, "A00": -1})


def test_registered_protocol_and_immutable_output_path(tmp_path: Path) -> None:
    protocol = data_protocol()
    assert protocol["train_days"] == 16
    assert protocol["holdout_first_day"] == 28
    assert protocol["holdout_days"] == 12
    assert protocol["model_k"] == 4
    assert protocol["truth_k"] == 16
    assert protocol["training_seeds"] == [103, 137, 211]
    assert protocol["bootstrap_draws"] == 2000
    frozen = tmp_path / "m3"
    frozen.mkdir()
    with pytest.raises(ValueError, match="outside"):
        run(frozen, frozen / "ablation")
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "receipt.json").write_text("{}")
    with pytest.raises(ValueError, match="fresh"):
        run(frozen, existing)
