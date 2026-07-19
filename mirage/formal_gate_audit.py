"""Aggregate the formal local-data acceptance gates A through F."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .formal_market_builder import _atomic_write, run_base_build


DEFAULT_RAW_REPORT = Path("data/audit/raw_coverage_report.json")
DEFAULT_DERIVED = Path("data/derived")
DEFAULT_LICENSE = Path("data/reference/license_decision.json")
DEFAULT_OUTPUT = DEFAULT_DERIVED / "formal_gate_report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(root: Path, manifest_path: Path) -> bool:
    if not manifest_path.is_file():
        return False
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            return False
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            return False
    return True


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate_gates(
    *,
    raw_report: Mapping[str, Any], full_gate: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, str]], provenance: Sequence[Mapping[str, Any]],
    expiry_points: Sequence[Mapping[str, str]], license_decision: Mapping[str, Any],
    derived_manifest_valid: bool, deterministic_rebuild: bool, gitignore_text: str,
) -> dict[str, Any]:
    episodes = Counter(row["episode_id"] for row in snapshots)
    raw_options = raw_report.get("tushare", {}).get("options", {})
    any_iv = sum(any(row.get(field, "") for field in ("atm_iv_1m", "atm_iv_3m", "atm_iv_6m")) for row in snapshots)
    rv_fallback = sum("rv_fallback" in row.get("source", "") for row in snapshots)
    full_iv = full_gate.get("iv_coverage", {})
    source_valid = all(
        (
            any(row.get(field, "") for field in ("atm_iv_1m", "atm_iv_3m", "atm_iv_6m"))
            and ("sse_510500_etf_iv_proxy" if row["underlying"] == "CSI500" else "cffex_mo_index_iv") in row.get("source", "")
        )
        or (
            not any(row.get(field, "") for field in ("atm_iv_1m", "atm_iv_3m", "atm_iv_6m"))
            and "rv_fallback" in row.get("source", "")
        )
        for row in snapshots
    )
    gate_a_checks = {
        "rows_72": len(snapshots) == 72,
        "episodes_12_x_6": len(episodes) == 12 and set(episodes.values()) == {6},
        "provenance_72": len(provenance) == 72,
        "raw_gate_ready": raw_report.get("gate_a", {}).get("ready") is True,
        "raw_manifest_hashes_recorded": all(len(row.get("raw_manifest_hashes", {})) == 2 for row in provenance),
        "derived_manifest_valid": derived_manifest_valid,
    }
    gate_b_checks = {
        "max_input_not_after_asof": all(row.get("max_input_date", "9999") <= row.get("as_of", "") for row in provenance),
        "strict_rv_counts": all(
            row.get("rv_counts") == {"return_20d": 20, "rv20": 20, "rv60": 60, "drawdown_levels": 126}
            for row in provenance
        ),
        "consecutive_rebuild_deterministic": deterministic_rebuild,
    }
    gate_c_checks = {
        "every_row_iv_or_explicit_rv_fallback": all(
            any(row.get(field, "") for field in ("atm_iv_1m", "atm_iv_3m", "atm_iv_6m"))
            or "rv_fallback" in row.get("source", "")
            for row in snapshots
        ),
        "counts_match_full_gate": (
            full_iv.get("any_iv") == any_iv
            and full_iv.get("total") == len(snapshots)
            and len(snapshots) - any_iv == rv_fallback
        ),
        "source_labels_valid": source_valid,
        "pricing_fallback_valid": full_gate.get("checks", {}).get("pricing_volatility_or_rv_fallback") is True,
    }
    gate_d_checks = {
        "exact_whitelist_nonempty": raw_options.get("exact_whitelist_contracts", 0) > 0,
        "raw_whitelist_ready": (
            raw_options.get("raw_prerequisite_ready") is True
            and raw_options.get("formal_whitelist_status") == "ready"
        ),
        "invalid_master_zero": raw_options.get("invalid_master_contracts") == 0,
        "active_join_failures_zero": raw_options.get("observed_active_join_failures") == 0,
        "selected_m_only": all(set(row.get("option_iv", {}).get("selected_versions", [])) <= {"M"} for row in provenance),
        "selected_adjusted_zero": full_gate.get("checks", {}).get("selected_adjusted_contracts_zero") is True,
    }
    gate_e_checks = {
        "points_present": bool(expiry_points),
        "dte_7_365": all(7 <= int(row["dte"]) <= 365 for row in expiry_points),
        "forward_ratio": all(0.75 <= float(row["forward_ratio"]) <= 1.25 for row in expiry_points),
        "mad_ratio": all(float(row["mad_ratio"]) <= 0.01 for row in expiry_points),
        "range_ratio": all(float(row["range_ratio"]) <= 0.03 for row in expiry_points),
        "iv_range": all(0.01 <= float(row["implied_vol"]) <= 1.5 for row in expiry_points),
        "settlement_m_contracts": all(
            row.get("price_field") == "settlement" and row.get("selected_version") == "M" and bool(row.get("contracts"))
            for row in expiry_points
        ),
    }
    gate_f_checks = {
        "research_authorization_confirmed": license_decision.get("status") == "user_confirmed_for_research",
        "confirmation_date_frozen": license_decision.get("confirmed_date") == "2026-07-13",
        "raw_redistribution_disabled": license_decision.get("raw_redistribution") is False,
        "raw_and_env_gitignored": all(
            token in {line.strip() for line in gitignore_text.splitlines() if line.strip() and not line.lstrip().startswith("#")}
            for token in ("data/raw/cffex/", "data/raw/tushare/", ".env")
        ),
        "release_scope_derived_code_hash_only": license_decision.get("raw_redistribution") is False,
    }
    gates = {}
    for letter, title, checks in (
        ("A", "formal panel and raw completeness", gate_a_checks),
        ("B", "leakage safety and deterministic rebuild", gate_b_checks),
        ("C", "volatility coverage and fallback", gate_c_checks),
        ("D", "contract whitelist and adjustment control", gate_d_checks),
        ("E", "parity and expiry-point quality", gate_e_checks),
        ("F", "research authorization and release boundary", gate_f_checks),
    ):
        gates[letter] = {"title": title, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    return {
        "ready": all(gate["status"] == "PASS" for gate in gates.values()),
        "gates": gates,
        "summary": {
            "rows": len(snapshots), "episodes": len(episodes), "any_iv": any_iv,
            "rv_fallback": rv_fallback, "expiry_points": len(expiry_points),
        },
    }


def run_audit(
    *, raw_report_path: Path = DEFAULT_RAW_REPORT, derived_root: Path = DEFAULT_DERIVED,
    license_path: Path = DEFAULT_LICENSE, output_path: Path = DEFAULT_OUTPUT,
    rebuild: bool = True,
) -> dict[str, Any]:
    deterministic = False
    if rebuild:
        run_base_build(output_root=derived_root, phase="full")
        first = tuple(_sha256(derived_root / name) for name in ("market_snapshots.csv", "market_snapshot_provenance.jsonl"))
        run_base_build(output_root=derived_root, phase="full")
        second = tuple(_sha256(derived_root / name) for name in ("market_snapshots.csv", "market_snapshot_provenance.jsonl"))
        deterministic = first == second
    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    full_gate = json.loads((derived_root / "gate_report.json").read_text(encoding="utf-8"))
    snapshots = _read_csv(derived_root / "market_snapshots.csv")
    provenance = [json.loads(line) for line in (derived_root / "market_snapshot_provenance.jsonl").read_text(encoding="utf-8").splitlines()]
    expiry_points = _read_csv(derived_root / "option_expiry_points.csv")
    license_decision = json.loads(license_path.read_text(encoding="utf-8"))
    result = evaluate_gates(
        raw_report=raw_report, full_gate=full_gate, snapshots=snapshots,
        provenance=provenance, expiry_points=expiry_points,
        license_decision=license_decision,
        derived_manifest_valid=verify_manifest(derived_root, derived_root / "manifest.sha256"),
        deterministic_rebuild=deterministic,
        gitignore_text=Path(".gitignore").read_text(encoding="utf-8"),
    )
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(output_path, (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit formal MIRAGE gates A-F locally")
    parser.add_argument("--no-rebuild", action="store_true")
    args = parser.parse_args(argv)
    report = run_audit(rebuild=not args.no_rebuild)
    print(json.dumps({"ready": report["ready"], "gates": {key: value["status"] for key, value in report["gates"].items()}, **report["summary"]}))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
