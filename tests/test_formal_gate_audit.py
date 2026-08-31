"""Synthetic acceptance tests for the aggregate A-F gate evaluator."""

from __future__ import annotations

from stochopia.formal_gate_audit import evaluate_gates


def _passing_inputs():
    snapshots, provenance = [], []
    for underlying in ("CSI500", "CSI1000"):
        for episode in range(6):
            for round_num in range(1, 7):
                has_iv = not (underlying == "CSI500" and episode == 0 and round_num <= 3)
                snapshots.append({
                    "episode_id": f"{underlying}_{episode}", "underlying": underlying,
                    "atm_iv_1m": "0.2" if has_iv else "", "atm_iv_3m": "", "atm_iv_6m": "",
                    "source": ("sse_510500_etf_iv_proxy" if underlying == "CSI500" else "cffex_mo_index_iv") if has_iv else "rv_fallback",
                })
                provenance.append({
                    "episode_id": f"{underlying}_{episode}", "as_of": "2025-01-31", "max_input_date": "2025-01-31",
                    "raw_manifest_hashes": {"tushare":"x", "cffex":"y"},
                    "rv_counts": {"return_20d":20,"rv20":20,"rv60":60,"drawdown_levels":126},
                    "option_iv": {"selected_versions": ["M"] if has_iv else []},
                })
    expiry = [{"dte":"91","forward_ratio":"1","mad_ratio":"0","range_ratio":"0","implied_vol":"0.2","price_field":"settlement","selected_version":"M","contracts":"C|P"}]
    return snapshots, provenance, expiry


def test_all_formal_gates_pass_for_frozen_synthetic_evidence():
    snapshots, provenance, expiry = _passing_inputs()
    result = evaluate_gates(
        raw_report={"gate_a":{"ready":True},"tushare":{"options":{"exact_whitelist_contracts":1614,"invalid_master_contracts":0,"observed_active_join_failures":0,"raw_prerequisite_ready":True,"formal_whitelist_status":"ready"}}},
        full_gate={"checks":{"pricing_volatility_or_rv_fallback":True,"selected_adjusted_contracts_zero":True},"iv_coverage":{"any_iv":69,"total":72}},
        snapshots=snapshots, provenance=provenance, expiry_points=expiry,
        license_decision={"status":"user_confirmed_for_research","confirmed_date":"2026-07-13","raw_redistribution":False},
        derived_manifest_valid=True, deterministic_rebuild=True,
        gitignore_text="data/raw/cffex/\ndata/raw/tushare/\n.env\n",
    )
    assert result["ready"] is True
    assert {letter: gate["status"] for letter, gate in result["gates"].items()} == {letter:"PASS" for letter in "ABCDEF"}


def test_gate_d_fails_on_active_join_failure():
    snapshots, provenance, expiry = _passing_inputs()
    result = evaluate_gates(
        raw_report={"gate_a":{"ready":True},"tushare":{"options":{"exact_whitelist_contracts":1614,"invalid_master_contracts":0,"observed_active_join_failures":1,"raw_prerequisite_ready":True,"formal_whitelist_status":"ready"}}},
        full_gate={"checks":{"pricing_volatility_or_rv_fallback":True,"selected_adjusted_contracts_zero":True},"iv_coverage":{"any_iv":69,"total":72}},
        snapshots=snapshots, provenance=provenance, expiry_points=expiry,
        license_decision={"status":"user_confirmed_for_research","confirmed_date":"2026-07-13","raw_redistribution":False},
        derived_manifest_valid=True, deterministic_rebuild=True,
        gitignore_text="data/raw/cffex/\ndata/raw/tushare/\n.env\n",
    )
    assert result["gates"]["D"]["status"] == "FAIL"
