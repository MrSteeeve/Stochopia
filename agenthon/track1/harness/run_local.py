"""Run the agent on public units and score each run with the unit's own checks.

    sudo -E python -m harness.run_local --units-dir <track1-coding-public>/units \
        --split dev --jobs 4 --name baseline

Model access comes from MODEL_ENDPOINT / MODEL_NAME / MODEL_TOKEN in this
process's environment; every T1A_* variable is passed through as well.
Results land in runs/<name>/: one directory per unit (input copy, agent
output, agent logs, scored copy, result.json) plus results.jsonl and
summary.md. Finished units are skipped unless --force.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .sandbox import Backend, ensure_mountpoints
from .splits import card, group, load_splits

ROOT = Path(__file__).resolve().parent.parent
_SCHEMA_WORDS = ("exist", "missing", "column", "schema", "key", "structure", "row_count", "rows", "format", "file", "shape", "length")


def _failed_tests(report_path: Path) -> list[dict]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    failed = []
    for t in report.get("tests", []):
        if t.get("outcome") in {"failed", "error"}:
            msg = ""
            for phase in ("setup", "call", "teardown"):
                crash = (t.get(phase) or {}).get("crash") or {}
                if crash.get("message"):
                    msg = crash["message"]
                    break
            failed.append({"test": t.get("nodeid", "").split("::", 1)[-1], "message": msg.strip()[:300]})
    return failed


def classify(res: dict) -> str:
    if res["passed"]:
        return "passed"
    if res["agent_timed_out"]:
        return "agent_timeout"
    status = res.get("agent_status")
    if status is None:
        return "agent_crash"
    if status in {"llm_error", "no_code"}:
        return status
    if not res["output_files"]:
        return "no_output"
    if res["reward"] is None:
        return "checker_error"
    text = " ".join(f"{f['test']} {f['message']}" for f in res["failed_tests"]).lower()
    if any(w in text for w in _SCHEMA_WORDS):
        return "missing_or_schema"
    return "wrong_values"


def run_unit(unit_dir: Path, run_root: Path, backend: Backend, env: dict[str, str], split: str, force: bool) -> dict:
    uid = unit_dir.name
    udir = run_root / uid
    result_path = udir / "result.json"
    if result_path.is_file() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    if udir.exists():
        shutil.rmtree(udir)
    input_dir, out_dir, logs_dir, check_dir = (udir / d for d in ("input", "output", "agent_logs", "check"))
    # The scoring platform does not mount checks/ into the agent's container.
    shutil.copytree(unit_dir, input_dir, ignore=shutil.ignore_patterns("checks", "__pycache__"))
    for d in (out_dir, logs_dir):
        d.mkdir(parents=True)
    c = card(unit_dir)
    agent_timeout = float(c.get("agent", {}).get("timeout_sec") or 1800.0)
    verifier_timeout = float(c.get("verifier", {}).get("timeout_sec") or 300.0)

    agent = backend.run_agent(input_dir, out_dir, logs_dir, env, agent_timeout + 60.0)
    (udir / "agent_stdout.txt").write_text(agent.stdout + "\n--- stderr ---\n" + agent.stderr, encoding="utf-8")

    # Score a copy, so output/ stays exactly what the agent produced.
    shutil.copytree(out_dir, check_dir)
    for stale in ("reward.json", "pytest_report.json"):
        (check_dir / stale).unlink(missing_ok=True)
    checker = backend.run_checker(unit_dir, check_dir, verifier_timeout + 60.0)
    (udir / "checker_stdout.txt").write_text(checker.stdout + "\n--- stderr ---\n" + checker.stderr, encoding="utf-8")

    reward = None
    reward_path = check_dir / "reward.json"
    if reward_path.is_file():
        try:
            data = json.loads(reward_path.read_text(encoding="utf-8"))
            if data.get("details") != "test runner not yet completed":
                reward = float(data.get("reward", 0.0))
        except (ValueError, TypeError):
            reward = None
    stats = {}
    stats_path = logs_dir / "agent_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    meta = c.get("metadata", {})
    res = {
        "unit": uid,
        "split": split,
        "category": meta.get("category", ""),
        "group": group(str(meta.get("category", "")), uid),
        "difficulty": meta.get("difficulty", ""),
        "passed": reward == 1.0,
        "reward": reward,
        "failed_tests": _failed_tests(check_dir / "pytest_report.json")[:25],
        "agent_status": stats.get("status"),
        "agent_returncode": agent.returncode,
        "agent_timed_out": agent.timed_out,
        "agent_seconds": round(agent.seconds, 1),
        "checker_seconds": round(checker.seconds, 1),
        "requests_used": stats.get("requests_used"),
        "runs": len(stats.get("attempts", [])),
        "llm_errors": stats.get("llm_errors", [])[:3],
        "output_files": sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file()),
    }
    res["failure_class"] = classify(res)
    result_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def _rate(rows: list[dict]) -> str:
    if not rows:
        return "-"
    k = sum(r["passed"] for r in rows)
    return f"{k}/{len(rows)} ({100.0 * k / len(rows):.1f}%)"


def summarize(results: list[dict], run_root: Path) -> str:
    lines = [f"# Run `{run_root.name}`", ""]
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_split[r["split"]].append(r)
    lines += ["## pass@1", "", "| split | passed |", "|---|---|"]
    for s in sorted(by_split):
        lines.append(f"| {s} | {_rate(by_split[s])} |")
    lines.append(f"| all | {_rate(results)} |")
    for title, key in (("Failure classes", "failure_class"), ("By group", "group"), ("By difficulty", "difficulty")):
        lines += ["", f"## {title}", ""]
        if key == "failure_class":
            counts = Counter(r[key] for r in results)
            lines += ["| class | units |", "|---|---|"] + [f"| {k} | {v} |" for k, v in counts.most_common()]
        else:
            buckets: dict[str, list[dict]] = defaultdict(list)
            for r in results:
                buckets[str(r[key])].append(r)
            lines += ["| value | passed |", "|---|---|"] + [f"| {k} | {_rate(v)} |" for k, v in sorted(buckets.items())]
    used = [r["requests_used"] for r in results if r.get("requests_used") is not None]
    secs = [r["agent_seconds"] for r in results]
    lines += [
        "",
        "## Cost",
        "",
        f"- mean requests per unit: {sum(used) / len(used):.1f}" if used else "- mean requests per unit: -",
        f"- mean agent seconds per unit: {sum(secs) / len(secs):.0f}" if secs else "",
        "",
        "## Units",
        "",
        "| unit | split | group | result | requests | runs | first failed check |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: (x["split"], x["unit"])):
        first = r["failed_tests"][0]["test"] if r["failed_tests"] else ""
        lines.append(
            f"| {r['unit']} | {r['split']} | {r['group']} | {r['failure_class']} | {r.get('requests_used')} | {r['runs']} | {first[:60]} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness.run_local")
    ap.add_argument("--units-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    ap.add_argument("--units", default="", help="comma-separated unit ids (overrides --split)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--name", default=time.strftime("run-%Y%m%d-%H%M%S"))
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--backend", choices=["ns", "docker"], default="ns")
    ap.add_argument("--python", default=sys.executable, help="ns backend: interpreter with the sandbox packages")
    ap.add_argument("--image", default="t1agent:dev", help="docker backend: agent image")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    splits = load_splits()
    split_of = {u: s for s, us in splits.items() for u in us}
    all_units = sorted(p for p in args.units_dir.iterdir() if (p / "instruction.md").is_file())
    if args.units:
        wanted = {u.strip() for u in args.units.split(",") if u.strip()}
        units = [u for u in all_units if u.name in wanted]
    elif args.split == "all":
        units = all_units
    else:
        units = [u for u in all_units if split_of.get(u.name) == args.split]
    if args.limit:
        units = units[: args.limit]
    if not units:
        print("no units selected (run `python -m harness.splits` first?)", file=sys.stderr)
        return 2

    if args.backend == "ns":
        ensure_mountpoints()
    backend = Backend(kind=args.backend, python=args.python, agent_image=args.image, agent_src=ROOT)
    env = {k: v for k, v in os.environ.items() if k in {"MODEL_ENDPOINT", "MODEL_NAME", "MODEL_TOKEN"} or k.startswith("T1A_")}
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        if k in os.environ and args.backend == "ns":
            env[k] = os.environ[k]
    if not env.get("MODEL_ENDPOINT"):
        print("warning: MODEL_ENDPOINT is not set; the agent will make no model calls", file=sys.stderr)

    run_root = args.runs_dir / args.name
    run_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(run_unit, u, run_root, backend, env, split_of.get(u.name, "unsplit"), args.force): u.name
            for u in units
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 - one broken unit must not stop the run
                print(f"{name}: harness error {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            results.append(res)
            print(f"[{len(results)}/{len(units)}] {name}: {res['failure_class']} (requests={res.get('requests_used')}, {res['agent_seconds']}s)", flush=True)

    with (run_root / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in sorted(results, key=lambda x: x["unit"]):
            fh.write(json.dumps(r) + "\n")
    md = summarize(results, run_root)
    (run_root / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md.split("## Units")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
