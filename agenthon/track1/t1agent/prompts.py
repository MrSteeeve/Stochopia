"""Prompt construction. The only place that decides what the model sees."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .checks import Problem
from .contract import Deliverable
from .executor import RunResult
from .workspace import Workspace

PACKAGES = (
    "numpy, pandas, scipy, statsmodels, scikit-learn, arch, polars, pyarrow, numba, "
    "matplotlib, seaborn, plotly, openpyxl, backtrader"
)

SYSTEM = f"""You are a senior quantitative developer. You solve one quantitative-finance coding task by \
writing a single self-contained Python 3.13 script.

Hard rules:
- Follow the task's formulas, conventions, parameters, seeds and method choices exactly. When the task \
names a method (e.g. a specific approximation, estimator or ddof), use that method, not an alternative.
- Read inputs only from the paths given in the task. There is no network access.
- Write every deliverable to exactly the path and file name the task specifies, with exactly the \
requested keys/columns, units and ordering. Create output directories with os.makedirs(..., exist_ok=True).
- Output files must contain plain numbers: convert numpy types to Python float/int, never write NaN or \
Infinity into JSON.
- Never write reward.json, reward.txt or pytest_report.json.
- Available packages: {PACKAGES}. No other third-party packages.
- Print a short summary of the key numbers at the end so they can be sanity-checked.
- Reply with exactly one ```python fenced block containing the complete script, nothing else."""

PLAN_INSTRUCTIONS = """Before any code is written, analyse the task and return ONLY a JSON object with:
{
  "deliverables": [{"path": "/app/output/<file>", "required_keys": [...], "required_columns": [...]}],
  "method": "<the exact method/formulas the task requires, one paragraph>",
  "conventions": ["<units, day counts, annualisation, ddof, rounding, sign conventions, ordering ...>"],
  "pitfalls": ["<specific mistakes that would make a strict checker fail>"]
}
Use top-level keys only in required_keys. Leave a list empty when the task does not specify it."""


def _preview_file(path: Path, max_chars: int = 900) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv", ".txt", ".md"}:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = [next(fh, "") for _ in range(6)]
            n_lines = sum(1 for _ in path.open("rb"))
            return "".join(lines)[:max_chars] + f"... ({n_lines} lines total)"
        if suffix == ".json":
            text = path.read_text(encoding="utf-8", errors="replace")
            return text[:max_chars] + ("..." if len(text) > max_chars else "")
        if suffix in {".parquet", ".pqt"}:
            import pandas as pd

            df = pd.read_parquet(path)
            return f"shape={df.shape}\ndtypes: {dict(df.dtypes.astype(str))}\n{df.head(3).to_string()[:max_chars]}"
        if suffix in {".xlsx", ".xls"}:
            import pandas as pd

            sheets = pd.read_excel(path, sheet_name=None, nrows=3)
            return "\n".join(f"sheet {k}: columns {list(v.columns)[:20]}" for k, v in sheets.items())[:max_chars]
    except Exception as exc:  # noqa: BLE001
        return f"(preview failed: {type(exc).__name__})"
    return f"({path.stat().st_size} bytes)"


def data_listing(ws: Workspace, max_files: int = 25) -> str:
    reverse = {src: dst for dst, src in ws.path_map.items()}
    rows = []
    for f in ws.data_files[:max_files]:
        named = reverse.get(f)
        where = f"{named}  (also at {f})" if named else str(f)
        rows.append(f"### {where}\n{_preview_file(f)}")
    if len(ws.data_files) > max_files:
        rows.append(f"... and {len(ws.data_files) - max_files} more files")
    return "\n\n".join(rows) if rows else "(no data files)"


def _deliverables_text(ds: list[Deliverable]) -> str:
    if not ds:
        return "(could not be extracted automatically; follow the task text)"
    return "\n".join(f"- {json.dumps(d.to_dict())}" for d in ds)


def task_block(ws: Workspace, deliverables: list[Deliverable]) -> str:
    return (
        f"# TASK\n{ws.instruction}\n\n"
        f"# INPUT FILES (already present at these paths)\n{data_listing(ws)}\n\n"
        f"# OUTPUT DIRECTORY\n{ws.out_dir} (the paths /app/output and /output both point here)\n\n"
        f"# DELIVERABLES DETECTED\n{_deliverables_text(deliverables)}\n"
    )


def plan_messages(ws: Workspace, deliverables: list[Deliverable]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task_block(ws, deliverables) + "\n" + PLAN_INSTRUCTIONS},
    ]


def code_messages(ws: Workspace, deliverables: list[Deliverable], plan: dict | None) -> list[dict[str, str]]:
    plan_text = json.dumps(plan, indent=1, ensure_ascii=False)[:5000] if plan else "(none)"
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": task_block(ws, deliverables)
            + f"\n# ANALYSIS\n{plan_text}\n\nWrite the complete script now.",
        },
    ]


def _run_report(run: RunResult | None, problems: list[Problem]) -> str:
    parts = []
    if run is not None:
        status = "TIMED OUT" if run.timed_out else f"exit code {run.returncode}"
        parts.append(f"Run: {status} after {run.seconds:.1f}s")
        if run.stdout.strip():
            parts.append(f"stdout (tail):\n{run.stdout}")
        if run.stderr.strip():
            parts.append(f"stderr (tail):\n{run.stderr}")
    if problems:
        parts.append("Output checks:\n" + "\n".join(str(p) for p in problems))
    return "\n\n".join(parts)


def repair_messages(
    ws: Workspace,
    deliverables: list[Deliverable],
    plan: dict | None,
    code: str,
    run: RunResult | None,
    problems: list[Problem],
) -> list[dict[str, str]]:
    report = _run_report(run, problems) or "The previous reply did not contain a python code block."
    return code_messages(ws, deliverables, plan) + [
        {"role": "assistant", "content": f"```python\n{code}\n```" if code else "(no code)"},
        {
            "role": "user",
            "content": f"{report}\n\nFix the root cause. Re-read the task for anything the script got wrong. "
            "Reply with the complete corrected script in one ```python block.",
        },
    ]


def review_messages(
    ws: Workspace,
    deliverables: list[Deliverable],
    plan: dict | None,
    code: str,
    run: RunResult,
    problems: list[Problem],
) -> list[dict[str, str]]:
    return code_messages(ws, deliverables, plan) + [
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        {
            "role": "user",
            "content": f"{_run_report(run, problems)}\n\nThe script ran. Review it line by line against the task: "
            "formulas, conventions, units, file names, keys, columns, ordering, and the warnings above. "
            "If everything is correct reply with exactly OK. Otherwise reply with the complete corrected "
            "script in one ```python block.",
        },
    ]


_PY_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = _PY_BLOCK.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    open_block = re.search(r"```(?:python|py)?\s*\n(.*)$", text or "", re.DOTALL)
    if open_block and "import" in open_block.group(1):
        return open_block.group(1).strip() + "\n"  # truncated reply: try it anyway
    return ""


def extract_json(text: str) -> dict | None:
    text = text or ""
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None
