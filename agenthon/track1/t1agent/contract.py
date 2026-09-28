"""Deterministic extraction of the deliverables an instruction asks for.

Best effort by design: it finds output file names reliably, and top-level JSON
keys / table columns when the instruction shows them. The plan call can add
to it; the checker only treats missing files and missing keys as errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_OUT_FILE = re.compile(
    r"(?<![\w.])(/app/output|/output)/([\w\-./]+\.(?:json|csv|parquet|pqt|png|html|txt|md|xlsx|npy|pkl))\b"
)
_EXT = r"(?:json|csv|parquet|pqt|png|html|txt|md|xlsx|npy|pkl)"
_OUTPUT_MARKER = re.compile(r"^#{1,6}[^\n]*\b(outputs?|deliverables?)\b|/app/output\b|(?<![\w/])/output\b", re.I | re.M)
_HEADING_OR_ITEM = re.compile(r"^\s*(#{2,6}\s|\d+[.)]\s|[-*+]\s)")
_EMPHASISED_FILE = re.compile(rf"`/?([\w\-./]+\.{_EXT})`|(?:\*\*|\b)([\w\-]+\.{_EXT})\b")
_NOT_DELIVERABLES = {"reward.json", "pytest_report.json", "reward.txt"}
_FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_KEY = re.compile(r'"([^"\n]{1,80})"\s*:')
_TABLE_NAME = re.compile(r"^\|\s*`([^`|]+)`\s*\|", re.M)
_INLINE_COLUMNS = re.compile(r"columns?\s*[:=]\s*((?:`[^`]+`[,\s]*(?:and\s+)?)+)", re.IGNORECASE)


@dataclass
class Deliverable:
    name: str  # path relative to the output directory
    keys: list[str] = field(default_factory=list)  # top-level JSON keys
    columns: list[str] = field(default_factory=list)  # CSV / parquet columns

    @property
    def fmt(self) -> str:
        return self.name.rsplit(".", 1)[-1].lower()

    def to_dict(self) -> dict:
        return {"name": self.name, "keys": self.keys, "columns": self.columns}


def top_level_keys(block: str) -> list[str]:
    """Keys at depth 1 of a JSON-like block that may contain placeholders."""
    keys: list[str] = []
    depth = 0
    i = 0
    while i < len(block):
        ch = block[i]
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == '"':
            end = block.find('"', i + 1)
            if end < 0:
                break
            if depth == 1:
                m = _KEY.match(block, i)
                if m and m.group(1) not in keys:
                    keys.append(m.group(1))
            i = end
        i += 1
    return keys


def _named_in_output_region(instruction: str, input_names: set[str]) -> dict[str, int]:
    """File names announced as headings or list items once the text starts talking about outputs."""
    marker = _OUTPUT_MARKER.search(instruction)
    if not marker:
        return {}
    found: dict[str, int] = {}
    offset = instruction.rfind("\n", 0, marker.start()) + 1
    for line in instruction[offset:].splitlines(keepends=True):
        if _HEADING_OR_ITEM.match(line):
            for m in _EMPHASISED_FILE.finditer(line):
                name = (m.group(1) or m.group(2)).strip("/")
                base = name.rsplit("/", 1)[-1]
                if base in _NOT_DELIVERABLES or base in input_names:
                    continue
                if name.startswith(("app/output/", "output/")):
                    name = name.split("output/", 1)[1]
                if "/" in name:
                    continue
                found.setdefault(name, offset + m.start())
        offset += len(line)
    return found


def extract(instruction: str, input_names: set[str] | None = None) -> list[Deliverable]:
    input_names = input_names or set()
    found: dict[str, int] = {}
    for m in _OUT_FILE.finditer(instruction):
        name = m.group(2).strip("/")
        if name.rsplit("/", 1)[-1] not in _NOT_DELIVERABLES:
            found.setdefault(name, m.start())
    for name, pos in _named_in_output_region(instruction, input_names).items():
        if not any(k == name or k.endswith("/" + name) for k in found):
            found[name] = pos
    ordered = sorted(found.items(), key=lambda kv: kv[1])
    deliverables: list[Deliverable] = []
    for idx, (name, pos) in enumerate(ordered):
        # The spec for a file sits between its first mention and the next file's.
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(instruction)
        window = instruction[pos : max(end, pos + 200)][:6000]
        d = Deliverable(name=name)
        if d.fmt == "json":
            for fm in _FENCE.finditer(window):
                if fm.group(1).lower() in {"json", ""} and "{" in fm.group(2):
                    d.keys = top_level_keys(fm.group(2))
                    break
        elif d.fmt in {"csv", "parquet", "pqt", "xlsx"}:
            cols = _TABLE_NAME.findall(window)
            if not cols:
                im = _INLINE_COLUMNS.search(window)
                if im:
                    cols = re.findall(r"`([^`]+)`", im.group(1))
            d.columns = [c.strip() for c in cols if c.strip() and "/" not in c]
        deliverables.append(d)
    return deliverables


def merge(base: list[Deliverable], planned: list[dict]) -> list[Deliverable]:
    """Add files the plan names; fill keys/columns the regex could not see."""
    by_name = {d.name: d for d in base}
    for item in planned or []:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("path") or item.get("name") or "").strip()
        if not raw or "." not in raw.rsplit("/", 1)[-1]:
            continue
        name = raw
        for root in ("/app/output/", "/output/"):
            if name.startswith(root):
                name = name[len(root) :]
        if name.startswith("/") or name.rsplit("/", 1)[-1] in _NOT_DELIVERABLES:
            continue
        d = by_name.get(name)
        if d is None:
            d = Deliverable(name=name)
            by_name[name] = d
            base.append(d)
        keys = [str(k) for k in item.get("required_keys") or [] if isinstance(k, (str, int))]
        cols = [str(c) for c in item.get("required_columns") or [] if isinstance(c, (str, int))]
        if not d.keys and keys:
            d.keys = keys
        if not d.columns and cols:
            d.columns = cols
    return base
