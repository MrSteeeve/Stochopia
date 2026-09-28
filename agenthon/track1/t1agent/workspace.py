"""Stage the task inside the agent container.

At scoring time the task directory is mounted at /input, but instruction.md
was written against the unit's own image, where environment/Dockerfile copied
the data to paths such as /app/data/x.csv or /app/curve_data.json. We replay
those COPY lines as symlinks so the paths in the instruction resolve, and make
/app/output and /output point at the same output directory.
"""

from __future__ import annotations

import os
import re
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

OUTPUT_ROOTS = ("/app/output", "/output")
_CANARY_LINE = re.compile(r"canary|BENCHMARK DATA SHOULD NEVER APPEAR", re.IGNORECASE)
_GUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
_ABS_PATH = re.compile(r"(?<![\w.])(/(?:app|input|data)/[\w./-]*[\w-])")
_SKIP_DIRS = {"checks", "tests", "solution", "__pycache__"}


@dataclass
class Workspace:
    task_dir: Path
    out_dir: Path
    instruction: str  # canary lines removed
    card: dict
    timeout_s: float
    data_files: list[Path]
    path_map: dict[str, Path]  # absolute path named by the task -> file under task_dir
    canaries: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def task_id(self) -> str:
        return str(self.card.get("task", {}).get("id") or self.task_dir.name)

    @property
    def category(self) -> str:
        return str(self.card.get("metadata", {}).get("category") or "")


def clean_instruction(text: str) -> tuple[str, list[str]]:
    """Drop the canary banner lines and return the GUIDs they carried.

    A canary GUID in an output file fails gate g2, so the model never sees it.
    """
    kept, canaries = [], []
    for line in text.splitlines():
        if _CANARY_LINE.search(line):
            canaries.extend(_GUID.findall(line))
            continue
        kept.append(line)
    return "\n".join(kept).strip() + "\n", canaries


def parse_copy_lines(dockerfile: Path) -> list[tuple[str, str]]:
    """Return (src relative to the Dockerfile dir, absolute dst) for COPY/ADD lines."""
    pairs: list[tuple[str, str]] = []
    if not dockerfile.is_file():
        return pairs
    workdir = "/"
    for raw in dockerfile.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(" ")
        head = head.upper()
        if head == "WORKDIR":
            workdir = rest.strip()
            continue
        if head not in {"COPY", "ADD"}:
            continue
        try:
            parts = [p for p in shlex.split(rest) if not p.startswith("--")]
        except ValueError:
            continue
        if len(parts) < 2:
            continue
        *srcs, dst = parts
        if not dst.startswith("/"):
            dst = workdir.rstrip("/") + "/" + dst
        for src in srcs:
            pairs.append((src, dst + ("/" if len(srcs) > 1 and not dst.endswith("/") else "")))
    return pairs


def build_path_map(task_dir: Path, instruction: str) -> tuple[dict[str, Path], list[str]]:
    env_dir = task_dir / "environment"
    mapping: dict[str, Path] = {}
    notes: list[str] = []
    for src, dst in parse_copy_lines(env_dir / "Dockerfile"):
        src_path = (env_dir / src).resolve()
        inside = src_path == env_dir.resolve() or env_dir.resolve() in src_path.parents
        if not src_path.exists() or not inside:
            continue
        if src_path.is_dir():
            for f in sorted(p for p in src_path.rglob("*") if p.is_file()):
                mapping[dst.rstrip("/") + "/" + f.relative_to(src_path).as_posix()] = f
        else:
            target = dst + src_path.name if dst.endswith("/") else dst
            mapping[target] = src_path
    # Paths the instruction names that no COPY line produced: match by file name.
    by_name: dict[str, list[Path]] = {}
    for f in list_data_files(task_dir):
        by_name.setdefault(f.name, []).append(f)
    for p in sorted(set(_ABS_PATH.findall(instruction))):
        if p in mapping or any(p == r or p.startswith(r + "/") for r in OUTPUT_ROOTS):
            continue
        if p.startswith("/input/"):
            continue
        name = p.rsplit("/", 1)[-1]
        if "." in name and len(by_name.get(name, [])) == 1:
            mapping[p] = by_name[name][0]
            notes.append(f"mapped {p} by file name")
    return mapping, notes


def list_data_files(task_dir: Path) -> list[Path]:
    files = []
    for f in sorted(task_dir.rglob("*")):
        rel = f.relative_to(task_dir).parts
        if not f.is_file() or (rel and rel[0] in _SKIP_DIRS):
            continue
        if f.name in {"instruction.md", "card.toml", "manifest.json", "Dockerfile"}:
            continue
        files.append(f)
    return files


def _link(dst: Path, src: Path, notes: list[str]) -> None:
    try:
        if dst.exists() or dst.is_symlink():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)
    except OSError as exc:
        notes.append(f"could not link {dst} -> {src}: {exc}")


def prepare(task_dir: Path, out_dir: Path, *, materialize: bool = True) -> Workspace:
    task_dir = task_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = (task_dir / "instruction.md").read_text(encoding="utf-8", errors="replace")
    instruction, canaries = clean_instruction(raw)
    card: dict = {}
    card_path = task_dir / "card.toml"
    if card_path.is_file():
        try:
            card = tomllib.loads(card_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            card = {}
        for line in card_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "canary" in line.lower():
                canaries.extend(_GUID.findall(line))
    timeout_s = float(card.get("agent", {}).get("timeout_sec") or 1800.0)
    path_map, notes = build_path_map(task_dir, instruction)
    ws = Workspace(
        task_dir=task_dir,
        out_dir=out_dir.resolve(),
        instruction=instruction,
        card=card,
        timeout_s=timeout_s,
        data_files=list_data_files(task_dir),
        path_map=path_map,
        canaries=sorted(set(canaries)),
        notes=notes,
    )
    if materialize:
        for dst, src in path_map.items():
            _link(Path(dst), src, ws.notes)
        for root in OUTPUT_ROOTS:
            root_path = Path(root)
            if root_path.resolve() != ws.out_dir and not root_path.exists():
                _link(root_path, ws.out_dir, ws.notes)
        os.makedirs(ws.out_dir, exist_ok=True)
    return ws
