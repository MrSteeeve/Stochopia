"""Reproduce the scoring mount layout for one agent run and one checker run.

Two backends:

* ``ns``: Linux mount namespaces (``unshare -m``), no Docker daemon needed.
  Needs root. Each run gets a private tmpfs over /app and /tmp and bind mounts
  for /input, /app/output, /output and a log directory, so runs are isolated
  and can execute in parallel. The network is NOT cut (the agent must reach
  the model endpoint), unlike the real sandbox.
* ``docker``: the documented ``docker run`` invocations; the checker runs in
  the unit's own image (needs ``finance-bench-sandbox:latest`` built first).

Layout, as the scoring harness uses it:
  agent:   task dir (without checks/) -> /input:ro ; out -> /app/output and /output
  checker: full unit -> /input:ro ; out -> /app/output and /output ;
           unit/checks -> /tests:ro (the reference-data verifiers read /tests) ;
           the unit's data at its image paths (/app/data/..., /app/x.csv), because
           several checks re-read the inputs from there
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

MOUNTPOINTS = ("/app", "/input", "/output", "/tests", "/agentlogs", "/logs/verifier")


@dataclass
class Proc:
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool


def _run(cmd: list[str], timeout_s: float, env: dict[str, str] | None = None) -> Proc:
    t0 = time.monotonic()
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
        return Proc(cp.returncode, cp.stdout[-20000:], cp.stderr[-20000:], time.monotonic() - t0, False)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return Proc(-9, out[-20000:], err[-20000:], time.monotonic() - t0, True)


def ensure_mountpoints() -> None:
    if os.geteuid() != 0:
        raise SystemExit("the ns backend needs root (mount namespaces); use --backend docker instead")
    for mp in MOUNTPOINTS:
        Path(mp).mkdir(parents=True, exist_ok=True)


def unit_data_links(unit_dir: Path) -> list[tuple[str, Path]]:
    """(path in the unit image, source file) for every COPY line of the unit's Dockerfile."""
    from t1agent.workspace import parse_copy_lines

    env_dir = unit_dir / "environment"
    links: list[tuple[str, Path]] = []
    for src, dst in parse_copy_lines(env_dir / "Dockerfile"):
        src_path = (env_dir / src).resolve()
        if not src_path.exists():
            continue
        if src_path.is_dir():
            for f in sorted(p for p in src_path.rglob("*") if p.is_file()):
                links.append((dst.rstrip("/") + "/" + f.relative_to(src_path).as_posix(), f))
        else:
            links.append((dst + src_path.name if dst.endswith("/") else dst, src_path))
    return links


def _ns_script(
    binds: list[tuple[Path, str, bool]],
    command: list[str],
    env: dict[str, str],
    links: list[tuple[str, Path]] = (),
) -> str:
    lines = ["set -e", "mount -t tmpfs tmpfs /app", "mkdir -p /app/output"]
    for src, dst, ro in binds:
        lines.append(f"mkdir -p {shlex.quote(dst)}")
        lines.append(f"mount --bind {shlex.quote(str(src))} {shlex.quote(dst)}")
        if ro:
            lines.append(f"mount -o remount,bind,ro {shlex.quote(dst)}")
    for dst, src in links:
        if dst == "/app" or dst.startswith(("/app/output", "/output", "/input", "/tests")):
            continue
        lines.append(f"mkdir -p {shlex.quote(str(Path(dst).parent))}")
        lines.append(f"ln -sfn {shlex.quote(str(src))} {shlex.quote(dst)}")
    # Last, so the bind sources above could still live under /tmp.
    lines.append("mount -t tmpfs tmpfs /tmp")
    lines.append("cd /app")
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    lines.append(f"exec env {exports} {' '.join(shlex.quote(c) for c in command)}")
    return "\n".join(lines) + "\n"


def run_ns(
    binds: list[tuple[Path, str, bool]],
    command: list[str],
    env: dict[str, str],
    timeout_s: float,
    links: list[tuple[str, Path]] = (),
) -> Proc:
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, dir="/var/tmp") as fh:
        fh.write(_ns_script(binds, command, env, links))
        script = fh.name
    try:
        return _run(["unshare", "-m", "--propagation", "private", "bash", script], timeout_s)
    finally:
        os.unlink(script)


@dataclass
class Backend:
    kind: str  # "ns" | "docker"
    python: str  # interpreter used inside the ns backend
    agent_image: str = ""
    docker_network: str = "bridge"
    agent_src: Path | None = None  # ns backend: directory containing the t1agent package

    def run_agent(self, input_dir: Path, out_dir: Path, log_dir: Path, env: dict[str, str], timeout_s: float) -> Proc:
        if self.kind == "ns":
            assert self.agent_src is not None
            full_env = {
                "PATH": f"{Path(self.python).parent}:/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "PYTHONPATH": str(self.agent_src),
                "T1A_LOG_DIR": "/agentlogs",
                **env,
            }
            binds = [(input_dir, "/input", True), (out_dir, "/app/output", False), (out_dir, "/output", False), (log_dir, "/agentlogs", False)]
            return run_ns(binds, [self.python, "-m", "t1agent", "solve", "--task-dir", "/input", "--out", "/app/output"], full_env, timeout_s)
        cmd = ["docker", "run", "--rm", "--network", self.docker_network]
        for k, v in {**env, "T1A_LOG_DIR": "/agentlogs"}.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [
            "-v", f"{input_dir}:/input:ro",
            "-v", f"{out_dir}:/app/output",
            "-v", f"{out_dir}:/output",
            "-v", f"{log_dir}:/agentlogs",
            self.agent_image, "solve", "--task-dir", "/input", "--out", "/app/output",
        ]
        return _run(cmd, timeout_s)

    def run_checker(self, unit_dir: Path, out_dir: Path, timeout_s: float) -> Proc:
        if self.kind == "ns":
            env = {
                "PATH": f"{Path(self.python).parent}:/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "OUTPUT_DIR": "/app/output",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            with tempfile.TemporaryDirectory(dir="/var/tmp") as logs:
                binds = [
                    (unit_dir, "/input", True),
                    (out_dir, "/app/output", False),
                    (out_dir, "/output", False),
                    (unit_dir / "checks", "/tests", True),
                    (Path(logs), "/logs/verifier", False),
                ]
                return run_ns(binds, ["bash", "/input/checks/test.sh"], env, timeout_s, unit_data_links(unit_dir))
        # The checks read inputs at their unit-image paths (/app/data/..., /app/x.csv),
        # so they run in the unit's own image, built FROM the shared sandbox base.
        image = f"t1unit-{unit_dir.name.lower()}:latest"
        if _run(["docker", "image", "inspect", image], 60).returncode != 0:
            built = _run(["docker", "build", "-q", "-t", image, str(unit_dir / "environment")], 1800)
            if built.returncode != 0:
                return built
        cmd = [
            "docker", "run", "--rm", "--network=none",
            "-e", "OUTPUT_DIR=/app/output", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{unit_dir}:/input:ro",
            "-v", f"{out_dir}:/app/output",
            "-v", f"{out_dir}:/output",
            "-v", f"{unit_dir / 'checks'}:/tests:ro",
            image, "bash", "/input/checks/test.sh",
        ]
        return _run(cmd, timeout_s)
