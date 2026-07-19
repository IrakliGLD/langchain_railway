"""Generate or verify the hashed production lock (requirements-lock.txt).

F10-SEC-01 B2.A.6. The lock records the complete production dependency
closure of requirements.txt, resolved exactly as the container installs it
(CPython 3.11, manylinux, wheels only), with the sha256 hash of every PyPI
release artifact for each pin so ``pip install --require-hashes`` succeeds
regardless of which compatible wheel pip selects.

The resolution flags are fixed (not inherited from the host interpreter), so
regeneration is byte-deterministic on any machine with PyPI access:

    python scripts/generate_requirements_lock.py            # rewrite the lock
    python scripts/generate_requirements_lock.py --check    # CI freshness gate

Dependabot and humans edit requirements.txt only; the CI ``--check`` step
fails until the lock is regenerated to match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
LOCK = ROOT / "requirements-lock.txt"

# Production container resolution target (python:3.11-slim, glibc x86_64).
RESOLUTION_FLAGS = [
    "--python-version", "311",
    "--implementation", "cp",
    "--abi", "cp311",
    "--platform", "manylinux2014_x86_64",
    "--platform", "linux_x86_64",
    "--platform", "any",
    "--only-binary=:all:",
]


# Marker environment of the production container. pip evaluates dependency
# markers against the GENERATING host (there is no override flag), so a
# Windows host resolves host-only packages such as colorama (a
# platform_system=="Windows" dependency of click) that a Linux host never
# sees. The reachability walk below re-evaluates every dependency edge
# against this fixed environment, making the lock byte-identical no matter
# where it is generated.
TARGET_MARKER_ENV = {
    "python_version": "3.11",
    "python_full_version": "3.11.15",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
    "sys_platform": "linux",
    "platform_system": "Linux",
    "os_name": "posix",
    "platform_machine": "x86_64",
}


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _top_level_requirements() -> list:
    from packaging.requirements import Requirement

    requirements = []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            requirements.append(Requirement(stripped))
    return requirements


def _reachable_for_target(report: dict) -> set[str]:
    """Walk requires_dist edges with markers evaluated for the container."""
    from packaging.requirements import Requirement

    metadata = {}
    for item in report["install"]:
        meta = item["metadata"]
        metadata[_normalize_name(meta["name"])] = meta.get("requires_dist") or []

    def _edge_applies(requirement: Requirement, active_extras: frozenset[str]) -> bool:
        if requirement.marker is None:
            return True
        for extra in active_extras or {""}:
            if requirement.marker.evaluate({**TARGET_MARKER_ENV, "extra": extra}):
                return True
        return False

    visited: set[tuple[str, frozenset[str]]] = set()
    queue: list[tuple[str, frozenset[str]]] = []
    for requirement in _top_level_requirements():
        queue.append((_normalize_name(requirement.name), frozenset(requirement.extras)))
    while queue:
        name, extras = queue.pop()
        if (name, extras) in visited:
            continue
        visited.add((name, extras))
        for raw in metadata.get(name, []):
            try:
                dep = Requirement(raw)
            except Exception:
                continue
            if not _edge_applies(dep, extras):
                continue
            dep_name = _normalize_name(dep.name)
            if dep_name in metadata:
                queue.append((dep_name, frozenset(dep.extras)))
    return {name for name, _extras in visited}


def resolve_closure() -> dict[str, str]:
    """Resolve requirements.txt for the container target without installing."""
    with tempfile.TemporaryDirectory() as td:
        report_path = Path(td) / "report.json"
        command = [
            sys.executable, "-m", "pip", "install",
            "--dry-run", "--quiet", "--ignore-installed",
            "--report", str(report_path),
            "-r", str(REQUIREMENTS),
            "--target", str(Path(td) / "target"),
            *RESOLUTION_FLAGS,
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise SystemExit(
                "dependency resolution failed:\n" + completed.stderr[-4000:]
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
    reachable = _reachable_for_target(report)
    closure: dict[str, str] = {}
    for item in report["install"]:
        meta = item["metadata"]
        name = _normalize_name(meta["name"])
        if name in reachable:
            closure[name] = meta["version"]
    return closure


def artifact_hashes(name: str, version: str) -> list[str]:
    """Return the sorted sha256 digests of every PyPI artifact for a pin."""
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)
    digests = sorted(
        {
            entry["digests"]["sha256"]
            for entry in data.get("urls", [])
            if entry.get("digests", {}).get("sha256")
        }
    )
    if not digests:
        raise SystemExit(f"no sha256 digests published for {name}=={version}")
    return digests


def _normalized_digest(path: Path) -> str:
    """sha256 over LF-normalized bytes so Windows/Linux checkouts agree."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def render_lock() -> str:
    source_digest = _normalized_digest(REQUIREMENTS)
    closure = resolve_closure()
    lines = [
        "# Generated by scripts/generate_requirements_lock.py - do not edit by hand.",
        f"# Source: requirements.txt (sha256={source_digest})",
        "# Resolution: cp311 / manylinux2014_x86_64 / wheels-only (production container).",
        "# Install with: pip install --require-hashes --only-binary :all: -r requirements-lock.txt",
        "",
    ]
    for name in sorted(closure):
        version = closure[name]
        hashes = artifact_hashes(name, version)
        lines.append(f"{name}=={version} \\")
        for index, digest in enumerate(hashes):
            terminator = "" if index == len(hashes) - 1 else " \\"
            lines.append(f"    --hash=sha256:{digest}{terminator}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail (exit 1) if requirements-lock.txt is stale instead of rewriting it",
    )
    args = parser.parse_args()

    rendered = render_lock()
    if args.check:
        current = LOCK.read_text(encoding="utf-8") if LOCK.is_file() else ""
        if current.replace("\r\n", "\n") != rendered:
            print(
                "requirements-lock.txt is stale; regenerate with "
                "'python scripts/generate_requirements_lock.py'",
                file=sys.stderr,
            )
            return 1
        print(f"requirements-lock.txt is current ({rendered.count('==')} pins verified)")
        return 0

    LOCK.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote {LOCK.name} with {len(rendered.splitlines())} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
