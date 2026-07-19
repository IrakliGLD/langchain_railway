"""Offline consistency checks for the hashed production lock (B2.A.6).

The authoritative freshness gate is the CI step running
``scripts/generate_requirements_lock.py --check`` (network resolution).
These tests catch the same drift offline: a requirements.txt edit without a
lock regeneration fails here first.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = (ROOT / "requirements.txt").read_text(encoding="utf-8")
LOCK = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")

_PIN_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==([^\s#\\]+)", re.MULTILINE)


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def _pins(text: str) -> dict[str, str]:
    return {_normalize(name): version for name, version in _PIN_RE.findall(text)}


def test_lock_source_digest_matches_requirements():
    # The generator embeds the sha256 of the requirements.txt it resolved.
    # An edited requirements.txt without a regenerated lock fails here.
    match = re.search(r"# Source: requirements\.txt \(sha256=([0-9a-f]{64})\)", LOCK)
    assert match, "lock is missing its source-digest header"
    current = hashlib.sha256((ROOT / "requirements.txt").read_bytes()).hexdigest()
    assert match.group(1) == current, (
        "requirements-lock.txt is stale; run "
        "'python scripts/generate_requirements_lock.py'"
    )


def test_every_top_level_pin_is_locked_at_the_same_version():
    top_level = _pins(REQUIREMENTS)
    locked = _pins(LOCK)
    assert top_level, "requirements.txt has no pins?"
    for name, version in top_level.items():
        assert locked.get(name) == version, (
            f"{name}=={version} from requirements.txt is not locked identically"
        )


def test_every_locked_pin_carries_artifact_hashes():
    lines = LOCK.splitlines()
    for index, line in enumerate(lines):
        if _PIN_RE.match(line):
            assert line.rstrip().endswith("\\"), f"pin without hash block: {line}"
            assert "--hash=sha256:" in lines[index + 1], (
                f"pin without sha256 hash: {line}"
            )


def test_lock_contains_no_unhashed_or_editable_entries():
    for line in LOCK.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("--hash="):
            continue
        assert _PIN_RE.match(line), f"unexpected lock entry: {line}"
        assert "-e " not in line and "git+" not in line
