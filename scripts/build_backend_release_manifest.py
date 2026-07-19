"""Write a deterministic manifest for one locally built backend image."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("Dockerfile", "requirements.txt", "requirements-lock.txt", "railway.json")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.release_identity import normalize_release_sha  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--image-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        git_sha = normalize_release_sha(args.git_sha, field_name="--git-sha")
        image_revision = normalize_release_sha(
            args.image_revision,
            field_name="--image-revision",
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    if git_sha != image_revision:
        parser.error("--image-revision must match --git-sha")

    payload = {
        "schema_version": "backend-release-manifest-v2",
        "git_sha": git_sha,
        "image_id": args.image_id,
        "image_revision": image_revision,
        "inputs": {name: _sha256(ROOT / name) for name in INPUTS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
