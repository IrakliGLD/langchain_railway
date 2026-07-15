"""Write a deterministic manifest for one locally built backend image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("Dockerfile", "requirements.txt", "railway.json")


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.git_sha) != 40 or any(c not in "0123456789abcdef" for c in args.git_sha.lower()):
        parser.error("--git-sha must be a full 40-character commit SHA")

    payload = {
        "schema_version": "backend-release-manifest-v1",
        "git_sha": args.git_sha.lower(),
        "image_id": args.image_id,
        "inputs": {name: _sha256(ROOT / name) for name in INPUTS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
