"""Fail closed when synthetic privacy canaries appear in exported logs.

This scanner deliberately knows only synthetic, repository-owned values.  It
never prints a matching log line or the canary value, so its own output is safe
to attach to release evidence.  Operators inject the canaries through a test
request, export the relevant Railway/Edge logs, and scan those files locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

CANARIES = {
    "email": b"enai-log-canary+20260717@example.invalid",
    "uuid": b"017f7110-4a77-4f88-9c21-000000000001",
    "bearer": b"Bearer eyJsb2dfY2FuYXJ5.eyJzeW50aGV0aWM.signature01",
    "prompt": b"ENAI_LOG_CANARY_PROMPT_20260717_DO_NOT_RETAIN",
    "sql": b"ENAI_LOG_CANARY_SQL_20260717_DO_NOT_RETAIN",
}
DEFAULT_MAX_FILE_MIB = 100


def scan_paths(paths: Iterable[Path], *, max_file_bytes: int) -> dict[str, object]:
    """Return a content-free aggregate report for regular, bounded files."""

    files_scanned = 0
    bytes_scanned = 0
    findings = {label: 0 for label in CANARIES}

    for candidate in paths:
        path = candidate.expanduser()
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"log input must be a regular non-symlink file: {path}")
        size = path.stat().st_size
        if size > max_file_bytes:
            raise ValueError(f"log input exceeds the {max_file_bytes}-byte safety limit: {path}")
        payload = path.read_bytes()
        files_scanned += 1
        bytes_scanned += len(payload)
        for label, canary in CANARIES.items():
            findings[label] += payload.count(canary)

    leaked = {label: count for label, count in findings.items() if count}
    return {
        "schema_version": 1,
        "clean": not leaked,
        "files_scanned": files_scanned,
        "bytes_scanned": bytes_scanned,
        "finding_counts": leaked,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan exported logs for Enai's synthetic privacy canaries.")
    parser.add_argument("log_files", nargs="+", type=Path)
    parser.add_argument(
        "--max-file-mib",
        type=int,
        default=DEFAULT_MAX_FILE_MIB,
        help=f"reject larger inputs (default: {DEFAULT_MAX_FILE_MIB} MiB)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_file_mib < 1 or args.max_file_mib > 1024:
        build_parser().error("--max-file-mib must be between 1 and 1024")
    try:
        report = scan_paths(
            args.log_files,
            max_file_bytes=args.max_file_mib * 1024 * 1024,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"schema_version": 1, "clean": False, "scan_error": str(exc)}))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
