"""Convert production logs into candidate golden-set fixtures (§5.3).

Parses ``ROUTING_FIXTURE_CANDIDATE {json}`` lines emitted by
``agent/fixture_candidates.py``, dedupes by query (first occurrence wins),
and emits cases in the golden-set shape with ``expected`` intentionally
blank — a human verifies and labels each candidate before adopting it into
``routing_golden_set.json``.

Keyless and stdlib-only (no config import — runs on a laptop against a
pasted log file).

Usage:
    python evaluation/harvest_fixture_candidates.py railway.log
    python evaluation/harvest_fixture_candidates.py railway.log --out candidates.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.fixture_candidates import MARKER


def parse_candidates(lines: list[str]) -> list[dict]:
    """Extract candidate payloads from log lines; dedupe by query."""
    seen_queries: set[str] = set()
    candidates: list[dict] = []
    for line in lines:
        marker_at = line.find(MARKER)
        if marker_at == -1:
            continue
        raw = line[marker_at + len(MARKER):].strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        query = str(payload.get("query", "")).strip()
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        candidates.append(payload)
    return candidates


def to_cases(candidates: list[dict]) -> list[dict]:
    cases = []
    for index, payload in enumerate(candidates, start=1):
        observed = json.dumps(
            payload.get("routed", {}), ensure_ascii=False, separators=(",", ":"),
        )
        cases.append({
            "id": f"cand_{index:03d}",
            "query": payload["query"],
            "note": (
                f"trigger={payload.get('trigger', '?')}; observed={observed}; "
                f"trace_id={payload.get('trace_id', '')}; "
                "VERIFY expected before adopting"
            ),
            "expected": {},
        })
    return cases


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logfile", help="log file containing candidate marker lines")
    parser.add_argument("--out", default=None, help="write JSON here instead of stdout")
    args = parser.parse_args(argv)

    lines = Path(args.logfile).read_text(encoding="utf-8", errors="replace").splitlines()
    cases = to_cases(parse_candidates(lines))
    document = {
        "version": 1,
        "description": (
            "Candidate fixtures harvested from production logs. Human-label the "
            "expected fields, then move adopted cases into routing_golden_set.json."
        ),
        "cases": cases,
    }
    rendered = json.dumps(document, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {len(cases)} candidate(s) to {args.out}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
