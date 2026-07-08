"""Routing golden-set eval — regression harness for Stage 0.2 routing (§5.3).

Runs the LIVE question analyzer plus the pipeline's finalize cross-check for
each fixture in ``routing_golden_set.json`` and scores the routed contract
fields (``query_type``, ``answer_kind``, ``render_style``, ``preferred_path``,
``top_tool``). This is the pre-deploy complement to the A5 production
disagreement counters: run it after every analyzer-prompt, cross-check, or
runtime-skill edit so yesterday's routing fixes cannot silently regress.

NOT part of the pytest suite (tests/ never calls a real LLM). Requires the
production env vars (SUPABASE_DB_URL is validated at config import but the DB
is never touched; the LLM key of the active MODEL_TYPE is actually used).

Usage:
    python evaluation/routing_golden_set.py               # full run, strict
    python evaluation/routing_golden_set.py --dry-run     # validate fixtures only (no env/LLM)
    python evaluation/routing_golden_set.py --id r003     # single case
    python evaluation/routing_golden_set.py --min-score 0.9

Exit code 0 when pass-rate >= --min-score (default 1.0), else 1.

Extending the set: whenever a routing failure is fixed (see
skills/pipeline-failure-diagnostics), add the failing question with its
now-correct expected fields. Only assert fields you are confident about.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_FIXTURE_PATH = Path(__file__).with_name("routing_golden_set.json")

# Field name -> how to read it off a finalized QuestionAnalysis.
_ASSERTABLE_FIELDS = ("query_type", "answer_kind", "render_style", "preferred_path", "top_tool")


def load_fixtures(path: Path = _FIXTURE_PATH) -> list[dict]:
    """Load and structurally validate the golden-set fixtures.

    Enum validity is checked lazily against the contracts module when
    available; structural rules (unique ids, non-empty queries, known
    expected-field names) always apply. Raises ValueError on any violation.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("routing_golden_set.json: 'cases' must be a non-empty list")

    seen_ids: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not case_id or case_id in seen_ids:
            raise ValueError(f"routing_golden_set.json: missing or duplicate id {case_id!r}")
        seen_ids.add(case_id)
        if not str(case.get("query", "")).strip():
            raise ValueError(f"{case_id}: empty query")
        expected = case.get("expected")
        if not isinstance(expected, dict) or not expected:
            raise ValueError(f"{case_id}: 'expected' must be a non-empty object")
        unknown = set(expected) - set(_ASSERTABLE_FIELDS)
        if unknown:
            raise ValueError(f"{case_id}: unknown expected fields {sorted(unknown)}")
    return cases


def validate_expected_enums(cases: list[dict]) -> None:
    """Check expected values against the runtime contract enums.

    Separate from load_fixtures so the pytest fixture-validation test can run
    it without the runner's env requirements leaking in.
    """
    from contracts.question_analysis import AnswerKind, PreferredPath, QueryType, RenderStyle

    valid = {
        "query_type": {e.value for e in QueryType},
        "answer_kind": {e.value for e in AnswerKind},
        "render_style": {e.value for e in RenderStyle},
        "preferred_path": {e.value for e in PreferredPath},
    }
    for case in cases:
        for field, value in case["expected"].items():
            if field == "top_tool":
                continue  # tool names are registry strings, not enums
            if value not in valid[field]:
                raise ValueError(
                    f"{case['id']}: expected.{field}={value!r} is not a valid "
                    f"{field} (valid: {sorted(valid[field])})"
                )


def _routed_fields(ctx) -> dict:
    """Read the post-finalize routed values off the context.

    Delegates to the runtime's single authority so the eval and the
    production fixture-candidate emitter can never drift apart.
    """
    from agent.fixture_candidates import routed_fields_snapshot

    return routed_fields_snapshot(ctx.question_analysis)


def run(cases: list[dict], min_score: float) -> int:
    # Imports deferred: config validates env at import time, and the analyzer
    # needs a live LLM key — --dry-run must work without either.
    from agent import pipeline, planner
    from models import QueryContext

    validate_expected_enums(cases)

    failures: list[str] = []
    for case in cases:
        ctx = QueryContext(query=case["query"])
        try:
            ctx = planner.prepare_context(ctx)
            ctx = planner.analyze_question_active(ctx)
            if ctx.question_analysis is None:
                failures.append(f"{case['id']}: analyzer returned no QuestionAnalysis "
                                f"(error={ctx.question_analysis_error})")
                print(f"FAIL {case['id']}: analyzer failure")
                continue
            pipeline._finalize_answer_kind(ctx)
        except Exception as exc:  # keep scoring the rest of the set
            failures.append(f"{case['id']}: exception {exc}")
            print(f"FAIL {case['id']}: exception {exc}")
            continue

        routed = _routed_fields(ctx)
        mismatches = {
            field: (expected_value, routed.get(field))
            for field, expected_value in case["expected"].items()
            if routed.get(field) != expected_value
        }
        if mismatches:
            detail = ", ".join(
                f"{f}: expected {e!r} got {g!r}" for f, (e, g) in mismatches.items()
            )
            failures.append(f"{case['id']}: {detail}")
            print(f"FAIL {case['id']} ({case.get('note', '')}): {detail}")
        else:
            print(f"pass {case['id']}: {case['query'][:60]}")

    total = len(cases)
    passed = total - len(failures)
    score = passed / total if total else 0.0
    print(f"\nRouting golden set: {passed}/{total} passed (score={score:.2f}, "
          f"min={min_score:.2f})")
    if failures:
        print("Disagreements:")
        for line in failures:
            print(f"  - {line}")
    return 0 if score >= min_score else 1


def main(argv: list[str] | None = None) -> int:
    # Windows consoles often default to cp1252, which cannot print the
    # Georgian/Russian fixture queries — degrade rather than crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="validate fixture structure only; no env or LLM needed")
    parser.add_argument("--id", dest="case_id", default=None,
                        help="run a single case by id")
    parser.add_argument("--min-score", type=float, default=1.0,
                        help="pass-rate below which the run exits 1 (default 1.0)")
    args = parser.parse_args(argv)

    cases = load_fixtures()
    if args.case_id:
        cases = [c for c in cases if c["id"] == args.case_id]
        if not cases:
            print(f"No case with id {args.case_id!r}")
            return 1
    if args.dry_run:
        print(f"routing_golden_set.json OK: {len(cases)} case(s), "
              f"assertable fields {_ASSERTABLE_FIELDS}")
        return 0
    continuity_cases = [c for c in cases if c.get("skip_unless_continuity")]
    if continuity_cases:
        cases = [c for c in cases if not c.get("skip_unless_continuity")]
        print(
            f"Skipping {len(continuity_cases)} continuity case(s) — the runner "
            "does not chain turns yet (architecture §5 contract-continuity slice)"
        )
    return run(cases, args.min_score)


if __name__ == "__main__":
    sys.exit(main())
