"""Paired legacy vs constants-first analyzer eval over report-track prompts.

Phase 3 of ``docs/superpowers/plans/2026-08-09-analyzer-prompt-cache-ordering.md``.
Gates activation of ``ENAI_ANALYZER_CONSTANTS_FIRST=report``.

Why this exists separately from ``routing_golden_set.py``: that set holds 18
Standard one-liners. Report tracks hand the analyzer a composite -- the leading
question, then ``Research track:``, ``Required coverage:`` bullets and
``Report context:`` (see ``agent/report_research_execution.py``
``build_report_track_analysis_query``). That shape has already produced four
routing misroutes, and the Standard golden does not contain a single instance
of it. Reordering the prompt is exactly the kind of edit that could disturb it
again.

Two things are measured, because routing agreement alone would miss the more
likely failure:

1. **Routing agreement.** The same routed contract fields the golden set
   scores, compared arm to arm rather than against an expected value -- there
   is no ground truth for these composites, only "did the order change it".

2. **Schema adherence.** Constants-first moves the output schema from the end
   of the prompt to the front, ~35,000 characters from the generation point.
   Every recurring analyzer failure in production has been a schema violation
   (job 83010f04: six candidate topics against a cap of five; job 3b92f462:
   prose in a 64-character identifier field, which cost a whole track its
   analysis). This counts sanitizer repairs and hard validation failures per
   arm, which is deterministic evidence rather than an LLM judgement.

Model nondeterminism produces disagreements on its own, so a case that differs
is repeated (``--repeats``, default 3) in BOTH arms before the difference is
attributed to prompt order. A field that also varies within an arm is reported
as unstable, not as a regression.

NOT part of the pytest suite: it calls a real LLM. Requires the production env
vars for the active MODEL_TYPE.

Usage:
    python evaluation/analyzer_prompt_order_pairs.py             # full run
    python evaluation/analyzer_prompt_order_pairs.py --dry-run   # fixtures only, no env/LLM
    python evaluation/analyzer_prompt_order_pairs.py --id t03
    python evaluation/analyzer_prompt_order_pairs.py --repeats 5

Exit code 0 when no case shows a stable routing disagreement and the
constants-first arm's schema-repair count does not exceed the legacy arm's.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

_FIXTURE_PATH = Path(__file__).with_name("analyzer_prompt_order_cases.json")

_ROUTED_FIELDS = (
    "query_type",
    "answer_kind",
    "render_style",
    "preferred_path",
    "top_tool",
)

# Emitted by core/llm_payloads.py when a payload had to be repaired, and by
# core/llm.py when repair was not enough. Both mean the model did not honour
# the schema; the second costs the track its analysis. Kept in sync by
# tests/test_analyzer_prompt_order_pairs.py, which reads the real log strings
# out of the sanitizer -- a marker that matches nothing would report a clean
# run no matter how badly adherence degraded.
_REPAIR_MARKERS = (
    "Sanitized over-long analyzer string",
    "Sanitized candidate_topics",
    "Sanitized unknown derived_metrics",
    "Question-analysis schema validation failed",
)


def load_fixtures(path: Path = _FIXTURE_PATH) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path.name}: 'cases' must be a non-empty list")

    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not case_id or case_id in seen:
            raise ValueError(f"{path.name}: missing or duplicate id {case_id!r}")
        seen.add(case_id)
        query = str(case.get("query", ""))
        if not query.strip():
            raise ValueError(f"{case_id}: empty query")
        # The whole point of this set is the composite shape. A one-liner here
        # would silently make the eval a duplicate of the Standard golden.
        if "Research track:" not in query:
            raise ValueError(
                f"{case_id}: not a report-track composite (no 'Research track:' line)"
            )
    return cases


class _RepairCounter(logging.Handler):
    """Count schema-repair warnings emitted while one arm runs."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.counts: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # a broken log record must not fail the eval
            return
        for marker in _REPAIR_MARKERS:
            if marker in message:
                self.counts[marker] += 1

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _analyze_once(query: str, mode: str) -> tuple[dict, str]:
    """Route one query under one ordering. Returns (routed_fields, error)."""
    import core.llm as llm_core
    from agent import pipeline, planner
    from agent.fixture_candidates import routed_fields_snapshot
    from models import AnswerMode, QueryContext

    previous_mode = llm_core.ANALYZER_CONSTANTS_FIRST_MODE
    llm_core.ANALYZER_CONSTANTS_FIRST_MODE = mode
    try:
        ctx = QueryContext(query=query, answer_mode=AnswerMode.REPORT.value)
        ctx = planner.prepare_context(ctx)
        ctx = planner.analyze_question_active(ctx)
        if ctx.question_analysis is None:
            return {}, f"analyzer returned nothing ({ctx.question_analysis_error})"
        pipeline._finalize_answer_kind(ctx)
        return routed_fields_snapshot(ctx.question_analysis), ""
    except Exception as exc:
        return {}, f"exception {type(exc).__name__}: {exc}"
    finally:
        llm_core.ANALYZER_CONSTANTS_FIRST_MODE = previous_mode


def _disable_response_cache():
    """Repeats must reach the model; a cache hit would report false stability."""
    import core.llm as llm_core

    originals = (llm_core._cache_get_or_reserve, llm_core._cache_set)
    llm_core._cache_get_or_reserve = lambda _key: (None, None)
    llm_core._cache_set = lambda *_args, **_kwargs: None
    return originals


def _restore_response_cache(originals) -> None:
    import core.llm as llm_core

    llm_core._cache_get_or_reserve, llm_core._cache_set = originals


def _unstable_fields(samples: list[dict]) -> set[str]:
    """Fields that differ between repeats of the SAME arm."""
    return {
        field
        for field in _ROUTED_FIELDS
        if len({sample.get(field) for sample in samples}) > 1
    }


def run(cases: list[dict], repeats: int) -> int:
    counter = _RepairCounter()
    logging.getLogger("Enai").addHandler(counter)
    cache_originals = _disable_response_cache()

    regressions: list[str] = []
    unstable: list[str] = []
    errors: list[str] = []
    repairs = {"legacy": 0, "constants_first": 0}

    try:
        for case in cases:
            arms: dict[str, list[dict]] = {}
            arm_errors: dict[str, str] = {}
            for arm, mode in (("legacy", "off"), ("constants_first", "all")):
                before = counter.total
                routed, error = _analyze_once(case["query"], mode)
                repairs[arm] += counter.total - before
                arms[arm] = [routed]
                if error:
                    arm_errors[arm] = error

            if arm_errors:
                detail = "; ".join(f"{arm}: {err}" for arm, err in arm_errors.items())
                errors.append(f"{case['id']}: {detail}")
                print(f"ERROR {case['id']}: {detail}")
                continue

            differing = {
                field
                for field in _ROUTED_FIELDS
                if arms["legacy"][0].get(field) != arms["constants_first"][0].get(field)
            }
            if not differing:
                print(f"agree {case['id']}: {case['query'].splitlines()[0][:58]}")
                continue

            # Repeat both arms before blaming the ordering.
            for arm, mode in (("legacy", "off"), ("constants_first", "all")):
                for _ in range(max(0, repeats - 1)):
                    before = counter.total
                    routed, error = _analyze_once(case["query"], mode)
                    repairs[arm] += counter.total - before
                    if not error:
                        arms[arm].append(routed)

            noisy = _unstable_fields(arms["legacy"]) | _unstable_fields(
                arms["constants_first"]
            )
            stable_differences = sorted(differing - noisy)
            if noisy & differing:
                unstable.append(
                    f"{case['id']}: {sorted(noisy & differing)} varies within an arm"
                )
                print(f"noisy {case['id']}: {sorted(noisy & differing)} unstable "
                      f"across {repeats} repeats — not attributable to order")
            if stable_differences:
                detail = ", ".join(
                    f"{field}: legacy={arms['legacy'][0].get(field)!r} "
                    f"new={arms['constants_first'][0].get(field)!r}"
                    for field in stable_differences
                )
                regressions.append(f"{case['id']}: {detail}")
                print(f"DIFFER {case['id']}: {detail}")
    finally:
        _restore_response_cache(cache_originals)
        logging.getLogger("Enai").removeHandler(counter)

    total = len(cases)
    print(f"\nReport-track prompt-order pairs: {total} case(s), {repeats} repeats "
          f"on disagreement")
    print(f"  stable routing differences : {len(regressions)}")
    print(f"  unstable (model variance)  : {len(unstable)}")
    print(f"  errors                     : {len(errors)}")
    print(f"  schema repairs  legacy={repairs['legacy']} "
          f"constants_first={repairs['constants_first']}")
    if counter.counts:
        for marker, count in counter.counts.most_common():
            print(f"    {marker}: {count}")
    for label, lines in (
        ("Stable differences", regressions),
        ("Unstable fields", unstable),
        ("Errors", errors),
    ):
        if lines:
            print(f"{label}:")
            for line in lines:
                print(f"  - {line}")

    schema_regressed = repairs["constants_first"] > repairs["legacy"]
    if schema_regressed:
        print(
            "\nSchema adherence got worse under constants-first. The schema now "
            "sits ~35,000 chars from the generation point; the plan's fallback "
            "is to move CONTRACT_RULES back behind the question, which still "
            "leaves a ~4,750-token prefix."
        )
    return 0 if not (regressions or errors or schema_regressed) else 1


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="validate fixture structure only; no env or LLM needed")
    parser.add_argument("--id", dest="case_id", default=None,
                        help="run a single case by id")
    parser.add_argument("--repeats", type=int, default=3,
                        help="runs per arm when a case disagrees (default 3)")
    args = parser.parse_args(argv)

    cases = load_fixtures()
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
        if not cases:
            print(f"No case with id {args.case_id!r}")
            return 1
    if args.dry_run:
        print(f"{_FIXTURE_PATH.name} OK: {len(cases)} report-track case(s), "
              f"compared on {list(_ROUTED_FIELDS)}")
        return 0
    return run(cases, max(1, args.repeats))


if __name__ == "__main__":
    sys.exit(main())
