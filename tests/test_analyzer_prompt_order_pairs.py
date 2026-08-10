"""Offline tests for the report-track prompt-order eval harness.

The harness itself calls a live model, so its logic has to be correct before
it is run: a comparison that cannot tell model noise from a prompt-order
regression would either block a safe change or wave through an unsafe one.
Every test here drives ``run()`` with a stubbed analyzer.
"""

import json
import logging
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest  # noqa: E402

from evaluation import analyzer_prompt_order_pairs as harness  # noqa: E402

_AGREED = {
    "query_type": "data_retrieval",
    "answer_kind": "timeseries",
    "render_style": "chart",
    "preferred_path": "tool",
    "top_tool": "get_prices",
}


def _case(case_id: str = "t01") -> dict:
    return {
        "id": case_id,
        "query": "What were prices?\nResearch track: Prices\nReport context: x",
    }


def _stub(monkeypatch, responses):
    """Feed ``run`` a scripted sequence of (routed_fields, error) per arm."""
    calls = {"legacy": 0, "constants_first": 0}

    def fake(query, mode):
        arm = "legacy" if mode == "off" else "constants_first"
        index = min(calls[arm], len(responses[arm]) - 1)
        calls[arm] += 1
        return responses[arm][index]

    monkeypatch.setattr(harness, "_analyze_once", fake)
    monkeypatch.setattr(harness, "_disable_response_cache", lambda: None)
    monkeypatch.setattr(harness, "_restore_response_cache", lambda _originals: None)
    return calls


def test_fixtures_are_all_report_track_composites():
    """A one-liner here would silently duplicate the Standard golden set."""
    cases = harness.load_fixtures()
    assert len(cases) >= 8
    for case in cases:
        assert "Research track:" in case["query"], case["id"]
        assert "Report context:" in case["query"], case["id"]


def test_a_one_line_query_is_rejected(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps({"cases": [{"id": "x", "query": "What were prices in May?"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="report-track composite"):
        harness.load_fixtures(path)


def test_agreement_passes(monkeypatch, capsys):
    _stub(monkeypatch, {
        "legacy": [(_AGREED, "")],
        "constants_first": [(_AGREED, "")],
    })
    assert harness.run([_case()], repeats=3) == 0
    assert "stable routing differences : 0" in capsys.readouterr().out


def test_a_reproducible_difference_fails(monkeypatch, capsys):
    """Same answer every repeat in each arm, but the arms disagree."""
    changed = {**_AGREED, "top_tool": "get_generation"}
    _stub(monkeypatch, {
        "legacy": [(_AGREED, "")],
        "constants_first": [(changed, "")],
    })
    assert harness.run([_case()], repeats=3) == 1
    out = capsys.readouterr().out
    assert "stable routing differences : 1" in out
    assert "get_generation" in out


def test_model_noise_is_not_reported_as_a_regression(monkeypatch, capsys):
    """The reviewer's point: a single-run difference proves nothing.

    Legacy itself flips between two answers across repeats, so the field is
    unstable within an arm and the difference is not attributable to order.
    """
    changed = {**_AGREED, "top_tool": "get_generation"}
    _stub(monkeypatch, {
        "legacy": [(_AGREED, ""), (changed, ""), (_AGREED, "")],
        "constants_first": [(changed, ""), (changed, ""), (changed, "")],
    })
    assert harness.run([_case()], repeats=3) == 0
    out = capsys.readouterr().out
    assert "unstable (model variance)  : 1" in out
    assert "stable routing differences : 0" in out


def test_an_analyzer_failure_is_an_error_not_an_agreement(monkeypatch, capsys):
    _stub(monkeypatch, {
        "legacy": [(_AGREED, "")],
        "constants_first": [({}, "analyzer returned nothing (schema invalid)")],
    })
    assert harness.run([_case()], repeats=3) == 1
    assert "errors                     : 1" in capsys.readouterr().out


def test_worse_schema_adherence_fails_even_when_routing_agrees(monkeypatch, capsys):
    """The failure mode most likely to be caused by moving the schema.

    Routing can agree while the model stops honouring field limits, because
    the sanitizer repairs the payload before routing ever sees it. Counting
    repairs is what makes that visible.
    """
    log = logging.getLogger("Enai")

    def fake(query, mode):
        if mode != "off":
            log.warning("Sanitized over-long analyzer string: 92 > 64 characters, dropped")
        return (_AGREED, "")

    monkeypatch.setattr(harness, "_analyze_once", fake)
    monkeypatch.setattr(harness, "_disable_response_cache", lambda: None)
    monkeypatch.setattr(harness, "_restore_response_cache", lambda _originals: None)

    assert harness.run([_case()], repeats=3) == 1
    out = capsys.readouterr().out
    assert "schema repairs  legacy=0 constants_first=1" in out
    assert "Schema adherence got worse" in out


def test_equal_schema_repair_counts_do_not_fail(monkeypatch, capsys):
    """Pre-existing repairs are not this change's fault."""
    log = logging.getLogger("Enai")

    def fake(query, mode):
        log.warning("Sanitized candidate_topics 6 -> 5 (dropped lowest-scoring)")
        return (_AGREED, "")

    monkeypatch.setattr(harness, "_analyze_once", fake)
    monkeypatch.setattr(harness, "_disable_response_cache", lambda: None)
    monkeypatch.setattr(harness, "_restore_response_cache", lambda _originals: None)

    assert harness.run([_case()], repeats=3) == 0
    assert "schema repairs  legacy=1 constants_first=1" in capsys.readouterr().out


def test_every_sanitizer_warning_is_actually_matched_by_a_marker():
    """A marker that matches nothing reports a clean run however bad it got.

    Read the real log strings out of the sanitizer rather than trusting the
    list, so a new repair path cannot go uncounted and a renamed one cannot
    quietly stop counting.
    """
    import re
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "core" / "llm_payloads.py"
    emitted = re.findall(r'"(Sanitized[^"]*)"', source.read_text(encoding="utf-8"))
    assert emitted, "no sanitizer warnings found; did the log strings move?"

    for message in emitted:
        assert any(
            message.startswith(marker) for marker in harness._REPAIR_MARKERS
        ), f"unmatched sanitizer warning: {message!r}"

    for marker in harness._REPAIR_MARKERS:
        if marker.startswith("Sanitized"):
            assert any(message.startswith(marker) for message in emitted), (
                f"marker matches nothing in the sanitizer: {marker!r}"
            )


def test_the_hard_failure_marker_matches_the_real_message():
    """The failure that costs a track its whole analysis must be counted."""
    import re
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "core" / "llm.py"
    assert re.search(
        r'"Question-analysis schema validation failed', source.read_text(encoding="utf-8")
    ), "the hard-failure log string moved; update _REPAIR_MARKERS"


def test_the_repair_counter_ignores_unrelated_warnings():
    counter = harness._RepairCounter()
    for message in (
        "Prompt budget applied (section-aware): label=question_analysis",
        "REPORT_CHART_REQUEST_DROPPED {}",
    ):
        counter.emit(
            logging.LogRecord("Enai", logging.WARNING, __file__, 1, message, None, None)
        )
    assert counter.total == 0

    counter.emit(
        logging.LogRecord(
            "Enai", logging.WARNING, __file__, 1,
            "Question-analysis schema validation failed: sql_hints.metric", None, None,
        )
    )
    assert counter.total == 1
