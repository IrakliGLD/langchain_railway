"""Byte-exact pins on the analyzer prompt, ahead of the cache-ordering change.

The analyzer prompt opens with ``UNTRUSTED_USER_QUESTION`` and appends the
output schema after every block, so the only constant-to-constant prefix two
unrelated questions share is 28 characters -- roughly seven tokens. Every one
of the ~10,000 constant tokens behind the question is unreachable by a prefix
cache.

The plan (``docs/superpowers/plans/2026-08-09-analyzer-prompt-cache-ordering.md``)
moves the constants in front. That is a behaviour change to the pipeline's
semantic centre, guarded by a selector rather than by an exhaustive golden --
the analyzer is an LLM over unbounded language and there is no golden to
freeze. What CAN be frozen is the legacy prompt itself, so "Standard is
byte-identical while the selector is off" is an assertion rather than a
promise. This module is that assertion, and the matrix it defines is reused by
the constants-first tests.
"""

import hashlib
import json
import os
from pathlib import Path

# Ensure config validation passes before importing modules that depend on config.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("NVIDIA_API_KEY", "test-nvidia-key")

import core.llm as llm_core  # noqa: E402
from contracts.question_analysis import QuestionAnalysis  # noqa: E402

# The production default. Pinned literally so a developer with
# ANALYZER_PROMPT_BUDGET_MAX_CHARS exported cannot silently change what these
# tests measure.
DEFAULT_ANALYZER_BUDGET_CHARS = 45_000

_FIXTURE = Path(__file__).parent / "fixtures" / "analyzer_prompt_legacy_hashes.json"

_HISTORY = [
    {"role": "user", "content": "What was the balancing price in April 2026?"},
    {"role": "assistant", "content": "It was 178.4 GEL/MWh."},
]
_PREVIOUS_CONTRACT = '{"top_tool":"get_prices","period":"2025"}'
_ANOMALY = "Evidence returned 2 rows; expected at least 12."

# One query per prompt family and per shape that changes block selection:
# scalar, comparison, threshold (filter guide), explanation, forecast,
# scenario, knowledge, a follow-up needing history, and the report-track
# composite in both a realistic and a maximum-length form. Report-track shape
# is deliberately present: it is what four routing misroutes traced to, and the
# Standard routing golden does not contain it.
_QUERIES: dict[str, tuple[str, list | None]] = {
    "scalar": ("What was the balancing price in May 2026?", None),
    "comparison": (
        "Compare deregulated and balancing prices in 2025 vs 2024.", None),
    "threshold": (
        "Which months in 2025 had a balancing price above 200 GEL/MWh?", None),
    "explanation": ("Why did the balancing price rise in July 2025?", None),
    "forecast": ("Forecast the balancing price for the next six months.", None),
    "scenario": ("What if hydro generation dropped 20% next winter?", None),
    "knowledge": (
        "What is the balancing market and how is its price set?", None),
    "clarify": ("and for last year?", _HISTORY),
    "report_track": (
        "How did the generation mix shift between 2023 and 2025?\n"
        "Research track: Generation mix and cross-border trade\n"
        "Required coverage:\n"
        "- What share did hydro, thermal and wind each contribute?\n"
        "- How did import and export volumes change over the same period?\n"
        "Report context: Prepare an analytical report on the Georgian "
        "electricity market covering 2023-2025.",
        None,
    ),
    # build_report_track_analysis_query bounds the primary question at 600
    # chars, the title at 160 and each coverage bullet at 300, so this is the
    # longest input the report path can hand the analyzer.
    "report_track_long": (
        "How did the generation mix shift between 2023 and 2025? " + "x" * 543
        + "\nResearch track: " + "y" * 160
        + "\nRequired coverage:\n"
        + "\n".join(f"- {'z' * 300}" for _ in range(5)),
        None,
    ),
}

_CONTEXTS: dict[str, tuple[str, str]] = {
    "": ("", ""),
    "+prev": (_PREVIOUS_CONTRACT, ""),
    "+anom": ("", _ANOMALY),
    "+prev+anom": (_PREVIOUS_CONTRACT, _ANOMALY),
}


def _matrix() -> list[tuple[str, str, list | None, str, str]]:
    """(case_id, query, history, previous_contract, evidence_anomaly_note)."""
    return [
        (f"{query_id}{context_id}", query, history, previous_contract, anomaly)
        for query_id, (query, history) in _QUERIES.items()
        for context_id, (previous_contract, anomaly) in _CONTEXTS.items()
    ]


ANALYZER_PROMPT_MATRIX = _matrix()


def render_legacy_case(
    query: str,
    history: list | None,
    previous_contract: str,
    anomaly: str,
) -> tuple[str, str, list[str]]:
    """Render one matrix case exactly as ``llm_analyze_question`` does today.

    Returns ``(prompt, budgeted_prompt, truncation_priority)``.
    """
    schema_hint = QuestionAnalysis.model_json_schema()
    prompt_context = llm_core._build_analyzer_prompt_context(query, history)
    blocks = llm_core._build_analyzer_prompt_blocks(
        query,
        prompt_context.history_str,
        prompt_context.effective_pre_type,
        prompt_context.prompt_profile,
        prompt_context=prompt_context,
        previous_contract=previous_contract,
        evidence_anomaly_note=anomaly,
    )
    prompt = llm_core._render_analyzer_prompt(blocks, schema_hint)
    priority = llm_core._select_analyzer_truncation_priority(
        query,
        prompt_context.effective_pre_type,
        prompt_context.prompt_profile,
        prompt_context=prompt_context,
    )
    budgeted = llm_core._enforce_prompt_budget(
        prompt,
        label="question_analysis",
        budget_override=DEFAULT_ANALYZER_BUDGET_CHARS,
        truncation_priority=priority,
    )
    return prompt, budgeted, priority


def common_prefix_length(texts: list[str]) -> int:
    """Longest prefix shared by every string in ``texts``.

    The common prefix of a set equals the common prefix of its lexicographic
    minimum and maximum, so this stays linear instead of quadratic in the
    number of cases.
    """
    if not texts:
        return 0
    low, high = min(texts), max(texts)
    limit = min(len(low), len(high))
    index = 0
    while index < limit and low[index] == high[index]:
        index += 1
    return index


def _budgeted_prompts() -> dict[str, str]:
    return {
        case_id: render_legacy_case(query, history, previous_contract, anomaly)[1]
        for case_id, query, history, previous_contract, anomaly in ANALYZER_PROMPT_MATRIX
    }


# --- the pin ---------------------------------------------------------------

def test_legacy_analyzer_prompt_is_byte_stable():
    """The mandate -- "no impact on Standard" -- as an assertion.

    Regenerate deliberately, never reflexively, when the contract or a catalog
    legitimately changes:

        python -c "import tests.test_analyzer_prompt_order as t; t.regenerate()"
    """
    expected = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    actual = {
        case_id: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for case_id, prompt in _budgeted_prompts().items()
    }

    assert set(actual) == set(expected["hashes"]), (
        "matrix membership changed; regenerate the fixture only if the new "
        "matrix is intended"
    )
    drifted = sorted(
        case_id
        for case_id, digest in actual.items()
        if expected["hashes"][case_id] != digest
    )
    assert not drifted, (
        f"legacy analyzer prompt changed for {drifted}. If the selector is "
        "off, Standard's prompt must be byte-identical to what shipped. When "
        "the contract or a catalog changed on purpose, regenerate with: "
        'python -c "import tests.test_analyzer_prompt_order as t; t.regenerate()"'
    )


def test_matrix_covers_every_prompt_family():
    """A narrowed matrix would weaken every other test here silently."""
    families = {
        llm_core._build_analyzer_prompt_context(query, history).prompt_family
        for _case_id, query, history, _prev, _anom in ANALYZER_PROMPT_MATRIX
    }
    assert families == {
        "data",
        "data_explanation",
        "forecast_scenario",
        "knowledge",
    }, families


# --- what the reorder is going to change -----------------------------------

def test_todays_cacheable_prefix_is_negligible():
    """28 characters. This is the defect the plan exists to fix."""
    shared = common_prefix_length(list(_budgeted_prompts().values()))
    assert shared < 200, (
        f"expected a negligible shared prefix today, measured {shared}"
    )


def test_the_output_schema_trails_every_block_today():
    """Moving only the question would leave 15k characters behind it.

    ``_render_analyzer_prompt`` appends the schema after the last block, so the
    single largest constant is the last thing in the prompt. Phase 2 has to
    move this too, and this test is what notices if it does not.
    """
    schema_text = llm_core._compact_json(QuestionAnalysis.model_json_schema())
    for case_id, query, history, previous_contract, anomaly in ANALYZER_PROMPT_MATRIX:
        prompt, _budgeted, _priority = render_legacy_case(
            query, history, previous_contract, anomaly
        )
        assert prompt.startswith("UNTRUSTED_USER_QUESTION:\n<<<"), case_id
        assert prompt.endswith(schema_text), case_id


# --- invariants the reorder depends on -------------------------------------

def test_header_blocks_are_never_truncation_candidates():
    """The constants-first header is only stable if it cannot be dropped.

    ``tests/test_question_analyzer_phase_c.py`` already asserts CONTRACT_* tags
    are disjoint from the truncation lists. This states the same requirement
    from the caching side, naming the exact blocks the header will contain, so
    adding one of them to a priority list fails here with the reason.
    """
    header_blocks = {
        "CONTRACT_QUERY_TYPE_GUIDE",
        "CONTRACT_ANSWER_KIND_GUIDE",
        "CONTRACT_RULES",
    }
    for profile in (
        llm_core._ANALYZER_TRUNCATION_DATA,
        llm_core._ANALYZER_TRUNCATION_KNOWLEDGE,
    ):
        assert header_blocks.isdisjoint(profile), (
            "a header block became truncation-eligible; the constants-first "
            "prefix breaks at the first block whose length varies"
        )


def test_every_block_keeps_its_label_and_delimiters():
    """Injection defence is the delimiters plus the system message, not order.

    The reorder moves blocks; it must never move a label away from its body.
    """
    for case_id, query, history, previous_contract, anomaly in ANALYZER_PROMPT_MATRIX:
        prompt_context = llm_core._build_analyzer_prompt_context(query, history)
        blocks = llm_core._build_analyzer_prompt_blocks(
            query,
            prompt_context.history_str,
            prompt_context.effective_pre_type,
            prompt_context.prompt_profile,
            prompt_context=prompt_context,
            previous_contract=previous_contract,
            evidence_anomaly_note=anomaly,
        )
        prompt = llm_core._render_analyzer_prompt(
            blocks, QuestionAnalysis.model_json_schema()
        )
        for name, body in blocks:
            assert f"{name}:\n<<<{body}>>>" in prompt, f"{case_id}: {name}"


def test_the_untrusted_question_survives_every_budget_path():
    """It is in neither truncation list, so no budget pressure can drop it."""
    for case_id, query, history, previous_contract, anomaly in ANALYZER_PROMPT_MATRIX:
        _prompt, budgeted, _priority = render_legacy_case(
            query, history, previous_contract, anomaly
        )
        assert "UNTRUSTED_USER_QUESTION:\n<<<" in budgeted, case_id


def test_the_emergency_fallback_keeps_the_schema():
    """No matrix case reaches the emergency path, so exercise it deliberately.

    All 40 cases are handled by ``_section_aware_truncate``. The path that
    matters for the reorder is the other one: when section-aware truncation
    cannot free enough room it raises, and ``_protected_section_fallback_truncate``
    rebuilds the prompt from prefix + surviving tagged sections + suffix.
    """
    query, history = _QUERIES["scenario"]
    schema_text = llm_core._compact_json(QuestionAnalysis.model_json_schema())
    prompt_context = llm_core._build_analyzer_prompt_context(query, history)
    blocks = llm_core._build_analyzer_prompt_blocks(
        query,
        prompt_context.history_str,
        prompt_context.effective_pre_type,
        prompt_context.prompt_profile,
        prompt_context=prompt_context,
    )
    prompt = llm_core._render_analyzer_prompt(
        blocks, QuestionAnalysis.model_json_schema()
    )
    priority = llm_core._select_analyzer_truncation_priority(
        query,
        prompt_context.effective_pre_type,
        prompt_context.prompt_profile,
        prompt_context=prompt_context,
    )

    # A budget no amount of eligible truncation can satisfy.
    trimmed = llm_core._enforce_prompt_budget(
        prompt,
        label="question_analysis",
        budget_override=5_000,
        truncation_priority=priority,
    )

    assert schema_text in trimmed
    assert "UNTRUSTED_USER_QUESTION:\n<<<" in trimmed


def test_untagged_text_between_sections_is_discarded_by_the_fallback():
    """The constraint that decides where the schema may be placed.

    ``_protected_section_fallback_truncate`` reconstructs from the text before
    the first tagged section, the surviving tagged sections, and the text after
    the last one. Anything untagged in between is dropped. So the reordered
    prompt may put the raw schema in the prefix, but never between two blocks.
    """
    from core.prompt_budget import _protected_section_fallback_truncate

    prompt = "\n\n".join(
        [
            "PREFIX KEEPS THIS",
            "CONTRACT_RULES:\n<<<rules body>>>",
            "INTERSTITIAL LOSES THIS",
            "UNTRUSTED_USER_QUESTION:\n<<<the question>>>",
            "SUFFIX KEEPS THIS",
        ]
    )

    rebuilt = _protected_section_fallback_truncate(
        prompt, 10, "probe", ["UNTRUSTED_CONVERSATION_HISTORY"]
    )

    assert "PREFIX KEEPS THIS" in rebuilt
    assert "SUFFIX KEEPS THIS" in rebuilt
    assert "CONTRACT_RULES:\n<<<rules body>>>" in rebuilt
    assert "INTERSTITIAL LOSES THIS" not in rebuilt


def test_most_of_the_matrix_truncates_under_the_default_budget():
    """Truncation is the norm here, not an edge case.

    36 of 40 profiles exceed the 40,500-char effective budget today, so the
    analyzer routes with a shortened tool catalog on almost every call. That is
    a separate defect from caching -- recorded here so raising the budget shows
    up as a deliberate, visible change rather than a quiet one.
    """
    truncated = [
        case_id
        for case_id, query, history, previous_contract, anomaly in ANALYZER_PROMPT_MATRIX
        if len(render_legacy_case(query, history, previous_contract, anomaly)[0])
        > int(DEFAULT_ANALYZER_BUDGET_CHARS * 0.90)
    ]
    assert len(truncated) == 36, (
        f"{len(truncated)} of {len(ANALYZER_PROMPT_MATRIX)} cases truncate; "
        "update this count deliberately when the analyzer budget changes"
    )


# --- constants-first ordering ----------------------------------------------

def render_constants_first_case(
    query: str,
    history: list | None,
    previous_contract: str,
    anomaly: str,
) -> tuple[str, str, list[str]]:
    """Same as :func:`render_legacy_case`, under the constants-first order."""
    schema_hint = QuestionAnalysis.model_json_schema()
    prompt_context = llm_core._build_analyzer_prompt_context(query, history)
    blocks = llm_core._build_analyzer_prompt_blocks(
        query,
        prompt_context.history_str,
        prompt_context.effective_pre_type,
        prompt_context.prompt_profile,
        prompt_context=prompt_context,
        previous_contract=previous_contract,
        evidence_anomaly_note=anomaly,
        order=llm_core._ANALYZER_ORDER_CONSTANTS_FIRST,
    )
    prompt = llm_core._render_analyzer_prompt(
        blocks, schema_hint, order=llm_core._ANALYZER_ORDER_CONSTANTS_FIRST
    )
    priority = llm_core._select_analyzer_truncation_priority(
        query,
        prompt_context.effective_pre_type,
        prompt_context.prompt_profile,
        prompt_context=prompt_context,
    )
    budgeted = llm_core._enforce_prompt_budget(
        prompt,
        label="question_analysis",
        budget_override=DEFAULT_ANALYZER_BUDGET_CHARS,
        truncation_priority=priority,
    )
    return prompt, budgeted, priority


def _constants_first_prompts() -> dict[str, str]:
    return {
        case_id: render_constants_first_case(
            query, history, previous_contract, anomaly
        )[1]
        for case_id, query, history, previous_contract, anomaly in ANALYZER_PROMPT_MATRIX
    }


def test_reordering_moves_blocks_without_changing_any_of_them():
    """A reorder must not become a rewrite.

    Set equality on block name to body, so a block that quietly loses content
    or gains an edit fails here rather than in production.
    """
    for case_id, query, history, previous_contract, anomaly in ANALYZER_PROMPT_MATRIX:
        prompt_context = llm_core._build_analyzer_prompt_context(query, history)
        arguments = dict(
            prompt_context=prompt_context,
            previous_contract=previous_contract,
            evidence_anomaly_note=anomaly,
        )
        legacy = llm_core._build_analyzer_prompt_blocks(
            query,
            prompt_context.history_str,
            prompt_context.effective_pre_type,
            prompt_context.prompt_profile,
            **arguments,
        )
        reordered = llm_core._build_analyzer_prompt_blocks(
            query,
            prompt_context.history_str,
            prompt_context.effective_pre_type,
            prompt_context.prompt_profile,
            order=llm_core._ANALYZER_ORDER_CONSTANTS_FIRST,
            **arguments,
        )
        assert dict(legacy) == dict(reordered), case_id
        assert [name for name, _ in legacy] != [
            name for name, _ in reordered
        ], f"{case_id}: order did not actually change"


def test_the_schema_leads_and_the_question_follows_the_contract_blocks():
    for case_id, query, history, previous_contract, anomaly in ANALYZER_PROMPT_MATRIX:
        prompt, _budgeted, _priority = render_constants_first_case(
            query, history, previous_contract, anomaly
        )
        assert prompt.startswith("Respond with JSON exactly matching this schema:"), case_id
        for name in (
            "CONTRACT_QUERY_TYPE_GUIDE",
            "CONTRACT_ANSWER_KIND_GUIDE",
            "CONTRACT_RULES",
        ):
            assert prompt.index(f"{name}:\n<<<") < prompt.index(
                "UNTRUSTED_USER_QUESTION:\n<<<"
            ), f"{case_id}: {name} must precede the question"


def test_the_previous_contract_still_directly_follows_the_question():
    """``tests/test_contract_continuity.py`` pins this for the legacy order.

    The block is only interpretable next to the question it qualifies, so the
    adjacency has to survive the reorder.
    """
    blocks = llm_core._build_analyzer_prompt_blocks(
        "and for 2023?",
        "",
        "single_value",
        "default",
        previous_contract='{"top_tool":"get_prices"}',
        order=llm_core._ANALYZER_ORDER_CONSTANTS_FIRST,
    )
    names = [name for name, _ in blocks]
    question = names.index("UNTRUSTED_USER_QUESTION")
    assert names[question + 1] == llm_core._ANALYZER_BLOCK_PREVIOUS_CONTRACT


def test_the_constant_prefix_survives_the_budget():
    """The measurement the whole plan turns on.

    Post-budget, across every family and context combination, including the
    longest report-track composite the report path can produce. Truncation
    fires in most of these; the header has to be untouched by it.
    """
    shared = common_prefix_length(list(_constants_first_prompts().values()))
    assert shared >= 30_000, (
        f"constants-first prefix collapsed to {shared} chars; a block whose "
        "length varies has moved into the header"
    )


def test_the_header_is_identical_across_prompt_families():
    """Families select different blocks, so the header must precede that."""
    by_family: dict[str, list[str]] = {}
    for case_id, query, history, previous_contract, anomaly in ANALYZER_PROMPT_MATRIX:
        family = llm_core._build_analyzer_prompt_context(query, history).prompt_family
        by_family.setdefault(family, []).append(
            render_constants_first_case(query, history, previous_contract, anomaly)[1]
        )
    header_lengths = {
        family: common_prefix_length(prompts)
        for family, prompts in by_family.items()
    }
    cross_family = common_prefix_length(
        [prompts[0] for prompts in by_family.values()]
    )
    assert cross_family >= 30_000, (
        f"header differs between families ({cross_family} chars shared, "
        f"within-family {header_lengths})"
    )


def test_constants_first_keeps_every_label_and_delimiter():
    for case_id, query, history, previous_contract, anomaly in ANALYZER_PROMPT_MATRIX:
        prompt_context = llm_core._build_analyzer_prompt_context(query, history)
        blocks = llm_core._build_analyzer_prompt_blocks(
            query,
            prompt_context.history_str,
            prompt_context.effective_pre_type,
            prompt_context.prompt_profile,
            prompt_context=prompt_context,
            previous_contract=previous_contract,
            evidence_anomaly_note=anomaly,
            order=llm_core._ANALYZER_ORDER_CONSTANTS_FIRST,
        )
        prompt = llm_core._render_analyzer_prompt(
            blocks,
            QuestionAnalysis.model_json_schema(),
            order=llm_core._ANALYZER_ORDER_CONSTANTS_FIRST,
        )
        for name, body in blocks:
            assert f"{name}:\n<<<{body}>>>" in prompt, f"{case_id}: {name}"


def test_the_schema_is_prefix_text_not_interstitial():
    """The emergency fallback keeps the prefix and drops untagged interstitials.

    Placing the schema first is only safe because nothing tagged precedes it.
    """
    prompt, _budgeted, priority = render_constants_first_case(
        *_QUERIES["scenario"], "", ""
    )
    schema_text = llm_core._compact_json(QuestionAnalysis.model_json_schema())
    first_tagged = prompt.index("CONTRACT_QUERY_TYPE_GUIDE:\n<<<")
    assert prompt.index(schema_text) < first_tagged

    trimmed = llm_core._enforce_prompt_budget(
        prompt,
        label="question_analysis",
        budget_override=5_000,
        truncation_priority=priority,
    )
    assert schema_text in trimmed
    assert "UNTRUSTED_USER_QUESTION:\n<<<" in trimmed


# --- the selector ------------------------------------------------------------

def test_standard_is_untouched_while_the_selector_is_off(monkeypatch):
    """The mandate. Off means byte-identical, not merely equivalent."""
    for mode in ("off", "report"):
        monkeypatch.setattr(llm_core, "ANALYZER_CONSTANTS_FIRST_MODE", mode)
        assert llm_core._analyzer_prompt_order(report_profile=False) == (
            llm_core._ANALYZER_ORDER_LEGACY
        ), mode


def test_report_opts_in_before_standard_does(monkeypatch):
    monkeypatch.setattr(llm_core, "ANALYZER_CONSTANTS_FIRST_MODE", "report")
    assert llm_core._analyzer_prompt_order(report_profile=True) == (
        llm_core._ANALYZER_ORDER_CONSTANTS_FIRST
    )


def test_all_covers_both_modes(monkeypatch):
    monkeypatch.setattr(llm_core, "ANALYZER_CONSTANTS_FIRST_MODE", "all")
    for report_profile in (False, True):
        assert llm_core._analyzer_prompt_order(report_profile) == (
            llm_core._ANALYZER_ORDER_CONSTANTS_FIRST
        )


def test_an_unknown_selector_value_reads_as_off(monkeypatch):
    """An ordering nobody asked for is the wrong way to fail."""
    monkeypatch.setenv("ENAI_ANALYZER_CONSTANTS_FIRST", "yes-please")
    import importlib

    import config

    reloaded = importlib.reload(config)
    try:
        assert reloaded.ANALYZER_CONSTANTS_FIRST_MODE == "off"
    finally:
        monkeypatch.delenv("ENAI_ANALYZER_CONSTANTS_FIRST", raising=False)
        importlib.reload(config)


def test_the_selector_defaults_to_off():
    """Phase 2 ships dark."""
    import config

    assert config.ANALYZER_CONSTANTS_FIRST_MODE == "off"
    assert llm_core._analyzer_prompt_order(report_profile=True) == (
        llm_core._ANALYZER_ORDER_LEGACY
    )


# --- the routing-affinity key -----------------------------------------------

def test_no_cache_key_is_sent_while_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(llm_core, "ENABLE_ANALYZER_PROMPT_CACHE_KEY", False)
    for order in (
        llm_core._ANALYZER_ORDER_LEGACY,
        llm_core._ANALYZER_ORDER_CONSTANTS_FIRST,
    ):
        assert llm_core._analyzer_prompt_cache_key(order) == ""


def test_no_cache_key_under_the_legacy_order(monkeypatch):
    """A shared key across a 28-character prefix routes calls together for nothing."""
    monkeypatch.setattr(llm_core, "ENABLE_ANALYZER_PROMPT_CACHE_KEY", True)
    assert llm_core._analyzer_prompt_cache_key(llm_core._ANALYZER_ORDER_LEGACY) == ""


def test_the_cache_key_is_stable_and_derived_from_the_schema(monkeypatch):
    """Adding a knowledge topic changes the header, so it must change the key.

    The topic names live inside the schema via KnowledgeTopicName, so a digest
    moves automatically where a hand-bumped version would be forgotten.
    """
    import hashlib

    monkeypatch.setattr(llm_core, "ENABLE_ANALYZER_PROMPT_CACHE_KEY", True)
    monkeypatch.setattr(llm_core, "_ANALYZER_SCHEMA_DIGEST", "")

    key = llm_core._analyzer_prompt_cache_key(
        llm_core._ANALYZER_ORDER_CONSTANTS_FIRST
    )
    expected = hashlib.sha256(
        llm_core._compact_json(QuestionAnalysis.model_json_schema()).encode("utf-8")
    ).hexdigest()[:12]

    assert key == f"enai-analyzer-constants_first-{expected}"
    assert key == llm_core._analyzer_prompt_cache_key(
        llm_core._ANALYZER_ORDER_CONSTANTS_FIRST
    ), "key must not vary between calls in a process"


def test_the_key_reaches_openai_and_no_other_provider():
    """It is an OpenAI argument; other providers would reject it."""
    from core.provider_invocation import ProviderInvocationRuntime

    kwargs = ProviderInvocationRuntime._invoke_kwargs(
        "openai", 30.0, None, "enai-analyzer-x"
    )
    assert kwargs["prompt_cache_key"] == "enai-analyzer-x"

    # Allow-listed, not deny-listed: an unknown provider gets nothing rather
    # than inheriting an argument its SDK would reject.
    for provider in ("gemini", "nvidia", "some_future_provider"):
        kwargs = ProviderInvocationRuntime._invoke_kwargs(
            provider, 30.0, None, "enai-analyzer-x"
        )
        assert "prompt_cache_key" not in kwargs, provider


def test_no_key_means_no_argument_at_all():
    """Every other stage must send exactly what it sent before."""
    from core.provider_invocation import ProviderInvocationRuntime

    for provider in ("openai", "gemini", "nvidia"):
        kwargs = ProviderInvocationRuntime._invoke_kwargs(provider, 30.0, None, "")
        assert "prompt_cache_key" not in kwargs, provider


def test_the_key_does_not_leak_to_the_next_stage_on_the_thread(monkeypatch):
    """Report tracks analyze on pooled threads; a leaked key would follow one."""
    monkeypatch.setattr(llm_core, "ANALYZER_CONSTANTS_FIRST_MODE", "all")
    monkeypatch.setattr(llm_core, "ENABLE_ANALYZER_PROMPT_CACHE_KEY", True)
    seen: list[str] = []

    def capture(*_args, **_kwargs):
        seen.append(llm_core._LLM_PROMPT_CACHE_KEY.get())
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(llm_core, "_invoke_with_openai_fallback", capture)
    monkeypatch.setattr(llm_core, "_cache_get_or_reserve", lambda _key: (None, None))

    for _attempt in range(2):
        try:
            llm_core.llm_analyze_question("What was the balancing price in May 2026?")
        except Exception:
            pass

    assert seen and all(key.startswith("enai-analyzer-") for key in seen)
    assert llm_core._LLM_PROMPT_CACHE_KEY.get() == "", "key outlived the analyzer call"


def regenerate() -> None:  # pragma: no cover - maintenance helper
    """Rewrite the legacy hash fixture. Never call this to make a test pass."""
    payload = {
        "note": (
            "SHA-256 of the budgeted analyzer prompt per matrix case. "
            "Regenerate only when the contract or a catalog legitimately "
            "changes; a diff here means Standard's prompt moved."
        ),
        "budget_chars": DEFAULT_ANALYZER_BUDGET_CHARS,
        "hashes": {
            case_id: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for case_id, prompt in sorted(_budgeted_prompts().items())
        },
    }
    _FIXTURE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
