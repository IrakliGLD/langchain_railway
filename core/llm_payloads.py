"""
LLM payload parsing and sanitization.

Extracted from ``core/llm.py`` (Q2, 2026-06-10) as a pure structural move:
JSON extraction from model output, relative-date coercion for analyzer period
fields, schema-aware null-list coercion, and QuestionAnalysis payload
sanitizers. Everything here is a pure function over its inputs -- none of it
reads config constants that tests monkeypatch on ``core.llm`` (that is the
criterion that kept other helpers behind; see core/llm_runtime.py note).
"""
import json
import logging
import re
from datetime import date as _date
from typing import Optional

from dateutil.relativedelta import relativedelta

from contracts.question_analysis import (
    _VALID_ROLES_BY_INTENT,
    AnswerKind,
    ChartFamily,
    ChartIntent,
    KnowledgeTopicName,
    MeasureTransform,
    PeriodInfo,
    PresentationMode,
    QuestionAnalysis,
    RenderStyle,
    SemanticRole,
    SeriesSplitMode,
    VisualGoal,
    VisualizationTimeGrain,
)

log = logging.getLogger("Enai")


def _extract_json_payload(raw_text: str) -> dict:
    """Extract a JSON object payload from model output.

    Tolerates three drift modes observed in production traces:
      1. Trailing text after the JSON object (Q7 trace 4e9b17da, 2026-05-16:
         ``Extra data: line 1 column 1407``) — Gemini occasionally appends
         commentary after the closing brace.
      2. Markdown fences around the JSON.
      3. Leading prose before the first ``{``.

    Strategy: strip fences, find the first ``{``, then ``raw_decode`` from
    that position. ``raw_decode`` returns the first complete JSON value and
    the index where it stopped — trailing content is silently discarded.
    Falls back to the prior ``find``/``rfind`` heuristic only if ``raw_decode``
    also fails (e.g., malformed object).
    """
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty model output")

    # Remove markdown fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    parsed = None
    start = text.find("{")
    if start != -1:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            parsed = None

    if parsed is None:
        # Fallback: strict full-text parse, then last-resort find/rfind slice.
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object found in model output")
            parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Structured output must be a JSON object")
    return parsed


def _compact_json(value) -> str:
    """Serialize JSON with stable compact formatting for prompt/cache efficiency."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_PERIOD_RAW_TEXT_MAX_LENGTH = (
    PeriodInfo.model_json_schema()
    .get("properties", {})
    .get("raw_text", {})
    .get("anyOf", [{}])[0]
    .get("maxLength", 120)
)


# ---------------------------------------------------------------------------
# Relative-date coercion for analyzer-emitted period fields.
#
# The contract (`contracts.question_analysis.ISODate`) constrains period
# fields to `^\d{4}-\d{2}-\d{2}$`, but Gemini has been observed to emit
# relative tokens like ``"12m"`` (12 months back) and ``"now"`` (today) —
# see Q3 production trace 7f6fc4b0 (2026-05-16). Pydantic rejection of the
# stray token previously crashed the full QuestionAnalysis validation and
# forced a ~8s heuristic fallback. The sanitizer normalizes recognized
# relative tokens to ISO before validation; un-coercible tokens cause the
# enclosing period block to be dropped (graceful degradation — downstream
# SQL planner re-derives the period from query text).
# ---------------------------------------------------------------------------
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RELATIVE_DATE_RE = re.compile(r"^\s*(\d+)\s*([dwmqy])\s*$", re.IGNORECASE)


def _coerce_relative_date(token: object, *, anchor: _date, role: str) -> Optional[str]:
    """Normalize an analyzer-emitted date token to ISO ``YYYY-MM-DD`` or None.

    Tolerated drift modes:
      - already-ISO string → returned verbatim
      - ``"now"`` / ``"today"`` (case-insensitive) → ``anchor.isoformat()``
      - ``"<N>d|w|m|q|y"`` (e.g. ``"12m"``) → calendar-aware offset using
        ``dateutil.relativedelta``. For ``role="start"`` the result is
        ``anchor - N units``; for ``role="end"`` the result is ``anchor``
        (intuitive when paired with ``start='Nm', end='now'`` for a
        rolling-window query).

    Returns ``None`` when the token is unrecognizable. The caller should
    drop the enclosing period block in that case rather than emit invalid
    ISO that would crash strict Pydantic validation.
    """
    if isinstance(token, _date):
        return token.isoformat()
    if not isinstance(token, str):
        return None
    text = token.strip().lower()
    if not text:
        return None
    if _ISO_DATE_RE.match(text):
        return text
    if text in {"now", "today"}:
        return anchor.isoformat()
    match = _RELATIVE_DATE_RE.match(text)
    if not match:
        return None
    n = int(match.group(1))
    unit = match.group(2).lower()
    if role == "end":
        # Relative tokens at end-of-window anchor to "now" — the natural
        # interpretation of e.g. start='12m', end='12m' is "12 months back
        # through now", not "a 24-month range centred on now".
        return anchor.isoformat()
    # role == "start": calendar-aware offset back from anchor.
    if unit == "d":
        delta = relativedelta(days=n)
    elif unit == "w":
        delta = relativedelta(weeks=n)
    elif unit == "m":
        delta = relativedelta(months=n)
    elif unit == "q":
        delta = relativedelta(months=n * 3)
    elif unit == "y":
        delta = relativedelta(years=n)
    else:
        return None
    return (anchor - delta).isoformat()


def _normalize_period_dates_inplace(container: dict, *, anchor: _date) -> bool:
    """Normalize ``start_date`` and ``end_date`` keys on a period-like dict.

    Returns True when both dates are present and ISO-valid (either already
    or after coercion). Returns False when either is missing or could not
    be coerced — caller should drop the period block.
    """
    if not isinstance(container, dict):
        return False
    start_raw = container.get("start_date")
    end_raw = container.get("end_date")
    if start_raw is None and end_raw is None:
        return False
    start_iso = _coerce_relative_date(start_raw, anchor=anchor, role="start") if start_raw is not None else None
    end_iso = _coerce_relative_date(end_raw, anchor=anchor, role="end") if end_raw is not None else None
    if not start_iso or not end_iso:
        return False
    # Enforce start <= end; if reversed after coercion (very rare), drop —
    # PeriodInfo._validate_date_order would otherwise crash Pydantic.
    try:
        if _date.fromisoformat(end_iso) < _date.fromisoformat(start_iso):
            return False
    except ValueError:
        return False
    container["start_date"] = start_iso
    container["end_date"] = end_iso
    return True


_QUESTION_ANALYSIS_SCHEMA = QuestionAnalysis.model_json_schema()
_QUESTION_ANALYSIS_SCHEMA_DEFS = _QUESTION_ANALYSIS_SCHEMA.get("$defs", {})

# Allowed values for ``knowledge.candidate_topics[*].name``. Used by the
# sanitizer to drop unknown topic names emitted by the LLM rather than fail
# the entire ``QuestionAnalysis`` validation (Q6 production trace b19e2464,
# 2026-05-16: "knowledge.candidate_topics.1.name 'regulatory_procedure'
# Input should be ... enum values").
_KNOWN_TOPIC_NAMES = frozenset(t.value for t in KnowledgeTopicName)


def _resolve_schema_node(schema: dict | None) -> dict:
    """Resolve simple local $ref pointers inside the QuestionAnalysis schema."""
    if not isinstance(schema, dict):
        return {}
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return schema
    return _QUESTION_ANALYSIS_SCHEMA_DEFS.get(ref.split("/")[-1], schema)


def _schema_allows_array(schema: dict | None) -> bool:
    """Return True when the schema node accepts an array value."""
    node = _resolve_schema_node(schema)
    if not isinstance(node, dict):
        return False
    if node.get("type") == "array":
        return True
    for option in node.get("anyOf", []):
        resolved = _resolve_schema_node(option)
        if isinstance(resolved, dict) and resolved.get("type") == "array":
            return True
    return False


def _coerce_null_lists_from_schema(value, schema: dict | None):
    """Recursively coerce nulls to [] for fields declared as arrays in the contract."""
    node = _resolve_schema_node(schema)
    if not isinstance(node, dict):
        return value

    if value is None and _schema_allows_array(node):
        return []

    if value is None:
        return value

    if "anyOf" in node:
        for option in node.get("anyOf", []):
            resolved = _resolve_schema_node(option)
            if not isinstance(resolved, dict):
                continue
            option_type = resolved.get("type")
            if option_type == "object" and isinstance(value, dict):
                return _coerce_null_lists_from_schema(value, resolved)
            if option_type == "array" and isinstance(value, list):
                return _coerce_null_lists_from_schema(value, resolved)
        return value

    node_type = node.get("type")
    if node_type == "object" and isinstance(value, dict):
        properties = node.get("properties", {})
        for key, child_schema in properties.items():
            if key in value:
                value[key] = _coerce_null_lists_from_schema(value.get(key), child_schema)
        return value

    if node_type == "array" and isinstance(value, list):
        item_schema = node.get("items")
        if item_schema is None:
            return value
        return [_coerce_null_lists_from_schema(item, item_schema) for item in value]

    return value


def _sanitize_question_analysis_payload(payload: dict) -> dict:
    """Best-effort cleanup for question-analysis payloads before model validation."""
    if not isinstance(payload, dict):
        return payload

    payload = _coerce_null_lists_from_schema(payload, _QUESTION_ANALYSIS_SCHEMA)

    def _pop_dict(source: dict, key: str) -> dict | None:
        value = source.pop(key, None)
        return value if isinstance(value, dict) else None

    def _sanitize_topic_candidates(raw_topics: object) -> list[dict]:
        if not isinstance(raw_topics, list):
            return []
        sanitized_topics: list[dict] = []
        for item in raw_topics:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                score = item.get("score", 0.5)
            else:
                name = str(item or "").strip()
                score = 0.5
            if not name:
                continue
            # Phase 3 (2026-05-16): drop unknown topic names rather than
            # crash the entire QuestionAnalysis validation. Q6 trace
            # b19e2464: Gemini emitted "regulatory_procedure" which is
            # not in ``KnowledgeTopicName``. Keeping the other valid
            # candidate topics preserves more analyzer signal than the
            # heuristic fallback would.
            if name not in _KNOWN_TOPIC_NAMES:
                continue
            try:
                score_val = max(0.0, min(1.0, float(score)))
            except (TypeError, ValueError):
                score_val = 0.5
            sanitized_topics.append({"name": name, "score": score_val})
        return sanitized_topics

    def _guess_tool_name_from_params_hint(params_hint: dict) -> str | None:
        if not isinstance(params_hint, dict):
            return None
        if params_hint.get("types") or params_hint.get("mode") in {"quantity", "share"}:
            return "get_generation_mix"
        entities = {
            str(entity).strip().lower()
            for entity in (params_hint.get("entities") or [])
            if str(entity).strip()
        }
        if entities:
            tariff_like = {
                "enguri", "vardnili", "gardabani", "regulated_hpp",
                "regulated_new_tpp", "regulated_old_tpp", "old_tpp_group",
            }
            if entities & tariff_like:
                return "get_tariffs"
            return "get_balancing_composition"
        if any(params_hint.get(key) is not None for key in ("metric", "currency", "granularity", "start_date", "end_date")):
            return "get_prices"
        return None

    classification = payload.get("classification")
    if isinstance(classification, dict):
        raw_ambiguities = payload.pop("ambiguities", None)
        if raw_ambiguities is not None and not classification.get("ambiguities"):
            classification["ambiguities"] = raw_ambiguities

    knowledge = payload.get("knowledge")
    if isinstance(knowledge, dict):
        raw_topics = payload.pop("candidate_topics", None)
        if raw_topics is not None and not knowledge.get("candidate_topics"):
            sanitized_topics = _sanitize_topic_candidates(raw_topics)
            if sanitized_topics:
                knowledge["candidate_topics"] = sanitized_topics
        # Phase 3 (2026-05-16): always sanitize the candidate_topics on the
        # knowledge object — the LLM normally emits them here directly
        # (not via the top-level merge above), and we need to drop unknown
        # enum values (Q6 trace b19e2464) before Pydantic validation.
        direct_topics = knowledge.get("candidate_topics")
        if isinstance(direct_topics, list):
            knowledge["candidate_topics"] = _sanitize_topic_candidates(direct_topics)

    tooling = payload.get("tooling")
    if isinstance(tooling, dict):
        raw_candidate_tools = payload.pop("candidate_tools", None)
        if isinstance(raw_candidate_tools, list) and not tooling.get("candidate_tools"):
            tooling["candidate_tools"] = raw_candidate_tools

        raw_params_hint = _pop_dict(tooling, "params_hint")
        if raw_params_hint is None:
            raw_params_hint = _pop_dict(payload, "params_hint")
        if raw_params_hint is not None:
            candidate_tools = tooling.get("candidate_tools")
            if isinstance(candidate_tools, list) and candidate_tools:
                first_candidate = candidate_tools[0]
                if isinstance(first_candidate, dict) and not first_candidate.get("params_hint"):
                    first_candidate["params_hint"] = raw_params_hint
            else:
                guessed_tool = _guess_tool_name_from_params_hint(raw_params_hint)
                if guessed_tool:
                    tooling["candidate_tools"] = [{
                        "name": guessed_tool,
                        "score": 0.5,
                        "reason": "sanitized params_hint",
                        "params_hint": raw_params_hint,
                    }]

    sql_hints = payload.get("sql_hints")
    if sql_hints is None:
        payload["sql_hints"] = {}
        sql_hints = payload["sql_hints"]
    if isinstance(sql_hints, dict):
        if sql_hints.get("dimensions") is None:
            sql_hints["dimensions"] = []
        period = sql_hints.get("period")
        if isinstance(period, dict):
            # Phase 2 (2026-05-16): coerce relative tokens like "12m" / "now"
            # to ISO YYYY-MM-DD before Pydantic sees them (Q3 trace 7f6fc4b0).
            # If coercion fails for either bound, drop the period — downstream
            # SQL planner re-derives the period from query text.
            anchor = _date.today()
            if not _normalize_period_dates_inplace(period, anchor=anchor):
                sql_hints.pop("period", None)
            else:
                raw_text = period.get("raw_text")
                if isinstance(raw_text, str) and len(raw_text) > _PERIOD_RAW_TEXT_MAX_LENGTH:
                    period.pop("raw_text", None)
        elif period is None:
            sql_hints.pop("period", None)

    # Apply the same coercion to tooling.candidate_tools[*].params_hint dates.
    # These also use the ISODate contract (Optional, so the field can be
    # nulled rather than dropping the whole hint).
    tooling_hint_anchor = _date.today()
    tooling_for_dates = payload.get("tooling")
    if isinstance(tooling_for_dates, dict):
        for candidate in tooling_for_dates.get("candidate_tools") or []:
            if not isinstance(candidate, dict):
                continue
            params_hint = candidate.get("params_hint")
            if not isinstance(params_hint, dict):
                continue
            start_raw = params_hint.get("start_date")
            end_raw = params_hint.get("end_date")
            if start_raw is not None:
                coerced = _coerce_relative_date(start_raw, anchor=tooling_hint_anchor, role="start")
                if coerced is None:
                    params_hint.pop("start_date", None)
                else:
                    params_hint["start_date"] = coerced
            if end_raw is not None:
                coerced = _coerce_relative_date(end_raw, anchor=tooling_hint_anchor, role="end")
                if coerced is None:
                    params_hint.pop("end_date", None)
                else:
                    params_hint["end_date"] = coerced

    vis = payload.get("visualization")
    if not isinstance(vis, dict):
        return payload

    raw_family = vis.get("preferred_chart_family")
    if isinstance(raw_family, str):
        try:
            vis["preferred_chart_family"] = ChartFamily(raw_family).value
        except ValueError:
            vis.pop("preferred_chart_family", None)

    raw_presentation = vis.get("primary_presentation")
    if isinstance(raw_presentation, str):
        try:
            vis["primary_presentation"] = PresentationMode(raw_presentation).value
        except ValueError:
            vis.pop("primary_presentation", None)

    raw_goal = vis.get("visual_goal")
    if isinstance(raw_goal, str):
        try:
            vis["visual_goal"] = VisualGoal(raw_goal).value
        except ValueError:
            vis.pop("visual_goal", None)

    raw_transform = vis.get("measure_transform")
    if isinstance(raw_transform, str):
        try:
            vis["measure_transform"] = MeasureTransform(raw_transform).value
        except ValueError:
            vis.pop("measure_transform", None)

    raw_time_grain = vis.get("time_grain")
    if isinstance(raw_time_grain, str):
        try:
            vis["time_grain"] = VisualizationTimeGrain(raw_time_grain).value
        except ValueError:
            vis.pop("time_grain", None)

    raw_split_mode = vis.get("series_split_mode")
    if isinstance(raw_split_mode, str):
        try:
            vis["series_split_mode"] = SeriesSplitMode(raw_split_mode).value
        except ValueError:
            vis.pop("series_split_mode", None)

    raw_max_series = vis.get("max_series")
    if raw_max_series is not None:
        try:
            max_series = int(raw_max_series)
        except (TypeError, ValueError):
            vis.pop("max_series", None)
        else:
            if 1 <= max_series <= 8:
                vis["max_series"] = max_series
            else:
                vis.pop("max_series", None)

    chart_requested = bool(vis.get("chart_requested_by_user"))
    chart_recommended = bool(vis.get("chart_recommended"))
    if (
        vis.get("primary_presentation")
        in {PresentationMode.CHART.value, PresentationMode.CHART_PLUS_TABLE.value}
        and not chart_requested
        and not chart_recommended
    ):
        chart_recommended = True
        vis["chart_recommended"] = True
    if not chart_requested and not chart_recommended:
        vis.pop("chart_intent", None)
        vis.pop("target_series", None)
        return payload

    chart_intent = None
    raw_intent = vis.get("chart_intent")
    if isinstance(raw_intent, str):
        try:
            chart_intent = ChartIntent(raw_intent)
            vis["chart_intent"] = chart_intent.value
        except ValueError:
            vis.pop("chart_intent", None)

    raw_roles = vis.get("target_series")
    sanitized_roles: list[str] = []
    if isinstance(raw_roles, list):
        for role in raw_roles:
            if not isinstance(role, str):
                continue
            try:
                sanitized_roles.append(SemanticRole(role).value)
            except ValueError:
                continue

    if sanitized_roles:
        vis["target_series"] = sanitized_roles
    else:
        vis.pop("target_series", None)

    if chart_intent is not None:
        allowed_roles = _VALID_ROLES_BY_INTENT.get(chart_intent, frozenset())
        if not vis.get("target_series") or any(
            SemanticRole(role) not in allowed_roles for role in vis["target_series"]
        ):
            vis.pop("chart_intent", None)
            vis.pop("target_series", None)
    else:
        vis.pop("target_series", None)

    return payload


def _sanitize_chart_hints(payload: dict) -> dict:
    """Backward-compatible alias for question-analysis payload sanitization."""
    return _sanitize_question_analysis_payload(payload)


