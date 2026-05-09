# Reading pipeline logs

The `/ask` request emits structured TRACE entries plus free-form INFO/WARNING lines. This is how to read them efficiently.

## What to grab first

When a user pastes a log, find these lines in order:

1. **The first `TRACE` for `stage_0_2_question_analyzer`**. This is the most important diagnostic line in the entire log. It contains:
   - `type=` (query_type the LLM picked)
   - `answer_kind=` and `render_style=` (the answer contract)
   - `confidence=` (LLM's self-assessed certainty)
   - `mode_disagree`, `conceptual_disagree` (heuristic vs LLM disagreement flags)
   - `error=` (true if the analyzer failed validation)
   - `duration_ms=` (router latency)

2. **Any `WARNING answer_kind cross-check disagreement` line**. Format:
   ```
   answer_kind cross-check disagreement: llm=<X> derived=<Y> chosen=<Z>
   ```
   If `chosen != llm`, the code overrode the model's contract. This is the most common silent quality bug.

3. **Any `WARNING Question analyzer failed`** — schema validation crashed. Look at the validation error; it tells you which field the LLM emitted incorrectly.

4. **`Prompt budget applied (section-aware)` WARNING** — input was truncated. Note the `truncated_sections` list and the `original` vs `final` chars.

5. **`stage_4_*_summary` TRACE entry** — final answer stage. Note `summary_source` (`structured_summary`, `citation_gate_fallback`, `legacy_text_fallback`, `deterministic_scenario_fallback`, `generic_renderer`). Anything ending in `_fallback` means quality degraded.

6. **`provenance_gate` TRACE_DETAIL** — `gate_passed=false` with `coverage<0.80` means numbers in the answer didn't match the evidence. Look at `unmatched_tokens` to see which numbers were ungrounded.

7. **`Finished request in X.XXs`** — total request time. Compare against latency tables below.

## Latency expectations (per stage, normal range)

| Stage | Normal | Concerning | Likely issue if exceeded |
|---|---|---|---|
| `stage_0_prepare_context` | < 5 ms | > 50 ms | shouldn't happen |
| `stage_0_2_question_analyzer` | 2–6 s | > 12 s | thinking_budget too high; or model name not resolving cleanly |
| `stage_0_3_vector_knowledge` | 0.3–1.5 s | > 4 s | embedding provider slow; switch to OpenAI text-embedding-3-small |
| `stage_0_6_tool_execute` | 50–300 ms | > 1 s | DB pool / query plan |
| `stage_3_analyzer_enrich` | 100–300 ms | > 1 s | pandas pipeline |
| `stage_4_summarize_data` | 8–30 s | > 60 s | LLM timeout; check for retry warning |
| `stage_4_conceptual_summary` | 5–20 s | > 30 s | summarizer prompt too large |
| `stage_5_chart_build` | 10–80 ms | > 500 ms | chart selector regex |

## Multi-stage anti-patterns to spot

**Silent retry on summarizer.** Look for:
```
WARNING Retrying langchain_google_genai.chat_models._chat_with_retry ... DeadlineExceeded: 504
```
A summarizer retry doubles total latency. Almost always caused by prompt size on Gemini 2.5-pro hitting the 60s server deadline.

**Cross-check override.** Look for `chosen=knowledge` when `llm=list` or `llm=comparison`. The code's `_SAFE_ANSWER_KINDS = {TIMESERIES, EXPLANATION, KNOWLEDGE}` clobbers narrower shapes.

**Fallback cascade.** When the analyzer fails validation, you'll see `analyzer_available=false`, then `semantic_locked=false`, then heuristic mode picks a path, then summarizer runs with reduced context. Trace the cascade backward to find the first failure (almost always the analyzer).

## Identifying which model handled a stage

```
Stage-specific LLM cached: model=<X> thinking_budget=<Y> max_retries=<Z>
```

The `max_retries` value identifies the call site:
- `max_retries=2` → router (analyzer)
- `max_retries=1` → summarizer
- (no retries arg) → planner / others

If you see two `Stage-specific LLM cached` lines per request with different models, that's normal — first the router model, then the summarizer model. If you see the same model twice, your `ROUTER_MODEL` and `SUMMARIZER_MODEL` are the same.

## Total-time arithmetic

Sum the `duration_ms` of named stages and compare to the `Finished request in X.XXs` line. The gap is "everything else" — middleware, auth, history seeding, response building. If the gap is > 2 s on a normal request, something between the stages is slow (rare; check for memory checks, history loads).

## When the log is truncated

Users sometimes paste only the start of a request. Tells:
- `Stage-specific LLM cached: model=...` for the summarizer is the LAST thing visible → the LLM call hadn't returned yet.
- No `Finished request` line → request still in flight or log was cut.
- Ask the user to grep for `Finished request|stage_4_|TRACE` in the same file for the same `trace_id`.
