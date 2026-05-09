# Failure taxonomy

Each failure pattern below: how to identify it, where it lives in the code, what NOT to do, what to actually fix. Patterns marked **OBSERVED** are ones we have hit and resolved; the post-mortem is in the listed commit.

## A. Schema crash in question analyzer

**Symptom**:
```
WARNING Question analyzer failed | source=llm_active error=Question-analysis schema validation failed
Input should be 'scalar', 'list', ... [type=enum, input_value='<X>']
```

**What's happening**: the LLM emitted a value for a field that isn't in the Pydantic enum. Most often `answer_kind` because `query_type` and `answer_kind` use overlapping vocabularies (`comparison`, `forecast`) but with different domains.

**Where to look**:
- [contracts/question_analysis.py](../../../contracts/question_analysis.py) — enum definitions
- [contracts/question_analysis_catalogs.py](../../../contracts/question_analysis_catalogs.py) — `QUESTION_ANALYSIS_ANSWER_KIND_GUIDE`
- [core/llm.py `_ANALYZER_CORE_RULES`](../../../core/llm.py) — the prompt that tells the LLM how to fill the contract

**Cascade**: schema crash → `analyzer_available=false` → heuristic fallback → `semantic_locked=false` → summarizer runs without firm contract → grounding gate fails → `citation_gate_fallback`. Always fix the schema crash; it short-circuits four downstream degradations.

**Don't**: add the bad value to the enum just to make the error go away (this corrupts downstream switches).

**Do**: clarify the prompt vocabulary; explicitly map `query_type` values to `answer_kind` values.

**OBSERVED** (2026-05-09): LLM emitted `answer_kind=data_explanation` (a `query_type` value). Fixed in commit `e8874f5` by adding explicit vocabulary distinction and mapping table to `_ANALYZER_CORE_RULES`.

## B. Cross-check override clobbers correct answer_kind

**Symptom**:
```
WARNING answer_kind cross-check disagreement: llm=<X> derived=<Y> chosen=<Y>
```
where `<X>` was correct and `<Y>` is "safer" (`KNOWLEDGE` or `TIMESERIES`). The narrative answer paraphrases items, drops list members, or hallucinates categories.

**Where**: `agent/pipeline.py::_cross_check_answer_kind` and `_SAFE_ANSWER_KINDS`.

**Why it exists**: the cross-check was added to catch low-confidence LLM mistakes by preferring "broad" shapes that can still display correctly (KNOWLEDGE narrative tolerates anything).

**Why it bites**: for legal/regulatory questions where the source enumerates items (eligible parties, requirements, documents), KNOWLEDGE narrative is *less* safe than LIST — narrative format encourages paraphrase. The "safe" set was tuned for data questions and accidentally applies to legal-text questions.

**Don't**: remove the cross-check entirely (data questions still benefit from it).

**Do**: narrow the override condition. When `llm_kind=LIST`, `confidence ≥ 0.85`, and `query_type ∈ {regulatory_procedure, conceptual_definition}`, trust the LLM. Otherwise keep current behavior.

**OBSERVED** (2026-05-09): "who can trade on the exchange during transitory model" — LLM correctly emitted `LIST`, code overrode to `KNOWLEDGE`, narrative output added "Commercial Importers (Traders)" which was not in the source. Fix in progress (Phase 3 of the 2026-05 fix series).

## C. Router misclassification

**Symptom**: `query_type` is wrong but no error. Examples:
- "trend and structure of power supply" classified as `conceptual_definition` instead of `data_retrieval` → routed to knowledge path, no chart.
- "who can trade…" classified as `conceptual_definition` instead of `regulatory_procedure` → wrong template (definitional, not enumeration).

**Where**: analyzer prompt at [core/llm.py](../../../core/llm.py); few-shot examples; query_type guidance from `QUESTION_ANALYSIS_QUERY_TYPE_GUIDE`.

**Don't**: add a hardcoded keyword override in `agent/pipeline.py` that catches the specific phrasing. That's a question-specific patch; the next variant will miss.

**Do**: add few-shot examples to the prompt covering the phrasing class. The LLM generalizes from examples.

## D. Cross-check disagreement on conceptual flag

**Symptom**: `conceptual_disagree=true` in TRACE — heuristic and LLM disagree on whether the question is conceptual.

**Where**: heuristic in `core/llm.py::classify_query_type` (rule-based) vs LLM-emitted classification.

**Most common cause**: the heuristic's keyword list is incomplete or the LLM is over-classifying as conceptual.

**Don't**: blindly defer to one or the other. Look at the question; whichever is right, that's the side to bias toward.

**Do**: log analysis to see which side is correct more often, then adjust the keyword list or the prompt accordingly.

## E. Summarizer 504 timeout + retry

**Symptom**:
```
WARNING Retrying ... DeadlineExceeded: 504 The request timed out
```
The retry succeeds but the request took ~2× normal latency.

**Where**: any path that calls `llm_summarize_structured` or `llm_summarize` on a large prompt with `gemini-2.5-pro`.

**Trigger**: prompt size > ~50K chars on 2.5-pro. Google's server deadline is ~60 s; pro is slow on long prompts.

**Don't**: increase `max_retries` (each retry hits the same wall).

**Do (in order of leverage)**:
1. Trim the prompt (`PROMPT_BUDGET_MAX_CHARS` env var, currently 63000 in user's env).
2. Switch `SUMMARIZER_MODEL` to `gemini-2.5-flash` for queries where pro quality isn't measurably better.
3. Set `max_retries=0` and let the OpenAI fallback at [core/llm.py:1924](../../../core/llm.py) handle 504s — caps worst-case at single-call deadline.

**OBSERVED** (2026-05-08): "why balancing electricity price changed" hit 86 s summarizer because of pro+63K prompt; first call timed out at 60s, retry took 25s, total ~86s.

## F. Grounding gate failure

**Symptom**:
```
TRACE_DETAIL provenance_gate ... gate_passed=false coverage=0.7143 unmatched_tokens=[...]
```
Followed by `summary_source=citation_gate_fallback` — the user got a generic apology answer instead of the analytical one the LLM produced.

**What it means**: the summarizer fabricated numbers (e.g. `8.19`, `16.8`) that don't appear in the evidence preview. The provenance gate at [agent/summarizer.py](../../../agent/summarizer.py) caught the hallucination and refused to ship the answer.

**Where**: the answer was generated from the LLM but couldn't be traced back to the data. Almost always one of:
1. The LLM did arithmetic in its head (computing percentages, deltas) instead of citing pre-computed values.
2. The data preview was truncated by `Prompt budget applied` and the LLM filled the gap with plausible-looking numbers.
3. `analyzer_available=false` (Pattern A) — the structured contract was missing so the summarizer had no firm shape.

**Don't**: lower the coverage threshold below 0.80 to "make it pass". The gate exists because hallucinated numbers degrade trust faster than a slightly-incomplete answer does.

**Do**:
- If caused by truncation: precompute the derived metric in `agent/analyzer.py::stage_3_analyzer_enrich` and surface it as a labeled stat in the `stats_hint`.
- If caused by analyzer fallback: fix Pattern A first.
- If caused by the LLM doing math: tighten the answer-composer skill to require quoting computed values verbatim from `STATISTICS`.

## G. Slow LLM stage (model-name surprise)

**Symptom**: a stage that should be fast (router on flash-lite) takes 18+ seconds repeatedly, with stable network. Reducing `thinking_budget` doesn't help — sometimes it makes it *worse* (silent retries via `max_retries`).

**Where**: model identifier resolution. If `gemini-3.1-flash-lite` doesn't exist as a SKU, the SDK may queue, fall back, or retry, producing erratic latency.

**Don't**: tune `thinking_budget` lower and lower as a remedy.

**Do**: verify the model name resolves at the API:
```bash
curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY" \
  | python -c "import sys,json; [print(m['name']) for m in json.load(sys.stdin)['models'] if 'flash' in m['name'].lower()]"
```
If absent, switch to a known-stable SKU (`gemini-2.5-flash-lite`).

**OBSERVED** (2026-05-08): `gemini-3.1-flash-lite` produced 18–57 s router latencies that did not respond to thinking_budget tuning. Switched to `gemini-2.5-flash-lite` → router dropped to 2–6 s.

## H. Prompt budget truncation hides the answer

**Symptom**:
```
WARNING Prompt budget applied (section-aware): label=summarize_structured 
        original=110698 final=62991 budget=63000 
        truncated_sections=['UNTRUSTED_DATA_PREVIEW', ...]
```
Followed by a pre_gate event with claims that don't match anything in the data → grounding gate failure.

**Where**: [core/llm.py::_enforce_prompt_budget](../../../core/llm.py).

**Why it bites**: `UNTRUSTED_DATA_PREVIEW` is the actual data the summarizer is supposed to ground in. When it's the section getting truncated, the LLM has no choice but to invent.

**Don't**: just raise the budget — the budget exists because the model has a context limit and overlong prompts cost retries.

**Do**:
1. Reduce upstream bloat — domain knowledge cap, conversation history truncation, vector knowledge top_k.
2. Re-rank: data preview should be the LAST thing dropped, not the first. Check `truncation_priority` in `_enforce_prompt_budget`.

## I. Heuristic vs analyzer mode disagreement

**Symptom**: `mode_disagree=true` — heuristic conceptual classifier and the LLM analyzer disagree on whether to enter `light` or `analyst` mode.

**Where**: [agent/planner.py](../../../agent/planner.py) heuristic vs the analyzer's `analyzer_mode`.

**Usually benign**, but watch for: when the two disagree AND the answer is wrong, the heuristic's path won. Trace which mode was actually taken (`mode=` in the TRACE) and whether the right path was eventually chosen.
