# Comprehensive Systems Audit - Current Repo State
**Date:** 2026-03-24  
**Auditor Role:** AI Systems Auditor (application architecture, prompt/runtime quality, retrieval quality, and production-readiness review)  
**Repo Audited:** `d:\Enaiapp\langchain_railway`  
**HEAD commit:** `f262a9d`

## Scope And Method
- This refresh is code-first and repo-evidenced.
- It supersedes the older 2026-03-16 audit for the current repository state because the March 20-24 vector, prompt, and summarizer changes materially changed behavior.
- This refresh did **not** re-audit the external `D:\export_enai` repo or live Railway/Supabase deployments.
- The current worktree has one unrelated local modification in `ingest_one_document.py`; it was treated as operator tooling, not as a production-path code change.

Verification performed for this refresh:
- `pytest -q` -> **265 passed, 0 failed**
- `pytest tests/test_vector_pipeline.py tests/test_vector_prompt_integration.py tests/test_vector_store.py tests/test_vector_retrieval.py tests/test_guardrails.py -q` -> **122 passed, 0 failed**
- `pytest --collect-only -q` -> **265 collected**

Observed local warnings during verification:
- FastAPI `@app.on_event("startup")` deprecation (`main.py:314`)
- Python 3.14 compatibility warnings from dependency stack
- local pytest cache warning on `.pytest_cache`

## 1) Executive Summary
- **Overall Score:** **8.9/10**
- **SQL-First Analytical Fit:** **9.3/10**
- **Regulatory / Vector Knowledge Fit:** **8.8/10**
- **Prompting / Grounding Fit:** **8.9/10**
- **Testing & QA:** **9.6/10**
- **Deployment & Ops Readiness:** **7.7/10**

Recommendation:
- The current codebase is strong for a **controlled production rollout** behind an authenticated gateway.
- The March 20-24 work materially improved the regulatory/vector retrieval path: query embedding, cross-language retrieval, bridge topics, reranking, prompt grounding, and traceability are now all substantially better than the March 16 baseline.
- It is still **not ready for direct public exposure as-is** because Railway-side trust is still shared-secret based, `/metrics` is unauthenticated, `/evaluate` is a privileged in-process workload, and some retrieval/prompt behaviors still need refinement.

Top current blockers:
- Railway request auth still depends on a shared `X-App-Key`, not a principal-aware identity boundary (`main.py:650-692`).
- `/metrics` is unauthenticated and exposes internal operational state (`main.py:350-378`).
- `/evaluate` is a privileged synchronous workload inside the serving process (`main.py:381-460`).
- Regulatory procedure questions are still classified by focus more accurately than by query type; there is no dedicated procedural query type in `classify_query_type()` (`core/llm.py:376-430`, `core/llm.py:433-486`).
- Vector retrieval still enforces document diversity only; it does not enforce section diversity, so repeated chunks from the same article can still crowd the prompt (`knowledge/vector_store.py:158-215`).

## 2) Major Closures Since The 2026-03-16 Audit

### Closed Or Materially Improved
1. **Prompt budget handling is now section-aware instead of naive head-tail trimming.**
   - `core/llm.py:2000-2145`
   - This is a major quality improvement. The prompt now preserves section boundaries and truncates lower-priority sections first instead of blindly cutting the middle.

2. **Structured-summary cache keys now include skill content hash.**
   - `core/llm.py:1859-1863`
   - `skills/loader.py:260-280`
   - This closes the earlier stale-cache issue where skill edits could leave the summarizer serving outdated guidance.

3. **`get_answer_template()` no longer falls back to the entire template file.**
   - `skills/loader.py:85-106`
   - Unknown or unmapped query types now receive a safe fallback string instead of the full `answer-templates.md` file.

4. **Provenance coverage is now practical for derived analytical claims.**
   - `config.py:106-109`
   - `agent/summarizer.py:390-411`
   - Default `PROVENANCE_MIN_COVERAGE` is now `0.8`, which is materially safer than the earlier `1.0` setting for real analytical narratives.

5. **Vector retrieval moved from hard topic filtering toward soft scoring and richer reranking.**
   - `knowledge/vector_store.py:333-436`
   - `knowledge/vector_retrieval.py:87-259`
   - The current stack now uses:
     - bridge topics
     - keyword/title/section/topic boosts
     - boosted-score gating
     - language fallback
     - sparse-corpus similarity relaxation

6. **Vector traces now log section-level evidence, not just document titles.**
   - `agent/pipeline.py:230-275`
   - This made recent debugging materially easier because logs now show the actual retrieved article/section.

7. **Conceptual answers now use hybrid grounding instead of pure generic background.**
   - `agent/summarizer.py:487-610`
   - `core/llm.py:1882-2029`
   - Retrieved passages are primary evidence when active vector evidence exists, while domain knowledge remains as secondary, topic-filtered background.

## 3) Current Strengths

### A) Pipeline Architecture
- The five-stage structure remains clean and readable:
  - Stage 0: cheap prep
  - Stage 0.2: LLM question analysis
  - Stage 0.3: vector knowledge retrieval
  - Stage 0.5: deterministic tool routing
  - Stage 0.7: LLM-assisted tool routing fallback
- Conceptual questions still short-circuit early instead of paying SQL/tool costs unnecessarily.

### B) Retrieval / Regulatory Knowledge
- The vector stack is now meaningfully stronger than it was on 2026-03-16:
  - cross-language retrieval support
  - bridge-topic extraction for export, import, registration, capacity, transitory model, etc.
  - soft topic scoring instead of hard SQL topic filtering
  - document title / section title / metadata reranking
  - top-section trace logging
- The code now reflects the March 20-24 operational learnings rather than the earlier baseline assumptions.

### C) Prompting / Grounding
- The structured summarizer prompt now distinguishes:
  - user question
  - external source passages
  - domain knowledge
  - statistics
  - data preview
  - conversation history
- The system message now clearly allows markdown inside the JSON `answer` field (`core/llm.py:1924`, `core/llm.py:1981`), fixing the earlier JSON-vs-markdown wording conflict.

### D) Testing
- Local test coverage is materially stronger than before:
  - **265 collected / 265 passed**
- The newest coverage includes:
  - vector prompt threading
  - vector-aware conceptual summarization
  - section-title logging
  - bridge-topic extraction
  - sparse-corpus retrieval fallback
  - reranking and document-diversity behavior

## 4) Current Open Issues

### P0 / High Severity

#### 1. Shared-secret gateway trust remains the main auth boundary inside Railway
- Evidence:
  - `main.py:650-692`
  - `config.py:28-37`
- `POST /ask` still authorizes via `X-App-Key == GATEWAY_SHARED_SECRET`.
- This is acceptable for a trusted proxy architecture, but it is not principal-aware and should still be treated as a hard production boundary concern.

#### 2. `/metrics` is still unauthenticated
- Evidence:
  - `main.py:350-378`
- The endpoint exposes:
  - metrics counters
  - cache stats
  - model info
  - DB pool state
  - resilience snapshot
- This remains a direct operational exposure.

#### 3. `/evaluate` is still a privileged in-process workload
- Evidence:
  - `main.py:381-460`
- Even though it uses a separate admin secret, it still runs evaluation loops synchronously in the web process and calls `process_query()` directly.
- This remains a reliability and operational-isolation concern.

### P1 / Medium Severity

#### 4. Procedure/regulation questions still lack a dedicated query-type path
- Evidence:
  - `core/llm.py:376-430`
  - `core/llm.py:433-486`
- `get_query_focus()` now recognizes `regulation`, but `classify_query_type()` still only returns:
  - `single_value`
  - `list`
  - `comparison`
  - `trend`
  - `table`
  - `unknown`
- So registration / eligibility / compliance questions often get a good **focus** but a weak **query type**.
- Result: prompt routing is better than before, but still not fully explicit for regulatory procedures.

#### 5. There is still no trade-specific answer-composer guidance section
- Evidence:
  - `skills/loader.py:116-124`
  - `core/llm.py:477-480`
- `get_query_focus()` can return `trade`, but `skills/loader.py` maps `"trade"` to an empty section.
- Import/export answers therefore depend on:
  - always-rules
  - retrieved passages
  - or regulation focus fallback by document type
- This is workable, but still weaker than balancing / tariff / regulation paths.

#### 6. Retrieval enforces document diversity but not section diversity
- Evidence:
  - `knowledge/vector_store.py:158-215`
  - `agent/pipeline.py:250-265`
- The code prevents one document from monopolizing all slots when competitors are strong, but it does **not** prevent repeated chunks from the same article or section within one document.
- This is a real prompt-efficiency issue for regulation-heavy documents.

#### 7. Feature-flag defaults are still rollout-aggressive
- Evidence:
  - `config.py:71-77`
- Both:
  - `ENABLE_QUESTION_ANALYZER_HINTS`
  - `ENABLE_VECTOR_KNOWLEDGE_HINTS`
  default to `true`.
- That undermines conservative rollout semantics for new deployments that forget to pin explicit env values.

#### 8. Topic and knowledge routing are still partly hardcoded
- Evidence:
  - `knowledge/__init__.py:50-179`
  - `knowledge/__init__.py:266-310`
- `TOPIC_MAP` is still code-defined, which means:
  - adding or tuning keyword-to-topic mappings requires code change
  - knowledge routing remains partly dependent on static code, not only curated content

#### 9. Skill-cache invalidation is still process-lifetime only
- Evidence:
  - `skills/loader.py:246-280`
- The cache-key hash now exists, which is good, but the hash itself is memoized for the process lifetime.
- That is acceptable in production deployments with restart-on-deploy, but still imperfect during long-running local dev sessions.

### P2 / Low Severity

#### 10. Startup lifecycle still uses deprecated FastAPI startup hooks
- Evidence:
  - `main.py:314-320`
- This is not a correctness issue today, but it remains technical debt and shows up in local test warnings.

#### 11. Typo compatibility is still intentionally preserved in vector topics
- Evidence:
  - `knowledge/vector_retrieval.py:147-151`
- The code still emits both:
  - `wholesale_market_participants`
  - `whoesale_market_participants`
- This is reasonable for backward compatibility with already-ingested data, but it confirms that topic hygiene in ingested metadata still needs cleanup.

## 5) Findings That Are No Longer Open
- The old prompt-budget middle-cut problem is closed by section-aware truncation.
- The old full-template fallback in `get_answer_template()` is closed.
- The old summarizer cache-staleness problem is materially reduced by `skill_hash` in the cache key.
- The old `PROVENANCE_MIN_COVERAGE = 1.0` strictness problem is closed in current defaults.
- The old vector debug visibility problem is closed by `top_sections` logging.
- The old hard failure of English-to-Georgian retrieval was addressed by language fallback and soft retrieval scoring.

## 6) Updated Scorecard
| Category | Current Score | Notes |
|---|---:|---|
| SQL-first analytical retrieval | 9.3 | Strong guarded SQL and typed-tool path; no major regression found |
| Vector / regulatory retrieval | 8.8 | Much stronger than March 16; still lacks section diversity and trade-specific guidance |
| Prompting / grounding | 8.9 | Major improvements landed; remaining issue is procedure-type specificity |
| Reliability / failure handling | 8.7 | Good resilience patterns; `/evaluate` still competes with serving path |
| Security / privacy / boundary control | 8.0 | Still limited by shared-secret gateway auth and open `/metrics` |
| Testing / QA | 9.6 | 265 passing tests, including new vector and prompt integration coverage |
| Deployment / ops readiness | 7.7 | Still needs metrics protection, better rollout defaults, and ops/runbook hardening |

## 7) Recommended Fix Order

### Phase 1: Production Boundary / Ops
1. Protect or remove `/metrics` from public exposure.
2. Move `/evaluate` out of the serving process or isolate it operationally.
3. Replace or strengthen the shared-secret Railway trust boundary if the service is exposed beyond a tightly controlled proxy.

### Phase 2: Answer Quality
4. Introduce a dedicated procedural/regulatory query type or equivalent summarizer path for registration / eligibility / compliance questions.
5. Add a trade-specific answer-composer guidance section instead of relying on empty trade focus.
6. Add section-diversity logic on top of the existing document-diversity logic in vector retrieval.

### Phase 3: Maintainability
7. Externalize more thresholds and retrieval tuning values from code into config.
8. Move hardcoded `TOPIC_MAP` logic closer to content metadata/frontmatter or another content-managed registry.
9. Clean up typo-compatibility tags after re-ingestion normalizes topic names.

### Phase 4: Technical Debt
10. Replace deprecated FastAPI startup hooks with lifespan handlers.
11. Tighten cache invalidation semantics for long-running local dev sessions if that workflow matters.

## 8) Bottom Line
- The current repo is materially better than the March 16 baseline in the areas that changed most: vector retrieval, prompt assembly, summarizer grounding, and test coverage.
- The current code does **not** show a fresh critical implementation bug in the March 20-24 commit range; the main remaining concerns are now:
  - production boundary / ops exposure
  - lack of a dedicated procedure query type
  - missing trade-specific guidance
  - section-level retrieval repetition
- For controlled deployment behind a trusted gateway, the code is in good shape.
- For wider exposure, the auth/metrics/evaluate issues should still be treated as real blockers.
