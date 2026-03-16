# Comprehensive Systems Audit - Current Repo State
**Date:** 2026-03-16 (refresh of 2026-03-13 baseline)
**Auditor Role:** AI Systems Auditor (LLM Architect + Security Reviewer + SRE)
**Repo Audited:** `d:\Enaiapp\langchain_railway`
**HEAD commit:** `a8088dd`

## Scope And Framing
- This refresh is repo-first and evidence-based. Only controls enforced in this repository are scored as fully closed.
- Prior audit notes that depended on upstream Supabase/auth/ops assumptions are no longer treated as full closures unless enforced in code or deployment artifacts in this repo.
- This refresh also incorporates supplemental upstream evidence reviewed outside this repo and summarized in `docs/active/EXPORT_ENAI_CHAT_AUTH_REVIEW.md`: the mother application in `D:\export_enai`, including the current `chat-with-enerbot` Supabase Edge Function source. Deployment parity with live Supabase remains an external assumption.
- Runtime and stack (inferred): Python, FastAPI, LangChain, SQLAlchemy, pandas.
- LLM providers (repo-evidenced): Gemini and OpenAI (`core/llm.py`).
- Retrieval model (repo-evidenced): SQL-first analytical retrieval with typed-tool fast paths, guarded SQL fallback, and markdown knowledge injection.
- Storage (repo-evidenced): Postgres/Supabase (`config.py`, `core/query_executor.py`).
- Deployment (repo-evidenced): Railway + Docker (`railway.json`, `Dockerfile`).
- This refresh also includes a frontend secret-exposure review of `D:\export_enai` and an OWASP LLM Top 10 treatment/gap mapping. The reviewed OWASP baseline is the 2025 OWASP Top 10 for LLM/GenAI Applications; the attached PDF could not be fully text-extracted in this environment, so the threat taxonomy was aligned against the official OWASP publication.
- Hard SLOs, rollback policy, and persistent telemetry export are not codified in repo.

## 1) Executive Summary
- **Overall Score:** **9.0/10** (was 8.8).
- **Analytical Text-to-SQL Fit Score:** **9.4/10** (was 9.2).
- **Typed-Tool / Agent Fit Score:** **8.7/10** (was 8.4).
- **Recommendation:** Close to controlled production behind the reviewed Supabase Edge Function gateway pattern, assuming the current upstream source matches deployment. Not ready for direct public exposure as-is because Railway-side trust is still shared-secret based and several privacy/ops controls remain open.
- **Current verification in this environment:**
  - `pytest -q` -> **148 passed, 0 failed** (was 92)
  - `python -m guardrails.redteam_gate --min-score 0.92 --min-block-rate 1.0 --max-false-block-rate 0.02 --min-grounding-detect-rate 1.0 --min-grounding-accept-rate 1.0` -> **pass_gate=true, score=1.0**
- **Main improvement pattern since the previous full audit:** core architectural gaps that were previously still open are now closed in repo: citation-grade provenance, formal red-team gating, agent-loop telemetry/provenance consistency, balancing-segment canonicalization, and expanded regression coverage.
- **Improvement pattern since the 2026-03-13 refresh:** reliability and correctness of the analytical pipeline: SQL dialect alignment, chart metric alias resolution, routing dead-zone closure, YoY/MoM causal context signals, semantic score observability, Pydantic contract limit hardening, and +56 new regression tests.
- **Frontend secret review result:** no evidence was found that server-side API keys or backend shared secrets are present in the checked-in `D:\export_enai` frontend source or checked-in built assets. The browser-visible configuration is limited to Supabase URL, Supabase anon key, and frontend feature flags. This is a source/build review, not live deployed frontend verification.

Top current blockers:
- Railway-side caller auth is still a shared `X-App-Key` trust boundary. Supplemental upstream review now shows the Supabase gateway is authenticated and quota-aware, but Railway itself still receives only proxy-level identity. Repo-level secret reuse is now closed by splitting gateway auth, session signing, and evaluate/admin secrets; deployment secret rotation is still required (`config.py:26-36`, `main.py:619-641`, `main.py:888-959`).
- `/metrics` is unauthenticated and exposes internal operational state (`main.py:588-616`).
- No explicit PII redaction or minimization pipeline exists before model/provider calls or logs.
- `/evaluate` remains a privileged synchronous workload inside the serving process (`main.py:619-713`).
- No rollback/disaster-recovery runbook exists in active docs, and deployment policy is only partially codified (`railway.json:1-8`, `main.py:568-585`).

## 2) Major Closures Since The Previous Full Audit
| Item | Current Status | Evidence |
|---|---|---|
| Citation-grade provenance from claims to exact SQL/tool cells | Closed | `models.py:57-84`, `agent/provenance.py:1-51`, `agent/summarizer.py:144-301`, `main.py:1042-1060` |
| Deterministic provenance restamping on analyzer share fallback | Closed | `agent/analyzer.py:465-489`, `tests/test_guardrails.py:262-314` |
| Agent-loop `data_exit` provenance contract | Closed | `agent/orchestrator.py:245-263`, `tests/test_orchestrator.py:54-74` |
| Agent-loop request token/cost telemetry | Closed | `agent/orchestrator.py:266-323`, `utils/metrics.py:116-220`, `tests/test_orchestrator.py:119-144` |
| Formal red-team score gate wired into CI | Closed | `guardrails/redteam_gate.py:84-181`, `.github/workflows/ci.yml:30-37`, `tests/security/test_redteam_gate.py:14-24` |
| Router semantic fallback + fallback-intent observability | Closed | `agent/router.py:43-182`, `agent/pipeline.py:68-167`, `utils/metrics.py:247-268`, `tests/test_metrics_observability.py:71-85` |
| Canonical balancing-segment contract across execution and prompt assets | Closed | `agent/tools/composition_tools.py:53-74`, `agent/sql_executor.py:49-76`, `context.py:165-169`, `tests/test_phase5_cleanup.py:73-97` |
| Expanded regression coverage for guardrails/orchestration/provenance | Closed | `tests/test_guardrails.py`, `tests/test_orchestrator.py`, `tests/test_phase5_cleanup.py`, `tests/test_metrics_observability.py` |

### Closures in 2026-03-16 Refresh
| Item | Current Status | Evidence |
|---|---|---|
| SQL dialect alignment (BigQuery → PostgreSQL) | Closed | `core/sql_generator.py:54` now reads `read='postgres'`; aligns AST parser with Supabase backend |
| Chart metric alias resolution (bidirectional) | Closed | `agent/chart_pipeline.py:17-31` builds `_CHART_METRIC_ALIASES` from `METRIC_VALUE_ALIASES`; filter expansion at `agent/chart_pipeline.py:203-218`; tests at `tests/test_guardrails.py:1194-1245` |
| METRIC_VALUE_ALIASES expanded to 12 metric families | Closed | `agent/analyzer.py:48-67` covers balancing, deregulated, guaranteed capacity, exchange rate, tariff, generation, and CPI; was 3 families (p_bal_gel, p_bal_usd, xrate) |
| Routing dead-zone closure (threshold 0.62 → 0.55) | Closed | `config.py:76-79` default lowered to 0.55; margin gate (0.08) at `agent/router.py:180` prevents ambiguous routing; tests at `tests/test_router.py:86-115` |
| Semantic score observability on router miss | Closed | `agent/router.py:13` `_last_semantic_scores` dict; populated at `agent/router.py:171-173`; included in pipeline trace at `agent/pipeline.py:291`; test at `tests/test_router.py:111-119` |
| YoY/MoM causal context signals in why-explanations | Closed | `agent/analyzer.py:194-201` (`_find_yoy_row`), YoY extraction in `_build_why_context` at `agent/analyzer.py:937-1072`; provenance concat at `agent/analyzer.py:1112-1118`; tests at `tests/test_guardrails.py:896-1175` |
| Pydantic contract limits hardened (10 → 25) | Closed | `contracts/question_analysis.py:140,173,174,223` `max_length=25` on entities, types, ambiguities; tests at `tests/test_guardrails.py:1250-1276` |
| Question analyzer active/shadow modes with Phase C hints | Closed | `agent/pipeline.py:160-202` stage 0.2 dispatch; `agent/planner.py:185-247` analyze functions; `agent/planner.py:344-447` `build_tool_invocation_from_analysis()`; tests at `tests/test_question_analyzer_shadow.py`, `tests/test_question_analyzer_phase_c.py` |

## 3) Updated Scorecard
| Category | 2026-03-13 | Current | Delta | Gap To 9.5 |
|---|---:|---:|---:|---:|
| A) SQL-First Analytical Data Access Quality | 9.0 | 9.2 | +0.2 | +0.3 |
| B) Agent & Tooling Design | 8.9 | 9.1 | +0.2 | +0.4 |
| C) Prompting & Guardrails | 9.1 | 9.1 | 0.0 | +0.4 |
| D) Memory & State Management | 8.4 | 8.4 | 0.0 | +1.1 |
| E) Data Privacy & Security | 8.2 | 8.2 | 0.0 | +1.3 |
| F) Reliability & Failure Handling | 8.9 | 9.0 | +0.1 | +0.5 |
| G) Observability & Evaluation | 9.0 | 9.1 | +0.1 | +0.4 |
| H) Cost, Latency & Performance | 8.3 | 8.3 | 0.0 | +1.2 |
| I) Testing & QA | 9.2 | 9.5 | +0.3 | +0.0 |
| J) Deployment & Ops Readiness | 7.6 | 7.6 | 0.0 | +1.9 |

Score notes (2026-03-16 refresh):
- **A) +0.2**: SQL dialect fix (`core/sql_generator.py:54` now `read='postgres'`) closes a correctness gap where BigQuery AST rules could produce false positives/negatives in table whitelist validation. 12-family alias expansion means chart filters correctly resolve LLM plan names to DB columns across all metric types.
- **B) +0.2**: Routing dead-zone closure (0.62 → 0.55) means queries with moderate semantic affinity no longer fall through to the expensive LLM SQL planner. The margin gate (0.08) prevents ambiguous routing at the lower threshold. Question analyzer Phase C hints provide a second routing path when deterministic rules miss.
- **F) +0.1**: Routing dead-zone closure reduces unnecessary fallback to the less-reliable LLM SQL path. YoY graceful degradation ensures missing historical data does not error. Zero-delta edge case handled correctly.
- **G) +0.1**: `_last_semantic_scores` trace logging means every router miss now has full per-tool score visibility in structured traces (`agent/pipeline.py:291`). YoY signal inclusion enriches causal analysis traces.
- **I) +0.3**: Test count rose from 92 → 148 (+61%). New coverage spans: YoY signals (3 tests), chart alias resolution (3 tests), Pydantic contract limits (3 tests), routing threshold/margin gate (3 tests), question analyzer contract (7 tests), question analyzer shadow/active modes (8 tests), trace observability (3 tests), price tool SQL (4 tests), query validation (expanded parametrized).
- Security (E) remains graded conservatively: upstream auth/quota closure is based on supplemental source review outside this repo and was not runtime-verified.
- Repo-level secret reuse is closed in code: preferred env names are `ENAI_GATEWAY_SECRET`, `ENAI_SESSION_SIGNING_SECRET`, and `ENAI_EVALUATE_SECRET`. Deployment rollout and secret rotation are still operational work.
- The main boundary gap remains Railway's reliance on a shared proxy secret rather than principal-aware identity.

## 4) Remaining Blockers By Category

### A) SQL-First Analytical Data Access Quality (9.2)
P0:
- None currently.

P1:
- Typed-tool coverage is still partial for long-tail intents, so free-form SQL planning remains part of the production path for uncovered analytical questions.

### B) Agent & Tooling Design (9.1)
P0:
- None currently.

P1:
- Multi-dataset arbitration in the agent loop remains heuristic and intentionally conservative (`agent/orchestrator.py:226-242`).

### C) Prompting & Guardrails (9.1)
P0:
- None currently.

P1:
- Guardrails are materially stronger than before, but they still rely on curated rule sets and fixed regression cases rather than provider-level adversarial fuzzing at runtime.

### D) Memory & State Management (8.4)
P0:
- None currently.

P1:
- Session memory, metrics, cache, circuit breakers, and backpressure controls are all process-local and in-memory (`utils/session_memory.py:19-131`, `utils/metrics.py:17-342`, `utils/resilience.py:20-172`, `core/llm.py` cache implementation).

### E) Data Privacy & Security (8.2)
P0:
- Repo-level caller authentication inside Railway is still a shared `X-App-Key` secret presented by the upstream proxy rather than principal-aware identity. Secret reuse inside Railway is now closed in code, but a compromise of the gateway secret or proxy still bypasses principal-aware identity (`config.py:26-36`, `main.py:619-641`, `main.py:888-959`, `utils/session_memory.py:23-97`).
- `/metrics` is unauthenticated and returns internal counters, cache stats, model info, DB pool state, and circuit-breaker/backpressure state (`main.py:588-616`).
- No explicit PII redaction/minimization step exists before prompts, provider calls, or structured security/error logging.

P1:
- Repo now prefers `ENAI_GATEWAY_SECRET`, `ENAI_SESSION_SIGNING_SECRET`, and `ENAI_EVALUATE_SECRET`, while still accepting the earlier split-secret names as fallback. Runtime risk remains until Railway env vars are added and the Supabase `CHAT_BACKEND_SECRET` value is rotated to the new gateway secret.
- Checked-in frontend review of `D:\export_enai` found no server-side secrets in browser code or checked-in built assets. Browser-exposed values are limited to the expected Supabase URL and anon key, while `SUPABASE_SERVICE_ROLE_KEY` and `CHAT_BACKEND_SECRET` are documented as server-side only (`D:\export_enai\src\lib\customSupabaseClient.js:6-21`, `D:\export_enai\.env.example:4-55`, `D:\export_enai\ENVIRONMENT.md:5-28`). No committed real `.env` file was found, and the checked-in bundle resolves only the placeholder `https://example.supabase.co` fallback.
- Supplemental upstream source review in `docs/active/EXPORT_ENAI_CHAT_AUTH_REVIEW.md` indicates `chat-with-enerbot` now enforces `Authorization`, active account status, and pre-backend quota before proxying, but those controls live outside this repo and were not runtime-verified in this environment.
- Operator-reported legacy JWT verification disablement on the Supabase function is less material if deployment matches the reviewed source because the function now performs explicit `auth.getUser()` checks in code; deployment parity should still be confirmed.
- Operational protection of `/metrics` and `/evaluate` still depends on deployment policy outside this repo.

### F) Reliability & Failure Handling (9.0)
P0:
- `/evaluate` runs synchronous evaluation loops inside the serving process and can contend with production request capacity (`main.py:619-713`).

P1:
- Breakers and backpressure are process-local only; multi-replica coordination does not exist in repo.
- FastAPI startup still uses deprecated `@app.on_event("startup")` (`main.py:552-553`).

### G) Observability & Evaluation (9.1)
P0:
- None currently.

P1:
- Metrics are in-memory JSON and no persistent export/alerting contract is codified in repo.

### H) Cost, Latency & Performance (8.3)
P0:
- None currently.

P1:
- LLM response cache remains process-local only.
- No load/performance budget gate exists in CI.

### I) Testing & QA (9.5)
P0:
- None currently.

P1:
- `/evaluate` calls `process_query()` directly and therefore does not exercise the full HTTP request path (firewall wrapper, rate limit handler, referer checks, session issuance, request headers, and `/ask` telemetry finalization) (`main.py:669`).
- No chaos or sustained load tests exist for multi-worker behavior, prolonged provider outage, or DB saturation.

### J) Deployment & Ops Readiness (7.6)
P0:
- No rollback/disaster-recovery runbook exists in active docs.

P1:
- Readiness/liveness endpoints exist in app (`main.py:568-585`) but Railway deployment policy does not encode health-check behavior (`railway.json:1-8`).
- Operational exposure of `/metrics` is not constrained in repo deployment artifacts.
- Startup lifecycle still uses deprecated FastAPI startup hooks.

## 5) OWASP LLM Top Threats - Treatment Snapshot

This section maps the major OWASP LLM/GenAI application threats to controls or gaps evidenced in `langchain_railway` and the reviewed upstream `D:\export_enai` source. "Treated" means there is meaningful code-level mitigation; it does not mean the threat is eliminated.

| OWASP threat | Why it matters here | Current treatment in code | Current status |
|---|---|---|---|
| Prompt Injection | Users can try to override instructions, exfiltrate prompts, or steer the model into unsafe tool/SQL behavior. | Stage-0 firewall blocks instruction override, prompt exfiltration, role hijack, and dangerous SQL intent (`guardrails/firewall.py:22-127`); `/ask` enforces that firewall before pipeline execution (`main.py:972-1012`); red-team cases are CI-gated (`guardrails/redteam_gate.py:32-145`, `.github/workflows/ci.yml:30-37`). | **Mostly treated, still partial**. Strong curated defenses exist, but not runtime adversarial fuzzing. |
| Sensitive Information Disclosure | Raw user prompts, history, metrics, and internal state can leak private or operationally useful data. | Upstream chat access is authenticated and own-row constrained by route protection, edge-function auth/quota checks, and RLS (`D:\export_enai\edge_functions\chat-with-enerbot.txt:99-142`, `D:\export_enai\database\baseline\security\rls_and_grants.sql:86-124`, `D:\export_enai\database\baseline\schema\functions\record_chat_turn_txn.sql:28-97`). | **Partial**. Strong upstream access control exists, but Railway still exposes `/metrics` and lacks explicit PII minimization before provider calls/logging. |
| Supply Chain Vulnerabilities | Compromised dependencies or packages can bypass application-layer controls entirely. | Frontend CI runs production npm audit (`D:\export_enai\package.json:14`, `D:\export_enai\.github\workflows\ci.yml:73`); Python deps in Railway are pinned in `requirements.txt`; Railway CI runs tests and red-team checks (`.github/workflows/ci.yml:30-37`). | **Partial**. Dependency discipline exists, but no repo-evidenced Python vulnerability scanner such as `pip-audit` is wired into CI. |
| Data and Model Poisoning | Compromised knowledge sources or tool inputs can corrupt answers. | The core path is SQL-first against allowlisted tables (`core/sql_generator.py:30-178`) rather than open retrieval over arbitrary external corpora; typed tool surfaces are allowlisted (`agent/orchestrator.py:37-45`, `agent/orchestrator.py:338-356`). | **Partially treated / lower applicability**. The architecture reduces poisoning exposure, but there is no formal poisoning-detection or source-trust scoring layer. |
| Improper Output Handling | Unsafe model output could become executable SQL or otherwise dangerous downstream input. | SQL is AST-parsed, table-whitelisted, read-only constrained, and destructive nodes are rejected (`core/sql_generator.py:30-178`); the agent loop forbids SQL-writing tools and limits tool calls to an allowlist (`agent/orchestrator.py:115-123`, `agent/orchestrator.py:328-356`). | **Well treated** for the current SQL/tool path. |
| Excessive Agency | Models with too much autonomy can perform unintended actions or compound errors. | Only four typed tools are exposed to the agent (`agent/orchestrator.py:37-103`); agent rounds are bounded (`agent/orchestrator.py:313-360`); the system prompt explicitly forbids SQL/tool-surface expansion (`agent/orchestrator.py:115-123`). | **Treated** for current scope. |
| System Prompt Leakage | Users may try to recover hidden prompts or internal rules. | The firewall blocks prompt-exfiltration phrases (`guardrails/firewall.py:30-35`); prompt templates clearly separate `UNTRUSTED_*` inputs from authoritative guidance (`core/llm.py:856-883`, `core/llm.py:1351-1369`, `core/llm.py:1463-1485`). | **Partial**. Prompt-structure discipline is good, but there is no dedicated output scrubber for accidental leakage. |
| Vector and Embedding Weaknesses | RAG/vector pipelines can retrieve malicious or low-trust content into the model context. | The dominant architecture is SQL-first analytical retrieval, not vector retrieval (`core/sql_generator.py`, `agent/orchestrator.py`). | **Low applicability** in the current repo. If vector search is added later, this area should be re-audited. |
| Misinformation | The model can produce unsupported or numerically wrong narratives even when data exists. | Structured summarization retries on failed grounding and falls back conservatively when grounding still fails (`agent/summarizer.py:423-473`); provenance is attached and enforced before final output (`agent/summarizer.py:471-473`, `models.py:57-80`). | **Mostly treated** for data-backed answers, but still subject to normal LLM residual risk on conceptual answers. |
| Unbounded Consumption | Attackers or buggy flows can burn model budget, saturate concurrency, or amplify failures. | `/ask` is rate-limited (`main.py:888-889`); request backpressure and circuit breakers are enforced (`main.py:944-953`, `utils/resilience.py:20-172`); prompt budget is capped (`core/llm.py:1524-1541`); upstream chat quota is enforced before and during persistence (`D:\export_enai\edge_functions\chat-with-enerbot.txt:130-142`, `D:\export_enai\database\baseline\schema\functions\record_chat_turn_txn.sql:46-76`). | **Mostly treated**. Controls are good, but resilience state is process-local and Railway still trusts a shared proxy secret. |

## 6) Verification Snapshot (Current)
- Local verification (2026-03-16):
  - `pytest -q` -> **148 passed, 0 failed** (was 92 at 2026-03-13)
  - `python -m guardrails.redteam_gate --min-score 0.92 --min-block-rate 1.0 --max-false-block-rate 0.02 --min-grounding-detect-rate 1.0 --min-grounding-accept-rate 1.0` -> **pass**
- CI evidence:
  - `.github/workflows/ci.yml:30-37` runs `tests/security`, the formal red-team gate, and the full test suite.
- Important regression areas now covered:
  - citation-grade provenance
  - provenance-gate fallback behavior
  - agent-loop provenance and telemetry
  - router/fallback metrics
  - balancing prompt-asset canonicalization
  - YoY/MoM causal context signals (3 tests)
  - chart metric alias bidirectional resolution (3 tests)
  - Pydantic contract limit validation (3 tests)
  - semantic router threshold and margin gate (3 tests)
  - question analyzer contract, shadow, and Phase C hint modes (20 tests)
  - trace observability for why-context and summarizer (3 tests)
- Supplemental upstream evidence reviewed:
  - `D:\export_enai\src\App.jsx` wraps `/chat` in `ProtectedRoute`, and `D:\export_enai\src\components\ProtectedRoute.jsx` blocks anonymous and non-active users.
  - `D:\export_enai\src\contexts\SupabaseAuthContext.jsx` fetches account status and signs blocked users out during auth initialization.
  - `D:\export_enai\src\pages\ChatPage.jsx` routes chat traffic through `supabase.functions.invoke('chat-with-enerbot', ...)`, not direct browser-to-Railway calls.
  - `D:\export_enai\src\lib\customSupabaseClient.js` exposes Supabase client configuration in the frontend, but not the Railway `X-App-Key`; the checked-in frontend env contract and bundle review found no server-side secrets in browser code (`D:\export_enai\.env.example`, `D:\export_enai\ENVIRONMENT.md`, `D:\export_enai\dist\assets\index-2c38a8f9.js`).
  - Reviewed `chat-with-enerbot` Edge Function code requires `Authorization`, calls `auth.getUser()`, enforces active account status, pre-checks chat quota, and injects `X-App-Key` on the Railway request.
  - Reviewed upstream DB path `record_chat_turn_txn` remains the authoritative own-row/quota enforcement path for chat persistence.
- Warnings observed during local verification:
  - FastAPI startup hook deprecation (`@app.on_event("startup")`)
  - local Python 3.14 compatibility warnings from dependency stack
  - deploy image remains Python 3.11 (`Dockerfile:1-23`), so the local warning is not an immediate deploy blocker

## 7) Architecture Fit Commentary
- The implementation is now strongly aligned with a SQL-first analytical assistant architecture: typed-tool fast path, guarded SQL fallback, stage-level tracing, structured summaries, provenance-backed numeric claims, and bounded tool-only agent orchestration.
- The biggest improvements since the 2026-03-13 audit are in analytical pipeline correctness: SQL dialect alignment ensures the AST parser matches the production database, chart metric alias resolution prevents silent column-filter failures when the LLM uses semantic names, the routing dead-zone closure eliminates a class of queries that unnecessarily fell through to the expensive SQL planner path, and YoY/MoM context signals enrich causal explanations with historical comparison data.
- Testing coverage rose 61% (92 → 148 tests), with new coverage spanning every major subsystem added since the original audit: question analyzer modes, trace observability, chart aliases, routing thresholds, Pydantic contract limits, and causal context signals.
- The main remaining risks are at the Railway trust boundary and operational layer: shared-secret proxy trust inside Railway, endpoint exposure, privacy redaction, and production runbook maturity.
- Supplemental upstream evidence summarized in `docs/active/EXPORT_ENAI_CHAT_AUTH_REVIEW.md` shows the mother application routes chatbot traffic through an authenticated Supabase Edge Function that injects `X-App-Key` server-side and enforces active-user/quota checks before Railway.
- Supplemental frontend review also supports the narrower conclusion that browser-visible configuration is currently limited to public Supabase client values rather than backend secrets.
- That materially reduces the earlier proxy-boundary concern, but Railway still trusts the proxy by shared secret rather than by end-user or service-principal identity.
- Assuming the reviewed upstream source matches deployment and Railway operational endpoints remain non-public, the current repo is close to production-grade for controlled rollout.
- If this service is directly exposed to end users as implemented in repo, the auth/secret/metrics findings above should be treated as hard blockers, not residual polish items.

## 8) Test Coverage Snapshot (148 pytest-collected)
| Test File | Count | Coverage Area |
|---|---:|---|
| `tests/test_guardrails.py` | 31 | Guardrails, provenance, YoY signals, chart aliases, Pydantic limits |
| `tests/test_main.py` | 24 | HTTP endpoints, auth, rate limiting, firewall |
| `tests/test_query_validation.py` | 23 | Query validation parametrized cases |
| `tests/test_router.py` | 11 | Deterministic routing, semantic fallback, threshold, margin gate |
| `tests/security/test_adversarial_suite.py` | 8 | Adversarial prompt injection |
| `tests/test_question_analysis_contract.py` | 7 | Pydantic contract validation |
| `tests/test_context.py` | 6 | Query context construction |
| `tests/test_orchestrator.py` | 5 | Agent loop, provenance, telemetry |
| `tests/test_question_analyzer_shadow.py` | 5 | Shadow analyzer mode |
| `tests/test_metrics_observability.py` | 4 | Metrics counters, router metrics |
| `tests/test_pipeline_agent_mode.py` | 4 | Pipeline agent mode |
| `tests/test_phase5_cleanup.py` | 4 | Balancing canonicalization |
| `tests/test_price_tools_sql.py` | 4 | Price tool SQL generation |
| `tests/test_trace_observability.py` | 3 | Trace logging for why-context, summarizer |
| `tests/test_session_memory.py` | 3 | Session tokens, signing, history |
| `tests/test_tool_adapter.py` | 3 | Tool execution adapter |
| `tests/test_question_analyzer_phase_c.py` | 3 | Phase C hint integration |
| `tests/security/test_redteam_gate.py` | 1 | Red-team CI gate |
| **Total** | **148** | (pytest-collected; some parametrized cases share function entries) |
