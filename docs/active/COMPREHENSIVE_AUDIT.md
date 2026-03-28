# Comprehensive Systems Audit - Current Repo State
**Date:** 2026-03-28  
**Auditor Role:** AI Systems Auditor (application architecture, prompt/runtime quality, retrieval quality, security boundaries, and production-readiness review)  
**Repo Audited:** `d:\Enaiapp\langchain_railway`  
**HEAD commit:** `01f37aa` (local working tree includes uncommitted post-HEAD audit refresh and routing fixes)

## Scope And Method
- This refresh is code-first and repo-evidenced.
- It supersedes the earlier refreshes because the local working tree now includes the auth / deployment hardening, regulation-routing work, section-diversity retrieval controls, and later fallback-classifier fixes that were added after those snapshots.
- This refresh did not re-audit external repos or live Railway/Supabase deployments.
- A supplemental boundary review was performed against the current frontend at `D:\export_enai` to assess whether the present public web app changes the deployment-safety conclusion for this backend.
- The current worktree still has one unrelated local modification in `ingest_one_document.py`; it was treated as operator tooling, not as a production-path code change.

Verification performed for this refresh:
- `pytest -q` -> **315 passed, 0 failed, 5 warnings**
- `pytest --collect-only -q` -> **315 collected**
- Frontend verification against `D:\export_enai`:
  - `npm run test` -> **105 passed, 0 failed**
  - `npm run build` -> **success** (non-blocking bundle-size warning only)

Observed local warnings during verification:
- FastAPI `@app.on_event("startup")` deprecation (`main.py:425`)
- Python 3.14 compatibility warnings from `langchain_core` / `google.genai`
- local pytest cache warning on `.pytest_cache`

## 1) Executive Summary
- **Overall Score:** **9.1/10**
- **SQL-First Analytical Fit:** **9.3/10**
- **Regulatory / Vector Knowledge Fit:** **9.1/10**
- **Prompting / Grounding Fit:** **9.3/10**
- **Security / Boundary Control:** **9.2/10**
- **Testing & QA:** **9.7/10**
- **Deployment & Ops Readiness:** **9.1/10**

Recommendation:
- The repo is materially stronger than the earlier audit state. The public bearer rate-limit bug is fixed, gateway session bucketing now requires a verified signed session token, bearer callers no longer share the old decorator-level IP throttle, auth rollout now has a compatibility path, `/evaluate` now requires both a non-production env and an explicit second opt-in, and regulation/procedure fallback routing is materially less brittle than in the previous refresh.
- The currently reviewed `export_enai` frontend improves the public-edge story rather than weakening it: the browser holds only the Supabase anon key, `/chat` remains authenticated-only, the backend shared secret stays in the Supabase `chat-with-enerbot` edge function, admin actions are server-side role-checked in edge functions, and the static frontend ships a real CSP/HSTS/header baseline via its `Caddyfile`.
- The remaining concerns are no longer primarily auth-boundary bugs. They are now concentrated in:
  - operational isolation for privileged local/admin-only work
  - rollout-default caution and content-managed routing
  - startup lifecycle modernization
- When paired with the current `export_enai` frontend and its Supabase edge-function boundary, the current backend is reasonable to deploy publicly. The main conditions are:
  - keep `/evaluate` disabled
  - keep browser access limited to the intended Supabase public datasets
  - keep chat and admin traffic flowing through authenticated edge functions rather than direct browser calls to privileged backend paths

Top current blockers:
- `/evaluate` remains an in-process privileged workload when explicitly enabled (`main.py:501-558`).
- Gateway and pre-auth fallbacks are still proxy-shaped when no valid session token is present (`main.py:277-350`, `main.py:813-851`).
- Feature-flag defaults remain rollout-aggressive because question-analysis and vector hints default to `true` (`config.py:79-87`).
- Topic and knowledge routing are still partly hardcoded in application code (`knowledge/__init__.py:50-197`, `knowledge/__init__.py:244-310`).

## 2) Major Closures In The Current Local Fix Set

### Closed Or Materially Improved
1. **Public bearer rate limiting now works as intended.**
   - `main.py:354-360`
   - `_check_user_rate_limit()` now records the first request when a subject bucket is created instead of returning early with no state update.

2. **The gateway limiter now only uses a session bucket when the session token is valid.**
   - `main.py:277-301`
   - `tests/test_main.py:428-520`
   - Invalid or forged `X-Session-Token` values fall back to forwarded IP / remote IP rather than creating attacker-controlled buckets.

3. **Gateway and bearer callers now use separate post-auth limits.**
   - `main.py:304-360`
   - `main.py:813-851`
   - `tests/test_main.py:399-554`
   - The old shared decorator-level limiter is gone. Requests now pass through:
     - a coarse remote-IP pre-auth guard
     - a post-auth gateway limiter keyed by verified session/IP
     - a post-auth per-user bearer limiter

4. **Auth mode is explicit, but rollout compatibility no longer silently disables existing bearer deployments.**
   - `config.py:46-57`
   - `main.py:133-151`
   - `tests/test_config.py:69-82`
   - `ENAI_AUTH_MODE=auto` now preserves the old "JWT secret present -> bearer enabled" behavior while warning operators to move to an explicit mode.

5. **Bearer mode still requires `SUPABASE_JWT_SECRET` when explicitly enabled.**
   - `config.py:155-199`
   - `tests/test_config.py:19-33`
   - `ENAI_AUTH_MODE=gateway_and_bearer` still hard-fails at startup if `SUPABASE_JWT_SECRET` is absent.

6. **`/evaluate` enablement is now fail-safe instead of relying only on production detection.**
   - `config.py:190-198`
   - `tests/test_config.py:36-50`
   - `tests/test_config.py:85-115`
   - `ENABLE_EVALUATE_ENDPOINT=true` is now only valid when:
     - `ENAI_DEPLOYMENT_ENV` is `development` or `test`
     - `ALLOW_EVALUATE_ENDPOINT=true`

7. **Endpoint auth / abuse-control coverage is materially stronger.**
   - `tests/test_main.py:376-554`
   - The suite now covers:
     - bearer-token success
     - bearer-token rate-limit failure
     - forged-session fallback
     - verified-session gateway bucketing
     - gateway per-session isolation
     - bearer per-user isolation behind one IP
     - bearer rejection in `gateway_only` mode
     - `/metrics` disabled/auth behavior

8. **Deployment docs now include the current auth-mode, evaluate-opt-in, and rate-limit contract.**
   - `docs/active/DEVELOPER_GUIDE.md`
   - `docs/active/TESTING_GUIDE.md`
   - This closes the earlier documentation drift where the code and operator guidance no longer matched after the re-audit fixes.

9. **Regulatory procedure questions now have a dedicated query type, template, and fallback path.**
   - `contracts/question_analysis.py`
   - `contracts/question_analysis_catalogs.py`
   - `core/llm.py`
   - `skills/answer-composer/references/answer-templates.md`
   - `skills/loader.py`
   - `utils/query_validation.py`
   - `tests/test_question_analysis_contract.py`
   - `tests/test_question_analyzer_phase_c.py`
   - `tests/test_guardrails.py`
   - `tests/test_vector_prompt_integration.py`
   - Registration, eligibility, required-document, and procedural regulation questions now route through an explicit `regulatory_procedure` path instead of relying only on regulation focus plus generic conceptual handling.

10. **The earlier trade-guidance blocker was stale; trade guidance is present and now regression-covered.**
   - `skills/answer-composer/references/focus-guidance-catalog.md`
   - `skills/loader.py`
   - `tests/test_guardrails.py`
   - Trade focus already resolved to a real guidance section; the suite now locks that behavior in with a direct regression test.

11. **Vector retrieval now enforces section diversity on top of document diversity.**
   - `knowledge/vector_store.py`
   - `tests/test_vector_store.py`
   - Competitive selection now caps repeated chunks from the same section, then backfills unseen sections before allowing section repeats when needed to preserve recall.

12. **Fallback procedural/regulatory classification no longer loses obvious requirements queries to the generic list fallback, and year-scoped procedural questions stay procedural unless they are explicitly quantitative.**
   - `core/llm.py`
   - `utils/query_validation.py`
   - `tests/test_guardrails.py`
   - The fallback classifier now prioritizes `regulatory_procedure` ahead of the broad `"what are the"` list branch, and year mentions no longer demote procedure/eligibility/document questions unless the query clearly asks for counts or totals.

## 3) Current Strengths

### A) Pipeline Architecture
- The staged pipeline remains clean and production-usable:
  - cheap prep
  - LLM question analysis
  - vector knowledge retrieval
  - deterministic routing
  - LLM-assisted fallback routing
- The guardrail, typed-tool, and agent-loop boundaries are still cleaner than typical single-file FastAPI + LLM stacks.

### B) Security / Abuse Control
- The request boundary is now materially stronger than the previous audit state:
  - explicit auth modes
  - compatibility-safe auth rollout via `ENAI_AUTH_MODE=auto`
  - mandatory JWT secret for bearer mode
  - working public per-user rate limits
  - verified-session gateway limiter bucketing
  - separate gateway and bearer post-auth throttles
  - coarse pre-auth IP guard ahead of auth
  - `/metrics` and `/evaluate` disabled by default
  - fail-safe startup validation for `/evaluate`

### C) Prompting / Grounding
- The structured summarizer prompt remains more resilient under large contexts because prompt budgeting applies headroom and timeout-aware retry behavior (`core/llm.py:2047-2143`).
- The provenance gate still uses coverage-based grounding rather than brittle all-or-nothing numeric claim rejection (`agent/summarizer.py:406-435`).

### E) Public Web Boundary
- The reviewed frontend at `D:\export_enai` does not expose backend secrets in the browser:
  - the client uses `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` only (`D:\export_enai\src\lib\customSupabaseClient.js`)
  - the chat backend secret stays server-side in the Supabase edge function (`D:\export_enai\edge_functions\chat-with-enerbot.txt`)
  - admin functions verify the authenticated user role server-side before using the service role (`D:\export_enai\edge_functions\admin-get-users.txt`, `D:\export_enai\edge_functions\is-admin.txt`)
- Anonymous public access is intentionally narrow:
  - `/public` is limited to the public dashboard surface (`D:\export_enai\src\App.jsx`, `D:\export_enai\src\pages\EscoDashboard.jsx`)
  - only the intended public datasets are browser-readable by anonymous users according to the checked-in data config and DB grants (`D:\export_enai\src\lib\dashboardDataConfig.js`, `D:\export_enai\database\baseline\security\rls_and_grants.sql`)
- The static frontend deploy baseline is stronger than average for a small app:
  - the checked-in `Caddyfile` sets CSP, HSTS, `X-Frame-Options`, `X-Content-Type-Options`, and a restrictive `Permissions-Policy` (`D:\export_enai\Caddyfile`)

### D) Testing
- Local test health is stronger than before:
  - **315 collected / 315 passed**
- Coverage now includes the previously missing auth, regulation-routing, trade-guidance, and vector section-diversity regression cases in addition to the existing routing, vector, provenance, and pipeline coverage.

## 4) Current Open Issues

### P0 / High Severity
- **No currently open P0 implementation bugs were identified in the current local code path.**

### P1 / Medium Severity

#### 1. `/evaluate` still runs privileged synchronous work inside the API process when enabled
- Evidence:
  - `main.py:501-558`
- The production enablement path is now blocked by config validation, which is good.
- But if an operator enables it in non-production, it still runs evaluation loops in-process and competes with serving capacity.
- Suggested solution:
  - Move evaluation execution behind a separate admin-only worker path or one-off job runner instead of running it inside the serving API process.
  - Keep the current config guardrails, but add an explicit operator workflow for "run evaluation out of band and store artifacts/results" so the main app never shares capacity with eval loops.

#### 2. Gateway and pre-auth fallbacks are still proxy-shaped when no valid session token is present
- Evidence:
  - `main.py:277-350`
  - `main.py:813-851`
- The spoofable-bucket bug is closed and bearer users no longer share the old decorator IP bucket.
- But requests without a valid signed gateway session still fall back to forwarded IP or remote IP, and the coarse pre-auth guard is still remote-IP based.
- That is now a residual operational tradeoff, not the earlier implementation bug.
- Suggested solution:
  - Prefer a trusted gateway-generated stable client identifier or signed session identifier for both gateway and coarse pre-auth throttling whenever the proxy is present.
  - Keep remote-IP fallback only as the last resort, and document that proxy-shaped throttling remains expected when no verified per-client signal exists.
  - Add one more regression test around any new gateway principal header so the limiter cannot silently drift back to shared proxy buckets.

#### 3. Feature-flag defaults are still rollout-aggressive
- Evidence:
  - `config.py:79-87`
- Both:
  - `ENABLE_QUESTION_ANALYZER_HINTS`
  - `ENABLE_VECTOR_KNOWLEDGE_HINTS`
  default to `true`.
- That remains a weak default for cautious rollouts.
- Suggested solution:
  - Flip both defaults to `false` and require explicit enablement in each environment.
  - If keeping them on by default is important for local development, gate that behavior on `ENAI_DEPLOYMENT_ENV=development` rather than using the same default everywhere.
  - Add startup logging that prints the effective values so rollout state is obvious from deploy logs.

#### 4. Topic and knowledge routing are still partly hardcoded
- Evidence:
  - `knowledge/__init__.py:50-197`
  - `knowledge/__init__.py:244-310`
- `TOPIC_MAP` is still embedded in code rather than being content-managed.
- Suggested solution:
  - Move topic aliases, routing rules, and reranking hints into versioned content metadata or config files instead of hardcoding them in Python.
  - Keep a thin typed loader/validator in code so invalid metadata fails fast at startup.
  - Add tests that assert required topic groups and aliases exist, so content-managed routing can change safely without code edits.

#### 5. The public web boundary still depends on correct Supabase grants / RLS and broad edge-function CORS is allowed
- Evidence:
  - `D:\export_enai\database\baseline\security\rls_and_grants.sql`
  - `D:\export_enai\src\lib\dashboardDataConfig.js`
  - `D:\export_enai\src\lib\cors.ts`
- The public deployment story is safe only if the Supabase project actually matches the checked-in grants/RLS contract.
- Edge functions currently allow `Access-Control-Allow-Origin: *`; this is acceptable because auth and server-side role checks remain authoritative, but it is still broader than an origin allowlist.
- Suggested solution:
  - Treat the checked-in grants/RLS SQL as a deployment contract and add a release-time verification step that confirms live Supabase grants match the intended public/private dataset split.
  - Tighten edge-function CORS to an explicit production origin allowlist unless cross-origin callers are intentionally required.
  - Keep chat/admin traffic on authenticated edge functions only, and avoid adding direct browser access to privileged backend endpoints.

### P2 / Low Severity

#### 6. Skill-cache invalidation is still process-lifetime only
- Evidence:
  - `skills/loader.py:260-280`
- The content hash exists, but it is still memoized for the process lifetime.
- Suggested solution:
  - Add file-mtime or content-hash revalidation on reload paths used in development, or expose an explicit cache-reset hook for long-running local sessions.
  - Keep production behavior simple if hot-reload is not needed there, but stop treating process restart as the only invalidation path during active prompt/skill iteration.

#### 7. Startup lifecycle still uses deprecated FastAPI startup hooks
- Evidence:
  - `main.py:425-431`
- This is not a correctness issue today, but it remains visible technical debt and produces test warnings.
- Suggested solution:
  - Replace `@app.on_event("startup")` with a FastAPI lifespan handler and move current startup initialization into that context.
  - Add one small lifecycle test so the migration removes the warning without changing current startup behavior.

#### 8. Frontend quality still has a few non-blocking hygiene warnings
- Evidence:
  - `D:\export_enai\src\components\ui\toast.jsx`
  - `D:\export_enai\src\pages\ChatPage.jsx`
  - `D:\export_enai\package.json`
- The frontend test suite passes, but it emits React `act(...)` warnings around `ChatPage` tests and an invalid `dismiss` prop warning in the toast component.
- The production build also emits a bundle-size warning for the main chunk. None of these are public-deployment blockers, but they are worth cleaning up.
- Suggested solution:
  - Fix the invalid DOM prop in the toast component and update the affected tests so async state changes are fully wrapped or awaited, removing the React `act(...)` warnings.
  - Split or defer the largest chat/dashboard bundle paths if the main chunk keeps growing, but treat that as a performance cleanup rather than a deployment blocker.

#### 9. Typo compatibility is still intentionally preserved in vector topics
- Evidence:
  - `knowledge/vector_retrieval.py:147-151`
- The code still emits both:
  - `wholesale_market_participants`
  - `whoesale_market_participants`
- This is acceptable for backward compatibility, but it confirms metadata hygiene still needs cleanup after re-ingestion.
- Suggested solution:
  - Re-ingest or migrate stored metadata to the correct topic name, then remove the typo alias after compatibility is no longer needed.
  - Until then, keep the compatibility shim explicit and covered by a targeted regression test so cleanup can happen deliberately rather than by drift.

## 5) Findings That Are Closed Or Re-Scoped
- The earlier public bearer rate-limit bug is closed in current code (`main.py:354-360`).
- The earlier spoofable gateway-session limiter bug is closed by verified session-token resolution before bucketing (`main.py:277-301`).
- The earlier bearer NAT-collision risk from the shared decorator limiter is closed by the post-auth gateway/public limiter split (`main.py:813-851`).
- The earlier bearer rollout regression is re-scoped: explicit modes are still preferred, but `ENAI_AUTH_MODE=auto` now provides a compatibility bridge instead of silently disabling bearer auth (`config.py:46-57`, `main.py:133-151`).
- The earlier `/evaluate` exposure concern is now re-scoped to operational isolation only; enablement now requires both a safe env and an explicit second opt-in (`config.py:190-198`).
- The earlier auth regression coverage gap is materially reduced by the expanded endpoint and config tests (`tests/test_main.py:376-554`, `tests/test_config.py:19-115`).
- The earlier lack of a dedicated procedural/regulatory path is closed by the new `regulatory_procedure` query type, template mapping, fallback classification, and conceptual evidence gating (`contracts/question_analysis.py`, `core/llm.py`, `utils/query_validation.py`, `skills/loader.py`).
- The later fallback collision where broad `"what are the"` list heuristics swallowed procedural requirements queries is closed in current local code, and year-scoped procedural questions are no longer demoted purely because they mention a year (`core/llm.py`, `utils/query_validation.py`, `tests/test_guardrails.py`).
- The earlier trade-guidance blocker was stale; trade guidance already existed and is now protected by direct regression coverage (`skills/answer-composer/references/focus-guidance-catalog.md`, `skills/loader.py`, `tests/test_guardrails.py`).
- The earlier retrieval crowding issue is materially reduced by section-level diversity controls with unseen-section-first backfill (`knowledge/vector_store.py`, `tests/test_vector_store.py`).
- The earlier lack of explicit production env guidance is closed in the active docs (`docs/active/DEVELOPER_GUIDE.md:67-126`, `docs/active/TESTING_GUIDE.md:67-89`).
- The public deployment conclusion is stronger when this backend is paired with the current `export_enai` frontend, because the browser never receives the backend shared secret and chat/admin paths stay behind authenticated Supabase edge functions (`D:\export_enai\src\lib\customSupabaseClient.js`, `D:\export_enai\edge_functions\chat-with-enerbot.txt`, `D:\export_enai\edge_functions\admin-get-users.txt`, `D:\export_enai\Caddyfile`).

## 6) Updated Scorecard
| Category | Current Score | Notes |
|---|---:|---|
| SQL-first analytical retrieval | 9.3 | Strong typed-tool and guarded SQL path; no fresh regression found |
| Vector / regulatory retrieval | 9.1 | Regulation routing is stronger and vector retrieval now adds section diversity without losing recall |
| Prompting / grounding | 9.3 | Timeout resilience, analyst grounding, and regulation-specific answer shaping remain materially improved |
| Reliability / failure handling | 8.9 | Better auth / config contract; `/evaluate` still competes with serving path when enabled |
| Security / privacy / boundary control | 9.2 | Verified gateway bucketing, per-user bearer limits, and fail-safe `/evaluate` gating materially strengthen the boundary |
| Testing / QA | 9.7 | 315 backend tests plus 105 frontend tests passed; remaining warnings are hygiene-level |
| Deployment / ops readiness | 9.2 | Backend hardening plus the current frontend/edge-function boundary make public deployment reasonable when Supabase grants and secrets are configured correctly |

## 7) Recommended Fix Order

### Completed In The Current Local Fix Set
1. Fix `_check_user_rate_limit()` so the current request is recorded when a bucket is first created.
2. Add endpoint tests for bearer success, bearer rate-limit failure, and `/metrics` disabled/auth behavior.
3. Revisit the gateway limiter model so one proxy path does not collapse into a single global bucket.
4. Require verified signed session tokens before creating session-specific gateway buckets.
5. Split gateway and bearer post-auth throttles so bearer callers are no longer constrained by the shared decorator IP bucket.
6. Treat `SUPABASE_JWT_SECRET` as mandatory when bearer auth is explicitly part of the boundary, while preserving `auto` compatibility during rollout.
7. Keep `/evaluate` disabled by default and require both a safe env and explicit opt-in when testing it locally.
8. Document explicit production env values for auth mode, evaluate gating, and rate limits.

### Next Phase: Maintainability / Ops
9. Externalize more topic/reranking configuration from code into content-managed metadata.
10. Replace deprecated FastAPI startup hooks with lifespan handlers.
11. Decide whether `/evaluate` should remain in-process for non-production use or move to a separate admin/worker path entirely.
12. Tighten cache invalidation semantics for long-running local development if that workflow matters.
13. Revisit rollout defaults for question-analysis and vector-knowledge hint toggles.
14. Consider tightening Supabase edge-function CORS from `*` to an explicit origin allowlist if the deployment model does not need broad cross-origin access.
15. Clean up the frontend `toast` DOM-prop warning, React `act(...)` test warnings, and main bundle-size warning in `export_enai`.

## 8) Bottom Line
- The auth and deployment concerns that dominated the earlier audit are now mostly closed in current code.
- The repo's remaining highest-value work is no longer procedure/trade/vector blocker closure; it is operational cleanup around `/evaluate`, startup lifecycle modernization, cautious rollout defaults, and content-managing more routing metadata.
- With the currently reviewed `export_enai` frontend in front of it, this system is reasonable to deploy publicly. The main residual caveats are operational rather than architectural: keep Supabase grants/RLS aligned with the checked-in contract, keep `/evaluate` off, and treat the edge-function layer as the only browser-facing path for chat and admin operations.
