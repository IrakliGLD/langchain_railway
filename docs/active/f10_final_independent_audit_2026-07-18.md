# F10 Final Independent Audit — 2026-07-18

## Decision

The ordered remediation sequence for the blockers below is documented in [`f10_blocker_remediation_plan_2026-07-18.md`](./f10_blocker_remediation_plan_2026-07-18.md).

**F10 is not closed.** The source revisions pass the available local quality gates and the public production surfaces are healthy, but the final release gate cannot be signed because:

1. the backend dependency audit reports unresolved known vulnerabilities and no named, time-bound security waiver exists;
2. the exact backend production revision cannot be derived from the deployed service, while frontend and Edge version labels are stale even though their deployed timestamps/source hashes indicate newer content;
3. authenticated production chat/admin/accessibility and live-database evidence could not be executed without operator credentials; and
4. compatibility defaults and rollout flags remain intentionally retained because the required two-release/telemetry evidence for safe removal is not recorded.

This audit does not convert missing evidence into a pass and does not invent a waiver.

## Revisions audited and push state

| Independent application | Audited repository revision | Push/deployment evidence |
|---|---|---|
| Backend (`langchain_railway`) | `921f823c0888271c38ec88c9e1aa09c73672fb45` on `refactor/review-phase-fixes` | Pushed to `origin/refactor/review-phase-fixes`. Public backend health is green, but the service exposes no immutable build identity, so production equality to this SHA is **unproven**. |
| Frontend/Supabase (`export_enai/_repo_sync`) | `4f22cb3a7b1b766d6802c90f613e096b75ef68a8` on `main` | Local `main` equals `origin/main`. The production asset timestamp aligns with this revision, but the HTML reports historical version `3a8ceda`; exact artifact-to-SHA equality is therefore **unproven**. |
| Supabase Edge sources | Frontend revision above; source-manifest digest `e0111671ca86dcc3a29c088290c60ec608ec4181d2ce2ba7f2bbe68b0b7ce85e` | Public preflights for all nine functions report this current digest; their human-readable Git version label is stale. |

The two repositories remain independent. No source, build, or runtime path in either application reads files from the other repository. Their integration boundary remains the versioned HTTP/JSON chat-gateway contract.

## Automated and public-production evidence

### Backend

| Gate | Result |
|---|---|
| Full pytest | `1,664` tests passed in the full run; five Windows temporary-directory setup errors were reproduced as environment permission failures. The affected modules then passed `9/9` using a repository-owned temporary root. Combined result: no product-test failure across the `1,669` collected tests. |
| Ruff | Pass (`All checks passed!`). |
| Dependency audit | **Fail:** `pip-audit` found 59 advisories across 11 installed requirement packages. Applicability varies, but the network/auth dependency set is not releasable without triage and remediation or an approved waiver. |
| Public health | `GET /healthz` 200; `GET /readyz` 200 with database and schema ready. |
| Public auth boundary | A contract-shaped unsigned `POST /ask` returned 401. Allowed-origin preflight returned the expected credentialed CORS policy. |
| Bounded readiness load | 40 requests at concurrency 8: 40/40 HTTP 200, p50 448 ms, p95 1,094.5 ms, max 1,260.4 ms. This validates only readiness/control capacity, not model/DB chat load. |

### Frontend, Edge, contracts, and database sources

| Gate | Result |
|---|---|
| Frontend unit/component suite | Pass: 458 tests, zero failures. |
| ESLint | Pass. |
| Production npm audit | Pass: zero reported production vulnerabilities. |
| Production build and artifact verification | Pass with safe CI placeholders; reproducibility manifest `31136a71bd9d04c10ef185d7f85c69f2dde5386b97bb6236ef53dfd046368a6b`. Vite warned that the main bundle is 616.49 kB (193.32 kB gzip). |
| Generated chat-gateway contract | Pass; generated v2 consumers are byte-current. |
| Edge verify | Pass with pinned Deno 2.1.4: manifest, format, lint, type-check, and 21 tests. |
| Database patch verification | Pass for F2, P3.B, P6.B, and FB.5 static verifiers. Live DB behavior was not rerun without a test database/credential. |
| Public production frontend | Dashboard and `/chat` return 200 with CSP and security headers. Exact SHA is not exposed correctly. |
| Public accessibility | Public/login axe smoke passed with no serious or critical finding. Authenticated dashboard/chat/admin scans were skipped because credentials were unavailable. |
| Deployed Edge source parity | All nine function preflights reported source digest `e0111671ca86dcc3a29c088290c60ec608ec4181d2ce2ba7f2bbe68b0b7ce85e`, equal to the current frontend manifest. `healthcheck` is green, but its `version` value is historical (`3244ed1`). |

## Adversarial findings

### F10-SEC-01 — unresolved backend dependency advisories

- **Severity:** High
- **Why this is a problem:** the current backend lock-by-requirements set includes known advisories in web/auth/provider-framework dependencies. `pip-audit` reported 59 advisory records across `langchain`, `langchain-core`, `langchain-community`, `langchain-openai`, `langchain-text-splitters`, `langsmith`, `litellm`, `protobuf`, `PyJWT`, `python-dotenv`, and `starlette`. Not every advisory is reachable in this application: for example, no production `litellm` import was found and direct bearer auth is disabled in production. That reduces exposure but does not make the untriaged set acceptable. Starlette is network-facing and PyJWT is directly imported, so the set cannot be dismissed wholesale.
- **Affected files/components:** `requirements.txt`; `main.py`/Starlette request handling; JWT verification code; `core/llm.py`, `core/llm_runtime.py`, and provider integrations.
- **Recommended fix:** isolate dependency modernization from behavioral work. Remove demonstrably unused packages first, upgrade FastAPI/Starlette and PyJWT to fixed compatible versions, then upgrade the LangChain family as one compatibility-tested group. Run the full backend, contract, provider, deadline, and golden suites plus `pip-audit` after each group. For any advisory retained, record package/advisory, reachability analysis, compensating control, named security owner, expiry date, and remediation ticket in an approved waiver.

### F10-REL-02 — deployed artifacts do not prove their exact source revision

- **Severity:** High
- **Why this is a problem:** a green health check does not prove that the tested source is the deployed source. The backend exposes no immutable SHA. Frontend HTML reports `3a8ceda` while the deployed asset timestamp is consistent with a later build. Edge functions share the current source digest, but `healthcheck.version` reports `3244ed1`. Incident response and rollback cannot reliably identify the artifact under test.
- **Affected files/components:** Railway build/deploy configuration, frontend `VITE_APP_VERSION`/production artifact generation, Edge `APP_VERSION`, release-evidence workflows, `/healthz`/`/readyz` metadata policy.
- **Recommended fix:** inject the full immutable Git SHA at build/deploy time, record it in the artifact manifest and release evidence, expose it through a deliberately public non-secret version field/header, and fail promotion if the deployed SHA/digest differs from the tested SHA/digest. Do not use a manually maintained version string as release identity.

### F10-E2E-03 — required authenticated production and live-database closure evidence is absent

- **Severity:** High (release-evidence blocker, not a newly demonstrated code defect)
- **Why this is a problem:** public endpoints cannot exercise actor assertion, entitlement/quota, paused-user behavior, conversation continuity, admin authorization/mutations, authenticated accessibility, or database migration constraints. These are explicitly part of the final gate.
- **Affected files/components:** deployed browser application; all Supabase Edge functions; backend `/ask`; production RLS/functions/patches; credentialed Playwright and database smoke workflows.
- **Recommended fix:** a release operator must run the credentialed smoke matrix against the exact promoted SHAs: normal chat, malformed response, backend/Edge timeout, browser abort/navigation, quota exhausted, paused user, optional degradation, critical failure, admin pagination/filter/mutation/refresh, cross-actor replay rejection, axe/keyboard/focus/screen-reader/200%-zoom checks, JSONB constraints, RLS/grants, and rollback. Store redacted results with deployment IDs.

### F10-ARCH-04 — rollout compatibility paths are still present after their cutover phase

- **Severity:** Medium
- **Why this is a problem:** `ENABLE_AGENT_LOOP` survives as an inert trace-shape switch although the loop was deleted, and the code default for `ENAI_GATEWAY_ACTOR_ASSERTION_MODE` remains `optional` even though P3 records production as `required`. Other P4/retrieval/contract compatibility paths remain behind flags. Stale switches enlarge the supported state space and can make a new or misconfigured deployment less strict than production.
- **Affected files/components:** `config.py`, `models.py`, `agent/pipeline.py`, gateway auth defaults/documentation, P4 rollout flags, v1/JSONB compatibility consumers, related tests.
- **Recommended fix:** do not delete them blindly during an audit. First record two stable production releases, counter/trace readings, consumer/version inventory, and rollback evidence as required by P8. Then remove each path in a dedicated behavior change: retire the inert agent-loop trace flag; make signed actor assertions required by default; remove v1/legacy JSON readers only after all consumers and rows are migrated; and remove experimental flags only after the selected behavior is permanently active. Until then, production must explicitly set the strict values and monitoring must alert on drift.

### F10-PERF-05 — frontend main bundle exceeds the configured advisory threshold

- **Severity:** Low
- **Why this is a problem:** a 616.49 kB main chunk can delay first interaction on constrained devices. This is an advisory build warning, not a failed functional gate.
- **Affected files/components:** Vite chunking/lazy-loading boundaries and large frontend dependencies.
- **Recommended fix:** capture route-level bundle composition, then lazy-load admin/chart-heavy routes and split only measured high-cost dependencies. Preserve contract and accessibility suites; do not mix this with the deferred Vite/esbuild major upgrade.

## Architecture conformance decision

The implemented query pipeline remains contract-driven and the deployed defaults previously attested by the operator are consistent with the intended gateway-only, one-replica design. F9 changed internal ownership without changing public schemas or the browser–Edge–backend boundary. The module map in `query_pipeline_architecture.md` was stale and is corrected by this audit to name `ApplicationRuntime`, `SessionRuntime`, `ProviderInvocationRuntime`, `PipelineStageOrchestrator`, typed plan validation, question interpretation, evidence derivation/finalization, and summary grounding.

The **documentation** needed that ownership correction. The **code/defaults** still need later, evidence-gated cleanup for the compatibility paths in F10-ARCH-04; changing those defaults without deployment/consumer evidence during an independent audit would be less safe than recording the blocker.

## A–F assessment (live-evidence weighted)

| Perspective | Grade | Evidence-based rationale |
|---|---:|---|
| Functional correctness and query pipeline | **B** | 1,669 backend tests are accounted for and 458 frontend tests pass; public health is green. No credentialed production query was available, so production correctness is not an A. |
| Architecture and maintainability | **B** | F9 boundaries exist and schemas remain independent, but the pipeline compatibility module is still large and stale compatibility switches remain. |
| Security and privacy | **C** | Public auth/CORS and frontend production audit pass, but backend dependency advisories and credentialed/live privacy evidence block stronger assurance. |
| Reliability, concurrency, and error handling | **B** | Deadline/session/DB coordination suites pass and readiness survives bounded concurrency. Provider-delivery and paid-chat failure injection was not rerun against production. |
| Performance and scalability | **B-** | Public readiness p95 is about 1.09 s and one-replica containment is intentional; representative model/DB load evidence is absent and the frontend bundle warning remains. |
| Frontend UX and accessibility | **B** | Unit/component and public axe smoke pass; authenticated, keyboard, assistive-technology, responsive, and 200% zoom evidence remains operator-dependent. |

**Overall: C+.** This is not a claim that the application is generally broken. It is the appropriate final-assurance grade while three High release blockers have neither been closed nor waived.

## Manual closure checklist

1. Deploy or identify the backend artifact built from `921f823c0888271c38ec88c9e1aa09c73672fb45` (or a reviewed descendant) and record its Railway deployment ID, image digest, full Git SHA, one-replica setting, and rollback deployment ID.
2. Rebuild/promote the frontend with `VITE_APP_VERSION=4f22cb3a7b1b766d6802c90f613e096b75ef68a8` (or the final promoted descendant) and make the generated artifact manifest the authority.
3. Deploy Edge functions from that same frontend revision with `APP_VERSION` set to the full promoted SHA. Confirm all nine functions report the expected source digest and version.
4. Remediate F10-SEC-01 and rerun `pip-audit`. If immediate remediation is impossible, obtain a named, approved, time-bound waiver containing advisory-level reachability and compensating controls.
5. Run the credentialed production/browser/database matrix in F10-E2E-03 and attach redacted evidence to the release record.
6. Run the backend release-evidence/SBOM workflow against the exact deployed SHA, review the SBOM and audit, and preserve the tested artifact digest.
7. Rehearse rollback, then verify `/healthz`, `/readyz`, a real signed chat request, conversation continuation, quota accounting, and admin refresh after promotion and after rollback.
8. Only after the P8 two-release/counter prerequisites are recorded, remove the compatibility paths in F10-ARCH-04 in independent changes with regression tests.

F10 may be marked complete only after these records are attached and the three High blockers are either closed or covered by named, approved waivers with explicit expiry dates.
