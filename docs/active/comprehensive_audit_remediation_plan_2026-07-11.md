# Comprehensive Audit Remediation Plan

**Date:** 2026-07-11
**Status:** P0, P1, P2, and P3 implementation tracks are independently committed. P3.A backend commit `5883228` and P3.B frontend commit `59b4d64` are deployed; the frontend formatting/manifest repair `3244ed1` is merged in frontend `main` at `5ae2b9b`. The operator confirmed completion of the P3.A/P3.B staging, production, smoke, soak, reconciliation, assertion-enforcement, and legacy-grant-revocation steps on 2026-07-13. P3 is complete under the supported gateway-only, one-backend-replica operating mode. Frontend-first package FB.1 (P6.3 and P6.7) is locally complete in independent frontend commit `b69831a`; deployment/browser smoke is not yet recorded here. FB.2 (P6.2) is locally complete in independent frontend commit `da652da`; its Supabase patch, bounded data migration, edge/frontend deployment, constraint validation, and browser smoke remain manual activation evidence. FB.3 (P6.4) is locally complete in independent frontend commit `c007fb1`; deployment and browser degraded/retry/cache smoke remain manual release evidence. FB.4 (P6.5) is implementation-complete in independent frontend commit `0624f91`; credentialed accessibility and manual assistive-technology release evidence remains open. FB.5 (P6.6) is implementation-complete in independent frontend commit `d588bd8`; its database patch/edge/frontend deployment, dedicated PostgreSQL regression, and browser/authorization smoke remain manual activation evidence. FB.6 (the independently executable P5.B subset) is implementation-complete in frontend commit `9bbe444`; edge/frontend deployment and the documented failure-injection smoke remain release evidence. The P5.A backend request-boundary deadline foundation is implementation-complete in independent backend commit `a199c88`, and its Edge-to-backend header complement is implementation-complete in independent frontend commit `a61de51`; P5.1 remains open for per-call DB/provider timeout and cooperative cancellation coverage plus end-to-end duplicate-execution/charge evidence. P5.2 database-gateway implementation is complete in backend commit `ba8dd11`; staging failure-injection and deployment evidence remain manual. Direct bearer remains intentionally disabled and horizontal scaling remains intentionally blocked until their later P5/P7 gates. H1 remains off until P4. Non-P3 production attestations that were not explicitly confirmed remain governed by their existing ledgers. See [p0_execution_ledger_2026-07-11.md](p0_execution_ledger_2026-07-11.md), [p0_manual_activation_and_followup_2026-07-12.md](p0_manual_activation_and_followup_2026-07-12.md), [p1_execution_ledger_2026-07-12.md](p1_execution_ledger_2026-07-12.md), and [p2_execution_ledger_2026-07-13.md](p2_execution_ledger_2026-07-13.md).
**Latest frontend P7 update (2026-07-14):** FB.7 (the independently executable P7.B subset) is implementation-complete in frontend commit `5fa8be1`. Browser and Edge operational logs are minimized behind allow-listed diagnostics, public healthcheck output is aggregated, frontend/edge release evidence tooling is added, production artifact verification and SBOM generation are wired into a protected GitHub workflow, and the P3.B generated-artifact verifier now normalizes LF/CRLF source fragments. Edge/frontend deployment, exact hosted artifact promotion, production smoke, and GitHub environment evidence remain manual release items. P7.2 live revocation, P7.4 scaling/state acceptance, and the P7 phase-wide exit gate remain open; the later P7.A local backend implementation is recorded in the 2026-07-15 update below.
**Latest backend update (2026-07-14):** P5.A request-boundary deadline foundation is implementation-complete in backend commit `a199c88`. `/ask` now converts the optional gateway budget header into one capped monotonic deadline, exposes safe remaining-budget/retry-owner metadata, returns typed `408 REQUEST_DEADLINE_EXCEEDED` responses, and checks remaining time before expensive pipeline stages. The versioned gateway contract, regression tests, and `p5a_backend_deadline_activation_runbook_2026-07-14.md` are included. Defaults activate without a new backend environment variable, and frontend commit `a61de51` sends the compatible budget header. This is not full P5.1 completion: per-call DB/provider timeout injection, cooperative cancellation of already-running calls, and end-to-end duplicate-charge/failure-injection evidence remain open.
**Latest frontend complement (2026-07-14):** Independent frontend commit `a61de51` now derives a backend budget from the existing Edge exchange timeout, reserves a five-second Edge response margin, caps the transmitted budget at `115000` milliseconds, and sends `X-Enai-Request-Budget-Ms` on the existing signed backend request. No browser runtime, database schema, backend source, or cross-repository file dependency changed. Edge deployment and staging/production failure-injection evidence remain manual; per-call DB/provider cancellation and duplicate-execution/charge evidence still keep P5.1 open.
**Latest P5.2 update (2026-07-14):** Backend commit `ba8dd11` routes fallback SQL, typed tools, vector operations, analyzer/pipeline enrichment, readiness, and schema reflection through one classified database gateway. Only transient SQLSTATE/infrastructure failures affect the breaker; syntax, schema, constraint, validation, and content errors do not. The gateway prevents connection acquisition while open and removes the vector path's duplicate string-matched breaker notification. No frontend or database migration is required.
**Latest backend phase update (2026-07-15):** P4.A implementation is complete in backend commits `b69ea2b`, `9da08e7`, `c2d19a8`, `286ecba`, and `7bb8d02`; the behavior-changing gates intentionally remain at `shadow`, `warn`, or `off` until their production evidence gates pass. P5.4/P5.5 are complete in `2c9e12b`, the explicit OpenAI timeout slice of P5.1 is complete in `d9ec12b`, and the versioned P6.A chat-gateway schema source is complete in `55e61c9`. P7.A's local privacy/runtime/packaging package is committed in `4e7d0bb`, but the P7.A track remains partial because live database identity/credential revocation, PUBLIC/network grants, exact-image promotion, dependency scan, one-replica topology, rollback, privacy/vendor-retention evidence, and any future shared-state implementation are still open. P8.A's first behavior-neutral extraction is complete in `c8bd654`, and its pytest/coverage/assessment debt slice is complete in `245aa1b`; the P8.A track remains incremental because further god-module extractions, remaining high-risk coverage, the Stage 0.7 production-counter decision, LIGHT latency evidence, and two-stable-release flag cleanup remain open.
**F0 deployment reconciliation (2026-07-16):** The operator confirmed a successful Railway production deployment of backend commit `4215050b7fceeeea539f0b9ff2addd088237e158` after the fixed-port/start-command/container/schema corrections in `87e35b3` through `4215050`. Deployment logs show gateway-only authentication, Uvicorn bound to `0.0.0.0:3000`, successful required-schema reflection, and a passing Railway `/readyz` healthcheck. The operator also confirmed the `enai_api_readonly` role creation/canary, read-only and forbidden-operation probes, production runtime-URL rotation, and successful readiness under that identity. This closes the deployed P0.10 healthcheck behavior and the live-runtime-identity portion of P7.2 only. PUBLIC/network-grant review, removal of any retained broad application credential, exact-artifact/SBOM evidence, rollback rehearsal, privacy/vendor-retention evidence, and control-plane autoscaling attestation remain open. Frontend/Supabase deployment evidence is unchanged unless separately recorded in its independent repository/runbooks.
**F1 backend runtime hardening (2026-07-16):** Backend commit `1488ff4` removes the direct-startup `main.py` re-import, rejects `ENAI_HTTP_WORKERS`, `WEB_CONCURRENCY`, or `UVICORN_WORKERS` values other than one while runtime state is process-local, and bounds readiness reflection with a success TTL, outage retry interval, and lock-protected single-flight refresh. Readiness remains fail-closed for stale or incomplete schema state. Focused concurrency, stale-cache, startup, and configuration regressions pass. Deployment cadence, replica/autoscaling control-plane evidence, and exact production-query-cost evidence remain manual gates; this does not authorize more than one Railway replica.
**F2 generated chat contract (2026-07-16):** Backend producer commits `040e90e` and `2e93098` publish the strict additive `chat-gateway-v2` schema while retaining `chat-gateway-v1` as the default/rollback contract. Independent frontend/Supabase consumer commit `7803c43`, corrected for checkout-stable schema hashing in `1838a79` and Railpack-safe web/Edge verification separation in `d1d6a1e` and checkout-stable Edge manifest hashing in `6c782d3` and pinned-Deno formatting in `ad30053`, vendors that released schema with immutable commit/hash provenance, deterministically generates browser/Edge DTO validators, negotiates `chat-edge-v3`, preserves multiple charts and public provenance/trust through the additive `complete_chat_operation_v3` persistence path, and renders/restores them in the UI. The repositories remain independently buildable and deployable: no frontend runtime or build step reads backend-repository files. Local implementation is complete; production activation still requires the ordered database/backend/Edge/browser rollout and smoke evidence recorded below.
**F3 canonical pipeline activation (2026-07-16):** Backend implementation is committed in `b03ba32` for deterministic request-scoped 0/5/25/100 P4 canaries and aggregate rollout telemetry; distinct live/history rendering of degraded, clarification, policy, and transient terminal outcomes is committed and pushed in independent frontend commit `aa96e6b`. Safe production defaults remain `shadow`/`warn`/`off`/`off`; no behavior-changing production gate has been enabled. Backend verification passed 66 focused P4 tests; the full suite produced 1,576 passes and only two Windows shared-temp setup errors, and the affected six-test module passed completely with a workspace-owned temp root. Frontend verification passed all 455 tests and scoped lint. The production build passed both generated-contract verifiers but stopped because local `VITE_SUPABASE_URL`/`VITE_SUPABASE_ANON_KEY` were absent; rerun the normal CI/Railway build with deployment public config. Repository-wide local lint is polluted by untracked historical `.worktrees`, while the four F3 files pass scoped lint. Frontend commit `aa96e6b` is pushed to its independent `main` branch; unrelated local artifacts were not staged. Production completion remains manual and sequential under [p4_f3_canonical_pipeline_activation_runbook_2026-07-16.md](p4_f3_canonical_pipeline_activation_runbook_2026-07-16.md). No F3 SQL patch or Edge Function deployment is required.
**Scope:** Backend at D:/Enaiapp/langchain_railway, frontend and Supabase assets at D:/export_enai, and their deployed integration
**Purpose:** Resolve every verified finding from the comprehensive audit and follow-up review without using severity labels as a substitute for dependency-aware prioritization

This remains the governing plan and does not authorize a production cutover by itself. Implementation evidence and unresolved gates are recorded in the linked execution ledger.

### Independent repository rule

- Track `Pn.A` belongs to the backend repository at `D:/Enaiapp/langchain_railway`.
- Track `Pn.B` belongs to the frontend/Supabase repository linked to `D:/export_enai`.
- Each track is implemented, verified, committed, deployed, and rolled back independently. A clean track must not wait for the other repository.
- A phase-wide exit gate is an integration/release gate only. It does not prevent an independently green repository track from being committed.
- Checks that require a live database, deployed edge function, production secret, traffic log, or hosting control plane are manual attestation items when those systems are unavailable. They are marked `Manual verification pending`, not treated as local code failures.
- Cross-repository behavior becomes active only after every dependency in the activation runbook is deployed with compatible versions.

### Status semantics and current blockers

Checkboxes in this plan now track the narrow acceptance item beside them. A checked item means the implementation or documentation artifact is complete according to local evidence; it does not imply production deployment, live-database execution, branch-protection configuration, or privacy-owner approval unless that item explicitly says so. Manual/live items stay unchecked until the operator records evidence.

| Area | Current state | Remaining blocker or manual gate |
|---|---|---|
| P0.A backend | Code is locally verified and committed in backend commit `e667f6b`; the later plan/ledger update is `675b497`. | Operator must record live flags, replica/worker count, access-log direct-bearer inventory, Railway deployment identity, `/readyz` behavior, gateway-secret parity, and platform restart/probe evidence. |
| P0.B frontend/Supabase | Code is locally verified and committed in frontend commit `3a8ceda`. | Operator/privacy owner must dispose of legacy privacy exports, apply/verify DB changes against a live database, deploy compatible Edge/browser artifacts, verify deployed hashes/RLS/grants/branch protection, and run browser/privacy smoke. |
| P1.A/P1.B | Local implementation is complete and committed; edge functions now have reproducible source layout and immutable deployment workflow. | Archive/remove obsolete local copies only after active deployment source is proven; record deployed backend/frontend/edge SHAs and staging/production source-hash evidence. |
| P2.A/P2.B | Local implementation is complete and committed; H1 remains intentionally off. | Production smoke/shadow evidence and P4 entry review remain manual; do not enable universal canonical-frame enforcement from P2 alone. |
| P3.A/P3.B | Operator confirmed staging, production, smoke, soak, reconciliation, assertion enforcement, and legacy-grant revocation on 2026-07-13. P3 is complete for gateway-only, one-backend-replica operation. | Direct bearer remains disabled, and horizontal scaling remains blocked until their P5/P7 gates pass. |
| P4.A/P4.B | P4.A behavior implementations are complete in `b69ea2b`, `9da08e7`, `c2d19a8`, `286ecba`, and `7bb8d02`. F3 adds locally verified deterministic per-actor canaries and aggregate assignment metrics; P4.B's generated contract/persistence path plus distinct terminal-outcome UI are committed in independent frontend commit `aa96e6b`. | Safe defaults remain `shadow`/`warn`/`off`/`off`. Frontend build-with-deployment-config, backend/frontend deployment, baseline observation, sequential 0→5→25→100 gates, golden/smoke evidence, and production exit attestation remain open. |
| P5.A/P5.B | Request-boundary deadline foundation is complete in backend `a199c88`; DB gateway in `ba8dd11`; provider/breaker/metrics work in `2c9e12b`; OpenAI timeout slice in `d9ec12b`; frontend timeout/retry in `9bbe444`; Edge budget propagation in `a61de51`. | P5.1 still needs remaining per-call DB/provider timeout propagation, cooperative cancellation, and end-to-end duplicate-execution/charge evidence. P5.3 session/shared-state work and live failure-injection/deployment evidence remain open. |
| P6.A/P6.B | P6.1 is locally complete: backend producer `chat-gateway-v2` is in `040e90e`/`2e93098`, and the independent generated frontend/Edge/persistence/UI consumer is in `7803c43` with checkout-stable hashing in `1838a79` and the Railpack-safe verification fix in `d1d6a1e` plus checkout-stable Edge manifest identity in `6c782d3` plus pinned-Deno source formatting in `ad30053`. P6.2-P6.7 frontend packages are locally complete or implementation-complete as recorded below. | Apply the F2 Supabase patch before Edge/browser v3 deployment, deploy compatible backend/Edge/browser artifacts in order, and record multiple-chart/history smoke. P6.2/P6.6 live DB work and P6.5 credentialed/manual accessibility evidence also remain manual/independent. |
| P7.A/P7.B | P7.A's local privacy/runtime/packaging package is complete in backend `4e7d0bb`; rate-limit state extraction supporting the future scaling boundary is in `c8bd654`. FB.7 is implementation-complete in frontend `5fa8be1`. | P7.A is still partial: live least-privilege identity/revocation, PUBLIC/network grants, dependency/image evidence, exact artifact promotion, one-replica topology, smoke/rollback, vendor retention, privacy-log canaries, and any approved multi-replica shared-state implementation remain open. |
| P8.A/P8.B | P8.A completed one behavior-neutral repository extraction in `c8bd654` and the pytest/coverage/assessment slice in `245aa1b`; P8.B has the prior generated-artifact/toast debt reductions. | Further backend/frontend extraction is incremental. Stage 0.7 counters, LIGHT latency/correctness evidence, two-release flag cleanup, and the Vite/esbuild major upgrade remain open. |

## 1. Sources and interpretation

This plan reconciles:

- the comprehensive backend/frontend audit;
- the user's follow-up verification, which confirmed approximately forty findings and challenged only severity framing;
- [graded_assessment_2026-07-08.md](graded_assessment_2026-07-08.md);
- [query_pipeline_architecture.md](query_pipeline_architecture.md); and
- the existing privacy, database, deployment, and operational documents in both code trees.

The follow-up review changes execution priority in three important ways:

1. H1 remains the keystone pipeline defect.
2. H2 is a hard prerequisite for H1, but its current live exposure is limited because H1 keeps most canonical-frame rendering on a cold path.
3. H12 is treated as Medium for planning because it is primarily an honesty/degradation problem rather than a security or data-corruption boundary.

The target is not a letter grade. The target is demonstrable analytical correctness, server-enforced authorization and entitlements, bounded failure behavior, reproducible releases, and documentation that describes the deployed default path.

## 2. Priority model

Execution order is determined by:

1. present production impact;
2. exploitability or silent analytical harm;
3. implementation and rollback risk;
4. dependency ordering; and
5. effort.

Relative effort:

- **S:** less than one engineering day;
- **M:** one to three engineering days;
- **L:** multi-day, cross-service, or schema/deployment work.

Change risk:

- **Low:** localized behavior with a straightforward rollback;
- **Medium:** answer, persistence, authorization, or deployment behavior changes;
- **High:** distributed state, migrations, concurrency, or multi-service cutovers.

## 3. Non-negotiable invariants

- Do not enable H1 on the normal path until H2 unit semantics and M2 provenance are corrected and tested.
- Browsers must never authoritatively increment quotas, choose billing periods, persist assistant answers, assign tiers, or assert account/admin status.
- Every paid or quota-controlled operation must have one actor-bound idempotency key and a durable state transition.
- Only active users may use protected services; only active administrators may perform administration.
- At least one active administrator must always remain.
- User-originated history is untrusted regardless of whether it arrived through a gateway.
- Audit history remains append-only and survives user deletion.
- A failed data request must not be relabelled as a conceptual success.
- Answer and chart must use the same period, filters, units, evidence identity, and provenance.
- One request owns one absolute deadline and one retry policy across browser, edge, backend, database, and model providers.
- Keep evidence re-analysis disabled and the backend at one worker/replica until their dedicated completion gates pass.
- Database migrations are additive first. Legacy grants and compatibility paths are revoked only after consumers have cut over and reconciliation is clean.
- Rollbacks must not restore a known authorization, quota, privacy, or provenance bypass.
- No behavior-changing implementation should begin in the frontend until one source tree is declared canonical.

## 4. Ownership to assign before implementation

| Workstream | Accountable role | Supporting roles |
|---|---|---|
| Analytics units, statistics, and evidence frames | Backend/AI lead | Data/analytics QA, frontend |
| Quotas, ledgers, RLS, and admin invariants | Database/Supabase lead | Security, edge/backend |
| Gateway identity, deadlines, sessions, and API contract | Backend/API lead | Platform, edge, frontend |
| Frontend state, calculations, charts, and accessibility | Frontend lead | QA, design/accessibility |
| Privacy exports, logging, retention, and audit policy | Privacy/Security owner | Platform, database |
| DB pool, deployment, containers, and least privilege | Platform/SRE lead | DBA, backend, security |
| Regression, adversarial, concurrency, and release qualification | QA/release lead | All implementation owners |

Every item must receive a named owner, reviewer, target milestone, and linked evidence before implementation starts.

## 5. Dependency and rollout order

The critical dependency chain is:

1. P0 containment and truthful release gates.
2. P1 canonical source/deployment foundations.
3. P2 analytical truth prerequisites.
4. P3 transactional entitlements, identity, and admin invariants.
5. P4 canonical pipeline activation.
6. P5 reliability, deadlines, and concurrency controls.
7. P6 frontend/API completion and accessibility.
8. P7 privacy, deployment, least privilege, and scaling hardening.
9. P8 structural refactoring and low-priority optimization.

P1 and P2 may run in parallel after P0. P3 depends on P1 for reproducible edge delivery. P4 depends on P2. P5 should use the idempotency and identity established in P3. P6 consumes the shared contracts created in P3/P4. P8 starts only after behavior is locked by regression tests.

Every major phase below is split into an `.A` backend track and a `.B` frontend/Supabase track. The detailed numbered packages remain the authoritative acceptance criteria; the track sections assign those packages and their cross-service obligations to a repository.

### Frontend-first continuation after P3

The backend application may remain frozen at P3.A while independently executable frontend/Supabase work continues. All changes in this sequence belong to the `EnaiDashboard` repository; they must not import, mount, read, generate from, or otherwise rely on files in the backend checkout. A cross-service contract must be versioned and committed into each repository through its normal release process.

If a frontend item discovers that the deployed HTTP contract lacks a required field or behavior, stop that item at its consumer boundary and record the backend dependency. Do not invent browser-derived authority, duplicate backend analytics, or add a permissive fallback merely to claim frontend completion.

Recommended frontend-only order, prioritizing impact and low change risk:

| Order | Frontend/Supabase package | Work that can proceed without backend source changes | Remaining dependency or gate |
|---|---|---|---|
| FB.0 | P0.B/P1.B/P2.B carry-forward attestations | Dispose of legacy privacy exports under owner approval; verify branch protection/required checks, deployed edge hashes, canonical hosting source, browser smoke, and P2 unit-contract smoke; archive obsolete local copies only after their active-use status is proven. | Some checks observe the deployed backend, but no backend code change is required. Do not treat an unobserved live check as passed. |
| FB.1 | P6.3 and P6.7 | **Locally complete in frontend commit `b69831a`.** Centralized safe UI errors/config/bootstrap/route boundaries and consolidated the toast implementation. | Uses the already deployed P3 safe error envelope. No backend code or new backend field was required. Deploy the frontend commit through the normal frontend release process and run browser smoke before production attestation. |
| FB.2 | P6.2 | **Locally complete in frontend commit `da652da`.** Legacy chart data/metadata readers, new-write shape enforcement, generated additive patch, bounded concurrent-safe migration, restricted quarantine, privacy export coverage, and regression tests are implemented. | Frontend/Supabase only. Apply the generated patch, deploy the edge/frontend commit, run migration batches to zero, validate constraints, and smoke legacy/new turns before production attestation. Live database execution was unavailable locally and remains a manual verification item. |
| FB.3 | P6.4 | **Locally complete in frontend commit `c007fb1`.** Dashboard loading is single-flight, actor-isolated, atomic, freshness/range-aware, stale-response safe, and explicit about optional degradation and retry. | Frontend-only implementation; P3 lease and billing semantics are unchanged. Deploy and browser-smoke cache hits, range changes, optional failure, critical refresh failure, actor change, and retry before production attestation. |
| FB.4 | P6.5 | **Implementation complete in frontend commit `0624f91`; release verification remains open.** Accessible control/state semantics, chart equivalents, modal fullscreen behavior, contrast corrections, and axe automation are implemented. | Frontend only. Deploy the commit, run the credentialed dashboard/chat/admin axe suites, and complete the documented keyboard/screen-reader/zoom/viewport matrix before production attestation. |
| FB.5 | P6.6 | **Implementation complete in frontend commit `d588bd8`; release verification remains open.** Database keyset pagination, hard page limits, service-role-only listing/aggregate RPCs, server prefix/status filtering, strict edge/browser envelopes, cancellation/stale-response protection, current-page mutation refresh, generated migration sources, and large-user regression coverage are implemented. | Frontend/Supabase only; no analytics-backend source, deployment, environment variable, or filesystem dependency is required. Apply the generated patch, deploy the edge then browser from the same commit, run the dedicated PostgreSQL regression, and complete staging/production smoke per the activation runbook. |
| FB.6 | P5.B independent subset | **Implementation complete in frontend commit `9bbe444`; release verification remains open.** Browser and edge calls have bounded local aborts, chat uses at most one typed safe retry with the same request ID, ambiguous delivery is not replayed, `Retry-After` is bounded/CORS-visible, and backend exchange stalls terminate locally. | Frontend/Supabase only; no analytics-backend source, environment variable, deployment, or filesystem dependency was added. Deploy all Edge Functions then the browser from the immutable commit and complete the activation runbook. End-to-end absolute deadline propagation and final retry ownership still require P5.A's compatible backend contract. |
| FB.7 | P7.B independent subset | **Implementation complete in frontend commit `5fa8be1`; release verification remains open.** Browser diagnostics now pass through one sanitized boundary, Edge operational metadata is allow-listed, public healthcheck responses are aggregate-only, frontend/edge packaging excludes local artifacts, production artifact verification/SBOM tooling is in place, and the protected release-evidence workflow builds from an exact commit SHA. | Frontend/Supabase only; no backend source, environment variable, deployment, or filesystem dependency was added. Deploy Edge Functions and the browser from the immutable commit, run the protected release-evidence workflow, complete smoke/rollback evidence per `docs/active/fb7_release_privacy_hardening_runbook_2026-07-14.md`, and coordinate any grant/network revocation with P7.A/P7.2 inventory first. |
| FB.8 | P8.B | Refactor frontend/edge modules incrementally behind characterization tests; consolidate test infrastructure; perform the deferred Vite/esbuild major upgrade as a separate change. | Start after the affected P6 behaviors are locked. No backend source change is required. |

Frontend packages that cannot be fully closed while backend work is frozen:

- **P4.B:** adapters, persistence slots, and disabled UI fixtures may be prepared, but activation and final consumer tests require P4.A's versioned canonical result/outcome/chart/provenance contract.
- **P6.1:** code-generation infrastructure may be prepared, but the generated DTO migration requires P6.A's complete OpenAPI/JSON Schema source. A hand-copied substitute is not acceptable.
- **P5.B integration gate:** frontend-local timeout/retry work may commit independently, but the one-deadline end-to-end acceptance requires P5.A.
- **P7.2/P7.4 shared gates:** frontend/Supabase hardening can proceed independently, but least-privilege revocation and multi-replica/shared-state acceptance require backend usage evidence and P7.A/P5.A coordination. They do not authorize one repository to read the other's files.

## 6. Phase summary

| Phase | Primary outcome | Findings | Effort | Risk |
|---|---|---|---|---|
| P0 | Immediate containment and truthful CI/release state | H4, H7, H8, H11, H16, H19, H20, M17, M24 | S–M | Low–Medium |
| P1 | One frontend source and reproducible edge delivery | H17, H18, M14 | M | Medium |
| P2 | Correct units, statistics, filters, scope, and frame provenance | H2, H3, M2, M10, M12, M13 | M–L | Medium |
| P3 | Server-owned entitlements, actor identity, admin invariants, and persistence | H5–H10, M5–M8, M16 | L | High |
| P4 | Universal canonical evidence and aligned charts/plans | H1, H12, M1, M3, M4 | L | Medium–High |
| P5 | Bounded retries, DB work, providers, metrics, and sessions | H13–H15, M11, M22, M23, L1 | L | Medium–High |
| P6 | Complete frontend contract, deterministic state, resilient UX, accessibility | M5, M7–M9, M19–M21, L4 | L | Medium |
| P7 | Privacy/logging, packaging, least privilege, and production attestation | H21, M15, M18, M23, L3 | M–L plus ops | Medium–High |
| P8 | Deep-module refactoring, coverage redistribution, and remaining debt | M25, L2, internal open items | Incremental | Medium |

## 7. P0 — Immediate containment and low-risk/high-impact fixes

**Goal:** remove current release blockers and cheap high-impact failures before beginning structural work.

Use separate, reviewable changes. Do not bundle all P0 items into one large change.

### P0.A — Backend application track

- Owns the backend portions of P0.1 and P0.2, plus P0.6, P0.7, the ASGI/backend half of P0.8, and P0.10.
- Commit gate: Ruff, the full backend suite, adversarial provenance/auth/body tests, startup/readiness tests, and `git diff --check` pass locally.
- Deployment activation: set explicit production auth/secrets/body limits, deploy one replica, and verify `/healthz`, `/readyz`, gateway authentication, and platform restart behavior.
- Live traffic, database schema, and Railway-control-plane checks are operator attestations and do not hold the backend commit.

### P0.B — Frontend/Supabase application track

- Owns the frontend/Supabase portions of P0.1 and P0.2, P0.3 through P0.5, the edge half of P0.8, and P0.9.
- Commit gate: clean install, lint, fail-on-console tests, production build/audit, edge source-contract tests, privacy tests, and changed-file checks pass locally.
- Deployment activation: apply the database patch, set Supabase/Railway variables, deploy all changed edge functions, deploy the frontend, and run the manual smoke/DB/privacy checks.
- Privacy-owner disposition and unavailable live DB/edge checks are operator attestations and do not hold the frontend commit.

### P0 shared activation dependency

`chat-with-enerbot` must be deployed with `CHAT_BACKEND_SECRET` equal to backend `ENAI_GATEWAY_SECRET`, and `CHAT_BACKEND_URL` must target the compatible backend deployment. See the P0 activation runbook before enabling traffic.

### P0.1 — Establish the issue ledger and baseline

- [x] Create a repository-local issue ledger for every H, M, and L identifier in the coverage matrix below.
- [ ] Assign named owners, reviewers, milestones, and external tickets where the project tracker is authoritative.
- [x] Record local flags, database pool settings, authentication mode, known deployment unknowns, and lock/runtime evidence.
- [ ] Record live replica/worker count, deployed edge hashes, production direct-bearer traffic, RLS/grants, and least-privilege identity from the deployed services.
- [x] Keep canonical evidence enforcement and evidence re-analysis off.
- [x] Capture clean backend and frontend test/lint/build baselines from available local runtimes.
- [ ] Capture pinned CI/deployment baselines from the production Node/Python runtimes.
- [x] Record which production checks cannot be verified locally: deployed RLS/grants, edge parity, provider timeouts, and live least-privilege identity.
- [x] Verify previously recorded internal fixes rather than reopening them without evidence.

**Done when:** every finding has a named owner, evidence link, acceptance test, rollback owner, and state of Open, In progress, Verified, or Deferred with approved reason.

### P0.2 — Restore truthful release gates

**Findings:** M24 and the internal branch-protection recommendation
**Effort/risk:** S / Low

- [x] Fix the four current frontend lint errors.
- [x] Eliminate unresolved React act warnings and the invalid dismiss DOM property warning.
- [x] Configure tests to fail on unexpected React errors, unhandled rejections, and console errors.
- [x] Require database regression tests for release/protected branches; missing TEST_DATABASE_URL must fail a release workflow.
- [ ] Turn on branch protection and make backend/frontend required checks non-bypassable for normal merges.
- [x] Record exact Node, npm, Python, and dependency-lock inputs used by local verification and intended CI.
- [ ] Record successful protected-branch CI evidence on pinned production runtimes.

**Acceptance:**

- npm ci, lint, tests, build, smoke checks, backend Ruff, security gates, and full backend tests pass from clean checkouts.
- No required release job silently skips DB or edge verification.
- No behavior change is introduced by the lint-only portion.

### P0.3 — Quarantine privacy exports and stop repository-local export output

**Finding:** H20
**Effort/risk:** S / Low, with Privacy approval

- [x] Inventory current export artifacts visible in the repository/export tree without copying payloads into new locations.
- [ ] Inventory repository history, shared archives, CI artifacts, and backups under Privacy-owner supervision.
- [ ] Move retained exports into encrypted, access-controlled incident/fulfilment storage.
- [x] Do not delete existing exports until the Privacy owner decides whether incident evidence or fulfilment records must be preserved.
- [x] Make future exports write outside the repository with restrictive ACLs, encryption, expiry, and an auditable deletion step.
- [x] Ignore privacy export paths, temporary reports, and real environment files; retain only reviewed examples.
- [ ] Rotate credentials only if investigation finds actual secret exposure.

**Acceptance:**

- Normal repository, CI artifacts, and project backups contain no live privacy export payload.
- A test export is created outside the repository, is access-restricted, and expires under policy.
- Secret/data scanning rejects export payloads and real environment files.

**Rollback:** disable exports if secure storage fails. Never return to repository-local raw export storage.

### P0.4 — Preserve append-only audit history

**Finding:** H19
**Effort/risk:** S–M / Low–Medium

- [x] Remove the deletion path that disables the audit-protection trigger or deletes subject/actor audit rows.
- [ ] Decide whether deleted users remain as UUIDs or become stable pseudonymous identifiers.
- [x] Ensure foreign keys do not force audit deletion in the committed additive patch and regression contract.
- [ ] Execute the audit-retention regression against a live `TEST_DATABASE_URL` and deploy/verify the migration.
- [x] Align the deletion response and privacy runbook with retained audit behavior.

**Acceptance:**

- User deletion succeeds while related audit events remain.
- Update/delete attempts against audit records remain denied.
- Trigger state remains enabled after successful and failed deletion.
- Code, database behavior, and privacy documentation agree.

### P0.5 — Require active administrators immediately

**Finding:** H8
**Effort/risk:** S / Low–Medium

- [x] Add active-status verification to every current admin edge function and is-admin.
- [x] Enumerate every administrative endpoint in one regression test.
- [x] Prepare a service-role break-glass procedure before deployment so a configuration mistake cannot permanently lock out operators.
- [x] Schedule centralization under P3 after reproducible edge sources exist.
- [ ] Deploy the matching edge functions and verify paused-admin 403 behavior plus deployed source hashes.

**Acceptance:**

- A paused or demoted admin with an otherwise valid token receives 403 from every administrative endpoint.
- Replaying an old token does not restore authority.
- Active admins retain expected access.

### P0.6 — Contain direct-bearer bypass

**Finding:** H7
**Effort/risk:** S configuration plus traffic analysis / Medium

- [ ] Inventory legitimate direct /ask bearer clients from trusted production access logs.
- [x] If none exist, explicitly set gateway-only mode and restrict backend network exposure where possible.
- [x] If direct bearer is required, do not leave it exempt: schedule it for the same active-status and transactional entitlement service as the edge path in P3.
- [x] Document the decision; do not rely on auto mode to choose the security boundary.
- [ ] Verify the deployed environment is gateway-only and record network exposure.

**Acceptance:**

- Paused or exhausted users cannot reach unmetered /ask through a direct token.
- Any gateway assertion rejects tampering, expiry, actor mismatch, and unauthorized replay.

**Rollback:** disable chat or restore the last trusted gateway artifact. Do not reopen unmetered direct bearer access.

### P0.7 — Close the low-effort provenance bypass

**Finding:** H11
**Effort/risk:** S / Low–Medium

- [x] Apply the existing claim-derivation mechanism to legacy fallback text.
- [x] Run fallback claims through the normal provenance gate.
- [x] Return an explicit degraded-evidence result when numeric claims cannot be grounded.
- [ ] Observe rejection/disagreement in shadow mode briefly before enforcement if production fallback volume is material.

**Acceptance:**

- An invented numeric fallback cannot pass as no_claims.
- Grounded numeric fallback passes.
- Nonnumeric safe fallback remains available.

### P0.8 — Bound request bodies before expensive parsing

**Finding:** H16, with H10 history trust completed in P3
**Effort/risk:** S–M / Low

- [x] Add ASGI and edge request-size limits.
- [ ] Verify the production proxy/platform request cap and memory/latency behavior under deployed load.
- [x] Preserve the live Q&A history contract with a typed question/answer turn model, forbid extra fields, allow at most three turns, and enforce per-field limits. The edge alone maps stored role/content rows into this contract.
- [x] Return stable 413/validation responses.
- [x] Add memory/latency tests for oversized and malformed bodies.

**Acceptance:**

- Oversized bodies are rejected before model/Pydantic work and without material memory growth.
- Oversized fields and turn counts produce stable validation errors.

### P0.9 — Correct the live frontend null-average bug

**Finding:** H4
**Effort/risk:** S / Low

- [x] Define missing-value semantics: finite zero is valid; null, undefined, NaN, and infinity are missing.
- [x] Apply the same denominator rule to price and tariff aggregation.
- [x] Define whether each published metric is a simple or weighted average.
- [ ] Confirm production telemetry/browser smoke after deploying the frontend commit.

**Acceptance:**

- [10, null, 20] produces 15.
- [0, null] produces zero with sample count one.
- All-missing input produces an explicit no-data result.
- Input order does not change the result.

### P0.10 — Make readiness and startup honest

**Finding:** M17
**Effort/risk:** S / Low

- [x] Return non-200 when any required readiness dependency, including schema reflection, is unavailable.
- [x] Distinguish optional/degraded components explicitly.
- [x] Let fatal startup exceptions terminate with a nonzero exit code.
- [x] Verify deployed platform startup and `/readyz` healthcheck behavior on the production Railway service.
- [x] Bound readiness reflection cost with a success TTL, an outage retry interval, and single-flight refresh while preserving stale-schema fail-closed behavior.
- [ ] Verify deployed readiness-query cadence/cost and control-plane probe behavior after the F1 backend batch is released.

**Acceptance:**

- Readiness returns 503 for missing required DB/schema state.
- Startup failure exits nonzero and triggers normal platform restart/failure detection.

### P0 exit gate

**Current state (2026-07-14):** Local P0 implementation is complete and committed for both repositories. The phase cannot be called production-complete until privacy-owner disposition, live database/edge/Railway/GitHub checks, branch protection, deployed source hashes, and browser smoke evidence are recorded.

- All P0 tests and release checks pass.
- Privacy exports are no longer generated under the repository.
- Inactive admins and direct-bearer callers cannot bypass current controls.
- The legacy summary path cannot bypass numeric provenance.
- Audit deletion no longer contradicts policy.
- No P0 rollback restores a known bypass.

## 8. P1 — Canonical source trees and reproducible edge delivery

**Goal:** make subsequent frontend and security work land in the code that is actually built and deployed.

### P1.A — Backend application track

- No backend source-tree migration is required in this phase.
- Publish and test the gateway request/auth/version contract consumed by `chat-with-enerbot`, including body limits and request correlation.
- Record the backend commit/deployment identity used by frontend integration tests.
- Completion and commit are independent of P1.B; incompatibility discovered by contract tests becomes a separately scoped backend change.

### P1.B — Frontend/Supabase application track

- Owns P1.1 through P1.3: reconcile the two frontend trees, port guarded auth behavior, and convert edge snapshots into deployable Supabase sources.
- P1.3 is required for reproducible activation of every later edge/database cutover. It is locally complete; manual dashboard copy/paste is retired and later cutovers must use the immutable workflow.
- Completion gate: one canonical Git source tree, deterministic auth tests, clean edge build/test/deploy from checkout, and immutable deployed-source evidence.

### P1.1 — Reconcile the two frontend trees

**Findings:** H18 and prerequisite for H17
**Effort/risk:** M / Medium

- [x] Declare the root `IrakliGLD/EnaiDashboard` Git repository the canonical source used by Vite/Caddy builds.
- [x] Freeze _repo_sync; implement no new work there.
- [x] Generate a file-level diff and classify each difference as accepted root behavior, reviewed fix to port, generated artifact, or obsolete experiment.
- [x] Port accepted behavior in small changes with tests.
- [x] Compare route, component, hook, asset, and test inventories.
- [ ] Archive/remove obsolete local export or mirror copies only after every unique difference has an explicit disposition and the active checkout/deployment source is recorded. Do not delete a directory merely because its local name is `_repo_sync` if it is the operator's current clone of the canonical GitHub repository.
- [x] Add CI enforcement that rejects reintroduction of the mirror.

**Acceptance:**

- No runtime, build, lint, CI, or deployment path references _repo_sync.
- Critical route smoke tests pass from the single canonical tree.
- The reconciliation record accounts for every divergent or mirror-only file.

### P1.2 — Port and verify the guarded auth implementation

**Finding:** H17
**Effort/risk:** M / Medium

- [x] Use the mirror implementation only as a reviewed reference.
- [x] Keep the Supabase auth listener synchronous and schedule async profile/admin work outside it.
- [x] Introduce a monotonic generation/run ID or abortable state machine.
- [x] Bound profile/admin lookups with explicit timeouts.
- [x] Clear protected client state synchronously on sign-out.

**Acceptance:**

- Delayed INITIAL_SESSION cannot resurrect state after SIGNED_OUT.
- Rapid sign-in, refresh, and sign-out ends deterministically signed out.
- Stale profile/admin responses cannot overwrite a newer session.
- Offline/time-out startup reaches a bounded explicit state.
- Reload, logout, paused-user, and two-tab browser scenarios pass.

### P1.3 — Replace manual edge snapshots with deployable sources

**Finding:** M14
**Effort/risk:** M / Medium

- [x] Move functions into the standard Supabase functions source layout.
- [x] Centralize CORS, active-admin/auth checks, logging, and error envelopes.
- [x] Pin exact remote dependencies and commit the runtime lockfile.
- [x] Add formatting, linting, typechecking, unit tests, local integration tests, and source-hash verification.
- [x] Add protected-environment CI deployment and rollback by immutable commit SHA.
- [x] Archive manual copy/paste deployment instructions.

**Acceptance:**

- A clean checkout can build, test, and deploy every edge function without renaming/copying files.
- Deployed source hashes match the approved commit.
- The previous signed artifact can be redeployed by hash.

### P1 exit gate

**Current state (2026-07-14):** P1.A and P1.B are locally complete and committed. Remaining blockers are deployment evidence, immutable source-hash proof, staging/production smoke, and cautious retirement of obsolete local copies after the active checkout/deployment source is proven.

- One frontend source tree exists.
- The auth race regression suite is green.
- Edge functions are executable, pinned, tested sources with immutable deployment evidence.
- Later quota/admin work can be delivered and rolled back reproducibly.

## 9. P2 — Analytical truth layer

**Goal:** correct unit semantics and statistical behavior before making canonical frames universal.

### P2.A — Backend application track

- Owns P2.1 through P2.4: the metric/unit registry, chronological statistics, frame provenance, and analytical scope/cache corrections.
- Produce a versioned machine-readable registry/schema and golden corpus as backend artifacts.
- Keep H1 canonical-frame enforcement off until this track's dimensional, order-invariance, scope, and provenance gates pass.

### P2.B — Frontend/Supabase application track

- Consume the versioned metric/unit definitions for labels, filters, chart axes, tables, exports, and validation; do not maintain a divergent manual conversion table.
- Add consumer-contract fixtures proving frontend display/filter behavior matches backend canonical values and declared precision.
- This track may commit independently, but deployment of new unit semantics must be version-compatible with P2.A or guarded by a compatibility flag.

### P2.1 — Create the metric/unit registry

**Finding:** H2
**Effort/risk:** M / Medium

- [x] Define storage unit, canonical unit, display unit, conversion, compatible aggregations, precision, and filter semantics for every metric family.
- [x] Correct GEL/MWh to tetri/kWh and USD/MWh to cents/kWh conversion.
- [x] Correct thousand MWh to MWh conversion.
- [x] Correct ratio to percentage conversion.
- [x] Convert before filtering, deriving, comparing, and rendering.
- [x] Reject incompatible unit comparisons rather than guessing.
- [x] Replace tests that currently encode raw values as display values.

**Acceptance:**

- 15 tetri/kWh filters equivalently to 150 GEL/MWh.
- 10 percent filters equivalently to ratio 0.10.
- Quantity and share round trips satisfy declared tolerances.
- Every adapter-to-render path has dimensional tests.
- Tariff behavior remains unchanged where already correct.

**Rollout:** dual-run conversions over the golden corpus. Keep H1 off until this is enforced and stable.

### P2.2 — Make statistics chronological and unit-aware

**Finding:** H3
**Effort/risk:** M / Medium

- [x] Parse, sort, and deduplicate by the documented period grain.
- [x] Count unique periods, not rows.
- [x] Use actual elapsed calendar time for CAGR.
- [x] Compute recent windows chronologically, regardless of source order.
- [x] Replace mixed numeric-column quick statistics with per-metric compatible results.
- [x] Correct the equal-values trend label.

**Acceptance:**

- Ascending and descending input produce identical results.
- Duplicate entity rows do not make a year appear complete.
- Missing years/months are reported and cannot silently alter CAGR.
- GEL, USD, rates, quantities, and shares are never averaged together.

### P2.3 — Repair frame provenance

**Finding:** M2
**Effort/risk:** S–M / Low

- [x] Remove the nonexistent ctx.provenance path.
- [x] Bind actual query hashes and all contributing source hashes to canonical frames.
- [x] Define how merged/derived evidence carries multiple references.

**Acceptance:**

- Every DB/tool-backed frame has nonempty source/query provenance.
- Derived frames preserve all contributing sources.
- Provenance survives serialization into answer/chart metadata.

### P2.4 — Correct remaining analytical scope/cache edge cases

**Findings:** M10, M12, M13
**Effort/risk:** S–M / Low–Medium

- [x] Keep correlation fallback within the requested period and expose N, period, and uncertainty.
- [x] Make structured forecast horizon authoritative; normalize case and supported word forms as fallback.
- [x] Include provider, exact embedding model, dimension, normalization/version, and corpus version in cache keys.

**Acceptance:**

- Correlation never includes dates outside the answer contract.
- 10-year, 10-Year, and supported word forms resolve identically.
- Model/dimension/corpus changes cause embedding cache misses.

### P2 exit gate

**Current state (updated 2026-07-15):** P2.A passed its backend gates and P2.B passed its local frontend gates. The P4 entry review has since produced the P4.A implementation commits; evidence finalization defaults to `shadow`, while enforcement/cutover still requires production evidence.

- Dimensional, statistical, filter, period-scope, and provenance golden tests pass.
- Shadow comparisons explain every intentional numeric change.
- H1 enforcement remains off until the P4 rollout gate explicitly confirms production shadow evidence.

## 10. P3 — Transactional entitlements, identity, admin invariants, and persistence

**Goal:** move all authoritative security, billing, and persistence decisions to trusted server/database boundaries.

**Current state (2026-07-13): Complete.** Backend commit `5883228` and frontend/Supabase commit `59b4d64` implement the two independent tracks. Frontend formatting/manifest repair `3244ed1` is merged into frontend `main` at `5ae2b9b`. The operator confirmed the additive and revoke migrations, immutable edge deployment, frontend deployment, staging/production smoke and soak, reconciliation, verified actor assertions, backend `required` mode, and post-revocation checks. Gateway-only auth and one backend replica remain supported safety constraints; they are not unfinished P3 work.

### P3.A — Backend application track

- Owns the backend portions of P3.2, P3.4, P3.5, P3.6, and P3.8.
- Verify signed actor/session/request context, treat all history as untrusted, preserve request identity, enforce safe typed errors, and keep direct bearer disabled until a later explicitly approved path can call the same active-status, entitlement, idempotency, and persistence authority.
- Commit independently behind compatible versioned contracts; do not enable a new direct-bearer mode until P3.B's database/edge authority is deployed.

**Backend implementation status (2026-07-13):**

- [x] Verify the P3.B HMAC over contract/request/actor/session/issue time, reject partial/tampered/stale/future assertions, and add bounded same-operation replay protection.
- [x] Provide an `optional` → `required` assertion rollout gate so the two repositories remain independently deployable.
- [x] Bind edge and direct-bearer session tokens to the authenticated actor; derive stable opaque sessions from verified edge actor/session pairs.
- [x] Preserve the external request ID, allocate a distinct backend span, and publish both identifiers without using them as authorization inputs.
- [x] Treat gateway/database/session/bearer history as untrusted and render it through escaped non-instructional prompt boundaries.
- [x] Reject unknown Ask v1 fields, explicitly reject `service_tier`, and publish a safe typed error envelope.
- [x] Deploy P3.B schema/edge authority, observe verified assertions, then set `ENAI_GATEWAY_ACTOR_ASSERTION_MODE=required` in backend staging and production.
- [x] Keep direct bearer disabled outside tests. Enabling it is a future product/scaling change that requires the same deployed authority; it is not required to close P3 safely.
- [x] Keep one backend replica until P5/P7 externalize replay/session state and multi-process tests pass. This remains an ongoing deployment invariant.

Deployment, verification, rollback, and complementary manual work are recorded in [`p3a_backend_activation_runbook_2026-07-13.md`](p3a_backend_activation_runbook_2026-07-13.md).

### P3.B — Frontend/Supabase application track

- Owns P3.1 and P3.3, the database/edge/persistence portions of P3.2, P3.4, P3.6, P3.7, and P3.8, plus browser removal of authoritative charging/persistence.
- Deliver additive schema/RLS/RPCs first, shadow and reconcile, switch edge persistence, switch the browser consumer, then revoke legacy grants.
- P1.B deployable edge sources are a hard prerequisite; P3.B must not be activated from untracked manual snapshots.

**Frontend/Supabase implementation and activation status (2026-07-13):**

- [x] Merge the P3.B entitlement/administration/frontend authority implementation and the required Deno-format/source-manifest repair into frontend `main`.
- [x] Apply the additive authority migration in staging and production.
- [x] Deploy the immutable edge source set and the matching frontend release.
- [x] Complete smoke, soak, stale-operation review, counter reconciliation, and failure/privacy checks.
- [x] Apply the final legacy-authority revoke migration and repeat the protected chat/dashboard/admin checks.

### P3.1 — Add entitlement and idempotency primitives

**Findings:** H5, H6 and prerequisites for M5–M7
**Effort/risk:** L / High

Design an additive entitlement-operation ledger containing:

- server-issued request ID;
- authenticated actor;
- operation kind;
- server-derived billing period;
- state such as reserved, completed, released, or failed-chargeable;
- charged units;
- server timestamps and expiry;
- result/turn reference and failure class.

Requirements:

- [x] Request IDs are unique and actor-bound.
- [x] Replays by the same actor are idempotent; cross-actor replays are denied.
- [x] Reservation atomically locks/checks status and quota.
- [x] Caller timestamps, IDs, tiers, and response metadata cannot affect billing.
- [x] Only trusted server/database functions mutate the ledger.
- [x] Current counters reconcile at cutover; do not fabricate per-request history.

### P3.2 — Move chat charging and persistence to the edge/server

**Finding:** H5, with H13 retry behavior completed in P5
**Effort/risk:** L / High

Target flow:

1. Authenticate actor and active status.
2. Establish authoritative request ID.
3. Atomically reserve chat entitlement using server time.
4. Call the backend once with signed actor/session/request context and bounded untrusted history.
5. Persist user/assistant turn and native metadata server-side.
6. Mark the operation completed before returning success.
7. Return the persisted result for idempotent replays.

Failure policy:

- release only when no paid request was sent;
- preserve ambiguous delivery as reserved/chargeable until reconciliation;
- never retry with a new ID;
- alert and reconcile stale reservations.

Cutover order:

1. additive schema/RLS and versioned v2 RPCs;
2. edge shadow decisions and reconciliation;
3. edge server persistence;
4. browser recognizes server persistence and stops writing;
5. revoke authenticated execution of the old authoritative increment/persistence path;
6. remove compatibility code in a later release.

**Acceptance:**

- Direct edge calls consume quota and store the turn even without browser persistence.
- Concurrent unique requests cannot exceed the limit.
- Concurrent retries with one ID produce one charge, one turn, and at most one model execution.
- Manipulating created_at, actor, tier, or metadata has no billing effect.
- Paused/exhausted users are rejected before model work.

### P3.3 — Enforce dashboard entitlements at the data boundary

**Finding:** H6
**Effort/risk:** L / High

- [x] Define exactly what consumes one unit.
- [x] Atomically reserve a unit and issue an actor-bound, short-lived query lease.
- [x] Require the lease on protected dataset, inventory, and hourly RPCs.
- [x] Ensure pagination under one lease charges once.
- [x] Revoke or guard raw unmetered RPC access.
- [x] Do not charge operations that never acquire a lease/data snapshot.

**Acceptance:**

- Direct protected RPC calls without a lease are denied.
- Cross-user and expired lease replay is denied.
- Concurrent reuse of a request ID charges once.
- Multi-page loads use one charge.

### P3.4 — Unify active-user and admin authorization

**Findings:** H7, H8, H9
**Effort/risk:** M–L / Medium–High

- [x] Create one server/database authority for active-user and active-admin checks.
- [x] Use it from every enabled production path. Direct bearer remains disabled rather than bypassing the authority.
- [x] Serialize role/status/delete operations.
- [x] Enforce at least one active administrator transactionally, including concurrent demotions.
- [x] Require recent authentication for destructive administration using the available session authentication timestamp.
- [x] Revoke sessions on pause/demotion.

**Acceptance:**

- Paused users cannot use protected services.
- Paused admins cannot call any admin endpoint.
- Self-pause/delete and last-active-admin removal are rejected.
- Concurrent attempts to remove each other leave at least one active admin.

### P3.5 — Treat every user-originated history as untrusted

**Finding:** H10
**Effort/risk:** M / Medium

- [x] Apply history firewalling, control-character handling, and explicit untrusted prompt boundaries regardless of transport.
- [x] Preserve role structure and maximum turn/field limits.
- [x] Add stored-history prompt-injection cases to the security gate.

**Acceptance:**

- Persistent injection remains inert through gateway, database history, and direct bearer paths.
- Benign multilingual history continues to work within the false-block threshold.

### P3.6 — Normalize turns, JSONB, actor/session identity, and request correlation

**Findings:** M5, M6, M7
**Effort/risk:** L / High

- [x] Add turn ID, request ID, explicit role/sequence ordinal, and deterministic ordering.
- [x] Store all new chart data/metadata as native JSONB through server-owned persistence.
- [x] Provide the temporary legacy chart-data reader. Complete coverage of legacy string scalars plus the idempotent bounded migration/quarantine remains explicitly owned by P6.2 and is not a P3 authority-cutover blocker.
- [x] Forward a signed actor assertion and bind session/conversation tokens to that actor.
- [x] Preserve the external request ID and create a separate internal span ID.
- [x] Define versioned Ask request/response schemas; stop silently ignoring service tier.

**Acceptance:**

- JSONB metadata round-trips without string scalars or field loss.
- Turn ordering is deterministic under equal timestamps and concurrent tabs.
- Cross-user session replay fails.
- One request ID joins browser, edge, backend, entitlement, persistence, and logs.
- Unknown contract fields follow an explicit reject/tolerate policy.

### P3.7 — Make failed administration durably auditable

**Finding:** M16
**Effort/risk:** M / Medium

- [x] Record an append-only attempt event before the mutation.
- [x] Commit expected rejection/failure outcomes without re-raising away the audit insert.
- [x] Use an idempotent saga for external Auth deletion plus database cleanup.
- [x] Correlate started, succeeded, failed, and compensated outcomes.

**Acceptance:**

- Every attempt has a durable outcome for expected rejection, Auth failure, timeout, and DB exception.
- Replaying an action ID is idempotent.
- Audit-protection triggers are never disabled.

### P3.8 — Standardize safe error behavior

**Finding:** M8
**Effort/risk:** M / Low–Medium

- [x] Check every Supabase data/error result in the P3 authority, entitlement, persistence, lease, and administration paths. The broader browser-wide safe-error/render audit remains P6.3.
- [x] Define fail-open/fail-closed decisions: profile/quota checks fail closed; optional history may degrade explicitly.
- [x] Publish a safe error envelope with code, safe message, retryable flag, and request ID.
- [x] Preserve real backend status categories instead of mapping all non-2xx responses to 502.

**Acceptance:**

- Fault injection does not expose SQL, PostgREST details, stack traces, URLs, provider bodies, or tokens.
- Suspension, quota exhaustion, validation failure, timeout, transient outage, and malformed response remain distinguishable.

### P3 exit gate

**Exit state (2026-07-13): Passed by committed automated evidence plus operator attestation.**

- [x] No authoritative entitlement or security decision depends on browser input.
- [x] Old direct quota/persistence grants are revoked after compatibility soak.
- [x] Active-status and last-admin invariants hold under concurrency.
- [x] Stored history is treated as untrusted.
- [x] Reconciliation reports no unexplained charge, turn, or audit mismatch.

P3 does not authorize direct bearer or multi-replica backend operation. It also does not claim completion of the P6 generated-contract, full legacy-JSON migration, or browser-wide error/UX work.

P3 finding disposition:

| Finding | P3 disposition | Later work that remains |
|---|---|---|
| H5, H6 | Closed: chat and dashboard authority is server/database-owned and legacy browser grants are revoked. | P5 improves deadline/retry behavior without reopening authority. |
| H7 | Closed for the supported deployment: production is gateway-only and verified assertions are required. | A direct-bearer product mode remains disabled unless a future design integrates the same authority. |
| H8, H9 | Closed: active-admin, recent-authentication, session-revocation, serialization, and last-active-admin invariants are enforced. | P6.6 bounds administration listing; it does not alter authority. |
| H10 | Closed: all history is bounded and untrusted at the backend prompt boundary. | P6.2 removes legacy persistence compatibility debt. |
| M5 | P3 identity/versioning portion closed. | P6.1 still replaces the hand-maintained bridge with generated complete DTOs. |
| M6 | Closed: actor, session, request, edge span, and backend span identities are preserved and bound. | P5/P7 externalize state before scaling. |
| M7 | P3 new-write/native-JSONB and deterministic-order portion closed. | P6.2 completes legacy scalar migration/quarantine. |
| M8 | P3 authority/edge safe-error portion closed. | P6.3 completes browser-wide mapping, bootstrap, render validation, and error boundaries. |
| M16 | Closed: administration attempts and terminal outcomes are durable and deletion is an idempotent saga. | P7 retains/minimizes operational logs without mutating the append-only audit record. |

## 11. P4 — Canonical pipeline activation and chart/plan alignment

**Entry condition:** P2 unit and provenance work is enforced and green.

### P4.A — Backend application track

- Owns P4.1 through P4.5: evidence finalization, plan enforcement, canonical chart sourcing, honest terminal outcomes, and complete re-analysis semantics.
- Activation order is off, shadow, staged enforce, then full enforce; evidence re-analysis remains disabled until its separate transition suite passes.
- Publish versioned result/outcome/chart/provenance schemas for P4.B.

### P4.B — Frontend/Supabase application track

- [x] Consume and persist the versioned canonical result, terminal-outcome, chart, filter, unit, and provenance fields without reconstructing evidence from raw rows.
- [x] Render degraded/clarification/policy/transient outcomes distinctly and add consumer contract tests for multiple charts and provenance.
- [x] Keep the frontend independently buildable: the vendored/generated contract is the only source artifact; no runtime/build filesystem dependency on the backend repository exists.
- [ ] Commit and deploy the F3 browser artifact, then record live and restored-history smoke before enabling honest terminal outcomes.

**Current state (2026-07-16):** P4.A behavior implementation is complete in backend commits `b69ea2b`, `9da08e7`, `c2d19a8`, `286ecba`, and `7bb8d02`. Backend F3 commit `b03ba32` adds deterministic request-scoped canaries for each behavior gate and `p4_rollout_events` telemetry. P4.B already consumes/persists the generated v2 contract and commit `aa96e6b` renders all non-success outcomes distinctly for live and restored messages; all 455 frontend tests and scoped F3 lint pass. Its browser deployment remains open. Evidence finalization still defaults to `shadow`, plan validation to `warn`, and honest terminal outcomes/re-analysis to off; production counter review and sequential activation remain open.

### P4.1 — Introduce one evidence finalization routine

**Finding:** H1
**Effort/risk:** M–L / Medium–High

- [x] Call one routine after normal primary execution, recovery execution, secondary merges, and any enrichment that changes evidence.
- [x] Invalidate stale frames before mutation.
- [x] Build the correct frame, bind provenance, and run evidence validation.
- [x] Return a typed finalization/gap result.
- [x] Add off, shadow, and enforce modes for the rollout.

**Backend implementation evidence:** Commit `b69ea2b` (P4.1, H1) adds `agent/evidence_finalizer.py` as the single finalization routine on every primary, recovery, merge, and enrichment evidence path; stale frames are invalidated in every mode; rollout modes are `off`, `shadow`, and `enforce`, with default `shadow` pending production evidence.

**Acceptance:**

- Every normal and recovery tool path produces the expected frame type.
- Frames rebuild after merges and never retain stale filters/rows.
- SCALAR, LIST, TIMESERIES, COMPARISON, FORECAST, and SCENARIO fixtures run validation.
- Deterministic shapes use the generic renderer with zero narrative LLM calls in enforce mode.
- Equivalent happy/recovery evidence produces equivalent canonical frames.

**Rollout:** off → shadow → `ENAI_EVIDENCE_FINALIZATION_MODE=enforce` with `ENAI_EVIDENCE_FINALIZATION_ENFORCE_PERCENT=0` → 5 → 25 → 100. Holdbacks stay in shadow. Roll back to shadow without discarding comparison telemetry. The same deterministic actor/session/request cohort mechanism independently gates plan validation, honest terminal outcomes, and re-analysis; see the F3 runbook.

### P4.2 — Make plan validation enforceable

**Finding:** M3
**Effort/risk:** M / Medium

- [x] Replace warning-only checks with a typed PlanValidationResult and gated enforcement stage.
- [x] Validate answer shape, tool capability, evidence roles, units, entities, and periods before any external call when enforcement is enabled.
- [x] Repair once, clarify, or reject; never loop indefinitely.

**Backend implementation evidence:** Commit `9da08e7` (P4.2, M3) introduces typed `PlanValidationResult` and a gated enforcement stage that clarifies before any tool/database call when enforcement is enabled; default remains `warn` pending rollout evidence.

**Acceptance:**

- Invalid plans make zero DB/tool calls.
- Repaired plans are revalidated exactly once.
- Documentation says warning-only until enforcement is actually deployed.

### P4.3 — Source charts from answer evidence

**Finding:** M1
**Effort/risk:** M / Medium

- [x] Prefer derived specifications or finalized canonical frames.
- [x] Permit raw ctx.df only through an explicit, measured fallback.
- [x] Attach evidence/filter/unit/provenance identity to each chart.

**Backend implementation evidence:** Commit `c2d19a8` (P4.3, M1) makes `_select_chart_source` prefer the finalized frame; any raw-DataFrame fallback is measured and surfaced on chart metadata rather than occurring silently.

**Acceptance:**

- Answer and chart have identical period, entity set, filters, unit, row identity, and provenance hash.
- Raw fallback is observable and cannot silently override canonical filtering.

### P4.4 — Introduce honest terminal outcomes

**Finding:** H12, planned as Medium
**Effort/risk:** M / Medium

- [x] Model conceptual answer, evidence unavailable, clarification required, policy blocked, and transient service failure separately.
- [x] Preserve the anti-retry-storm behavior without pretending unavailable data is conceptual evidence.
- [x] Prohibit numeric claims on evidence-unavailable outcomes.

**Backend implementation evidence:** Commit `286ecba` (P4.4, H12) adds the `TerminalOutcome` taxonomy and maps a data-primary SQL failure to a deterministic evidence-unavailable answer while preserving anti-retry-storm behavior; the user-facing behavior gate remains default-off pending rollout evidence.

**Acceptance:**

- Invalid/irrelevant SQL on a data-primary request produces a transparent degraded outcome, not a plausible conceptual substitution.
- Permanent failures are not retried as transient failures.

### P4.5 — Complete re-analysis semantics

**Finding:** M4
**Effort/risk:** L / Medium

- [x] Restart from an explicit stage checkpoint with a fresh dependent context.
- [x] Recompute response mode, clarification/knowledge short-circuits, retrieval tier, derived metrics, evidence plan, frames, and rendering.
- [x] Preserve only immutable request/auth state.
- [x] Keep re-analysis default-off until production activation is approved; mode-transition tests are implemented.

**Backend implementation evidence:** Commit `7bb8d02` (P4.5, M4) reruns the full post-analysis slice during re-analysis; data-to-knowledge and data-to-clarification reclassification terminate correctly instead of continuing with stale data-path state. Re-analysis remains default-off pending production activation.

**Acceptance:**

- Data-to-knowledge, data-to-clarification, and changed-answer-kind cases rerun every dependent stage.
- No stale vector, metric, frame, or render state survives.
- Re-analysis runs at most once and records its reason.

### P4 exit gate

- [ ] Universal frame shadow metrics are understood and enforcement is stable.
- [ ] Deterministic answers and their charts use the same canonical evidence.
- [ ] Invalid plans do not execute.
- [ ] Data failures are honest and the deployed frontend renders every non-success outcome distinctly.
- [ ] Re-analysis has two weeks of anomaly evidence, runs at most once, and preserves no stale dependent state.
- [ ] Architecture documentation and the activation ledger describe the deployed mode, percentages, and artifact SHAs rather than the target mode.

**F3 implementation evidence:** Backend commit `b03ba32` assigns requests deterministically per gate using verified actor → signed session → request identity precedence, partial traffic without a stable identity fails closed, invalid gate/percentage configuration is rejected, and aggregate metrics retain no identity material. Local backend evidence is 66 focused P4 passes plus the effective full 1,578-test regression result described in the 2026-07-16 status update. Independent frontend commit `aa96e6b` is supported by 455 full-suite passes, 23 focused ChatPage/outcome passes, and clean scoped lint. Local production-build completion is blocked only by absent public deployment config; generated contract verification passed before that stop. Production exit remains open under the F3 activation runbook.

## 12. P5 — Reliability, cost, concurrency, providers, metrics, and sessions

### P5.A — Backend application track

- Owns P5.2 through P5.6 and the backend deadline/retry/idempotency work in P5.1.
- Enforce one absolute request deadline, one retry owner, bounded global DB work, independent provider breakers, thread-safe metrics, actor-bound sessions, and one-replica mode until shared state is proven.
- Load/failure tests and the scaling assertion form this repository's independent completion gate.

### P5.B — Frontend/Supabase application track

- Owns browser/edge deadline propagation, abort behavior, retry classification, `Retry-After`, request-ID reuse, and removal of duplicate client retries under P5.1.
- Edge/database calls must consume the P3 idempotency identity and return safe retryable classifications; they must not mint a new billing identity on retry.
- Keep frontend/edge retries conservative until P5.A's deadline and idempotency contracts are deployed.

**Track status (2026-07-14):** The independently executable frontend reliability subset is complete in frontend commit `9bbe444`, the backend request-boundary deadline foundation is complete in independent backend commit `a199c88`, and Edge-to-backend budget propagation is complete in independent frontend commit `a61de51`. The repositories remain deployable independently and share only the versioned HTTP header contract. P5.1 remains open until per-call DB/provider timeout, cooperative cancellation, and end-to-end cancellation/duplicate-charge evidence are complete.

### P5.1 — Establish one deadline, retry owner, and idempotency identity

**Finding:** H13
**Effort/risk:** L / Medium–High

- [x] Define one absolute monotonic deadline at the trusted request boundary.
- [x] Propagate the request budget across browser, edge, and backend request-boundary contracts.
- [ ] Propagate remaining time into DB and model-provider calls and ensure already-running work cooperatively stops where possible.
- [x] Assign browser/edge/backend retry ownership through the published safe response/header contract.
- [x] Add explicit browser and edge timeouts, abort signals, bounded `Retry-After` handling, and status classification.
- [ ] Add explicit DB/provider timeouts, jitter policy, and provider-side cancellation/reconciliation.
- [x] Make all permitted browser/edge retries reuse the same actor-bound request ID.
- [ ] Prove end-to-end that timeout/ambiguous-delivery cases do not duplicate model execution or charges outside the idempotency policy.

**Acceptance:**

- A request cannot produce duplicate model calls/charges outside the idempotency policy.
- Every request terminates within the declared budget plus a small tolerance.
- Permanent 4xx/validation/quota failures are not retried.
- Ambiguous delivery is reconciled rather than blindly reissued.

**Frontend implementation evidence (2026-07-14):** Independent frontend commit `9bbe444` removes the six-attempt English-message retry loop and hidden-tab resume behavior; gives the shared Edge Function client a caller-linked 30-second default timeout; gives chat one 135-second browser-local budget; and gives the chat Edge Function a 120-second default backend exchange timeout with an optional Supabase-only `CHAT_BACKEND_TIMEOUT_MS` override. Only typed `ENTITLEMENT_UNAVAILABLE` and `REQUEST_IN_PROGRESS` outcomes may receive one bounded retry, and that retry reuses the original actor-bound request ID. Caller abort, browser timeout, raw network failure, backend/model timeout, validation, quota, persistence, and other ambiguous or permanent failures are not replayed. `Retry-After` is CORS-visible, parsed and capped, and cannot cross the remaining browser budget. Verification passed with lint, all `430` JS/JSX tests, a production build, pinned Deno format/lint/type-check gates, all `16` Deno tests, immutable edge manifest `7d564fcb063fb666f4391ec4fdd5b1dbd593bbe54b9807344505e3000ec964d3`, all generated database-patch verifiers, and a zero-vulnerability production dependency audit. A release-gate audit also pinned generated SQL patches to LF and added a regression contract so Windows checkouts no longer report false patch staleness. No backend runtime source was changed. Edge/frontend deployment, effective timeout recording, normal/offline/safe-retry/timeout/navigation/admin smoke, and request-ID/backend-execution counts remain manual release evidence in frontend `docs/active/fb6_request_reliability_activation_runbook_2026-07-14.md`. Backend commit `a199c88` supplies the request-boundary deadline and retry-owner contract, and frontend commit `a61de51` propagates the compatible Edge-to-backend budget header. The remaining per-call cancellation and duplicate-execution evidence must close before phase-wide P5.1 acceptance can be claimed.

**Backend implementation evidence (2026-07-14):** Independent backend commit `a199c88` adds the optional `X-Enai-Request-Budget-Ms` gateway header, caps it to a backend-controlled maximum, derives one monotonic deadline from request ingress, exposes `X-Enai-Deadline-Remaining-Ms` and `X-Enai-Retry-Owner`, returns safe typed deadline errors, and checks remaining time before expensive pipeline stages. It also corrects zero-second evidence-loop timeout handling, publishes the backward-compatible contract, and adds activation/rollback instructions in `docs/active/p5a_backend_deadline_activation_runbook_2026-07-14.md`. Verification passed with Ruff, `182` focused API/pipeline tests, `280` guardrail tests, and the full `1,422`-test suite; two initial pytest fixture errors caused by denied access to the Windows shared temp directory passed when rerun against a repository-local temp directory. No frontend source or deployment was changed.

**Additional backend P5.1 slice (2026-07-15):** Commit `d9ec12b` (H13) adds an explicit OpenAI request timeout with a default of `120` seconds. This closes the OpenAI per-request timeout slice only; remaining-time propagation into every DB/provider call, cooperative cancellation/reconciliation, jitter policy, and end-to-end duplicate-execution/charge evidence remain open.

**Frontend deadline-propagation evidence (2026-07-14):** Independent frontend commit `a61de51` preserves the existing `CHAT_BACKEND_TIMEOUT_MS` behavior, derives the backend request budget as the smaller of the Edge timeout minus `5000` milliseconds and the published `115000`-millisecond backend maximum, and sends it through `X-Enai-Request-Budget-Ms`. The immutable Edge source manifest is `fb8ead8dd412618adb398668e1df51085a287be926db73e62b94d809a14b9001`; the independent rollout and rollback procedure is `docs/active/fb8_backend_deadline_propagation_runbook_2026-07-14.md`. Verification passed with the full Edge manifest/format/lint/type-check gate and all `19` Deno tests, all `442` frontend tests, frontend lint, production build and artifact verification, all generated database-patch verifiers, and a zero-vulnerability production dependency audit. The credentialed Supabase privacy check was not run because live credentials were unavailable. No backend repository file was read at runtime or added as a build/deployment dependency.

### P5.2 — Route all database work through one gateway

**Finding:** H14
**Effort/risk:** M–L / Medium

- [x] Route typed tools, vector queries, fallback SQL, and availability probes through one DB gateway.
- [x] Define transient infrastructure versus syntax/schema/content/validation failures.
- [x] Allow only transient failures to affect the circuit breaker.
- [x] Protect all DB-backed paths when the breaker is open.

**Acceptance:**

- Invalid SQL does not open the DB breaker.
- Repeated connection/timeout failures do.
- An open breaker prevents all DB-backed tool paths from reaching the database.

**Backend implementation evidence (2026-07-14):** Independent backend commit `ba8dd11` adds `core/db_gateway.py`, classifies transient SQLSTATE and SQLAlchemy infrastructure failures, records non-transient SQL errors as infrastructure-reachable outcomes without incrementing failures, and guards all runtime engine connection/transaction acquisition. A source-architecture regression prevents future direct `ENGINE.connect()`/`begin()` bypasses outside the gateway. Verification passed with Ruff, a `510`-test broad main/guardrail/vector/security regression set, `15` focused gateway/security tests, and `7` readiness tests. A subsequent full `1,436`-test run was intentionally interrupted at 53% after no failures because the interactive validation wait was taking too long. Staging deployment and live failure injection remain manual under `docs/active/p5a_database_gateway_activation_runbook_2026-07-14.md`.

### P5.3 — Align global DB work with the connection budget

**Finding:** H15 and internal pool/backpressure item #7
**Effort/risk:** L plus capacity decision / Medium–High

- [ ] Replace per-request executors with a process-wide bounded coordinator/semaphore.
- [ ] Reserve capacity for control/readiness work.
- [ ] Include pool checkout/queue time in the request deadline.
- [ ] Support cancellation of DB-side work where possible.
- [ ] Set concurrency from measured Supabase/PgBouncer capacity, not local worker count.

**Acceptance:**

- Active DB work never exceeds the configured budget.
- Timed-out requests leave no running secondary jobs.
- Load tests show bounded queue/pool wait and no connection starvation.

### P5.4 — Unify provider configuration and breaker ownership

**Finding:** M11
**Effort/risk:** M / Low–Medium

- [x] Create one provider registry for credentials, model, timeout, retry, cost, and independent breaker.
- [x] Validate every selected provider at startup.
- [x] Ensure NVIDIA and OpenAI do not share state.

**Backend implementation evidence (2026-07-15):** Commit `2c9e12b` (P5.4, M11) gives NVIDIA its own breaker, validates the selected OpenAI provider key at startup, and ensures stage overrides never swap the configured provider.

**Acceptance:**

- OpenAI, Gemini, and NVIDIA breakers open/reset independently.
- Startup fails when a selected provider lacks required credentials.

### P5.5 — Make metrics thread-safe and secondary work observable

**Findings:** M22, L1
**Effort/risk:** S–M / Low

- [x] Use a thread-safe collector/backend or lock updates and snapshots.
- [x] Emit per-source Stage 0.8 metrics with a primary/secondary dimension.
- [x] Store hashes/counters rather than raw query/answer content.

**Backend implementation evidence (2026-07-15):** Commit `2c9e12b` (P5.5, M22 and L1) makes metric updates and snapshots thread-safe and adds per-source tool observability without storing raw query/answer content.

**Acceptance:**

- Multithreaded tests record exact expected totals and consistent snapshots.
- Secondary calls/latency/failures appear without double-counting.

### P5.6 — Bind sessions to actors and enforce the scaling constraint

**Finding:** M23
**Effort/risk:** M containment; L externalization / High

- [ ] Bind session token, conversation, history, and contract snapshot to subject/auth mode.
- [ ] Serialize turns per session.
- [ ] Prevent expired tokens from recreating empty sessions.
- [x] Add a startup assertion rejecting unsupported configured HTTP worker counts while process-local state remains active.
- [ ] Retain deployment/control-plane evidence that Railway runs exactly one replica with autoscaling disabled; source worker enforcement cannot prove replica topology.
- [ ] Introduce a repository abstraction and external shared state before scaling.

**Acceptance:**

- Cross-user token replay fails.
- Concurrent turns have deterministic order.
- Expired sessions cannot be resurrected.
- In-memory mode refuses unsupported scaling.
- Two-process tests share state when the external backend is enabled.

### P5 exit gate

**Current state (2026-07-15):** P5 is partially complete. P5.1 request-boundary/edge/browser work and the explicit OpenAI timeout slice (`d9ec12b`), P5.2 database gateway (`ba8dd11`), and P5.4/P5.5 (`2c9e12b`) are implemented. P5.1 remaining per-call DB/provider propagation/cooperative cancellation and duplicate-charge evidence, P5.3 global DB coordination, and P5.6 shared-state/scaling acceptance remain open. Keep one backend replica.

- Total work, time, retries, DB concurrency, provider breaker state, and session ownership are bounded and observable.
- Load and failure-injection tests pass.
- One-replica enforcement remains until external state passes its integration suite.

## 13. P6 — Frontend/API completion, deterministic state, and accessibility

### P6.A — Backend application track

- Owns the producer half of P6.1: publish versioned OpenAPI/JSON Schema with complete charts, public provenance/trust state, request ID, safe errors, pagination, and approved session fields.
- Maintain backward-compatible contract fixtures during the consumer migration and reject or explicitly tolerate unknown fields.
- The backend commit is independent; removal of an older contract version waits for P6.B adoption evidence.

### P6.B — Frontend/Supabase application track

- Owns the consumer half of P6.1 and P6.2 through P6.7: generated DTOs/runtime validation, persistence compatibility, safe bootstrap/errors, coherent dashboard snapshots, accessibility, bounded admin listing, and toast consolidation.
- Generate rather than hand-copy shared DTOs and add consumer-driven contract tests against P6.A artifacts.
- Completion gate includes browser smoke/e2e, no unexpected console diagnostics, accessibility review, and bounded large-user tests.

### P6.1 — Adopt a generated/versioned API contract

**Finding:** M5
**Effort/risk:** M–L / Medium

- [x] Make backend OpenAPI/JSON Schema the versioned contract source.
- [x] Generate edge/frontend DTOs and validate requests/responses at runtime.
- [x] Include complete charts, answer provenance/trust state, request ID, and approved session fields.
- [x] Remove or formally implement service tier as a server-derived field.
- [x] Keep internal telemetry out of the public DTO.

**Current state (2026-07-16):** P6.1/F2 local implementation is complete. Backend commits `040e90e` and `2e93098` add strict, closed `chat-gateway-v2` response/error schemas, complete chart identity/metadata, multiple charts, public provenance/trust, request correlation, safe session state, deterministic export/drift tests, and retain v1 for rollback. Independent frontend commits `7803c43`, `1838a79`, `d1d6a1e`, `6c782d3`, and `ad30053` vendor the released backend artifact from full commit `2e9309825c4f09c4f13e165b81d692ed7c86d2ed` with schema SHA-256 `b35fbb934c769e9bf04338a9cde87854270061ff62d716678d737562796582f5`, generate strict browser/Edge consumers, rejects missing/renamed/malformed/unknown fields, adds request correlation, introduces the additive `chat-edge-v3`/`complete_chat_operation_v3` persistence path, and proves multiple-chart/provenance survival for live and reloaded history UI. Backend focused contract/API tests (`125`) passed; the frontend full JavaScript suite (`449`) and production build passed. Deno was unavailable locally, so Deno format/lint/type/test remains a CI/deployment evidence gate; generated source identity is pinned by manifest `9b25997e5a37d6048580e7a7e6eb290dba33a7c46e4ed7c1a330601aa8ba777d`. Commit `1838a79` canonicalizes CRLF/LF before hashing and generated-file comparison. Commit `d1d6a1e` keeps `supabase/` excluded from the Railway static build, verifies only browser artifacts during that build, and preserves strict full browser/Edge verification in CI and the Edge deployment workflow; its missing-Supabase regression, focused contract tests, Edge manifest check, lint, and production build passed. It supersedes the incompatible `.dockerignore` exception attempted in `08b4db9`, which Railpack flattened instead of preserving at its repository path. Commit `6c782d3` canonicalizes manifest input text to LF before hashing, closes the Windows-generated/Linux-verified manifest mismatch, regenerates the embedded Edge identity, and adds an LF/CRLF regression. Commit `ad30053` applies the pinned Deno 2.1.4 formatter to the two Edge v3 imports, regenerates source identity, and passes the complete local `edge:verify` chain including 20 Deno tests.

**F2 activation blockers/manual sequence:**

1. Merge/deploy backend commit `2e9309825c4f09c4f13e165b81d692ed7c86d2ed` first; v1 remains available during migration.
2. In the live Supabase SQL editor, apply frontend patch `database/patches/2026-07-16_f2_chat_gateway_v2_persistence.sql` from frontend commit `ad30053` before deploying Edge v3. This is an additive production change, not a test-only script.
3. After confirming existing rows satisfy the shape check, run `alter table public.chat_history validate constraint chat_history_response_payload_shape_check;`.
4. Run the frontend commit's protected `Deploy Supabase edge functions` workflow so `chat-with-enerbot` and its generated/shared sources deploy together from the immutable commit.
5. Deploy the browser frontend from the same frontend commit only after the database and Edge rollout succeed.
6. Smoke one response with multiple charts, reload chat history, and confirm both charts plus the public grounding/provenance indicator survive; confirm `chat_history.response_payload->>'contract_version' = 'chat-gateway-v2'` for the assistant row.
7. Roll back browser first and Edge second if needed. The additive column, constraint, and v3 function may remain; do not remove v1 until a later explicit retirement gate.

**Acceptance:**

- Consumer-driven contract tests fail on missing, renamed, malformed, or silently ignored fields.
- Multiple charts and provenance survive backend → edge → persistence → UI.

### P6.2 — Complete chat persistence compatibility

**Finding:** M7, frontend half of P3.6
**Effort/risk:** M / Medium

- [x] Stop pre-stringifying JSONB for all new P3 server-authoritative persistence.
- [x] Read every supported legacy chart-data and chart-metadata string scalar during the compatibility window; unsupported or malformed shapes are safely contained.
- [x] Render new and legacy turns using explicit turn/sequence order with deterministic fallbacks.
- [x] Migrate valid legacy JSON strings idempotently in bounded, `SKIP LOCKED` batches; quarantine malformed and unsupported values behind service-role-only access.

**Frontend implementation evidence (2026-07-14):** Independent frontend commit `da652da` normalizes one layer of legacy chart-data and chart-metadata JSON before unit-contract enforcement; rejects new string/scalar chart payloads at the edge and database boundaries; adds a generated additive Supabase patch, `NOT VALID` shape constraints, a bounded concurrent-safe migration function, a restricted quarantine table, privacy-export coverage, and an operator activation runbook. Verification passed with `npm run lint`, all `385` JS/JSX tests, all `9` Deno edge tests, complete Deno type-checking, the edge-source manifest check, the generated P6.B patch verifier, and a production build. The dedicated SQL regression is wired into CI but was not executed locally because no live `TEST_DATABASE_URL` was available; applying the patch, running batches to `has_more = false`, validating both constraints, and browser smoke remain manual release evidence rather than local code blockers.

**Acceptance:**

- New metadata is JSON object/array/null, never a JSON string scalar.
- Re-running migration changes nothing.
- Legacy rows remain readable until the compatibility window closes.

### P6.3 — Centralize safe frontend error handling and bootstrap failures

**Findings:** M8, M19
**Effort/risk:** M / Low–Medium

- [x] Map typed errors to safe copy plus request ID.
- [x] Suppress intentional abort noise.
- [x] Validate public configuration at build/deploy time.
- [x] Render a safe config-error screen instead of throwing before React.
- [x] Add route error boundaries and an explicit 404.
- [x] Validate dates, charts, and API payloads before rendering.

**Frontend implementation evidence (2026-07-14):** Independent frontend commit `b69831a` introduces centralized public-config, public-error, and render-safety modules; applies safe typed copy to authentication, administration, chat, dashboard, and builder failure paths; contains malformed history/admin/chart payloads; adds a bootstrap configuration screen, route error boundary, and explicit 404; and makes the CI production build supply deliberate non-secret validation placeholders. `npm run lint`, all `376` JS/JSX tests, and a production `npm run build` passed. A build with missing required public configuration was also verified to fail closed. No backend source or contract change was needed. Frontend deployment and browser smoke remain release-operation evidence, not implementation blockers.

**Acceptance:**

- No raw PostgREST text, stack, URL, DB identifier, or TypeError reaches the UI.
- Missing config renders a diagnostic rather than a blank root.
- Malformed payloads are contained and the app shell remains usable.

### P6.4 — Introduce coherent dashboard snapshots

**Finding:** M9
**Effort/risk:** M / Medium

- [x] Stabilize callback dependencies and use a single-flight state machine.
- [x] Prevent usage updates from triggering duplicate loads.
- [x] Commit critical datasets atomically.
- [x] Mark optional partial data with freshness/degraded state.
- [x] Prevent stale or canceled requests from overwriting newer state.

**Frontend implementation evidence (2026-07-14):** Independent frontend commit `c007fb1` introduces one atomic dashboard snapshot object with range/TTL metadata and versioning; deduplicates identical in-flight loads; binds request identity to dataset, range, and critical-commit policy; reloads the complete required set when a primary dataset is stale; prevents older/canceled responses and late React updaters from committing; preserves the previous coherent snapshot on critical or lease refresh failure; removes failed optional data rather than mixing generations; exposes a safe degraded-state banner and explicit retry; and clears data plus load/error state on actor changes. P3 lease acquisition and billing authority are unchanged. Verification passed with full lint, all `397` JS/JSX tests, `29` focused snapshot/dashboard tests, a production build, and mounted-page degraded-state coverage. Deployment and browser smoke remain release-operation evidence.

**Acceptance:**

- One action creates one entitlement check and one load.
- Older requests cannot overwrite newer results.
- Partial critical failure preserves the previous coherent snapshot.
- Optional degradation is visible.

### P6.5 — Reach an accessibility baseline

**Finding:** M20
**Target:** WCAG 2.2 AA for affected routes
**Effort/risk:** M–L / Low–Medium

- [x] Name every icon-only action in the affected route/component scope.
- [x] Expose toggle state semantically.
- [x] Provide chart summaries and bounded, paged accessible data tables/equivalents.
- [x] Implement fullscreen charts as real dialogs with focus trapping/restoration and Escape.
- [x] Implement focus restoration, route headings, live status/error semantics, and contrast corrections; verify login/public-dashboard contrast locally.
- [ ] Complete manual touch-size, keyboard, screen-reader, 200-percent zoom, and responsive viewport verification on the deployed build.
- [x] Add automated axe scans for login, public/authenticated dashboards, chat, and admin.
- [ ] Run the credentialed authenticated/dashboard/chat/admin scans against staging and archive release evidence.

**Frontend implementation evidence (2026-07-14):** Independent frontend commit `0624f91` adds stable accessible names and pressed state, route headings and live/error semantics, chart canvas summaries plus value/unit/period table equivalents paged at 100 rows, and a Radix modal fullscreen chart with Escape, focus trapping, and explicit focus restoration. It also corrects failing dark/light contrast tokens and active-control surfaces, adds WCAG 2.0/2.1/2.2 A/AA axe coverage to the existing live-browser workflow, and documents activation and manual review in frontend `docs/active/fb4_accessibility_activation_runbook_2026-07-14.md`. Verification passed with `npm run lint`, all `406` JS/JSX tests, a production build, and a local production-preview axe run with no serious/critical findings on login or the public dashboard. Authenticated dashboard/chat and admin scans skipped locally because live smoke credentials were intentionally unavailable; the keyboard/screen-reader/touch/zoom/viewport matrix also remains release evidence and is not represented as complete.

**Acceptance:**

- No serious/critical automated accessibility violations on login, dashboards, chat, and admin.
- Keyboard users can operate every affected control.
- Canvas charts have value/unit/period equivalents.
- Dialog focus behavior is correct at mobile, tablet, desktop, and 200 percent zoom.

### P6.6 — Bound the admin list

**Finding:** M21
**Effort/risk:** M–L / Medium

- [x] Use cursor pagination, hard page limits, server filtering/search, and minimal projections.
- [x] Return role/status required for safe operations.
- [x] Compute global aggregates separately.
- [x] Add UI cancellation/stale-response protection and virtualization only if measured.

**Frontend implementation evidence (2026-07-14):** Independent frontend commit `d588bd8` replaces Auth population enumeration and four population-wide browser joins with a service-role-only PostgreSQL keyset page RPC capped at 100 rows (UI default 50) plus a separate global aggregate RPC. The edge contract validates an opaque cursor, bounded search/status input, minimal role/status/limit/usage output, and safe error envelopes; it authenticates the active administrator before parsing and retains only a bounded first-page response for the old no-body browser during edge-first rollout. The UI replaces rather than accumulates pages, performs server prefix/status filtering, cancels superseded requests, rejects stale results, and refreshes the operator's latest cursor when an earlier-page mutation finishes late. A generated additive patch, source-drift CI checks, edge/source manifest `bcb80a6e90fdf8f90d89b1f67036fe71c16100dd9936f8d2a6040d08309074e9`, contract/unit tests, and a transactional 10,000-user cursor regression are included. Frontend lint, the complete JS/JSX suite, the production build, the generated-patch verifier, pinned Deno format/lint/type-check, all 12 Deno tests, and the production dependency audit passed. The live PostgreSQL test was not run because no dedicated `TEST_DATABASE_URL` was available; database/edge/browser deployment plus database, authorization, pagination, filter, mutation, and cancellation smoke remain manual release evidence documented in frontend `docs/active/fb5_bounded_admin_activation_runbook_2026-07-14.md`.

**Acceptance:**

- No request loads the entire user population.
- Pagination has no duplicates/omissions under the documented consistency model.
- A 10,000-user test keeps response and browser memory bounded.

### P6.7 — Remove localized toast debt

**Finding:** L4
**Effort/risk:** S / Low

- [x] Keep one toast implementation.
- [x] Strip internal callbacks before spreading DOM/Radix props.
- [x] Test create, update, dismiss, suppressed notifications, and cleanup.

**Frontend implementation evidence (2026-07-14):** Commit `b69831a` removes the duplicate hook, keeps one external toast store, delegates Radix close lifecycle to the store, prevents suppressed aborts from creating blank notifications, and covers create/update/dismiss/close/global cleanup/final-unmount behavior.

### P6 exit gate

**Current state (2026-07-16):** P6.1 is locally complete across independent backend commits `040e90e`/`2e93098` and frontend commits `7803c43`/`1838a79`/`d1d6a1e`/`6c782d3`/`ad30053`; the former generated-consumer, Railway build-context, cross-checkout Edge manifest, and Deno formatting blockers are closed. P6.2 through P6.7 are locally implemented as recorded above, but the phase-wide release gate remains open until the F2 ordered live activation/multiple-chart history smoke, P6.2/P6.6 live database work, and P6.5 credentialed/manual accessibility evidence are recorded.

- The frontend consumes the complete validated API.
- Auth/dashboard/chat state is deterministic.
- Errors are safe and recoverable.
- Accessibility and large-admin-list tests pass.
- Browser smoke/e2e checks produce no unexpected console errors or warnings.

## 14. P7 — Privacy, deployment, least privilege, packaging, and scaling hardening

### P7.A — Backend application track

- Owns backend logging/telemetry minimization in P7.1, runtime database identity in P7.2, backend container/dependency artifacts in P7.3, backend shared-state/scaling work in P7.4, and its P7.5 attestations.
- Prove the runtime database role, one-replica topology, non-root/pinned artifact, secret rotation, denial probes, and deploy/rollback identity.

#### P7.A backend subtask status (2026-07-15)

| Subtask | Status | Implemented/fixed | Still required |
|---|---|---|---|
| P7.1 logging and public telemetry | **Partial** | `4e7d0bb`: allow-listed public response metadata; protected/internal telemetry removed from caller-visible DTOs; private actor/session/IP values hashed; routine raw query/answer/SQL previews redacted; raw fixture capture default-off, sampled, opt-in, and limited to development/test. | Attest that token/cost/model/stage telemetry storage is access-controlled; define and approve vendor retention, access, export, deletion, and incident procedures; run deployed synthetic log-canary scans. |
| P7.2 least-privilege database identity | **Production runtime identity verified; external hardening open** | `4e7d0bb` supplies the role/probes; the operator confirmed the production `enai_api_readonly` canary, denial probes, runtime-URL rotation, and passing readiness on 2026-07-16. No staging database is available. | Inventory and safely revoke/re-grant unnecessary `PUBLIC` privileges, restrict network access, remove any retained broad application connection secret after rollback, and retain the production evidence. |
| P7.3 packaging and deployment artifacts | **Local implementation complete; live evidence open** | `4e7d0bb`: strict `.dockerignore`; separate development dependencies; pinned Python image digest; non-root UID/GID; explicit runtime-only `COPY`; Docker selected as the Railway path; exact-SHA SBOM/audit/manifest/archive/checksum workflow. | Run the clean image build and dependency audit, inspect the image, promote the exact tested artifact, smoke it, and retain rollback evidence. |
| P7.4 state and scaling | **Partial/deferred; one worker enforced in source** | `c8bd654`: process-local sliding-window rate-limit state is behind `InMemoryRateLimitRepository`. The F1 batch passes the app object directly to Uvicorn and rejects configured worker counts other than one. | Prove exactly one live Railway replica and disabled autoscaling in the control plane. If scaling is approved later, implement shared session, continuity, rate-limit, and idempotency repositories with TTL/failure policies and pass multi-process/failure tests. Multi-replica operation remains prohibited. |
| P7.5 dependency/deployment assurance | **Partial** | `4e7d0bb`: exact-SHA evidence workflow, SBOM/audit inputs, immutable manifest/archive/checksums, non-root/excluded-artifact checks, rollback/smoke runbook; unavailable local scanning is explicitly `Unverified`. | Execute and review the workflow for staging/production; verify deployed RLS, edge hashes, secrets, grants, replica count, artifact identity, rollback commands, and named owners. |
| P7 exit gate | **Open** | Local backend and frontend hardening packages exist independently. | Complete privacy inventory, least-privilege proof, clean promoted-artifact proof, deployment parity, topology, smoke, and rollback attestations. |

### P7.B — Frontend/Supabase application track

- Owns browser/edge/Supabase logging minimization in P7.1, Supabase grants/RLS and network controls in P7.2, frontend/edge packaging and immutable deployment in P7.3, any frontend shared-state integration in P7.4, and its P7.5 attestations.
- Prove deployed edge hashes, RLS/grants, privacy retention/export operations, browser bundle contents, production dependency scans, and frontend/edge rollback artifacts.
- Each repository can complete independently; the phase exit gate requires both production attestations.

**Frontend/Supabase implementation evidence (2026-07-14):** Independent frontend commit `5fa8be1` completes the local FB.7 subset without reading from or depending on backend repository files. The implementation adds a sanitized browser diagnostics boundary, replaces routine production browser console output, minimizes Edge operational logs to allow-listed non-PII metadata, keeps exact actor/target identifiers only in protected append-only admin audit records, returns aggregate public healthcheck status, adds release artifact hashing and verification, adds a protected release-evidence workflow with exact-SHA build inputs and SBOM generation, adds strict frontend packaging excludes, and records operator rollout/rollback steps in `docs/active/fb7_release_privacy_hardening_runbook_2026-07-14.md`. Verification passed with frontend lint, all `442` JS/JSX tests, production build, production artifact verification, zero-vulnerability production dependency audit, CycloneDX production SBOM generation, pinned Deno format/lint/type-check/test gates with all `18` Deno tests passing, immutable Edge manifest verification at `fa037f173529bc3a51388c41e7a51bce1d8cfd41da59305668e0fdcf3b254cc4`, and all generated database-patch verifiers. Manual release evidence still required: run the GitHub release-evidence workflow against the final commit/environment, deploy the exact frontend artifact and Edge Functions, run staging/production smoke, retain rollback evidence, and coordinate any Supabase grant/network changes with P7.A/P7.2.

### P7.1 — Minimize public telemetry and production logging

**Finding:** M15
**Effort/risk:** M / Medium

- [x] Define an allow-listed public metadata DTO.
- [ ] Keep token/cost/model/stage telemetry in protected observability storage. The code/public-response boundary is complete; vendor IAM/storage protection remains to be attested.
- [x] Hash or minimize actor/session identifiers.
- [x] Remove raw query/answer previews from routine logs and make fixture capture opt-in, sampled, and access-controlled.
- [ ] Define vendor log retention, access, export, deletion, and incident response.

**Current state (2026-07-15):** Frontend/Supabase FB.7 minimizes browser/edge operational logging. Backend commit `4e7d0bb` adds one allow-listed public metadata projection, strips token/cost/stage/claim/session internals from caller-visible metadata, hashes private actor/session/IP identifiers, applies a default-deny trace/log sanitizer, makes raw fixture capture opt-in/sampled/local-only and default-off, and adds regression scans. Runtime vendor access, retention, export/deletion, and incident evidence remain operational attestations.

**Acceptance:**

- Automated log scans find no email, raw query, answer preview, reusable token, or unnecessary UUID.
- Operators retain correlation through non-PII request/span IDs.
- Privacy inventory covers Supabase/Railway/application logs.

### P7.2 — Attest least-privilege database identity

**Finding:** H21
**Effort/risk:** M plus ops / Medium–High

- [x] Enumerate every relation/schema/function used by typed tools, fallback SQL, vector retrieval, reflection, and readiness.
- [x] Create the dedicated runtime role with only required usage/select grants; production creation/canary was operator-confirmed on 2026-07-16.
- [x] Run the fail-closed runtime identity/read-only checks and denial probes; production execution was operator-confirmed and staging is unavailable.
- [ ] Revoke unnecessary PUBLIC grants and restrict network access.
- [x] Canary the dedicated role, rotate the production runtime URL, and verify deployed identity/read-only behavior.
- [ ] Remove any retained broad application connection secret after the rollback window; do not revoke Supabase-managed owner roles blindly.
- [x] Add non-destructive least-privilege probes and their operator attestation procedure.

**Backend implementation evidence (2026-07-15):** Commit `4e7d0bb` aligns `scripts/least_privilege_api_role.sql` with `config.STATIC_ALLOWED_TABLES`, creates no source-controlled password, enforces read-only/time/connection defaults, checks runtime identity in readiness and protected metrics, and adds rollback-safe allowed/denied probes. The manual sequence and blockers are in `p7a_backend_privacy_runtime_activation_runbook_2026-07-15.md`.

**Acceptance:**

- Expected analytics/vector queries pass.
- Writes, DDL, privilege escalation, and forbidden schema reads fail.
- Production reports the expected role and grants.
- Old credentials are revoked after the rollback window.

### P7.3 — Harden packaging and deployment artifacts

**Finding:** M18 and L3
**Effort/risk:** M / Low–Medium

- [x] Add a strict .dockerignore for environment files, VCS, caches, tests, reports, exports, and local artifacts.
- [x] Split runtime and development dependencies.
- [x] Use a non-root runtime and pinned base image/digest.
- [x] Choose Docker as the authoritative Railway deployment path; remove the competing Nixpacks build path.
- [x] Correct stale runtime/driver comments.
- [x] Add exact-SHA SBOM, dependency-audit, image-manifest/archive, checksum, and excluded-artifact evidence generation. Actual workflow execution remains manual.

**Current state (2026-07-15):** Frontend artifact verification remains independently implemented in FB.7. Backend commit `4e7d0bb` adds the local packaging and exact-SHA evidence implementation. Docker was unavailable locally and `pip-audit` is not installed in the current environment, so clean image build/inspection, the advisory result, exact artifact promotion, and rollback proof remain Unverified/manual rather than Passed.

**Acceptance:**

- Clean no-cache build succeeds.
- Runtime is non-root.
- Image inspection finds no environment file, VCS data, privacy export, report, source mirror, or dev-only test dependency.
- The tested artifact is the promoted artifact.

### P7.4 — Externalize state only when scaling is required

**Finding:** M23 continuation
**Effort/risk:** L / High

- [ ] Keep one replica while state is in memory.
- [ ] If scaling is approved, implement shared session, continuity, rate-limit, and idempotency repositories with TTL and explicit failure policy.
- [ ] Test multi-process correctness and infrastructure failure modes.

**Acceptance:**

- Two replicas share actor-bound state and limits consistently.
- Fail-open/fail-closed behavior is documented and tested.
- Scaling configuration cannot be enabled accidentally without the shared store.

**Current backend state (updated 2026-07-16):** Commit `c8bd654` hides process-local sliding-window state behind `InMemoryRateLimitRepository`, making a future shared implementation possible without moving authentication/key derivation. The F1 runtime batch now rejects configured HTTP worker counts other than one, but it does not implement multi-replica state and does not authorize scaling. Exactly one Railway replica with autoscaling disabled remains a manual production topology gate.

### P7.5 — Complete dependency and deployment assurance

- [ ] Run production dependency advisory scans under explicit authorization.
- [x] Treat unavailable scanning as Unverified, never Passed; the local missing `pip-audit` result is recorded that way.
- [ ] Verify deployed RLS policies, edge hashes, secrets, grants, and replica count.
- [ ] Record rollback commands and owners without embedding secrets.

### P7 exit gate

**Current state (2026-07-15):** P7.A's local privacy/runtime/packaging package is complete in `4e7d0bb`, and its process-local rate-limit repository extraction is in `c8bd654`; P7.B's frontend/Supabase subset is locally complete in `5fa8be1`. P7.A and the phase-wide P7 gate remain partial: live least-privilege identity/credential revocation, PUBLIC/network grants, exact artifact promotion, dependency/image evidence, one-replica topology, smoke/rollback, privacy-log canaries, vendor retention/access/export/deletion attestations, and any approved multi-replica shared-state implementation remain blockers.

- Privacy/logging inventory is complete.
- Least privilege is proven in staging and production.
- Production artifacts contain no sensitive/local material.
- Deployment parity and state topology are attested.

## 15. P8 — Structural refactoring and remaining debt

Start only after the affected behaviors have characterization and regression tests.

### P8.A — Backend application track

- Owns backend extractions in P8.1 and backend/pytest/coverage/counter debt in P8.2.
- Extract one stable deep interface per change with behavior-neutral golden, provenance, tool-call, and timing checks.

#### P8.A backend subtask status (2026-07-15)

| Subtask | Status | Implemented/fixed | Still required |
|---|---|---|---|
| P8.1 deep-module extraction | **First extraction complete; track open** | `c8bd654`: extracted process-local rate-limit storage/locking/eviction from `main.py` behind `InMemoryRateLimitRepository`; retained authentication, key derivation, limits, schemas, one-replica semantics, and golden/provenance/deadline behavior; added boundary tests. | Continue one behavior-neutral extraction at a time for remaining oversized interpretation, evidence, derivation, rendering, session/entitlement, `main.py`, `core/llm.py`, planner, pipeline, and summarizer responsibilities. |
| P8.2 L2 pytest authority | **Complete** | `245aa1b`: removed `pytest.ini`; `pyproject.toml` is the sole discovery/addopts authority. | No implementation blocker; retain the single authority. |
| P8.2 L3 runtime documentation | **Complete for current Docker path** | `4e7d0bb` and `245aa1b`: corrected runtime guidance and retained Docker as the authoritative Railway path. | Revalidate documentation whenever the base image or deployment path changes. |
| P8.2 coverage redistribution | **Partial** | `245aa1b`: global CI floor raised from 70% to 80% after measured 82.11%; focused 95% floors added for rate-limit state and query executor; final local suite `1,549` passed. | Add risk-driven tests for remaining weak/high-risk `main.py`, `core/llm.py`, `core/llm_runtime.py`, and additional pipeline branches without chasing coverage numerically. |
| P8.2 Stage 0.7 strategy decision | **Open/manual evidence** | Counter names and collection procedure documented. No strategy was deleted. | Record representative production counter deltas/traces and obtain owner approval before any removal. |
| P8.2 LIGHT retrieval overlap | **Open/manual evidence** | Existing behavior preserved and required latency/correctness measurements documented. | Collect per-tier latency, cost, relevance, grounding, provenance, and answer-quality evidence before changing retrieval. |
| P8.2 rollout-flag retirement | **Open** | Flags and required telemetry/release ledger identified. No flag was removed. | Complete two stable production releases for each candidate behavior, retain telemetry, and prove rollback before removal. |
| P8.2 A-F reassessment | **Complete** | `245aa1b`: current evidence-based backend assessment recorded; overall grade **B**. | Re-run after the remaining live P7/P8 evidence and structural work materially change the risk profile. |
| P8.A track completion | **Open** | The first extraction and the current test/debt slice are committed independently. | Finish additional incremental extractions, remaining high-risk coverage, and the three evidence-dependent cleanup decisions above. |

**Backend implementation evidence (2026-07-15):** Commit `c8bd654` performs one behavior-neutral extraction of the process-local sliding-window store behind `InMemoryRateLimitRepository`; authentication, key derivation, limits, schemas, and one-replica semantics remain unchanged. Commit `245aa1b` makes `pyproject.toml` the sole pytest authority, adds 95% focused state/database-boundary coverage gates, raises the measured global floor from 70% to 80%, and records deferred live decisions plus a current A-F assessment in `p8a_backend_assessment_and_deferred_gates_2026-07-15.md`. Final local evidence is `1,549` passing tests, `82.11%` production-code coverage, and clean Ruff/diff checks.

### P8.B — Frontend/Supabase application track

- Owns frontend/edge component-module extractions, duplicate state/toast cleanup, test-infrastructure consolidation, and the deferred Vite/esbuild major upgrade.
- The historical P3.B LF/CRLF generated-artifact verifier drift was closed in FB.7 by normalizing source fragments before generation and adding a regression contract. Remaining P8.B work should focus on behavior-neutral module extraction, test-infrastructure consolidation, and the deferred Vite/esbuild major upgrade.
- Keep refactors separate from behavior changes and verify bundle, accessibility, consumer-contract, and browser behavior per extraction.

### P8.1 — Split the god modules around stable deep interfaces

**Finding:** M25
**Effort/risk:** incremental L / Medium

Recommended boundaries:

- interpretation and contract finalization;
- evidence planning and validation;
- evidence execution/finalization;
- analytical derivation;
- render dispatch and grounding;
- DB gateway;
- model provider registry;
- session/entitlement repositories.

Rules:

- [x] No big-bang rewrite in the completed P8.A slice.
- [x] One extraction per change with behavior-neutral acceptance in `c8bd654`.
- [x] Public schemas, tool-call counts, golden outputs, provenance, and timing/deadline tests remain stable for the completed extraction.
- [x] Add boundary tests and focused coverage floors for the completed rate-limit/query-executor slice.
- [x] Keep the structural extraction separate from P7 behavior changes and the P8 test/debt commit.

These checks close only the completed slice; every future extraction must independently satisfy the same rules.

### P8.2 — Close low-level debt and internal carry-forward items

- [x] L2: choose `pyproject.toml` as the single pytest configuration authority and remove `pytest.ini`.
- [x] L3: retain corrected Docker/runtime documentation and authoritative Docker deployment guidance.
- [x] Raise the global coverage floor from 70% to 80% and enforce 95% focused coverage for the extracted rate-limit and query-executor boundaries.
- [ ] Continue redistributing coverage into remaining weak/high-risk areas: `main.py`, `core/llm.py`, `core/llm_runtime.py`, and additional pipeline orchestration branches.
- [ ] Take and document the Stage 0.7 production-counter reading before deleting strategies.
- [ ] Reassess LIGHT-tier retrieval overlap only after latency evidence and correctness gates.
- [ ] Remove temporary rollout flags after two stable releases and archive their telemetry.
- [x] Re-run the A-F assessment using current evidence rather than historical test counts; current overall backend grade is B in `p8a_backend_assessment_and_deferred_gates_2026-07-15.md`.

**Remaining P8.A blockers (2026-07-15):** Stage 0.7 production counters were unavailable; LIGHT-tier production latency/cost/correctness evidence was unavailable; and two stable releases have not been attested for temporary flags. No strategy, retrieval behavior, or flag was removed. Further god-module extraction remains intentionally incremental.

## 16. Finding coverage matrix

### High findings

| ID | Short description | Planned phase | Notes |
|---|---|---|---|
| H1 | Happy path bypasses canonical evidence | P4 | Blocked by H2 and M2 |
| H2 | Incorrect canonical units/filter units | P2 | Fix before H1; current exposure mainly recovery |
| H3 | Order-dependent/mixed statistics | P2 | Live analytical correctness |
| H4 | Frontend null prices counted as zero | P0 | Quick live correctness fix |
| H5 | Chat quota/persistence bypass | P3 | Highest real authorization/cost risk |
| H6 | Dashboard quota bypass | P3 | Enforce at data boundary |
| H7 | Direct bearer bypass | P0 containment, P3 permanent | Inventory callers first |
| H8 | Paused admins retain authority | P0 containment, P3 centralization | Enumerate every endpoint |
| H9 | All admins can be removed | P3 | Transactional invariant |
| H10 | Persisted history treated as trusted | P3 | Transport trust is not content trust |
| H11 | Legacy fallback bypasses provenance | P0 | Existing claim derivation makes this low effort |
| H12 | Data failure becomes conceptual | P4 | Planned as Medium per feedback |
| H13 | Retry amplification/no total deadline | P5 | Uses P3 idempotency |
| H14 | DB breaker broken both ways | P5 | One DB gateway |
| H15 | Pool oversubscription/orphan work | P5 | Includes internal pool/backpressure item |
| H16 | Unbounded body before auth | P0 | Proxy/ASGI plus typed limits |
| H17 | Auth listener race/deadlock | P1 | Port reviewed fix into canonical tree |
| H18 | Divergent frontend trees | P1 | Prerequisite for frontend work |
| H19 | Deletion purges append-only audit | P0 | Code should follow privacy contract |
| H20 | Raw privacy exports in repo | P0 | Privacy-led handling |
| H21 | Least-privilege deployment unverified | P7 | Discovery starts P0 |

### Medium findings

| ID | Short description | Planned phase |
|---|---|---|
| M1 | Charts use raw ctx.df | P4 |
| M2 | Frame provenance refs empty | P2 |
| M3 | Plan validation warning-only | P4 |
| M4 | Re-analysis incomplete | P4 |
| M5 | No shared API contract; charts/tier/provenance drift | P3 and P6 |
| M6 | Actor/session/request identity lost | P3 |
| M7 | JSONB double encoding and turn ordering | P3 and P6 |
| M8 | Supabase errors ignored/raw errors exposed | P3 and P6 |
| M9 | Dashboard churn/mixed snapshots | P6 |
| M10 | Correlation ignores requested period | P2 |
| M11 | Provider validation/breaker inconsistency | P5 |
| M12 | Forecast horizon parsing changes intent | P2 |
| M13 | Embedding cache not model-aware | P2 |
| M14 | Manual/unpinned edge deployment | P1 |
| M15 | Internal telemetry/content overexposed | P7 |
| M16 | Failed admin audits roll back | P3 |
| M17 | Readiness/startup over-report success | P0 |
| M18 | Unsafe container context/runtime packaging | P7 |
| M19 | Missing error boundaries/config diagnostics | P6 |
| M20 | Accessibility gaps | P6 |
| M21 | Unbounded admin listing | P6 |
| M22 | Metrics not thread-safe | P5 |
| M23 | Session ownership and scaling fragility | P5 and P7 |
| M24 | Frontend lint/test/release gate not clean | P0 |
| M25 | Oversized mixed-responsibility modules | P8 |

### Low findings

| ID | Short description | Planned phase |
|---|---|---|
| L1 | Secondary evidence observability incomplete | P5 |
| L2 | Duplicate pytest configuration | P8 |
| L3 | Stale Docker/runtime comments | P7 |
| L4 | Duplicate toast stores/invalid DOM prop | P0 warning cleanup and P6 consolidation |

## 17. Test and verification policy

Every behavior fix begins with a regression test that fails for the verified mechanism, not merely the reported symptom.

### Backend gates

- Ruff clean.
- Full backend suite green.
- Security/adversarial suite and scored red-team gate green for auth/history/provenance changes.
- Routing golden set green for analyzer/planner changes.
- Dimensional and order-invariance tests for analytics.
- DB integration tests for migrations, RLS, entitlement, leases, grants, and concurrency.
- Load/deadline/cancellation tests for P5.
- Coverage may increase; it must never be lowered to make a phase pass.

### Frontend gates

- Clean npm install.
- Lint with zero errors.
- Unit/component tests with no unresolved React warnings.
- Mandatory DB regression tier on release branches.
- Build and production smoke checks.
- Contract tests against generated schemas.
- Browser tests for loading, empty, malformed, timeout, abort, unauthorized, paused, quota-exhausted, and partial-data states.
- Accessibility automation plus keyboard/screen-reader review.

### Deployment and security gates

- Edge typecheck/unit/integration tests.
- Immutable deployed-source hash.
- Migration dry run, backup, reconciliation, and idempotent re-run.
- Least-privilege denial probes.
- Container/image inspection.
- Dependency advisory scan under explicit approval.
- Production canary metrics and documented rollback owner.

## 18. Rollout and rollback discipline

- Behavioral migrations use additive schema and versioned v2 functions first.
- Shadow decisions must compare structured fields/hashes, not retain raw user content.
- Material pipeline changes use off, shadow, and enforce modes with a scheduled flag-removal date.
- Do not deploy H1 and H2 in reverse order.
- Do not cut the frontend off the legacy persistence RPC until edge/server persistence is proven.
- Do not revoke old grants until upgraded consumers are observed and reconciliation is clean.
- Do not drop additive entitlement/audit data during rollback.
- Security rollback fails closed or disables the affected feature.
- Every canary defines success thresholds, abort thresholds, and a named decision owner before traffic begins.

## 19. Documentation changes

Update documentation in the same phase as deployed behavior:

- Add an immediate erratum to the pipeline architecture stating that canonical frames, validation, chart sourcing, and frame provenance are not universal today.
- Publish the authoritative storage/canonical/display unit table and filter semantics in P2.
- Document evidence invalidation/finalization and canonical chart sourcing in P4.
- Describe plan validation as warning-only until enforcement is live.
- State that legacy narrative fallback is subject to the same provenance gate.
- Define entitlement reservation, charge/release, idempotency, dashboard leases, and reconciliation in P3.
- Publish the versioned API contract, actor/session/request identity model, and public metadata allow-list.
- Document deadlines, retry ownership, DB error classification, pool/work budgets, and cancellation in P5.
- Keep re-analysis experimental/default-off until P4 tests pass.
- Preserve the one-replica requirement until P7 shared-state acceptance passes.
- Keep the privacy document's append-only audit rule; change code that contradicts it.
- Update the module map only after each P8 extraction is deployed.

## 20. Final completion gate

The remediation program is complete only when:

- all normal evidence paths produce correctly converted, validated, provenance-bound frames;
- deterministic answers and charts use identical canonical evidence;
- statistics are order-, period-, and unit-correct;
- fallback numeric claims cannot bypass grounding;
- browsers cannot bypass chat/dashboard quotas or authoritatively write billing state;
- direct bearer and admin paths enforce active status and admin invariants;
- audit history survives deletion and privacy exports/logs follow approved retention;
- total request time, retries, DB work, provider breakers, and sessions are bounded;
- frontend source, edge sources, API schemas, and production artifacts are reproducible;
- accessibility and large-data UI gates pass;
- least privilege and deployed RLS/grants are attested;
- every finding in the coverage matrix has a regression test or production attestation;
- temporary compatibility paths and rollout flags are removed after their stated windows; and
- a fresh independent audit finds no unresolved Critical or High item without an approved, time-bound release waiver.

## 21. First execution package

The recommended first package, in this exact order, is:

1. create the owner/evidence ledger and lock required CI checks;
2. fix the red frontend lint gate and warning policy;
3. quarantine repository-local privacy exports;
4. stop audit-log deletion;
5. require active admin status everywhere;
6. decide and contain direct-bearer access;
7. close the legacy fallback provenance bypass;
8. add request-body/history bounds;
9. correct frontend null-price aggregation;
10. make readiness/startup fail honestly;
11. reconcile the frontend source tree and port the guarded auth behavior; and
12. convert edge snapshots into reproducible sources before beginning the entitlement cutover.

This package maximizes immediate risk reduction while avoiding the higher-risk H1 and quota-schema cutovers until their prerequisites and rollback mechanisms exist.
