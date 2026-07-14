# Comprehensive Audit Remediation Plan

**Date:** 2026-07-11
**Status:** P0, P1, P2, and P3 implementation tracks are independently committed. P3.A backend commit `5883228` and P3.B frontend commit `59b4d64` are deployed; the frontend formatting/manifest repair `3244ed1` is merged in frontend `main` at `5ae2b9b`. The operator confirmed completion of the P3.A/P3.B staging, production, smoke, soak, reconciliation, assertion-enforcement, and legacy-grant-revocation steps on 2026-07-13. P3 is complete under the supported gateway-only, one-backend-replica operating mode. Frontend-first package FB.1 (P6.3 and P6.7) is locally complete in independent frontend commit `b69831a`; deployment/browser smoke is not yet recorded here. FB.2 (P6.2) is locally complete in independent frontend commit `da652da`; its Supabase patch, bounded data migration, edge/frontend deployment, constraint validation, and browser smoke remain manual activation evidence. Direct bearer remains intentionally disabled and horizontal scaling remains intentionally blocked until their later P5/P7 gates. H1 remains off until P4. Non-P3 production attestations that were not explicitly confirmed remain governed by their existing ledgers. See [p0_execution_ledger_2026-07-11.md](p0_execution_ledger_2026-07-11.md), [p0_manual_activation_and_followup_2026-07-12.md](p0_manual_activation_and_followup_2026-07-12.md), [p1_execution_ledger_2026-07-12.md](p1_execution_ledger_2026-07-12.md), and [p2_execution_ledger_2026-07-13.md](p2_execution_ledger_2026-07-13.md).
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
| FB.3 | P6.4 | Make dashboard loading single-flight, atomic, freshness-aware, and stale-response safe while preserving P3 leases. | Frontend/Supabase only; do not change lease or billing semantics. |
| FB.4 | P6.5 | Complete WCAG 2.2 AA automation and manual keyboard/screen-reader/zoom review. | Frontend only. |
| FB.5 | P6.6 | Add bounded cursor-based admin listing, server-side search/filtering, cancellation, and large-user tests in Supabase edge/database plus UI. | Frontend/Supabase only; no analytics-backend change is required. |
| FB.6 | P5.B independent subset | Remove duplicate browser retries; add bounded browser/edge aborts, safe retry classification, `Retry-After` handling, and same-request-ID reuse. | End-to-end absolute deadline propagation and final retry ownership cannot close until P5.A publishes/implements the compatible backend deadline contract. |
| FB.7 | P7.B independent subset | Minimize browser/edge logs, harden frontend/edge packaging, scan the production bundle/dependencies, and preserve deploy/rollback evidence. | Supabase least-privilege changes must be checked against the backend relation/function inventory before grants or network access used by the backend are changed. P7 phase-wide attestation still needs P7.A. |
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

- [ ] Create one ticket for every H, M, and L identifier in the coverage matrix below.
- [ ] Record current flags, replica/worker count, database pool settings, deployed edge hashes, authentication mode, and direct-bearer traffic.
- [ ] Keep canonical evidence enforcement and evidence re-analysis off.
- [ ] Capture clean backend and frontend test/lint/build baselines from pinned runtimes.
- [ ] Record which production checks cannot be verified locally: deployed RLS/grants, edge parity, provider timeouts, and live least-privilege identity.
- [ ] Verify previously recorded internal fixes rather than reopening them without evidence.

**Done when:** every finding has a named owner, evidence link, acceptance test, rollback owner, and state of Open, In progress, Verified, or Deferred with approved reason.

### P0.2 — Restore truthful release gates

**Findings:** M24 and the internal branch-protection recommendation
**Effort/risk:** S / Low

- [ ] Fix the four current frontend lint errors.
- [ ] Eliminate unresolved React act warnings and the invalid dismiss DOM property warning.
- [ ] Configure tests to fail on unexpected React errors, unhandled rejections, and console errors.
- [ ] Require database regression tests for release/protected branches; missing TEST_DATABASE_URL must fail a release workflow.
- [ ] Turn on branch protection and make backend/frontend required checks non-bypassable for normal merges.
- [ ] Record exact Node, npm, Python, and dependency-lock inputs used by CI.

**Acceptance:**

- npm ci, lint, tests, build, smoke checks, backend Ruff, security gates, and full backend tests pass from clean checkouts.
- No required release job silently skips DB or edge verification.
- No behavior change is introduced by the lint-only portion.

### P0.3 — Quarantine privacy exports and stop repository-local export output

**Finding:** H20
**Effort/risk:** S / Low, with Privacy approval

- [ ] Inventory current export artifacts, repository history, shared archives, CI artifacts, and backups without copying payloads into new locations.
- [ ] Move retained exports into encrypted, access-controlled incident/fulfilment storage.
- [ ] Do not delete existing exports until the Privacy owner decides whether incident evidence or fulfilment records must be preserved.
- [ ] Make future exports write outside the repository with restrictive ACLs, encryption, expiry, and an auditable deletion step.
- [ ] Ignore privacy export paths, temporary reports, and real environment files; retain only reviewed examples.
- [ ] Rotate credentials only if investigation finds actual secret exposure.

**Acceptance:**

- Normal repository, CI artifacts, and project backups contain no live privacy export payload.
- A test export is created outside the repository, is access-restricted, and expires under policy.
- Secret/data scanning rejects export payloads and real environment files.

**Rollback:** disable exports if secure storage fails. Never return to repository-local raw export storage.

### P0.4 — Preserve append-only audit history

**Finding:** H19
**Effort/risk:** S–M / Low–Medium

- [ ] Remove the deletion path that disables the audit-protection trigger or deletes subject/actor audit rows.
- [ ] Decide whether deleted users remain as UUIDs or become stable pseudonymous identifiers.
- [ ] Ensure foreign keys do not force audit deletion.
- [ ] Align the deletion response and privacy runbook with retained audit behavior.

**Acceptance:**

- User deletion succeeds while related audit events remain.
- Update/delete attempts against audit records remain denied.
- Trigger state remains enabled after successful and failed deletion.
- Code, database behavior, and privacy documentation agree.

### P0.5 — Require active administrators immediately

**Finding:** H8
**Effort/risk:** S / Low–Medium

- [ ] Add active-status verification to every current admin edge function and is-admin.
- [ ] Enumerate every administrative endpoint in one regression test.
- [ ] Prepare a service-role break-glass procedure before deployment so a configuration mistake cannot permanently lock out operators.
- [ ] Schedule centralization under P3 after reproducible edge sources exist.

**Acceptance:**

- A paused or demoted admin with an otherwise valid token receives 403 from every administrative endpoint.
- Replaying an old token does not restore authority.
- Active admins retain expected access.

### P0.6 — Contain direct-bearer bypass

**Finding:** H7
**Effort/risk:** S configuration plus traffic analysis / Medium

- [ ] Inventory legitimate direct /ask bearer clients from trusted access logs.
- [ ] If none exist, explicitly set gateway-only mode and restrict backend network exposure where possible.
- [ ] If direct bearer is required, do not leave it exempt: schedule it for the same active-status and transactional entitlement service as the edge path in P3.
- [ ] Document the decision; do not rely on auto mode to choose the security boundary.

**Acceptance:**

- Paused or exhausted users cannot reach unmetered /ask through a direct token.
- Any gateway assertion rejects tampering, expiry, actor mismatch, and unauthorized replay.

**Rollback:** disable chat or restore the last trusted gateway artifact. Do not reopen unmetered direct bearer access.

### P0.7 — Close the low-effort provenance bypass

**Finding:** H11
**Effort/risk:** S / Low–Medium

- [ ] Apply the existing claim-derivation mechanism to legacy fallback text.
- [ ] Run fallback claims through the normal provenance gate.
- [ ] Return an explicit degraded-evidence result when numeric claims cannot be grounded.
- [ ] Observe rejection/disagreement in shadow mode briefly before enforcement if production fallback volume is material.

**Acceptance:**

- An invented numeric fallback cannot pass as no_claims.
- Grounded numeric fallback passes.
- Nonnumeric safe fallback remains available.

### P0.8 — Bound request bodies before expensive parsing

**Finding:** H16, with H10 history trust completed in P3
**Effort/risk:** S–M / Low

- [ ] Add proxy/ASGI and edge request-size limits.
- [ ] Preserve the live Q&A history contract with a typed question/answer turn model, forbid extra fields, allow at most three turns, and enforce per-field limits. The edge alone maps stored role/content rows into this contract.
- [ ] Return stable 413/validation responses.
- [ ] Add memory/latency tests for oversized and malformed bodies.

**Acceptance:**

- Oversized bodies are rejected before model/Pydantic work and without material memory growth.
- Oversized fields and turn counts produce stable validation errors.

### P0.9 — Correct the live frontend null-average bug

**Finding:** H4
**Effort/risk:** S / Low

- [ ] Define missing-value semantics: finite zero is valid; null, undefined, NaN, and infinity are missing.
- [ ] Apply the same denominator rule to price and tariff aggregation.
- [ ] Define whether each published metric is a simple or weighted average.

**Acceptance:**

- [10, null, 20] produces 15.
- [0, null] produces zero with sample count one.
- All-missing input produces an explicit no-data result.
- Input order does not change the result.

### P0.10 — Make readiness and startup honest

**Finding:** M17
**Effort/risk:** S / Low

- [ ] Return non-200 when any required readiness dependency, including schema reflection, is unavailable.
- [ ] Distinguish optional/degraded components explicitly.
- [ ] Let fatal startup exceptions terminate with a nonzero exit code.

**Acceptance:**

- Readiness returns 503 for missing required DB/schema state.
- Startup failure exits nonzero and triggers normal platform restart/failure detection.

### P0 exit gate

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

**Current state (2026-07-13):** P2.A passes all backend gates (1,385 tests). P2.B passes lint, production build, the internal contract-integrity gate, and all 342 frontend tests. The local P2 implementation gate is closed; production smoke/shadow evidence remains a manual deployment attestation. H1 remains off until the P4 entry review.

- Dimensional, statistical, filter, period-scope, and provenance golden tests pass.
- Shadow comparisons explain every intentional numeric change.
- H1 remains off until the P4 entry review explicitly confirms P2.

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

- Consume and persist the versioned canonical result, terminal-outcome, chart, filter, unit, and provenance fields without reconstructing evidence from raw rows.
- Render degraded/clarification/policy/transient outcomes distinctly and add consumer contract tests for multiple charts and provenance.
- Frontend code may commit before enforcement, but UI behavior activates only when P4.A advertises a compatible contract/mode.

### P4.1 — Introduce one evidence finalization routine

**Finding:** H1
**Effort/risk:** M–L / Medium–High

- [ ] Call one routine after normal primary execution, recovery execution, secondary merges, and any enrichment that changes evidence.
- [ ] Invalidate stale frames before mutation.
- [ ] Build the correct frame, bind provenance, and run evidence validation.
- [ ] Return a typed finalization/gap result.
- [ ] Add off, shadow, and enforce modes for the rollout.

**Acceptance:**

- Every normal and recovery tool path produces the expected frame type.
- Frames rebuild after merges and never retain stale filters/rows.
- SCALAR, LIST, TIMESERIES, COMPARISON, FORECAST, and SCENARIO fixtures run validation.
- Deterministic shapes use the generic renderer with zero narrative LLM calls in enforce mode.
- Equivalent happy/recovery evidence produces equivalent canonical frames.

**Rollout:** off → shadow → 5 percent enforce → 25 percent → 100 percent. Roll back to shadow without discarding comparison telemetry.

### P4.2 — Make plan validation enforceable

**Finding:** M3
**Effort/risk:** M / Medium

- [ ] Replace warning-only checks with a typed PlanValidationResult.
- [ ] Validate answer shape, tool capability, evidence roles, units, entities, and periods before any external call.
- [ ] Repair once, clarify, or reject; never loop indefinitely.

**Acceptance:**

- Invalid plans make zero DB/tool calls.
- Repaired plans are revalidated exactly once.
- Documentation says warning-only until enforcement is actually deployed.

### P4.3 — Source charts from answer evidence

**Finding:** M1
**Effort/risk:** M / Medium

- [ ] Prefer derived specifications or finalized canonical frames.
- [ ] Permit raw ctx.df only through an explicit, measured fallback.
- [ ] Attach evidence/filter/unit/provenance identity to each chart.

**Acceptance:**

- Answer and chart have identical period, entity set, filters, unit, row identity, and provenance hash.
- Raw fallback is observable and cannot silently override canonical filtering.

### P4.4 — Introduce honest terminal outcomes

**Finding:** H12, planned as Medium
**Effort/risk:** M / Medium

- [ ] Model conceptual answer, evidence unavailable, clarification required, policy blocked, and transient service failure separately.
- [ ] Preserve the anti-retry-storm behavior without pretending unavailable data is conceptual evidence.
- [ ] Prohibit numeric claims on evidence-unavailable outcomes.

**Acceptance:**

- Invalid/irrelevant SQL on a data-primary request produces a transparent degraded outcome, not a plausible conceptual substitution.
- Permanent failures are not retried as transient failures.

### P4.5 — Complete re-analysis semantics

**Finding:** M4
**Effort/risk:** L / Medium

- [ ] Restart from an explicit stage checkpoint with a fresh dependent context.
- [ ] Recompute response mode, clarification/knowledge short-circuits, retrieval tier, derived metrics, evidence plan, frames, and rendering.
- [ ] Preserve only immutable request/auth state.
- [ ] Keep re-analysis disabled until mode-transition tests pass.

**Acceptance:**

- Data-to-knowledge, data-to-clarification, and changed-answer-kind cases rerun every dependent stage.
- No stale vector, metric, frame, or render state survives.
- Re-analysis runs at most once and records its reason.

### P4 exit gate

- Universal frame shadow metrics are understood and enforcement is stable.
- Deterministic answers and their charts use the same canonical evidence.
- Invalid plans do not execute.
- Data failures are honest.
- Architecture documentation describes the deployed mode, not the target mode.

## 12. P5 — Reliability, cost, concurrency, providers, metrics, and sessions

### P5.A — Backend application track

- Owns P5.2 through P5.6 and the backend deadline/retry/idempotency work in P5.1.
- Enforce one absolute request deadline, one retry owner, bounded global DB work, independent provider breakers, thread-safe metrics, actor-bound sessions, and one-replica mode until shared state is proven.
- Load/failure tests and the scaling assertion form this repository's independent completion gate.

### P5.B — Frontend/Supabase application track

- Owns browser/edge deadline propagation, abort behavior, retry classification, `Retry-After`, request-ID reuse, and removal of duplicate client retries under P5.1.
- Edge/database calls must consume the P3 idempotency identity and return safe retryable classifications; they must not mint a new billing identity on retry.
- Keep frontend/edge retries conservative until P5.A's deadline and idempotency contracts are deployed.

### P5.1 — Establish one deadline, retry owner, and idempotency identity

**Finding:** H13
**Effort/risk:** L / Medium–High

- [ ] Define one absolute monotonic deadline at the trusted request boundary.
- [ ] Propagate remaining time to browser/edge/backend/DB/model calls.
- [ ] Assign retry ownership to one layer.
- [ ] Add explicit provider and edge timeouts, abort signals, jitter, Retry-After handling, and status classification.
- [ ] Make all permitted retries reuse the same actor-bound request ID.

**Acceptance:**

- A request cannot produce duplicate model calls/charges outside the idempotency policy.
- Every request terminates within the declared budget plus a small tolerance.
- Permanent 4xx/validation/quota failures are not retried.
- Ambiguous delivery is reconciled rather than blindly reissued.

### P5.2 — Route all database work through one gateway

**Finding:** H14
**Effort/risk:** M–L / Medium

- [ ] Route typed tools, vector queries, fallback SQL, and availability probes through one DB gateway.
- [ ] Define transient infrastructure versus syntax/schema/content/validation failures.
- [ ] Allow only transient failures to affect the circuit breaker.
- [ ] Protect all DB-backed paths when the breaker is open.

**Acceptance:**

- Invalid SQL does not open the DB breaker.
- Repeated connection/timeout failures do.
- An open breaker prevents all DB-backed tool paths from reaching the database.

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

- [ ] Create one provider registry for credentials, model, timeout, retry, cost, and independent breaker.
- [ ] Validate every selected provider at startup.
- [ ] Ensure NVIDIA and OpenAI do not share state.

**Acceptance:**

- OpenAI, Gemini, and NVIDIA breakers open/reset independently.
- Startup fails when a selected provider lacks required credentials.

### P5.5 — Make metrics thread-safe and secondary work observable

**Findings:** M22, L1
**Effort/risk:** S–M / Low

- [ ] Use a thread-safe collector/backend or lock updates and snapshots.
- [ ] Emit per-source Stage 0.8 metrics with a primary/secondary dimension.
- [ ] Store hashes/counters rather than raw query/answer content.

**Acceptance:**

- Multithreaded tests record exact expected totals and consistent snapshots.
- Secondary calls/latency/failures appear without double-counting.

### P5.6 — Bind sessions to actors and enforce the scaling constraint

**Finding:** M23
**Effort/risk:** M containment; L externalization / High

- [ ] Bind session token, conversation, history, and contract snapshot to subject/auth mode.
- [ ] Serialize turns per session.
- [ ] Prevent expired tokens from recreating empty sessions.
- [ ] Add a startup/deployment assertion for unsupported multi-worker/replica configuration.
- [ ] Introduce a repository abstraction and external shared state before scaling.

**Acceptance:**

- Cross-user token replay fails.
- Concurrent turns have deterministic order.
- Expired sessions cannot be resurrected.
- In-memory mode refuses unsupported scaling.
- Two-process tests share state when the external backend is enabled.

### P5 exit gate

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

- [ ] Make backend OpenAPI/JSON Schema the versioned contract source.
- [ ] Generate edge/frontend DTOs and validate requests/responses at runtime.
- [ ] Include complete charts, answer provenance/trust state, request ID, and approved session fields.
- [ ] Remove or formally implement service tier as a server-derived field.
- [ ] Keep internal telemetry out of the public DTO.

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

- [ ] Stabilize callback dependencies and use a single-flight state machine.
- [ ] Prevent usage updates from triggering duplicate loads.
- [ ] Commit critical datasets atomically.
- [ ] Mark optional partial data with freshness/degraded state.
- [ ] Prevent stale or canceled requests from overwriting newer state.

**Acceptance:**

- One action creates one entitlement check and one load.
- Older requests cannot overwrite newer results.
- Partial critical failure preserves the previous coherent snapshot.
- Optional degradation is visible.

### P6.5 — Reach an accessibility baseline

**Finding:** M20
**Target:** WCAG 2.2 AA for affected routes
**Effort/risk:** M–L / Low–Medium

- [ ] Name every icon-only action.
- [ ] Expose toggle state semantically.
- [ ] Provide chart summaries and accessible data tables/equivalents.
- [ ] Implement fullscreen charts as real dialogs with focus trapping/restoration and Escape.
- [ ] Verify focus, headings, live status/errors, contrast, touch size, zoom, and responsive behavior.
- [ ] Add automated scans plus keyboard/screen-reader review.

**Acceptance:**

- No serious/critical automated accessibility violations on login, dashboards, chat, and admin.
- Keyboard users can operate every affected control.
- Canvas charts have value/unit/period equivalents.
- Dialog focus behavior is correct at mobile, tablet, desktop, and 200 percent zoom.

### P6.6 — Bound the admin list

**Finding:** M21
**Effort/risk:** M–L / Medium

- [ ] Use cursor pagination, hard page limits, server filtering/search, and minimal projections.
- [ ] Return role/status required for safe operations.
- [ ] Compute global aggregates separately.
- [ ] Add UI cancellation/stale-response protection and virtualization only if measured.

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

- The frontend consumes the complete validated API.
- Auth/dashboard/chat state is deterministic.
- Errors are safe and recoverable.
- Accessibility and large-admin-list tests pass.
- Browser smoke/e2e checks produce no unexpected console errors or warnings.

## 14. P7 — Privacy, deployment, least privilege, packaging, and scaling hardening

### P7.A — Backend application track

- Owns backend logging/telemetry minimization in P7.1, runtime database identity in P7.2, backend container/dependency artifacts in P7.3, backend shared-state/scaling work in P7.4, and its P7.5 attestations.
- Prove the runtime database role, one-replica topology, non-root/pinned artifact, secret rotation, denial probes, and deploy/rollback identity.

### P7.B — Frontend/Supabase application track

- Owns browser/edge/Supabase logging minimization in P7.1, Supabase grants/RLS and network controls in P7.2, frontend/edge packaging and immutable deployment in P7.3, any frontend shared-state integration in P7.4, and its P7.5 attestations.
- Prove deployed edge hashes, RLS/grants, privacy retention/export operations, browser bundle contents, production dependency scans, and frontend/edge rollback artifacts.
- Each repository can complete independently; the phase exit gate requires both production attestations.

### P7.1 — Minimize public telemetry and production logging

**Finding:** M15
**Effort/risk:** M / Medium

- [ ] Define an allow-listed public metadata DTO.
- [ ] Keep token/cost/model/stage telemetry in protected observability storage.
- [ ] Hash or minimize actor/session identifiers.
- [ ] Remove raw query/answer previews from routine logs and make fixture capture opt-in, sampled, and access-controlled.
- [ ] Define vendor log retention, access, export, deletion, and incident response.

**Acceptance:**

- Automated log scans find no email, raw query, answer preview, reusable token, or unnecessary UUID.
- Operators retain correlation through non-PII request/span IDs.
- Privacy inventory covers Supabase/Railway/application logs.

### P7.2 — Attest least-privilege database identity

**Finding:** H21
**Effort/risk:** M plus ops / Medium–High

- [ ] Enumerate every relation/schema/function used by typed tools, fallback SQL, vector retrieval, reflection, and readiness.
- [ ] Create a dedicated runtime role with only required usage, select, and execute grants.
- [ ] Deny writes, DDL, role changes, and unrelated/auth/storage schema access.
- [ ] Revoke unnecessary PUBLIC grants and restrict network access.
- [ ] Canary, rotate the production connection secret, verify deployed identity, then revoke the old broad credential.
- [ ] Add least-privilege probes to release attestation.

**Acceptance:**

- Expected analytics/vector queries pass.
- Writes, DDL, privilege escalation, and forbidden schema reads fail.
- Production reports the expected role and grants.
- Old credentials are revoked after the rollback window.

### P7.3 — Harden packaging and deployment artifacts

**Finding:** M18 and L3
**Effort/risk:** M / Low–Medium

- [ ] Add a strict .dockerignore for environment files, VCS, caches, tests, reports, exports, and local artifacts.
- [ ] Split runtime and development dependencies.
- [ ] Use a non-root runtime and pinned base image/digest.
- [ ] Choose the authoritative deployment path: Docker or Railway/Nixpacks. Test parity only if both remain supported.
- [ ] Correct stale Pydantic/driver comments.
- [ ] Generate an SBOM/image manifest where policy allows.

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

### P7.5 — Complete dependency and deployment assurance

- [ ] Run production dependency advisory scans under explicit authorization.
- [ ] Treat unavailable scanning as Unverified, never Passed.
- [ ] Verify deployed RLS policies, edge hashes, secrets, grants, and replica count.
- [ ] Record rollback commands and owners without embedding secrets.

### P7 exit gate

- Privacy/logging inventory is complete.
- Least privilege is proven in staging and production.
- Production artifacts contain no sensitive/local material.
- Deployment parity and state topology are attested.

## 15. P8 — Structural refactoring and remaining debt

Start only after the affected behaviors have characterization and regression tests.

### P8.A — Backend application track

- Owns backend extractions in P8.1 and backend/pytest/coverage/counter debt in P8.2.
- Extract one stable deep interface per change with behavior-neutral golden, provenance, tool-call, and timing checks.

### P8.B — Frontend/Supabase application track

- Owns frontend/edge component-module extractions, duplicate state/toast cleanup, test-infrastructure consolidation, and the deferred Vite/esbuild major upgrade.
- Normalize historical generated-artifact verification across LF/CRLF checkouts; the P3.B verifier currently false-fails on Windows even when Git reports no SQL content drift. The new P6.B verifier already contains this portability guard.
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

- [ ] No big-bang rewrite.
- [ ] One extraction per change with behavior-neutral acceptance.
- [ ] Public schemas, tool-call counts, golden outputs, provenance, and timing budgets remain stable.
- [ ] Add import/boundary tests and focused coverage floors.
- [ ] Never mix structural refactoring with behavior changes in the same change.

### P8.2 — Close low-level debt and internal carry-forward items

- [ ] L2: choose one pytest configuration authority.
- [ ] L3: retain corrected Docker/runtime documentation.
- [ ] Raise coverage where risk is highest: main.py, core/llm.py, llm_runtime.py, query executor, and pipeline orchestration.
- [ ] Take and document the Stage 0.7 production-counter reading before deleting strategies.
- [ ] Reassess LIGHT-tier retrieval overlap only after latency evidence and correctness gates.
- [ ] Remove temporary rollout flags after two stable releases and archive their telemetry.
- [ ] Re-run the A–F assessment using current evidence rather than historical test counts.

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
