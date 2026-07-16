# P0 Execution and Finding Ledger

**Date:** 2026-07-11
**Status:** P0 code is locally verified; production startup/readiness and the dedicated read-only runtime identity are operator-confirmed, while the remaining live gates stay explicit below
**Plan:** [comprehensive_audit_remediation_plan_2026-07-11.md](comprehensive_audit_remediation_plan_2026-07-11.md)
**Manual activation:** [p0_manual_activation_and_followup_2026-07-12.md](p0_manual_activation_and_followup_2026-07-12.md)

Repository commits:

- backend `refactor/review-phase-fixes`: `e667f6b`;
- frontend `fix/p0-remediation`: `3a8ceda` in `D:/Enaiapp/p0_frontend_commit_repo`, with `origin` set to `https://github.com/IrakliGLD/EnaiDashboard.git`.

This is the repository-local issue ledger because no external ticket system was placed in scope. Each ID below is one tracked item. Accountable and rollback roles come from the plan; individual assignee names still require the project owner's assignment, so P0.1 is not closed.

For P0 work, `Verified locally` means source plus automated local acceptance passed. It does not claim that a database migration, edge snapshot, Railway service, GitHub protection rule, or production environment was deployed or inspected. Backend and frontend/Supabase tracks are committed and released independently; an unavailable live check is recorded as manual follow-up and does not hold the other repository.

## P0 execution state

| Work item | State | Evidence | Remaining gate / rollback owner |
|---|---|---|---|
| P0.1 ledger and baseline | Locally complete; manual ownership pending | This ledger; source/test baselines, runtime versions, lock hashes, independent repository rule, and activation runbook | Assign individual owners/reviewers and milestones; capture production flags, replicas, access logs, edge hashes, RLS/grants, and least-privilege identity. These are manual attestations. QA/release lead owns rollback. |
| P0.2 truthful release gates (M24) | Verified locally and committed | The active export tree passes clean `npm ci`, lint, 228 fail-on-console tests, build, and production audit. Frontend commit `3a8ceda` passes lint, all 318 tests, build, and the production audit using the identical clean-installed lock graph. CI cannot skip DB tests; the negative console fixture exits nonzero. | Push/review the frontend branch, obtain pinned Node 20/Python 3.11 CI results, then run DB, post-deploy/browser smoke jobs and verify branch protection. These do not hold the backend commit. QA/release lead. |
| P0.3 privacy exports (H20) | Code verified locally; manual disposition pending | Active code uses AES-256-GCM with authenticated metadata, pre-RPC real-directory/symlink checks, external storage, random filenames, bounded expiry, and authenticated purge | Two pre-existing raw artifacts remain in `D:/export_enai/privacy_exports`; Privacy owner must decide approved storage/key/retention/evidence disposition before anyone moves or deletes them. Never roll back to raw repo-local output. |
| P0.4 append-only audit (H19) | Code verified locally; manual DB verification pending | Active baseline/patch preserve subject/actor audit rows; static contract and success/failure/trigger-state DB regression are installed | Run against `TEST_DATABASE_URL`, review the stable-UUID decision, deploy the migration, and verify it using the activation runbook. Database/Supabase lead. |
| P0.5 active admins (H8) | Code verified locally; manual edge verification pending | Six edge snapshots require `role=admin` and `status=active`; endpoint inventory contract passes | Deploy the recorded snapshots, verify hashes and paused-admin 403s, and complete P1.3 deployable edge sources. Database/Supabase lead. |
| P0.6 direct bearer containment (H7) | Verified locally | `auto` rejected; default is gateway-only; bearer mode is test-only until P3; config/API tests pass | Confirm deployed env/network exposure and inventory access logs. Backend/API lead; rollback must remain gateway-only. |
| P0.7 fallback provenance (H11) | Verified locally | Full legacy text is gated; single-digit, leading-decimal, scientific, post-line-eight, missing-evidence, partial-coverage, and post-reference-cap bypasses have regressions. Extreme scientific exponents remain compact, oversized coefficients are exact-tuple hashed, and pathological numeric claims remain fail-closed. | Deploy and monitor provenance rejection volume. Backend/AI lead. |
| P0.8 request/history bounds (H16) | Verified locally | 256 KiB bounded ASGI pre-read/replay, bounded config ceiling, full-stack chunked 413, typed three-turn Q&A, malformed JSON/UTF-8 rejection, fatal UTF-8 edge decoding, and blank-history filtering pass. Edge limits count Unicode code points; executable contract math proves the maximally escaped query-only input fits 32 KiB and the edge-created three-turn payload fits the backend's 256 KiB floor. | Verify proxy/platform cap, memory/latency under load, and deployed edge snapshot. Edge function behavior remains source-contract tested until P1 makes it directly executable. Backend/API lead. |
| P0.9 null averages (H4) | Verified locally and committed | The active frontend and frontend commit `3a8ceda` use an unweighted finite-observation denominator for price/tariff; null, zero, all-missing, order, and tariff regressions pass | Deploy the frontend commit and confirm production telemetry. Frontend lead. |
| P0.10 readiness/startup (M17) | Production baseline verified; F1 hardening locally verified | Railway startup on port 3000 and `/readyz` passed in production. F1 removes duplicate direct-startup module initialization, enforces one configured worker, and bounds schema reflection with a 60-second success TTL, 10-second failure retry interval, and lock-protected single-flight refresh; stale/incomplete schema remains fail-closed. | Deploy the F1 batch; verify reflection cadence/query cost, restart behavior, and exactly one replica with autoscaling disabled. Platform/SRE lead. |

## Baseline and production unknowns

- Backend flags remain safe in source: `ENABLE_EVIDENCE_REANALYSIS=false`; canonical happy-path enforcement H1 remains open and was not enabled.
- Backend source rejects configured HTTP worker counts other than one. The production logs showed one Uvicorn process and a `1/1` healthcheck, but Railway replica/autoscaling control-plane configuration is still not recorded.
- SQLAlchemy pool budget is `pool_size=3`, `max_overflow=2`; actual deployed database limits are not locally observable.
- Authentication defaults to `gateway_only`; non-test `gateway_and_bearer` now fails startup. Existing direct-bearer traffic is not locally observable.
- `D:/export_enai` is the active build/export tree but is not a Git worktree. The linked frontend Git repository is `D:/export_enai/_repo_sync`; it contains a newer canonical UI and chat edge v3.3 implementation. Formal tree reconciliation remains P1/H18.
- The path-by-path audited port was installed and committed as frontend `3a8ceda` on `fix/p0-remediation` in `D:/Enaiapp/p0_frontend_commit_repo`. It is based on canonical `main` commit `07c6024`, preserves canonical-only UI coverage and the chat-history role tie-breaker, and points to the same GitHub `origin` as the frontend repository.
- The original frontend worktree remains untouched on `docs/progressive-thinking-text-spec` with unrelated untracked logs, build output, local settings, and a plan. The P0 commit was created in the independent clean clone because linked-worktree creation was unavailable. Only the 36 audited paths were staged; `git add -A` was not used.
- Deployed edge hashes, live RLS/PUBLIC/network grants, provider cancellation beyond the explicit OpenAI timeout, branch protection, and production access logs remain manual verification items. The dedicated production runtime identity and read-only/denial probes were operator-confirmed on 2026-07-16.
- Backend: Ruff and `git diff --check` pass; all 1,348 tests pass in one run using a writable pytest temp root.
- Frontend active export tree: a clean copied source tree completed `npm ci`, lint, 228 fail-on-console tests, and a production build (2,865 modules).
- Frontend canonical-repository port: lint passes; all 318 tests pass with the fail-on-console guard; the production build completes with 2,879 modules; the identical clean-installed lock graph resolves React Router 7.18.1 and `ws` 8.21.0 and reports zero production vulnerabilities. The port used a junction to the already clean-installed dependency tree, so this is not evidence of a second clean `npm ci` in the canonical tree.
- The remaining all-dependency audit finding is in development-only `esbuild` through Vite/tsx and requires a breaking Vite 8 upgrade. Production audit excludes it; track the major toolchain upgrade in P8 rather than forcing it into P0.
- Local validation used Node 25.2.1/npm 11.6.2 and Python 3.14.0. CI/deployment pin Node 20.19.1 and Python 3.11, so the required pinned-runtime result must come from CI or matching local runtimes. Python 3.14 emitted the known LangChain/Pydantic-v1 compatibility warning.
- Lock evidence: frontend `package-lock.json` SHA-256 `B735CDC58E9AA29FC6BB4D57FA1E3453D2975C191F942C0BDB85734CAEB64913`; backend `requirements.txt` SHA-256 `4A4098406E084882ADDF94718C48DE8C3D5ED70E789324D9F3FD17B0C59F874C`.
- `TEST_DATABASE_URL` is absent locally. CI now fails instead of skipping, but the SQL regression gate has not run here.
- Post-deploy smoke and live-browser workflows are fail-closed when required secrets are absent, but neither was run locally because no deployed URLs/test identity were supplied.

## Independent audit-loop discoveries

| Severity | Discovery | Resolution / state |
|---|---|---|
| High | Legacy provenance ignored single-digit/scientific values, text after line eight, absent evidence, and tokens after the citation-reference cap | Fixed in the existing extractor/provenance gate; adversarial regressions pass. |
| High | Fixed-point normalization expanded compact scientific tokens such as `1e100000` into 100,001-character strings and could amplify CPU/memory across provenance processing | Fixed with exact bounded canonicalization, compact exponents, hashed oversized coefficients, guarded rounding, and repeated-token/adversarial regressions. |
| High | Reflected database views could expand the application SQL allow-list beyond the nine explicitly granted relations | Fixed: reflection is availability metadata only and readiness verifies every documented required column. |
| High | The locked production graph contained vulnerable React Router and `ws` versions | Fixed with semver-compatible lock updates; production audit now reports zero vulnerabilities. |
| Medium | Exception-based streamed-body limiting was swallowed by FastAPI and returned 400 instead of stable 413 | Fixed with bounded pre-read/replay and a full-stack chunked request test. |
| Medium | Privacy purge could follow a substituted directory link, and earlier metadata/storage validation occurred too late | Fixed with real-path/symlink revalidation before the RPC, before write, and before purge; authenticated metadata tamper tests pass. |
| Medium | A blank legacy `chat_history` user row caused the typed backend request to fail wholesale | Fixed by filtering blank persisted questions at the edge before Q&A mapping. |
| Medium | Startup-only reflection could stay degraded after recovery; per-probe reflection could overload the database, especially during an outage | Fixed locally: `/readyz` uses a bounded success cache, failure retry interval, and single-flight refresh, detects drift after the TTL, and fails closed when stale refresh fails. Deployment cadence evidence remains open. |
| Medium | JavaScript UTF-16 length and backend Unicode-code-point limits could disagree for non-BMP chat text | Fixed with code-point-aware edge validation/truncation and executable inbound/outbound byte-boundary contracts. |
| High | Copying the active export snapshot over the canonical frontend repository would regress the chat v3.3 equal-timestamp role tie-breaker and remove canonical-only UI tests | Avoided in frontend commit `3a8ceda` with a path-by-path port: chat is v3.4 with both protections, and Builder/Chat/Dashboard retain 30/13/17 tests. |
| Medium | The new console gate exposed asynchronous React updates in canonical-only Builder, Chat, and Dashboard tests (including a Radix announcement during an intentional 1.5-second wait) | Fixed at test lifecycle roots with awaited `act` boundaries; all 60 affected UI tests and all 318 frontend tests pass without console diagnostics. |
| Advisory High (development tooling) | Current Vite/tsx `esbuild` graph remains flagged and the available fix requires a breaking Vite 8 upgrade | Open for P8 toolchain modernization; it is absent from the production dependency audit and production bundle. Do not expose development servers to untrusted networks. |

## Complete finding registry

Every row links to the [finding coverage matrix](comprehensive_audit_remediation_plan_2026-07-11.md#16-finding-coverage-matrix); its planned phase supplies the acceptance and rollback procedure. The accountable role is also the rollback owner unless the phase section states otherwise.

| ID | Finding | Phase | Accountable / rollback role | State |
|---|---|---|---|---|
| H1 | Happy path bypasses canonical evidence | P4 | Backend/AI lead | Open |
| H2 | Incorrect canonical units/filter units | P2 | Backend/AI lead | Open |
| H3 | Order-dependent or mixed statistics | P2 | Backend/AI lead | Open |
| H4 | Frontend null prices counted as zero | P0 | Frontend lead | Verified locally |
| H5 | Chat quota/persistence bypass | P3 | Database/Supabase lead | Open |
| H6 | Dashboard quota bypass | P3 | Database/Supabase lead | Open |
| H7 | Direct bearer bypass | P0/P3 | Backend/API lead | P0 containment verified locally; permanent fix open |
| H8 | Paused admins retain authority | P0/P3 | Database/Supabase lead | In progress |
| H9 | All admins can be removed | P3 | Database/Supabase lead | Open |
| H10 | Persisted history treated as trusted | P3 | Backend/API lead | Open |
| H11 | Legacy fallback bypasses provenance | P0 | Backend/AI lead | Verified locally |
| H12 | Data failure becomes conceptual success | P4 | Backend/AI lead | Open |
| H13 | Retry amplification and no total deadline | P5 | Backend/API lead | Open |
| H14 | DB breaker fails open/closed incorrectly | P5 | Platform/SRE lead | Open |
| H15 | Pool oversubscription and orphan work | P5 | Platform/SRE lead | Open |
| H16 | Unbounded body before parsing/auth | P0 | Backend/API lead | Verified locally |
| H17 | Auth listener race/deadlock | P1 | Frontend lead | Open |
| H18 | Divergent frontend trees | P1 | Frontend lead | Open |
| H19 | User deletion purges append-only audit | P0 | Database/Supabase lead | In progress |
| H20 | Raw privacy exports in project tree | P0 | Privacy/Security owner | Blocked |
| H21 | Least-privilege deployment unverified | P7 | Platform/SRE lead | Open |
| M1 | Charts use raw `ctx.df` | P4 | Backend/AI lead | Open |
| M2 | Frame provenance references are empty | P2 | Backend/AI lead | Open |
| M3 | Plan validation is warning-only | P4 | Backend/AI lead | Open |
| M4 | Evidence re-analysis is incomplete | P4 | Backend/AI lead | Open |
| M5 | No shared API contract; charts/tier/provenance drift | P3/P6 | Backend/API + Frontend leads | Open |
| M6 | Actor/session/request identity is lost | P3 | Backend/API lead | Open |
| M7 | JSONB double encoding and turn ordering | P3/P6 | Backend/API + Frontend leads | Open |
| M8 | Supabase errors ignored or raw errors exposed | P3/P6 | Database/Supabase + Frontend leads | Open |
| M9 | Dashboard churn and mixed snapshots | P6 | Frontend lead | Open |
| M10 | Correlation ignores the requested period | P2 | Backend/AI lead | Open |
| M11 | Provider validation/breaker inconsistency | P5 | Platform/SRE lead | Open |
| M12 | Forecast horizon parsing changes intent | P2 | Backend/AI lead | Open |
| M13 | Embedding cache not model-aware | P2 | Backend/AI lead | Open |
| M14 | Manual or unpinned edge deployment | P1 | Platform/SRE lead | Open |
| M15 | Internal telemetry/content overexposed | P7 | Privacy/Security owner | Open |
| M16 | Failed admin audits roll back | P3 | Database/Supabase lead | Open |
| M17 | Readiness/startup over-report success | P0 | Platform/SRE lead | Verified locally |
| M18 | Unsafe container context/runtime packaging | P7 | Platform/SRE lead | Open |
| M19 | Missing error boundaries/config diagnostics | P6 | Frontend lead | Open |
| M20 | Accessibility gaps | P6 | Frontend lead | Open |
| M21 | Unbounded admin listing | P6 | Frontend lead | Open |
| M22 | Metrics not thread-safe | P5 | Backend/API lead | Open |
| M23 | Session ownership and scaling fragility | P5/P7 | Backend/API lead | Open |
| M24 | Frontend lint/test/release gate not clean | P0 | QA/release lead | Code gates verified locally; pinned/deployed gates open |
| M25 | Oversized mixed-responsibility modules | P8 | Backend/AI lead | Open |
| L1 | Secondary-evidence observability incomplete | P5 | Platform/SRE lead | Open |
| L2 | Duplicate pytest configuration | P8 | QA/release lead | Open |
| L3 | Stale Docker/runtime comments | P7 | Platform/SRE lead | Open |
| L4 | Duplicate toast stores/invalid DOM prop | P0/P6 | Frontend lead | P0 warning cleanup verified; consolidation open |
