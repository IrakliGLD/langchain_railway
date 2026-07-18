# F10 Blocker Remediation Plan — 2026-07-18

## 1. Objective and completion rule

Close the three High F10 release blockers and the Medium compatibility-cleanup blocker without coupling the two applications or combining unrelated high-risk changes.

F10 is complete only when:

- backend dependencies have no unresolved Critical/High advisory, or every retained advisory has an approved, named, time-bound waiver;
- the exact tested backend image, frontend artifact, and Edge source revision are the artifacts running in production;
- authenticated browser, Edge, backend, database, security, load, and accessibility evidence is attached to those exact revisions;
- every expired compatibility path is removed, while non-expired rollout paths have documented evidence and deadlines; and
- the independent closure audit is rerun and records no unwaived Critical/High finding.

| F10 blocker | Closing phases |
|---|---|
| `F10-REL-02` exact deployment identity | B0, B1, B3 |
| `F10-SEC-01` backend dependency advisories | B2.A, B3.A, B6 |
| `F10-E2E-03` authenticated/live evidence | B4.A, B4.B, B6 |
| `F10-ARCH-04` expired compatibility paths | B5.A, B5.B, B6 |

## 2. Non-negotiable boundaries

1. `D:\Enaiapp\langchain_railway` and `D:\export_enai\_repo_sync` remain independent Git repositories and deployment units.
2. Neither repository may read, copy, import, or build files from the other. Shared behavior is transferred only through versioned contract artifacts and HTTP/JSON.
3. Backend and frontend commits are reviewed, committed, pushed, deployed, and rolled back independently.
4. Do not upgrade the complete backend dependency graph in one commit. Each dependency group gets characterization, implementation, audit, and rollback evidence.
5. Do not run `npm run test:db` against production. Those SQL regressions seed `auth.users`, exercise mutations, and alter constraints inside transactions; the repository explicitly requires a dedicated test database.
6. Do not enable production failure-injection endpoints merely to satisfy an audit. Ambiguous delivery, malformed payload, timeout, and cancellation branches are exercised in isolated integration tests; production smoke uses safe real requests and denial probes.
7. No waiver is implicit. A waiver must identify an owner, approver, exact scope, compensating control, expiry date, and remediation ticket.

## 3. Execution order

| Phase | Track | Impact | Risk/effort | Purpose |
|---|---|---:|---:|---|
| B0 | Shared operations | High | Low / Small | Freeze candidate identities, owners, and evidence locations. |
| B1.A | Backend | High | Low–Medium / Medium | Make backend build/image/deployment identity verifiable. |
| B1.B | Frontend/Edge | High | Low / Small–Medium | Fix stale version labels and prove browser/Edge artifact identity. |
| B2.A | Backend | High | Medium–High / Large | Remove or remediate vulnerable dependencies in isolated slices. |
| B2.B | Frontend/Edge complement | Medium | Low / Small | Revalidate generated contracts and integration after backend dependency changes. |
| B3 | Both, independently promoted | High | Medium / Medium | Promote exact attested candidates and preserve rollback artifacts. |
| B4.A | Backend evidence | High | Medium / Medium | Run backend security, load, failure, and authenticated production gates. |
| B4.B | Frontend/Edge/DB evidence | High | Medium / Medium–Large | Run credentialed UI/Edge/database/accessibility gates safely. |
| B5.A | Backend cleanup | Medium–High | Medium / Medium | Remove expired flags and compatibility modes after prerequisites. |
| B5.B | Frontend/Edge cleanup | Medium | Medium / Medium | Remove obsolete protocol/data readers only after consumer/data proof. |
| B6 | Independent closure | High | Low / Medium | Repeat F10 against exact production artifacts and close or waive every blocker. |

Do not begin B3 promotion until B1 and B2 are green. B4 tests the promoted B3 artifacts. B5 items that require two stable releases remain time-gated; they must not be forced into the first release.

## 4. Phase B0 — Freeze candidates, ownership, and evidence

**Goal:** eliminate moving-target ambiguity before code changes.

**Implementation status (2026-07-18):** repository-side B0 ledger implemented in [`f10_b0_release_evidence_ledger_2026-07-18.md`](./f10_b0_release_evidence_ledger_2026-07-18.md). Audit correction: B1 development may proceed because new workflow/deployment/rollback IDs cannot exist until B1/B2 create candidates and B3 promotes them. Unassigned owners, protected evidence storage, candidate IDs, and green candidate CI remain mandatory B3 preflight gates; no High waiver is permitted while waiver authority is unassigned.

### Actions

1. Create one release ledger row for each independent application containing:
   - repository and branch;
   - candidate full Git SHA;
   - GitHub Actions run IDs;
   - Railway service/environment/deployment ID;
   - Supabase project reference and Edge deployment run ID;
   - artifact/image/source digest;
   - operator, approver, deployment time, observation window, and rollback ID.
2. Assign named owners for:
   - backend dependency/security decisions;
   - backend Railway deployment;
   - frontend Railway deployment;
   - Supabase Edge/database operations;
   - accessibility verification;
   - final waiver approval.
3. Define protected evidence locations. Retain manifests, SBOM, audit JSON, image digest, Edge deployment evidence, Playwright reports, database output, smoke output, and rollback records together.
4. Freeze each candidate during its test/promotion window. A later code commit creates a new candidate and invalidates prior promotion evidence, except for documentation-only commits explicitly recorded as such.

### Exit gate

- Both applications have separate candidate rows, owners, and rollback targets.
- No evidence record infers a SHA from a timestamp or short version label.

## 5. Phase B1 — Immutable release identity

This phase is first because it is high impact, relatively low risk, and makes every later test attributable to an exact artifact.

### B1.A — Backend identity

**Repository implementation status (2026-07-18): complete in backend commit `dc0620538b949ae6e2879b9b73d7956ddd04d5e2`; deployment evidence pending.** The Docker build now requires a full source SHA, embeds it in `/app/release-identity.json`, and labels the image with `org.opencontainers.image.revision`. Runtime startup accepts the embedded identity as authoritative, rejects a conflicting `ENAI_RELEASE_SHA`, and requires a resolved identity in staging/production. The protected `GET /versionz` endpoint exposes only schema version, application version, and full Git SHA through the existing `ENAI_EVALUATE_SECRET` boundary. The release-evidence workflow verifies the embedded identity and OCI label before emitting the v2 manifest, which records both the source SHA and image revision. Focused verification passed `173` tests and the full suite passed `1,685` tests; Ruff is green. Docker execution is unverified on this workstation because Docker is not installed and must be exercised by the protected workflow. B1.A's production exit gate remains open until the workflow artifact, Railway deployment/image identity, protected `/versionz` result, and rollback identity are recorded.

**Implementation:**

1. Define one canonical 40-character release identity, for example `ENAI_RELEASE_SHA`, supplied by the build/promotion system—not manually edited source code.
2. Validate it at production startup. Reject a configured non-hex/non-40-character value. Tests may use an explicit test identity.
3. Embed the SHA as the OCI image revision label and include it in `backend-release-manifest.json` with the immutable image digest.
4. Prefer build-once/promote-by-digest: the `Backend release evidence` workflow builds and attests the image once, and Railway deploys that same digest. If Railway must rebuild from Git, record its source SHA and image digest and add a post-deploy comparison to the attested manifest.
5. Add a safe deployment-identity verification surface. Preferred order:
   - protected operations/version endpoint using the existing operations authentication boundary; or
   - deployment log/manifest evidence retrieved through Railway.
   Do not add secrets, environment values, or dependency details to public health responses.
6. Keep `/healthz` and `/readyz` contracts stable; release identity is not a readiness condition except that malformed required production identity should fail startup.

**Tests:**

- valid/missing/malformed release identity by environment;
- manifest SHA equals OCI revision label;
- protected version response contains only allow-listed fields;
- `/healthz` and `/readyz` remain backward compatible;
- Docker non-root/excluded-file checks remain green.

**Exit gate:** the exact backend SHA and image digest can be mechanically compared before and after deployment.

#### B1.A activation and evidence steps

1. Push the B1.A commit, then run **Backend release evidence** with its full 40-character SHA. The workflow supplies that SHA explicitly as the image build argument.
2. For a normal Railway Git-triggered Docker build, do not manually invent `ENAI_RELEASE_SHA`: Railway supplies `RAILWAY_GIT_COMMIT_SHA`, and the Dockerfile consumes it. If Railway is configured to omit Git metadata or a non-Git builder is used, supply the exact candidate SHA as the Docker build argument `ENAI_RELEASE_SHA`.
3. A runtime `ENAI_RELEASE_SHA` variable is optional for the Docker image. If defined, it must equal the embedded SHA exactly or startup fails. Non-Docker staging/production execution must set it because no embedded identity file exists.
4. After deployment, call `GET /versionz` with `X-App-Key` set to the existing `ENAI_EVALUATE_SECRET`. Compare its `git_sha` byte-for-byte with the candidate SHA. Do not expose this secret or response through a public health monitor.
5. Confirm `/healthz` and `/readyz` remain green, then record the Railway deployment ID, source SHA, runtime image digest, release-workflow run/artifact, manifest image ID/revision, one-replica setting, previous rollback deployment/digest, operator, approver, and timestamps in the B0 ledger.
6. If Railway rebuilt from Git rather than promoting the workflow image, the SHA comparison proves source identity but the separately built image digest will differ. Record both digests explicitly; do not claim byte-identical promotion. Build-once/promote-by-digest remains the preferred later control when the platform workflow supports it.

### B1.B — Frontend and Edge identity

**Repository implementation status (2026-07-18): complete in independent frontend commit `55b15f342796df03dafe29ce77e6c19226cf9fa8`; deployment evidence pending.** Production browser builds now require a full SHA and automatically bind Railway Git metadata; stale explicit/Railway identities fail before build. The static manifest hashes all promoted files, post-deploy smoke requires exact full-SHA and deterministic aggregate equality and verifies every deployed file, and the Edge workflow derives `APP_VERSION` from the resolved immutable checkout before verifying source and version headers on all nine functions. Verification passed `464` frontend tests, ESLint, a zero-vulnerability production audit, two byte-stable production builds, Deno 2.1.4 format/lint/type-check, and `21` Edge tests. The new Edge source aggregate is `973efd2764f9ab31d35789a7cc17edad9ac8dc5c5da9679344cda3fceb2fddcc`. See the independent repository's `docs/active/b1b_immutable_frontend_edge_identity_2026-07-18.md`. B1.B's production exit remains open until exact frontend release, Railway, Edge deploy, and post-deploy smoke evidence plus rollback identities are recorded.

**Implementation:**

1. Keep `VITE_APP_VERSION` as a build input and require the full candidate SHA in production release builds.
2. Ensure the production artifact manifest records the same full SHA and hashes every promoted static asset.
3. Make post-deploy smoke fetch the deployed metadata/assets and compare them with the release-evidence manifest. A timestamp is never sufficient.
4. In `Deploy Supabase edge functions`, set or generate `APP_VERSION` from the workflow's resolved full SHA before deployment; never retain a manually entered historical version.
5. Continue verifying `X-Enai-Edge-Source` for all nine functions. Require both:
   - expected aggregate source digest; and
   - expected full release SHA/version.
6. Preserve the existing generated chat-gateway contract. Do not read the backend working tree during generation; use the vendored, versioned contract source already present in the frontend repository.

**Tests:**

- reject short/malformed workflow `git_ref`;
- generated artifact version equals requested full SHA;
- stale `VITE_APP_VERSION` and stale Edge `APP_VERSION` cause smoke failure;
- deployed Edge digest differs from expected digest causes deployment failure;
- frontend build remains reproducible from a clean checkout.

**Exit gate:** browser and Edge evidence independently maps the deployed content to the exact frontend SHA.

### B1 rollback

- Backend: redeploy the previous attested image digest and confirm its protected release identity plus `/readyz`.
- Frontend: redeploy the previous preserved static artifact/commit.
- Edge: redeploy the previous full SHA through the protected workflow and verify its previous source digest.

## 6. Phase B2 — Backend dependency remediation

### B2.A — Backend security slices

**Goal:** close `F10-SEC-01` without breaking provider, prompt, routing, deadline, or response contracts.

#### B2.A.1 — Produce an advisory/reachability ledger

**Repository implementation status (2026-07-18): complete.** See [`f10_b2a1_dependency_advisory_ledger_2026-07-18.md`](./f10_b2a1_dependency_advisory_ledger_2026-07-18.md) and `docs/evidence/f10_b2/`. The pinned cp311 closure (101 packages) carries 94 OSV records (~34 CVEs) in 11 packages; 42 records belong to pins with zero production imports, every record has a reachable fix or exits the closure, and no waiver is required. Documented deviation: this workstation's Python 3.14 cannot execute `pip-audit -r` against cp311-only pins, so the identical OSV database was queried directly against the exact resolved closure; the authoritative `pip-audit` JSON remains the protected release-evidence workflow artifact at the B2 candidate SHA. The ledger also records that the local test environment diverges from the production pin set on 76 of 101 packages, so local green suites are diagnostic evidence only.

1. Run `pip-audit -r requirements.txt --format json` and preserve the JSON.
2. Produce a forward and reverse dependency graph in a disposable Python 3.11 environment.
3. For every advisory record:
   - direct or transitive package;
   - importing production module, if any;
   - vulnerable feature and whether it is reachable;
   - fixed versions and compatibility constraint;
   - proposed action: remove, upgrade, replace, or waive.
4. Treat absence of an import as evidence for removal, not automatically as proof that a transitive package is unreachable.

#### B2.A.2 — Remove unused direct dependencies

**Repository implementation status (2026-07-18): complete in commits `1eefac2` (litellm), `6f50b3b` (langchain + langchain-community), `9b9017c` (statsmodels).** Each removal was verified by a clean cp311 dry-run re-resolution: the closure shrank 101 → 87 → 72 → 70 packages with zero additions and zero version movement, and the OSV record count fell 94 → 73 → 52. All 52 remaining records belong to the B2.A.3–B2.A.5 upgrade slices. Deviation: the `tenacity` direct pin was **retained as a commented transitive version hold** — dropping it floats the closure to 8.5.0 under the old langchain-core runtime, which would be untested drift; it is re-evaluated at B2.A.5. Verification on this workstation: targeted suite 1,661 passed, Ruff clean, no repo reference (code, tests, Dockerfile, CI) to any removed package. The container-faithful full pytest on Python 3.11 and the SBOM/pip-audit artifact remain CI/release-workflow evidence at the candidate SHA (`Manual verification pending`).

Start with the lowest-risk changes. The audit found no production import of `litellm` and no direct use of several umbrella LangChain packages; verify again before editing.

- Remove one unused direct package/group per commit.
- Re-resolve the environment from a clean virtual environment.
- Run import smoke, full pytest, Ruff, SBOM, and `pip-audit` after every removal.
- Do not remove a transitive package by pinning around a provider wrapper that still requires it.

#### B2.A.3 — Upgrade auth and utility dependencies

**Repository implementation status (2026-07-18): complete (PyJWT + python-dotenv).** `PyJWT==2.13.0` and `python-dotenv==1.2.2` close all 15 remaining auth/utility records (52 → 37); the disputed no-fix record `PYSEC-2025-183` ends at 2.10.1 and does not flag 2.13.0. `tests/test_auth_negative.py` pins the PyJWT 2.13 rejection matrix — alg=none, HS512-with-correct-secret, wrong secret, malformed/segment-count/garbage tokens, unknown and malformed `crit` headers, expiry, missing `sub`/`exp`/`aud`, wrong audience — and documents the current policy that audience is enforced while issuer is not, plus gateway-only behavior (valid bearer rejected while `ENABLE_PUBLIC_BEARER_AUTH` is off, gateway secret unaffected). Full suite 1,706 green including `tests/security`, Ruff clean, cp311 closure identical except the two intended version movements. **Protobuf is deliberately not upgraded here:** it is held at 4.25.9 by the legacy `google-generativeai` chain, which `langchain-google-genai==0.0.11` requires; it is remediated structurally in B2.A.5 where the entire chain leaves the closure.

- Upgrade PyJWT to an advisory-fixed version compatible with the explicit algorithm/claim policy.
- Upgrade `python-dotenv` and `protobuf` through separately reviewable changes where they remain installed.
- Add negative JWT tests for unsupported algorithms, malformed headers, critical headers, expiry, audience/issuer policy, and gateway-only behavior even though public bearer auth remains disabled.

#### B2.A.4 — Upgrade FastAPI/Starlette

- Choose a supported compatible FastAPI/Starlette pair that closes the reported Starlette advisories.
- Characterize middleware order, request path handling, exception envelopes, CORS, request-size limits, rate limiting, lifespan/startup, `/healthz`, and `/readyz` before upgrading.
- Run HTTP contract, security adversarial, red-team, readiness saturation, and container smoke after the upgrade.

#### B2.A.5 — Modernize the LangChain/provider group

Treat `langchain-core`, `langchain-openai`, `langchain-google-genai`, `langsmith`, and their transitive packages as one constrained compatibility problem but commit adaptations in reviewable slices.

1. Capture provider request/response characterization for OpenAI, Gemini, and NVIDIA using fakes—no paid calls in unit tests.
2. Preserve:
   - provider selection and independent breakers;
   - one bounded native attempt;
   - remaining-budget timeout propagation;
   - ambiguous-delivery classification;
   - prompt text/structured schema;
   - tool/model-call counts, golden answers, provenance, and public contracts.
3. Prefer adapting behind `ProviderInvocationRuntime` rather than spreading new SDK APIs through pipeline code.
4. If the modern LangChain stack remains vulnerable or forces excessive compatibility code, replace individual wrappers with direct provider SDK adapters behind that same interface. Do not perform a simultaneous pipeline rewrite.

#### B2.A.6 — Make the fix durable

- Separate top-level requirements from a reproducible compiled lock with hashes, or adopt an equivalent reproducible Python locking mechanism.
- Add a required CI Critical/High `pip-audit` gate and retain the release-workflow SBOM/audit artifacts.
- Add dependency review/renovation automation with grouped provider-framework updates.
- Never suppress the entire audit command because one advisory needs a waiver; apply a narrowly scoped, documented ignore whose ID matches the waiver ledger.

**B2.A exit gate:**

- full backend suite, focused security/coverage gates, Ruff, Docker build, non-root check, SBOM, and release manifest pass;
- no unresolved Critical/High advisory, unless each exception has an approved waiver;
- query-pipeline schemas, golden outputs, provenance, provider-attempt counts, deadlines, and public API remain unchanged.

### B2.B — Frontend/Edge integration complement

Backend library upgrades should not require frontend behavior changes if the HTTP contract remains stable.

Run from the independent frontend repository:

```text
npm ci
npm run contract:chat-gateway
npm run test
npm run lint
npm run edge:verify
npm run build
npm run artifact:verify
```

If the backend public schema changes, stop and create a new versioned schema first, vendor/export it through the established process, regenerate consumers, and deploy backend compatibility before Edge adoption. Do not silently edit both repositories around an unversioned response.

## 7. Phase B3 — Promote exact attested candidates

### B3.A — Backend

1. Push the candidate backend commit.
2. Run **Backend release evidence** with the full 40-character candidate SHA and protected `production` environment.
3. Review the SBOM and `pip-audit` artifact; do not promote a failed/unwaived High result.
4. Record manifest SHA, OCI image digest, workflow run ID, approver, and rollback digest.
5. Deploy that exact digest/SHA to Railway with one replica and autoscaling disabled.
6. Verify protected release identity, `/healthz`, `/readyz`, schema readiness, non-root runtime, and expected port.

### B3.B — Frontend and Edge

1. Push the candidate frontend commit.
2. Run **Frontend release evidence** with the full candidate SHA.
3. Promote the exact verified static artifact, or verify the Railway rebuild byte-for-byte against its manifest.
4. Run **Deploy Supabase edge functions** with the same full frontend SHA and deploy all tracked functions.
5. Confirm every function reports the expected source digest and full `APP_VERSION`.
6. Record the browser deployment ID, Edge workflow run, Supabase project, source digest, and previous rollback SHA.

### B3 ordering

Where the contract is unchanged, backend and frontend remain independently deployable. For the final evidence window, deploy backend first, verify compatibility, then frontend/Edge. Roll back only the failing application unless the versioned contract matrix proves a coordinated rollback is necessary.

## 8. Phase B4 — Complete the missing production and database evidence

### B4.A — Backend evidence

Run against the exact B3 backend artifact:

- complete pytest and Ruff gates from a clean checkout/container;
- security adversarial and formal red-team score gates;
- dependency audit and SBOM review;
- authenticated signed `/ask`, unsigned/invalid gateway rejection, request-size and typed-error checks;
- normal, slow-query, statement-timeout, pool exhaustion, breaker-open, readiness-under-saturation, request cancellation, and simultaneous primary/secondary evidence load tests;
- backend-before-provider-send, provider-response-lost, DB-timeout, and secondary-timeout integration tests using deterministic fakes or an isolated environment;
- proof that one actor-bound request ID cannot create duplicate provider attempts/charges;
- one-replica/autoscaling configuration and no orphan DB/secondary work.

Production failure smoke should remain safe: real signed happy path, invalid signature/replay denial, ordinary browser abort, and readiness saturation within an approved limit. Destructive/provider-ambiguity injection belongs in the isolated harness.

Before any production chat load, approve a maximum concurrency, request count, provider-spend ceiling, test window, abort thresholds, and rollback owner. Stop immediately on duplicate operation/provider attempts, elevated production error rate, readiness degradation, database saturation, or spend above the ceiling.

### B4.B — Frontend, Edge, database, and accessibility evidence

#### GitHub environment/secrets

Configure the existing workflows rather than inventing a parallel runner:

- `SMOKE_APP_URL`
- `SMOKE_FUNCTIONS_BASE_URL`
- `SMOKE_HEALTHCHECK_PATH`
- `SMOKE_SUPABASE_URL`
- `SMOKE_SUPABASE_ANON_KEY`
- `SMOKE_USER_EMAIL` / `SMOKE_USER_PASSWORD`
- `SMOKE_ADMIN_EMAIL` / `SMOKE_ADMIN_PASSWORD`
- `SMOKE_EXPECT_ADMIN`
- `TEST_DATABASE_URL` for a disposable database only
- existing `SUPABASE_ACCESS_TOKEN` and `SUPABASE_PROJECT_REF` for Edge deployment

Use separate synthetic accounts for active user, paused user, quota-exhausted user, and administrator. Never use a real customer's account.

The current workflows already support the normal user and administrator credentials. Extend the protected workflow contract with dedicated paused/quota smoke credentials only if those states cannot be created safely and reversibly through existing test setup. Secret values must never enter artifacts, screenshots, or logs.

#### Workflow order

1. Run frontend **CI** at the candidate SHA.
2. Run **Frontend release evidence** at that SHA.
3. Deploy the exact browser artifact and all Edge functions.
4. Run **Post Deploy Smoke** with `SMOKE_REQUIRE_AUTH=true` and expected full SHA.
5. Run **Live Browser Proof** for authenticated user and admin paths.
6. Preserve Playwright reports, screenshots/traces, console output, Edge evidence, and exact deployment IDs.

#### Required browser/Edge states

- normal question/answer and conversation continuation;
- malformed backend payload handled safely;
- timeout and browser abort/navigation;
- quota-exhausted and paused-user denial;
- optional degradation and critical failure messaging;
- admin authorization, bounded pagination/filtering, mutation, reconciliation, and refresh;
- no unexpected console error;
- keyboard order, visible focus, screen-reader labels/live regions, touch targets, responsive viewports, and 200% zoom;
- credentialed axe scan with no serious/critical violation.

Malformed/timeout/quota state generation should use isolated mocks or dedicated synthetic accounts, not ad-hoc mutation of real users.

#### Database evidence without an existing test database

Preferred options, in order:

1. create a temporary Supabase project, apply the committed baseline/patches, run `npm run test:db`, archive output, then destroy the project; or
2. run a local disposable Supabase/PostgreSQL environment with the required `auth` schema and committed baseline, then run the same command.

Do **not** point `TEST_DATABASE_URL` at live production. After disposable tests pass, run only these production-safe checks:

- read-only schema/constraint/index/function existence queries;
- RLS and explicit grant inventory;
- dedicated runtime role allowed/denied SELECT probes;
- bounded admin read through the application;
- controlled mutations through Edge using synthetic accounts, followed by application-level cleanup/reconciliation.

Global grant changes or migration repair in production require a maintenance window, backup, reviewed rollback SQL, and explicit operator approval.

**B4 exit gate:** every missing F10-E2E-03 item has redacted evidence tied to the exact B3 identities.

## 9. Phase B5 — Compatibility and expired-flag cleanup

First create a compatibility registry classifying every candidate as:

- **expired:** dead implementation or completed migration; remove now;
- **rollout:** behavior still collecting two-release evidence; retain with owner/deadline;
- **resilience fallback:** intentionally supported failure behavior; do not remove merely because its name says “legacy”;
- **protocol/data compatibility:** remove only after consumer/data inventory proves zero use.

### B5.A — Backend

1. Remove `ENABLE_AGENT_LOOP`, its pipeline imports/branches, `agent_loop_blocked_by_policy`, and tests that exist only to preserve the deleted loop's trace shape. Confirm no public response schema changes.
2. After all deployed Edge traffic is proven signed, remove the `optional` value and then retire `ENAI_GATEWAY_ACTOR_ASSERTION_MODE` rather than only changing its default. Actor assertion becomes unconditionally required for gateway requests; retain signature freshness/replay tests.
3. Remove legacy secret-name fallbacks only after Railway environment inventory confirms canonical `ENAI_*` names are present and one rollback deployment has been rehearsed.
4. For evidence finalization, plan validation, honest terminal outcomes, and evidence re-analysis:
   - activate one behavior at a time using the existing canary runbook;
   - record two stable production releases, counters, incidents, quality/deadline metrics, and rollback evidence;
   - then make the selected behavior the only path and delete its rollout switch/holdback code.
5. Do not remove the legacy SQL or safe summary fallback solely because it is called legacy. Remove only when production counters, correctness evaluation, and a safer terminal behavior prove it unnecessary.

### B5.B — Frontend/Edge/database

1. Remove chat-gateway v1 compatibility only after access logs and release inventory show every supported backend/Edge/browser combination uses v2 and rollback no longer depends on v1.
2. Remove JSON/text compatibility readers only after a production read-only query proves all relevant rows satisfy the JSONB constraints and quarantine/reconciliation counts are zero.
3. Remove deprecated Edge environment names only after Supabase secret inventory confirms canonical names and a rollback deployment succeeds.
4. Regenerate the Edge source manifest and generated contract consumers after every source cleanup; run full Edge and frontend gates.

### B5 exit gate

- expired paths are deleted rather than permanently default-off;
- every retained path has a classification, named owner, removal criterion, and deadline;
- two-release candidates have attached evidence;
- rollback remains possible through an attested release, not through dormant unsafe code.

## 10. Phase B6 — Final independent closure

Run a new F10 audit from clean checkouts at the exact production SHAs/digests.

Required closure matrix:

| Gate | Required result |
|---|---|
| Backend quality/security | Full tests, Ruff, red-team, container, SBOM and Critical/High dependency audit green. |
| Frontend/Edge | Tests, lint, production audit, build, artifact verification, generated contract and full Edge verification green. |
| Database | Disposable full regression green; production schema/RLS/grant/runtime-role attestations green. |
| Production smoke | Authenticated browser→Edge→backend→provider/DB flow green at recorded identities. |
| Failure/reliability | Deadlines, cancellation, ambiguity reconciliation, idempotency and no-orphan checks green. |
| Load | Representative chat plus readiness/control capacity within documented budgets. |
| Accessibility | Credentialed axe and manual keyboard/focus/screen-reader/touch/responsive/zoom evidence green. |
| Architecture | `query_pipeline_architecture.md` matches effective defaults and deployed behavior. |
| Compatibility | Expired paths removed; retained paths meet registry rules. |
| Waivers | No Critical/High without named approval and unexpired date. |

Re-run the A–F assessment using these artifacts. Mark F10 complete only after the evidence ledger links every gate to an immutable run/deployment/artifact.

## 11. Waiver template

Use only when remediation cannot safely complete before a release:

```text
Waiver ID:
Finding/advisory:
Affected exact artifact SHA/digest:
Reachable vulnerable behavior:
Reason remediation is deferred:
Compensating controls:
Named owner:
Approver:
Remediation ticket and target release:
Approved at:
Expires at (recommended maximum 30 days):
Rollback/disable action:
```

A waiver cannot cover an unknown deployed SHA. Establish identity first.

## 12. Recommended first implementation batch

Start with the lowest-risk, highest-leverage work:

1. B0 release ledger and owners.
2. B1.A protected backend build identity plus manifest/image label.
3. B1.B full frontend/Edge SHA propagation and stale-version failure tests.
4. Independently audit, commit, push, and deploy B1.A and B1.B.
5. Begin B2.A.1 advisory/reachability inventory and B2.A.2 unused dependency removal.

Do not combine B1 identity work with the LangChain/FastAPI upgrades. The identity work must be available to prove which dependency-remediated artifact is eventually promoted.
