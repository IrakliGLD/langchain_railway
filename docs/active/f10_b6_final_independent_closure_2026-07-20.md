# F10 B6 Final Independent Closure Re-Audit - 2026-07-20

> **Historical pre-merge audit.** This record intentionally captures the
> pre-merge decision and findings from 2026-07-20. It is superseded by
> [`f10_b6_final_independent_closure_2026-07-21.md`](./f10_b6_final_independent_closure_2026-07-21.md),
> which records the merged `0684dc1` deployment and final B6 decision.

## Decision

**F10 is not closed.** The dependency/security remediation is green and the
source revisions remain well tested, but the B6 release-evidence contract is
not satisfied at the artifacts currently running in production.

This re-audit found no unwaived Critical/High dependency advisory. It does not
convert missing, stale, contradictory, or differently-versioned evidence into
a pass.

## Production identities observed

| Surface | Identity/evidence observed on 2026-07-20 | Result |
|---|---|---|
| Backend source | Ledger-attested protected `/versionz`: `2f2a31053dfa391fbb0958ae858141c6f3e26ff9`; GitHub/Railway combined status is successful; fresh public `/healthz` and `/readyz` both returned 200 | Source identity and health pass; exact runtime image digest is not recorded |
| Browser | Fresh `release-manifest.json`: `04dd0143ff4fe2c9c3dca4742072f51ee64f540a`, aggregate `aff3a48bf94f97fb53dd7eaa3054f6a3ecba4516a332893ca575f528a56b2d84` | Healthy and internally consistent |
| Supabase Edge | Fresh healthcheck/header identity: `193896745b5fd985da6618c366e750b73887dd75`, source aggregate `973efd2764f9ab31d35789a7cc17edad9ac8dc5c5da9679344cda3fceb2fddcc` | Healthy, but version differs from browser |
| Database | Disposable regression and production read-only RLS/grant/runtime-role results are recorded in B4 | Current fixed-source disposable-run identity is not recorded unambiguously |

The repositories remain independent deployment units and no cross-repository
build/read dependency was found.

## Findings

### F10-REL-02 - current production artifacts no longer share the attested release identity

- **Severity:** High (release blocker)
- The repository's own public smoke was rerun with the live browser SHA
  `04dd0143...`. It verified all 15 served files against the deployed manifest,
  then failed because Edge returned `19389674...` instead of `04dd0143...`.
- Commit `04dd0143...` changes browser source (`src/components/ui/toast.jsx`),
  so it is not a documentation-only descendant covered by the earlier frozen
  identity at `19389674...`. Under the B0 freeze rule, the older integrated
  promotion/smoke evidence cannot attest the new browser artifact.
- The backend release-evidence image/SBOM is recorded at `699b703e...`, while
  production was separately rebuilt by Railway and reports source
  `2f2a3105...`. The ledger states that the digests differ but does not record
  the runtime image digest. This does not meet the plan's instruction to
  record both digests explicitly when Railway rebuilds from Git.
- **Closure action:** deploy all Edge functions at `04dd0143...` (or promote a
  newer reviewed SHA to both browser and Edge), rerun release evidence and the
  authenticated smoke at that exact SHA, and append the workflow/deployment
  IDs. Record the Railway backend runtime image digest and bind it to the
  protected source identity, SBOM/audit artifact, deployment ID, and rollback
  target.

### F10-E2E-03 - exact-current integrated, load, and database evidence is incomplete

- **Severity:** High (release-evidence blocker)
- Authenticated Post Deploy Smoke #240 is recorded at `19389674...`, not at the
  browser artifact now serving `04dd0143...`. The fresh public smoke proves the
  current mismatch and therefore cannot substitute for an authenticated
  browser-to-Edge-to-backend-to-provider/database pass at one identity.
- The approved representative chat-load envelope (concurrency 2, at most 20
  requests, at most USD 2) is explicitly marked **not run**. No waiver using
  the F10 waiver template exists. Unit saturation tests and a single happy-path
  chat do not satisfy the B6 `Load` row.
- The ledger associates the disposable DB green run with frontend
  `ac87530...`, then records the migration fix at later commit `a4c24a2...`.
  Static P6.B generation verifies at current source, but the immutable full
  disposable regression run for the fixed/current source is not identified.
- Rollback is documented as available but explicitly not rehearsed. That is a
  residual operational gap, not proof of rollback behavior.
- **Closure action:** after realigning browser/Edge identity, rerun the
  authenticated Post Deploy Smoke and Live Browser Proof, the full disposable
  DB regression at the fixed SHA, and the bounded representative chat load.
  Archive exact run/artifact/deployment IDs. If the load gate is intentionally
  deferred, create a complete named, approved, expiring waiver; an informal
  operator acceptance is not a waiver.

### F10-ARCH-04 - an expired path and a B6-due compatibility decision remain open

- **Severity:** Medium
- The compatibility registry classifies the `google.generativeai` fallback as
  unreachable/expired but defers deletion to 2026-08-31. B5's rule is explicit:
  expired paths are removed now; deadlines are for legitimate rollout or
  protocol/data compatibility paths, not dead implementation.
- The dormant `gateway_and_bearer` surface has a registry deadline of
  "Decision by B6 closure", but no keep/remove decision is recorded. Production
  remains safely `gateway_only`, so this is not a demonstrated auth bypass; it
  is an unmet compatibility-governance gate.
- **Closure action:** remove the inert legacy Gemini branch in its own tested
  cleanup change. Record and implement the B6 bearer-surface decision, or give
  the retained surface a concrete classification, owner, evidence-based
  removal criterion, and future date consistent with the registry rules.

### F10-DOC-06 - architecture and closure ledgers contradict source or each other

- **Severity:** Medium (B6 architecture/evidence gate)
- `query_pipeline_architecture.md` says `ENABLE_AGENT_LOOP` and
  `agent_loop_blocked_by_policy` survive, but B5 and source inspection confirm
  both were removed.
- The B5 registry lists `PLAN_VALIDATION_MODE` default as `shadow`; effective
  code and the architecture document correctly use `warn`.
- The B4 accessibility table says the admin scan was skipped and manual checks
  remain pending, while later paragraphs claim all admin scans and light manual
  accessibility are done.
- **Closure action:** reconcile the architecture and evidence documents to the
  effective code/defaults and immutable run evidence. Do not mark an unchecked
  manual item complete through summary prose.

### F10-A11Y-07 - the manual accessibility closure row is not evidenced

- **Severity:** Medium (release-evidence blocker)
- Automated credentialed axe evidence is recorded and it found a real WCAG
  4.1.2 defect that was fixed at `04dd0143...`.
- The required visible-focus, NVDA/name-role, live-region, and authenticated
  responsive/touch checks remain unchecked in the detailed checklist. The
  summary's phrase "operator light manual a11y" does not identify observations,
  browser/AT versions, pages, viewport/zoom settings, artifact SHA, or evidence
  location.
- **Closure action:** complete and attach the manual checklist against the
  realigned production SHA, then make the table, checklist, and exit summary
  agree.

### F10-QA-08 - generated SQL/hash checks are not clean-checkout portable on Windows

- **Severity:** Low (test-tool defect)
- In a clean `core.autocrlf=true` checkout, 464/465 frontend tests passed; the
  sole failure was the vendored schema raw-byte hash. `db:f2:verify` failed for
  the same line-ending reason. Both pass in the existing checkout and in Linux
  CI, and the generated v2 consumer verifier passes.
- **Closure action:** normalize CRLF/LF before hashing/comparison, or enforce LF
  for the generated contract/SQL sources through `.gitattributes` and test that
  policy on Windows.

## B6 closure matrix

| Gate | Re-audit result | Evidence/shortfall |
|---|---|---|
| Backend quality/security | **Partial** | 1,710 tests accounted for, Ruff green, security 24/24, red-team 1.0, fresh pip-audit 2.10.1 reports no vulnerabilities, lock check green. Exact production runtime image digest/container-SBOM binding remains absent. |
| Frontend/Edge | **Fail** | Lint, build, artifact verify, generated v2 contract, Edge manifest, and fresh npm audit (0 vulnerabilities) pass. Clean Windows suite is 464/465 due the line-ending tool defect. Browser `04dd0143...` and Edge `19389674...` are not one release identity; full Edge verification was not freshly rerun locally because pinned Deno 2.1.4 is unavailable. |
| Database | **Partial** | Static F2/P3.B/P6.B/FB.5 verification and production RLS/grant/runtime-role attestation pass. The immutable full disposable run tied to the later migration fix/current SHA is not identified. |
| Production smoke | **Fail** | Current public smoke verifies browser files but fails Edge version equality. Earlier authenticated smoke is tied to `19389674...`. |
| Failure/reliability | **Pass with residual operational risk** | Deadline, cancellation, ambiguity, idempotency, no-orphan, breaker, and saturation cases are green in the 1,710 backend suite; rollback was not rehearsed. |
| Load | **Fail** | Approved representative chat load is explicitly not run and has no formal waiver. |
| Accessibility | **Fail** | Credentialed axe is recorded green after the toast fix, but detailed records conflict and required manual checks remain unchecked. |
| Architecture | **Fail** | Architecture still documents the removed agent-loop flag/trace field. |
| Compatibility | **Fail** | One admitted expired path remains; the bearer-surface decision due at B6 is absent. |
| Waivers | **Pass for dependencies; not available to excuse other failed gates** | Fresh backend and frontend dependency audits are clean. No Critical/High dependency waiver is needed. No complete load/evidence waiver exists. |

## Fresh verification performed

Backend clean checkout: `2f2a31053dfa391fbb0958ae858141c6f3e26ff9`.

- `ruff check .` - pass.
- `pytest -q tests/security` - 24 passed.
- formal red-team gate - score 1.0, no hard failures.
- full `pytest` - 1,695 passed, 1 timeout and 14 temp-directory setup errors;
  all affected cases then passed (20/20) with a workspace-owned `--basetemp`,
  accounting for all 1,710 tests. Failures were environment/concurrency effects,
  not reproduced product defects.
- `python scripts/generate_requirements_lock.py --check` - pass after network
  access was allowed.
- `pip-audit 2.10.1 -r requirements-lock.txt --no-deps --disable-pip` - exit 0,
  no known vulnerabilities.
- Fresh public `/healthz` and `/readyz` - HTTP 200.

Frontend clean checkout/browser identity:
`04dd0143ff4fe2c9c3dca4742072f51ee64f540a`.

- ESLint - pass.
- frontend tests - 464 passed, 1 Windows line-ending hash failure.
- generated chat-gateway v2 web/full consumer checks - pass.
- Edge source manifest - pass at `973efd27...`.
- F2/P3.B/P6.B/FB.5 static patch generation checks - pass in the established
  checkout; clean Windows F2 verification exposes the line-ending defect.
- production build with exact SHA - pass; artifact verification - pass.
- `npm audit --omit=dev --audit-level=high` - 0 vulnerabilities.
- fresh public smoke - browser root/security headers/routes and all 15 served
  manifest files pass; Edge version equality fails (`04dd0143...` expected,
  `19389674...` received).

Skipped/not independently reproducible in this environment:

- protected backend `/versionz` (requires the operator secret; ledger evidence
  retained);
- pinned-Deno full Edge verification (runtime unavailable; prior immutable run
  evidence retained);
- disposable database regression (no disposable database credential supplied;
  prior operator evidence reviewed);
- authenticated browser/admin/axe replay (no synthetic credentials supplied;
  prior workflow evidence reviewed);
- Docker runtime/container smoke (Docker unavailable; prior release-workflow
  evidence reviewed, with the runtime-digest gap reported above);
- representative provider/DB chat load (explicitly unrun in the ledger).



| Perspective | Grade | Rationale |
|---|---:|---|
| Functional correctness and query pipeline | **A-** | Broad backend/frontend contracts and all 1,710 backend tests are accounted for; the current integrated production identity is not one frozen release. |
| Architecture and maintainability | **B** | Repository boundary and deep runtime modules remain sound, but an expired provider branch and stale architecture text remain. |
| Security and privacy | **B+** | Fresh Python/npm audits are clean; auth and production DB controls are strong. Exact runtime image/SBOM binding and current integrated evidence limit assurance. |
| Reliability, concurrency, and error handling | **B+** | Failure semantics are extensively tested; representative live load and rollback rehearsal are missing. |
| Performance and scalability | **B-** | One-replica containment and backpressure tests are good; representative chat load is absent and the main browser chunk remains about 617 kB. |
| Frontend UX and accessibility | **B** | Automated accessibility caught and closed a real critical issue; manual AT/focus/live-region evidence is incomplete and the browser/Edge identities drifted. |

**Overall: B-.** Source quality is materially stronger than in the 2026-07-18
audit, but final production assurance is still below the F10 closure standard.

## Closure recommendation

Do not mark F10 complete. The smallest safe closure sequence is:

1. choose one current frontend SHA and deploy browser plus all Edge functions at
   that exact SHA;
2. rerun current-SHA release evidence, authenticated smoke, Live Browser Proof,
   axe, and the detailed manual accessibility checklist;
3. rerun the full disposable DB regression at the fixed/current frontend SHA;
4. run the approved representative chat-load envelope or attach a complete,
   approved, expiring waiver;
5. record the backend Railway runtime image digest and bind it to deployment,
   protected SHA, SBOM/audit evidence, and rollback target;
6. remove the expired Gemini fallback, decide the dormant bearer surface, and
   reconcile the architecture/registry/evidence documents;
7. rerun B6 once more against those exact immutable identities.
