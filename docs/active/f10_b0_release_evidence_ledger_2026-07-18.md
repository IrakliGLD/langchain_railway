# F10 B0 Release and Evidence Ledger — 2026-07-18

## Status

**B0 repository implementation is complete; B0 operational exit is blocked.** Repository identities, public deployment evidence, evidence locations, and the required ownership schema are recorded. Named human owners, private frontend workflow IDs, Railway deployment/image IDs, and rollback deployment IDs are unavailable and must not be guessed.

Do not treat this document as approval to promote B1 artifacts. The unchecked manual fields in §7 must be completed first.

## 1. Scope and independence

| Application | Canonical local repository | Remote repository | Deployment unit |
|---|---|---|---|
| Backend | `D:\Enaiapp\langchain_railway` | `https://github.com/IrakliGLD/langchain_railway.git` | Independent Railway backend service |
| Frontend/Supabase | `D:\export_enai\_repo_sync` | `https://github.com/IrakliGLD/EnaiDashboard.git` | Independent Railway static frontend plus Supabase Edge/database |

Neither application may import, read, copy, or build the other repository's files. The only supported integration is the versioned browser → Supabase Edge → backend HTTP/JSON gateway contract.

## 2. Baseline source identities

These are the baselines observed before the B0 documentation commit. B1 implementation will create new candidates and must add new ledger rows rather than overwriting this history.

| Field | Backend baseline | Frontend/Supabase baseline |
|---|---|---|
| Branch | `refactor/review-phase-fixes` | `main` |
| Full source SHA | `f3482a53aac284f931507316541a364fd253dcf2` | `4f22cb3a7b1b766d6802c90f613e096b75ef68a8` |
| Remote equality at collection | Local HEAD equaled `origin/refactor/review-phase-fixes` before uncommitted B0 documentation | Local HEAD equaled `origin/main` |
| Audited application-code ancestor | `921f823c0888271c38ec88c9e1aa09c73672fb45`; `f3482a5` changed audit documentation only | Same SHA as source baseline |
| Local non-phase artifacts excluded | `.tmp_pytest_f3/` | `builderpage-multix-test-output.log`, `builderpage-test-output.log`, `docs/superpowers/plans/`, `test-output.log` |

## 3. CI and release-workflow evidence

| Application | Workflow/run | Result | Interpretation/action |
|---|---|---|---|
| Backend | CI run [`29643895335`](https://github.com/IrakliGLD/langchain_railway/actions/runs/29643895335), job `88078743162` | **Failed** at `Enforce focused state and database-boundary coverage` | B1 must not use this as green release evidence. The exact command passed locally (`14` tests, `98.08%` coverage) under Python 3.14/pytest 9, but CI uses Python 3.11 and pinned pytest 8; therefore the CI failure is not locally reproduced or explained. Obtain the authorized job log, reproduce with the pinned CI runtime, fix the root cause in a separate phase-appropriate commit, and require a successful rerun. |
| Backend | `Backend release evidence` for the baseline SHA | **Not recorded** | Release operator must run it for the final B1/B2 candidate, review SBOM/audit, and record workflow run/artifact IDs. |
| Frontend | CI/release/deploy runs for `4f22cb3...` | **Private/unavailable from the public API** | Repository/environment owner must record CI, Frontend release evidence, Edge deploy, Post Deploy Smoke, and Live Browser Proof run IDs. Do not infer success from deployment health alone. |

## 4. Public deployment observations

| Surface | Observation | Identity conclusion |
|---|---|---|
| Backend `https://enai.galdava.com/healthz` | HTTP 200 during F10 | Service is live; exact Git SHA/image digest remains unproven. |
| Backend `https://enai.galdava.com/readyz` | HTTP 200 with database and schema ready during F10 | Readiness is green; it is not release-identity evidence. |
| Frontend `https://dashboard.galdava.com` | HTTP 200 with production security headers; HTML version `3a8ceda` | Deployed version label is stale; exact frontend SHA remains unproven. |
| Supabase project | Public project ref `qvmqmmcglqmhachqaezt` | Project identity only; not a deployment revision. |
| All nine Edge functions | `X-Enai-Edge-Source=e0111671ca86dcc3a29c088290c60ec608ec4181d2ce2ba7f2bbe68b0b7ce85e` | Source digest equals the current frontend manifest; `healthcheck.version=3244ed1` remains stale. |

## 5. Evidence storage contract

For every future candidate, preserve the following immutable records:

| Evidence | Required location/retention |
|---|---|
| Source identity | Full SHA in this append-only ledger plus repository commit URL. |
| CI | GitHub workflow run URL and conclusion at the exact SHA. |
| Backend image | `backend-release-manifest.json`, OCI digest/revision label, SBOM, `pip-audit` JSON, checksums, and release-workflow run; retain beyond the default 30-day artifact window in the approved release archive. |
| Frontend artifact | Release manifest, asset hashes, full `VITE_APP_VERSION`, workflow run, Railway deployment ID, and promoted/rollback artifact identity. |
| Edge artifact | Full workflow SHA, `deployed-git-sha.txt`, `deployed-edge-source-manifest.json`, all function headers/hashes, Supabase project, deployment time, and rollback SHA. |
| Database | Disposable DB regression output plus production read-only schema/RLS/grant/runtime-role evidence; redact connection strings and credentials. |
| Browser/accessibility | Post-deploy smoke and Live Browser Proof runs, Playwright report, redacted traces/screenshots, manual accessibility checklist, and console review. |
| Operations | Railway/Supabase deployment IDs, environment, operator, approver, observation window, incidents, rollback rehearsal and outcome. |
| Waiver | Approved waiver record from the F10 template; never place secrets or raw prompts in it. |

Evidence artifacts must not contain credentials, raw prompts, user content, database URLs, provider keys, access tokens, session cookies, or unredacted personal data.

## 6. Required named ownership

The following assignments require explicit human names or accountable organizational identities. Repository usernames are not automatically approval authority.

| Responsibility | Named owner | Approver | Status |
|---|---|---|---|
| Backend dependency/security remediation | **UNASSIGNED** | **UNASSIGNED** | Manual blocker |
| Backend Railway build/deploy/rollback | **UNASSIGNED** | **UNASSIGNED** | Manual blocker |
| Frontend Railway build/deploy/rollback | **UNASSIGNED** | **UNASSIGNED** | Manual blocker |
| Supabase Edge/database operations | **UNASSIGNED** | **UNASSIGNED** | Manual blocker |
| Credentialed production smoke accounts | **UNASSIGNED** | **UNASSIGNED** | Manual blocker |
| Accessibility evidence | **UNASSIGNED** | **UNASSIGNED** | Manual blocker |
| Critical/High waiver authority | **UNASSIGNED** | **UNASSIGNED** | Manual blocker |
| Final F10 release decision | **UNASSIGNED** | **UNASSIGNED** | Manual blocker |

One person may hold several roles if explicitly recorded, but the waiver approver should not approve their own unresolved security exception where an independent reviewer is available.

## 7. Manual completion checklist

- [ ] Replace every `UNASSIGNED` ownership field with a named accountable owner/approver.
- [ ] Record the current backend Railway service, environment, deployment ID, source SHA, image digest, replica/autoscaling setting, and rollback deployment ID.
- [ ] Record the current frontend Railway service, environment, deployment ID, source SHA/artifact manifest, and rollback deployment ID.
- [ ] Record frontend private GitHub CI and release-evidence run IDs for the current or superseding candidate.
- [ ] Record the current Supabase Edge deployment workflow run, deployment timestamp, project ref, source digest, full version, and rollback SHA.
- [ ] Choose the protected evidence archive and confirm retention/access policy.
- [ ] Download the authorized backend failed-job log, reproduce with Python 3.11 plus `requirements.txt`/`requirements-dev.txt`, and close the focused-coverage CI failure at the next candidate SHA. The Python 3.14 local pass is diagnostic evidence, not a substitute.

## 8. Append-only candidate template

Copy this table for B1 and every later candidate. Never replace historical rows.

| Field | Value |
|---|---|
| Application/repository | |
| Branch and full Git SHA | |
| Candidate purpose | |
| CI run/conclusion | |
| Release-evidence run | |
| Artifact/image/source digest | |
| Railway/Supabase deployment ID | |
| Environment and replica topology | |
| Operator and approver | |
| Deployment/observation timestamps | |
| Smoke/load/accessibility/DB evidence | |
| Advisories/waivers | |
| Rollback artifact/deployment ID | |
| Rollback rehearsal/result | |
| Final decision | |

## 9. B0 exit decision

| Gate | Result |
|---|---|
| Independent repositories and integration boundary recorded | Pass |
| Exact baseline repository SHAs recorded | Pass |
| Public deployment/source evidence recorded without inferring exact deployment SHAs | Pass |
| Evidence schema and append-only candidate template defined | Pass |
| Named human owners/approvers assigned | **Blocked** |
| Private workflow, Railway, artifact, and rollback identities recorded | **Blocked** |
| Backend CI green at the candidate SHA | **Blocked** |

**Decision:** commit the B0 ledger so the missing operational evidence is explicit and reviewable. Do not mark B0 fully closed or start B1 promotion until the blocked rows are completed. B1 code design may be inspected, but implementation must not be represented as an approved release candidate while B0 remains operationally blocked.
