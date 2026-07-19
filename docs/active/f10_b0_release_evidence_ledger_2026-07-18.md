# F10 B0 Release and Evidence Ledger — 2026-07-18

## Status

**B0 repository implementation is complete and B1 development may proceed.** Repository identities, public deployment evidence, evidence locations, and the required ownership schema are recorded. Named human owners, private frontend workflow IDs, Railway deployment/image IDs, and rollback deployment IDs are unavailable and must not be guessed; they are promotion evidence that must be completed before B3, not information that can exist before B1 creates new candidates.

Do not treat this document as approval to promote B1 artifacts. B1 implementation, local verification, audit, commit, and push are allowed; B3 production promotion remains blocked until the applicable manual fields in §7 are completed.

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
| Backend dependency/security remediation | Irakli | Irakli | Assigned 2026-07-19 |
| Backend Railway build/deploy/rollback | Irakli | Irakli | Assigned 2026-07-19 |
| Frontend Railway build/deploy/rollback | Irakli | Irakli | Assigned 2026-07-19 |
| Supabase Edge/database operations | Irakli | Irakli | Assigned 2026-07-19 |
| Credentialed production smoke accounts | Irakli | Irakli | Assigned 2026-07-19 |
| Accessibility evidence | Irakli | Irakli | Assigned 2026-07-19 |
| Critical/High waiver authority | Irakli | Irakli | Assigned 2026-07-19 |
| Final F10 release decision | Irakli | Irakli | Assigned 2026-07-19 |

One person may hold several roles if explicitly recorded, but the waiver approver should not approve their own unresolved security exception where an independent reviewer is available. **Recorded 2026-07-19: Irakli holds every role as sole operator of both repositories.** No waiver currently exists or is requested — the B2 dependency closure audits clean — so the self-approval constraint on waivers is presently moot; if a future Critical/High waiver is ever needed, seek an independent reviewer before self-approval.

## 7. Manual completion checklist

- [x] Replace every `UNASSIGNED` ownership field with a named accountable owner/approver. *(2026-07-19: all roles assigned to Irakli, §6.)*
- [ ] Before B3, record the candidate backend Railway service, environment, deployment ID, source SHA, image digest, replica/autoscaling setting, and rollback deployment ID.
- [ ] Before B3, record the candidate frontend Railway service, environment, deployment ID, source SHA/artifact manifest, and rollback deployment ID.
- [ ] Before B3, record frontend private GitHub CI and release-evidence run IDs for the superseding candidate.
- [ ] Before B3, record the candidate Supabase Edge deployment workflow run, deployment timestamp, project ref, source digest, full version, and rollback SHA.
- [ ] Choose the protected evidence archive and confirm retention/access policy.
- [x] Download the authorized backend failed-job log, reproduce with Python 3.11 plus `requirements.txt`/`requirements-dev.txt`, and close the focused-coverage CI failure at the next candidate SHA. The Python 3.14 local pass is diagnostic evidence, not a substitute. *(Closed 2026-07-19 at `b628f788…`, CI run `29678536060` fully green. Root causes, fixed in order: a finite fake clock patched onto the shared `time` module collided with Python 3.11's logging timestamps (`14c8a42`); the workflow env diverged from the canonical test literals, 401ing the /ask contract tests (`948cddf`); and pydantic 2.9.2 rendered JSON schemas differently from the 2.12.4 that generated the committed contract artifacts (`b628f78`).)*

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
| Named roles and no-waiver default defined for development | Pass |
| Named human owners/approvers assigned for promotion | **Assigned 2026-07-19 (Irakli, all roles — §6)** |
| Candidate workflow, Railway, artifact, and rollback identities recorded | **Deferred until B1/B2 candidates exist and B3 runs** |
| Backend CI green at the candidate SHA | **Required for each B1/B2 commit before promotion** |

**Decision:** B0 is closed for repository development. Proceed with B1 implementation and commit its independent tracks. Do not begin B3 production promotion until owners, candidate deployment identities, rollback identities, green candidate CI, and the protected evidence archive are recorded.

## 10. B1 append-only implementation candidates

These rows record repository candidates only. Empty operational fields are deliberate blockers; they must not be inferred from local tests or public health.

| Field | Backend B1.A | Frontend/Supabase B1.B |
|---|---|---|
| Application/repository | `IrakliGLD/langchain_railway` | `IrakliGLD/EnaiDashboard` |
| Branch and full Git SHA | `refactor/review-phase-fixes`; `dc0620538b949ae6e2879b9b73d7956ddd04d5e2` | `main`; `55b15f342796df03dafe29ce77e6c19226cf9fa8` |
| Candidate purpose | Immutable backend image/runtime/manifest identity | Immutable browser artifact and Edge source/version identity |
| Local verification | `1,685` full tests; `173` focused; Ruff; workflow YAML | `464` frontend tests; ESLint; production audit 0; deterministic build; Deno format/lint/check; `21` Edge tests |
| Generated identity | Runtime/image identity is the candidate SHA; v2 manifest emitted by protected workflow | Edge aggregate `973efd2764f9ab31d35789a7cc17edad9ac8dc5c5da9679344cda3fceb2fddcc`; browser aggregate is build-config dependent and emitted by the release workflow |
| CI/release run | **PENDING** | **PENDING** |
| Railway/Supabase deployment ID | **PENDING** | **PENDING** |
| Operator/approver/evidence archive | Irakli / Irakli; protected archive selection pending | Irakli / Irakli; protected archive selection pending |
| Rollback artifact/deployment | **PENDING** | **PENDING** |
| Promotion decision | Blocked before B3 | Blocked before B3 |

## 11. B2 append-only implementation candidates (supersede §10 for promotion)

| Field | Backend B2 | Frontend/Supabase B2.B |
|---|---|---|
| Application/repository | `IrakliGLD/langchain_railway` | `IrakliGLD/EnaiDashboard` |
| Branch and full Git SHA | `refactor/review-phase-fixes`; `b628f7881175fc12e47da0f87570411d65c0e789` | `main`; `ae9b68f9779a4a2c22a0b0c14307f0eb837ae231` (unchanged by B2 — no frontend code change was required) |
| Candidate purpose | F10-SEC-01 closure: dependency closure audits **zero advisories, zero waivers** (94 → 0); hashed 68-pin lock installed with `--require-hashes`; CI advisory + lock-freshness gates | B2.B integration attestation of the unchanged frontend against the remediated backend |
| Local verification | Full suite 1,710 green incl. `tests/security`; Ruff; exact CI coverage commands green | 465/0 tests; ESLint; contract verify (no schema change); edge manifest `973efd27…`; identity-bearing build + artifact verify; prod npm audit 0 |
| CI run/conclusion | [`29678536060`](https://github.com/IrakliGLD/langchain_railway/actions/runs/29678536060) **success — every step green, including the first-ever pass of the full-test/coverage-floor step** | Owner attestation pending (repository CI is private; pinned Deno v2.1.4 `edge:verify` runs on every push) |
| Release-evidence run | **PENDING** — operator runs `Backend release evidence` with this exact SHA | **PENDING** — `Frontend release evidence` at this exact SHA |
| Railway/Supabase deployment ID | **PENDING** (B3) | **PENDING** (B3) |
| Rollback artifact/deployment | **PENDING** (record previous deployment before promoting) | **PENDING** |
| Promotion decision | Ready for B3 preflight once release-evidence artifacts are recorded | Ready for B3 preflight |

Ledger commits after `b628f7881175fc12e47da0f87570411d65c0e789` are documentation-only and are explicitly recorded as such per §B0-4; they do not create a new candidate.

**B3.A promotion progress (2026-07-19):** PR #126 merged the candidate into `main`; the promotion identity is the merge commit **`699b703eca38e2b9fb34d65a2db622499a8f1b0b`** (content-identical to `2ce375d2…`). CI on `main` at that SHA: run [`29686008664`](https://github.com/IrakliGLD/langchain_railway/actions/runs/29686008664) **success** — the first green `main` CI in this repository. `Backend release evidence` dispatched by Irakli against the protected `production` environment at that exact SHA: run [`29686830193`](https://github.com/IrakliGLD/langchain_railway/actions/runs/29686830193) **success**, immutable artifact `backend-production-699b703eca38e2b9fb34d65a2db622499a8f1b0b` (178,563,880 bytes: image archive, SBOM, pip-audit JSON, release manifest, checksums) — retain beyond the 30-day artifact window per §5. Public `/healthz` and `/readyz` returned 200 at recording time.

**Railway deployment attestation (2026-07-19, read from the Railway dashboard):**

| Field | Value |
|---|---|
| Project | `Enai` — `b15aea53-f966-462b-8de3-9d3f744c432d` |
| Service | `enerbot` — `c274d6d3-88af-494f-b38d-2c286409ac0e` |
| Environment | `production` — `475e29b6-2565-4bde-9ca4-b8144ca643ed` |
| Active deployment | `79b9fb13` (short) — status **ACTIVE, Deployment successful**, started 2026-07-19 15:54 GMT+4 |
| Deployed source | **Merge pull request #126** (= merge commit `699b703eca38e2b9fb34d65a2db622499a8f1b0b`), repo `IrakliGLD/langchain_railway`, branch `main` |
| Build model | Railway Git-triggered Docker build (auto-deploy on push to `main` enabled). Source SHA identity is proven; the Railway image digest is separately built and therefore differs from the attested release-evidence image — not a byte-identical promotion, per plan §B1.A-6/-18. |
| Replicas / region | **1 Replica**, EU West (Amsterdam). Multi-region/horizontal scaling is plan-gated (Pro-only) and not enabled → one-replica constraint satisfied. |
| Port | 3000 (matches Dockerfile `EXPOSE 3000`) |
| Rollback target | Previous deployment = **Merge pull request #125** (= `bd0a91f897…`), now `REMOVED` in history; redeploy it to roll back. |
| Startup evidence | Deploy logs: "Application startup complete", "Uvicorn running on 0.0.0.0:3000", "Schema reflection complete: 14 views", `/healthz` + `/readyz` 200. |

**B3.A identity verification — PASSED (2026-07-19).** Protected `GET /versionz` (auth via `ENAI_EVALUATE_SECRET`/`X-App-Key`) returned `git_sha = 2f2a31053dfa391fbb0958ae858141c6f3e26ff9`, `application_version 20.0`, `schema_version backend-release-identity-v1`. That SHA is the **current `main` head** (merge #135), which auto-deployed after #126; it is `699b703…` **+ one documentation-only ledger commit** (`git diff 699b703..2f2a3105` = 30 lines in this ledger file, zero code). Backend CI green at `2f2a3105` (run [`29689035560`](https://github.com/IrakliGLD/langchain_railway/actions/runs/29689035560)). The returned SHA equals the deployed `main` head and is content-identical to the tested candidate → F10-REL-02 identity requirement satisfied for the backend. **Deployed backend identity of record: `2f2a31053dfa391fbb0958ae858141c6f3e26ff9`.** (A 404 would have meant the endpoint was absent; the earlier 401 confirmed the endpoint exists and simply rejected a wrong key.)

**B3.B frontend/Edge live state (2026-07-19, curl-verified against production):**

| Surface | Evidence |
|---|---|
| Browser app | Railway project `EnaiDashboard`, service `EnaiDashboard` (dashboard.galdava.com), node@20.19.1, EU West, **1 Replica**, auto-deploy on push to `main`. ACTIVE deployment = commit **`ac87530185fd9387117b2746d13538ac6bad1c1b`** ("Add B5.B … registry"), Deployment successful. `dashboard.galdava.com` → HTTP 200. |
| Edge functions | Live `X-Enai-Edge-Source` = **`973efd2764f9ab31d35789a7cc17edad9ac8dc5c5da9679344cda3fceb2fddcc`** (equals the current repo manifest — post-B1.B source), `X-Enai-Edge-Version` = **`ae9b68f9779a4a2c22a0b0c14307f0eb837ae231`** (full 40-char SHA). `healthcheck` body: `status ok, check_count 6, failed_check_count 0`. The audit's stale `healthcheck.version=3244ed1` is resolved. |

**Identity reconciliation:** the tested frontend candidate is `ae9b68f9…` (B2.B/B4.B evidence); Edge is deployed at exactly that SHA. The browser app auto-deployed to `ac87530…`, which is `ae9b68f9…` **+ 3 documentation-only commits** (`22c3f65`, `75e22b0`, `ac87530` — all touch only `docs/active/*.md`). The built browser artifact and the Edge source are unchanged (Vite builds `src/`, the Edge manifest hashes `supabase/functions/` — neither includes `docs/`), so the two SHAs are content-identical for the deployed artifact; only the embedded version label differs. Optional B3.B polish to report one SHA everywhere: re-run **Deploy Supabase edge functions** at `ac87530…` (same source digest `973efd27…`, version label updates to `ac87530…`).

Remaining for B3.B closure: **Frontend release evidence** + **Post Deploy Smoke** workflow dispatches at the chosen frontend SHA, and their run IDs recorded here.

**B4.B disposable-database regression — GREEN, and it caught a real bug (2026-07-19).** A dedicated disposable Supabase test project (`ufosbrdhjrkaagjaltno`, region `eu-central-1`, **never** production `qvmqmmcglqmhachqaezt`) was provisioned, the committed baseline applied (76 files, baseline-only — patches are folded into the baseline), and `TEST_DATABASE_URL` pointed at it. Frontend CI `Run database regression tests` (`npm run test:db`, 11 SQL regressions covering the F10-E2E-03 database surface) is now green — CI run #270 at `ac87530`. **The regression surfaced a latent production defect** in `migrate_chat_history_jsonb_v1`: it updated `chart_data` and `chart_metadata` in two separate `UPDATE`s, so a row with both fields stored as legacy JSON strings tripped the sibling `NOT VALID` shape check on the first update — the real P6.B migration would have failed on any such production row. Fixed in frontend commit `a4c24a2fa2f4056e8995010d119f6f965759ab10` (single atomic UPDATE per row; patch regenerated; `db:p6b:verify` green). This is exactly the class of latent defect the F10-E2E-03 live-database requirement exists to catch. **Production follow-up:** if P6.B was already applied to production with the two-UPDATE function, re-apply the corrected `migrate_chat_history_jsonb_v1` (a `create or replace`) before running/continuing the migration; ties into the B5.B P6.B compatibility-removal item.

**Superseding backend candidate (2026-07-19):** the B5.A expired-flag removal (`a99e51cc9c7ff7824879dab3edf9affeffd6b4f0`, code) and compatibility registry advance the candidate to `refactor/review-phase-fixes` @ **`2ce375d29b060d512bfa746035ec38db508f3d8e`** — CI run [`29679422065`](https://github.com/IrakliGLD/langchain_railway/actions/runs/29679422065) **success, all gates green**. No promotion evidence existed for the prior candidate, so nothing is invalidated; all B2/B4.A repository evidence carries forward (dependency closure unchanged — the removal touched no pins; zero advisories; no public schema change, contract drift gates green). B3 promotion should use this SHA. Ledger commits after it are again documentation-only unless stated otherwise.
