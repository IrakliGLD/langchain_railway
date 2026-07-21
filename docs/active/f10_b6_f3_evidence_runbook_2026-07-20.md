# F10 B6 — Phase F3 Integrated Evidence Runbook — 2026-07-20

Gathers every closure-grade evidence artifact **at the single frozen identity**
from Phase F2. Assistant drafts this runbook and verifies/records each run;
operator dispatches the workflows and reads the Railway image digest.

## 0. PRECONDITION — hard gate (F2 freeze must be live)

F3 evidence is **only valid at the frozen identity**. Re-verify all three
surfaces before running anything below; if any reports a non-frozen SHA, **stop**
— you would just re-capture the REL-02 drift.

```bash
# browser (expect fc44fd40946bb0772ab4f178ac376196bec21498)
curl -fsS https://dashboard.galdava.com/release-manifest.json | grep app_version
# edge (expect fc44fd4… ; the deploy workflow already checks all nine)
curl -fsS https://qvmqmmcglqmhachqaezt.supabase.co/functions/v1/healthcheck
# backend (expect the F2.a merge SHA)
git fetch origin && git rev-parse --short origin/main
curl -H "X-App-Key: <evaluate-admin secret>" https://enai.galdava.com/versionz   # git_sha
```

**Historical gate status as of 2026-07-20: NOT MET.** That precondition was
subsequently satisfied and superseded by the final identities recorded below.

## 1. Frozen identities under test

- **Frontend / browser / Edge** — `fc44fd40946bb0772ab4f178ac376196bec21498`
- **Backend source** — `0684dc172eb2bb10a17a2e80941a6940b0882f2d`
- **Railway deployment** — `e2e73849-c47d-4f4f-8073-76edd2e0df95`
- **Railway runtime image manifest** —
  `sha256:319f6774f8197acd88941abae1b81a57bb10d19d98fa28a82e9d5b63c3b5336b`

## 2. Evidence items (all at the frozen identity)

| # | Item | How (all `workflow_dispatch`) | What it proves | Gate |
|---|---|---|---|---|
| 1 | **Frontend release evidence** | frontend Actions → *Build frontend release evidence* · `git_ref=fc44fd40946bb0772ab4f178ac376196bec21498` · `environment=production` | immutable artifact (`artifact:verify`) + prod dep audit + CycloneDX SBOM at the frozen SHA; uploads `frontend-release-production-<sha>` | REL-02 / E2E-03 |
| 2 | **Backend release evidence** | backend Actions → *Backend release evidence* · `git_ref=<merge SHA>` · `environment=production` | reproducible image whose embedded **and** OCI-labeled revision == SHA, non-root runtime, excluded `.git/.env/tests/docs`, current lock, SBOM, `pip-audit`, release manifest; uploads `backend-production-<sha>` | REL-02 / E2E-03 |
| 3 | **Railway image-digest binding** | operator reads Railway → Deployments for the F2.a deploy | binds the **deployed runtime** image digest to the source SHA + records the rollback-target deployment ID | REL-02 |
| 4 | **Authenticated Post Deploy Smoke** | frontend Actions → *Post Deploy Smoke* · `environment=production` (dispatch on `main`, which is `fc44fd4`) | live app serves `app_version==fc44fd4` + matching artifact `sha256`, and the authenticated browser→Edge path works (`SMOKE_REQUIRE_AUTH=true`, `SMOKE_EXPECT_VERSION=fc44fd4`) | E2E-03 |
| 5 | **Live Browser Proof + axe (all three)** | frontend Actions → *Live Browser Proof* | authenticated dashboard + protected-RPC pipeline with no runtime errors (`test:e2e:live`); axe login/public + dashboard/chat + **admin** (`test:e2e:a11y` with `SMOKE_ADMIN_*`) | E2E-03 / A11Y-07 |
| 6 | **Full disposable-DB regression** | re-run exactly as B4.B (`f10_b4b_*` / `d218174`) against a throwaway Postgres, at the merge SHA — **never** against production `qvmqmmcglqmhachqaezt` | schema + migrations + RLS + SECURITY DEFINER RPCs regress clean on a disposable DB | E2E-03 |
| 7 | **`/versionz`** | `curl -H "X-App-Key: <secret>" https://enai.galdava.com/versionz` | `git_sha == <merge SHA>` (deployed backend identity) | REL-02 |
| 8 | **Manual a11y checklist (F1.5)** | authenticated session at `fc44fd4`: visible-focus ring, AT name+role on tabs/chart/chat-input/toast, live-region structure, responsive/touch at 375 & 768 px | records the honest observations (browser+AT versions, pages, viewport/zoom, SHA) the B4 doc marked PENDING | A11Y-07 |

## 3. Assistant verification role (after each run)

Re-read the live identities, confirm the run/artifact targeted `fc44fd4`
(frontend) or final main `0684dc1` (backend), and record run IDs + artifact names into the
F3 evidence table below. **Any surface reporting a non-frozen SHA halts F3** and
re-opens F2. The **final identities are backend `0684dc1` / frontend `fc44fd4`**
(the pre-freeze `d52a97e` was superseded by the F1.4-follow-on edge-manifest fix;
see [`f10_b6_f2_freeze_evidence`](./f10_b6_f2_freeze_evidence_2026-07-20.md)).

Results (2026-07-20). The two release-evidence workflows and the edge deploy
each **reject mutable refs and assert `checkout HEAD == git_ref`**, so a green
run is itself proof the artifact was built at the exact frozen SHA.

| # | Item | Run / artifact ID | Target SHA | Status |
|---|---|---|---|---|
| 1 | Frontend release evidence | `frontend-release-evidence` run **#2** (artifact `frontend-release-production-<fc44fd4>`) | `fc44fd4` | ✅ green (44s) |
| 2 | Backend release evidence | run **`29812847819`** / job `88577462455`; artifact `8488198995`, `backend-production-0684dc172eb2bb10a17a2e80941a6940b0882f2d`, artifact digest `sha256:ab884b891de35f688ceef49644cde6190de2705c00c5f512acf4336a4ebfbd12` | `0684dc1` | ✅ green — exact checkout, embedded + OCI-labeled identity, non-root, lock, SBOM, pip-audit |
| 3 | Railway runtime image digest | deployment `e2e73849-c47d-4f4f-8073-76edd2e0df95`; manifest `sha256:319f6774f8197acd88941abae1b81a57bb10d19d98fa28a82e9d5b63c3b5336b` | `0684dc1` | ✅ active, source/embedded SHA bound, healthcheck green; rollback target `feff19e0-69d6-42e9-818a-757d76090e2e` |
| 4 | Post Deploy Smoke | run **`29812892495`** / #247, job `88577596025` | `fc44fd4` | ✅ green (43s); exact artifact, healthcheck, authenticated `/is-admin` |
| 5 | Live Browser Proof + axe ×3 | run **#8** | `fc44fd4` | ✅ green (1m19s) |
| 6 | Disposable-DB regression | frontend CI run **`29701996275`** / #271, job `88232468171`; `Run database regression tests` green (54s) | `a4c24a2`, byte-identical DB trees at `fc44fd4` | ✅ post-fix immutable run + continuity proof recorded in §7 |
| 7 | `/versionz` | protected request from Railway Console on active deployment, using the container's evaluate-admin secret without displaying it | `0684dc1` | ✅ `application_version=20.0`, `git_sha=0684dc172eb2bb10a17a2e80941a6940b0882f2d`, schema `backend-release-identity-v1` |
| 8 | Manual a11y checklist | programmatic authenticated pass (§6) + SR-listen disposition | `fc44fd4` | ✅ structural pass green; SR-listen **dispositioned** (`F10-B6-A11Y-01`, approved 2026-07-21) |

## 4. Exit

All eight items green at the **one** frozen identity, no surface reporting a
non-frozen SHA, and the table filled with run IDs + the Railway image digest.
Then Phase F4 (finalize the §8 load waiver with the now-known SHAs; rehearse
rollback) and Phase F5 (re-run the B6 closure matrix against the frozen
immutable identities).

## 5. Phase F4 — Load waiver + rollback rehearsal

### 5.1 §8 load waiver — APPROVED

[`f10_b6_load_waiver_2026-07-20.md`](./f10_b6_load_waiver_2026-07-20.md),
waiver `F10-B6-LOAD-01`, **approved by Irakli 2026-07-21**, expiring 2026-08-20.
Final SHAs filled (backend `0684dc1` + Railway deployment
`e2e73849-c47d-4f4f-8073-76edd2e0df95` + runtime digest
`sha256:319f6774…c3b5336b`, frontend `fc44fd4`); five compensating controls;
remediation ticket to run the envelope
before expiry.

### 5.2 Rollback rehearsal — PASS (2026-07-21)

Rehearsed live on the production backend (single replica, EU West) via Railway's
**Rollback** action:

| Step | Deployment | Result |
|---|---|---|
| Baseline | `65cf93b` (deployment `ffc9ec32`) | `healthz: ok`, `readyz: ready`+schema |
| Roll **back** to #135 | `2f2a310` (deployment `4a7bdbc7`) | ACTIVE in ~49s; `healthz: ok`, `readyz: ready`+schema (confirmed twice) |
| Roll **forward** to #136 | `65cf93b` (redeploy) | ACTIVE; `healthz: ok`, `readyz: ready`+schema |

Rollback restores build **and** variables; net-neutral here because `#136` was
the active deployment (no newer var-change deploy existed), so the roll-forward
restored the exact current variable snapshot. Health was continuous throughout
(the previous deployment serves until the replacement passes health checks).
**Rollback path validated end-to-end in both directions; the frozen `65cf93b`
identity is restored and healthy.** Rollback target of record: `2f2a310` (#135,
deployment `4a7bdbc7`).

**Phase F4 COMPLETE.**

## 6. Item 8 — programmatic authenticated a11y pass (2026-07-21)

Run through the in-app browser on an authenticated **admin** session at
`dashboard.galdava.com` (frozen `fc44fd4`), inspecting accessibility *structure*
only (no personal/admin data captured). Complements the already-green automated
axe scans (Live Browser Proof #8: login/public + dashboard/chat + admin, WCAG
2a/2aa/21aa/22aa) with the manual-checklist items axe can't cover.

| Surface | Widths | Horiz. overflow | Unnamed controls | Targets < 24px | Structure notes |
|---|---|---|---|---|---|
| Dashboard home | 375, 730 | none | **0 / 34** | **0** | 12 role=tab; no h1–h3 (minor) |
| Chat | 730 | none | **0 / 12** | **0** | input named "Ask ENAI Analyst"; **`role="log" aria-live="polite"`** message region (responses are announced) |
| Admin | 730, 768 | none | **0 / 39** | **0** | table has **7 `th` + `caption`**; at 768 the wide table scrolls in its own `overflow-x` container (page never overflows) |

- **Accessible names:** 0 unnamed interactive controls on any surface.
- **Touch targets:** 0 controls under 24×24 px on any surface/width.
- **Responsive:** no horizontal page overflow at 375 / 730 / 768; wide admin
  table uses an internal scroll container.
- **Live regions:** chat message container is `role="log"`/`aria-live="polite"`
  → chat responses announce to screen readers.
- **Visible focus:** styled controls use `focus-visible:ring-2 ring-offset-2`
  (buttons) / `focus-visible:border-primary` (chat input); keyboard `Tab`
  produced a visible orange focus indicator (screenshot evidence). A few icon
  buttons (sidebar/account/theme) carry no custom focus class and fall back to
  the browser-default outline — still visible.

**Minor, non-blocking observations (not WCAG violations; axe passed):** dashboard
home has no `h1–h3` heading elements (limits SR heading navigation); admin
mutations have no `aria-live` status region.

**Screen-reader *listening* pass — DISPOSITIONED.** The one part not
programmatically verifiable (a human NVDA/VoiceOver listen confirming the
announcements are spoken correctly) is formally deferred for this release by
[`F10-B6-A11Y-01`](./f10_b6_a11y_disposition_2026-07-21.md) (approved by Irakli
2026-07-21): the dashboard is data-viz-heavy, the admin panel is admin-only, and
there is no current AT user base or WCAG-AA obligation; the aria-live chat is the
priority surface to attest first if revisited. The structural accessibility above
remains in force.

The disposable-DB stamp, runtime-digest binding, and final merge/re-verification
are complete. The SR listening item is governed by the approved, dated
`F10-B6-A11Y-01` disposition rather than represented as performed evidence.

## 7. Item 6 — disposable-DB carry-forward proof (E2E-03)

The disposable-DB regression pack (`npm run test:db` → `database/tests/*.sql`
against a DB built from `database/baseline`) was last run green as the **B4.B
disposable-database regression** (operator, throwaway Supabase test project
`ufosbrdhjrkaagjaltno` — **never** prod `qvmqmmcglqmhachqaezt`), which caught and
verified the fix for the P6.B `migrate_chat_history_jsonb_v1` double-UPDATE bug
(fixed in frontend `a4c24a2`).

**Immutable continuity binding that run to `fc44fd4`** — the two directories the
pack depends on are **byte-identical** (content-addressed git tree hashes) at
`a4c24a2` (the fix commit) and the frozen `fc44fd4`:

| Directory | tree hash @ `a4c24a2` | tree hash @ `fc44fd4` | |
|---|---|---|---|
| `database/baseline` | `3a7abc6e1ba1…` | `3a7abc6e1ba1…` | **identical** |
| `database/tests` | `b28e0e15ac83…` | `b28e0e15ac83…` | **identical** |

(`git diff a4c24a2..fc44fd4 -- database/baseline database/tests` is empty; full
SHAs `a4c24a2fa2f4056e8995010d119f6f965759ab10` and
`fc44fd40946bb0772ab4f178ac376196bec21498`.) Because the baseline the DB is built
from and the regression SQL are bit-for-bit the same, the green result holds at
`fc44fd4` with certainty.

This satisfies the audit's route (a) — the disposable-DB run identity (B4.B,
throwaway `ufosbrdhjrkaagjaltno`) + the exact source/hash continuity proof
binding it to `fc44fd4`. **E2E-03 is closed by this carry-forward.**

*Optional fresh confirmation:* `TEST_DATABASE_URL` is now a frontend GitHub
secret, so `frontend ci.yml`'s `test:db` step runs the pack against the throwaway
DB on any frontend CI trigger (next frontend push, or a no-merge frontend PR off
`fc44fd4`). Note the backend G5 merge does **not** run this — `test:db` is a
frontend job. A fresh stamp is belt-and-suspenders, not required for closure.

## 8. REL-02 — Railway runtime image digest (resolved 2026-07-21)

**Correction:** the earlier claim that Railway "has no separate Docker digest"
was wrong. Railway **does** expose the runtime image digest in the deployment's
**Build log** (`exporting to docker image format` step). So B1.A is satisfied by
**capturing** the digest — no control amendment is required.

The final post-merge deployment is the current release identity. The earlier
`65cf93b`/`feff19e0` deployment is retained only as the rehearsed rollback target;
its digest must not be read as the active runtime identity:

| Field | Value |
|---|---|
| Source SHA (embedded `ENAI_RELEASE_SHA`) | `0684dc172eb2bb10a17a2e80941a6940b0882f2d` |
| Railway deployment ID | `e2e73849-c47d-4f4f-8073-76edd2e0df95` |
| **Runtime image manifest digest** | `sha256:319f6774f8197acd88941abae1b81a57bb10d19d98fa28a82e9d5b63c3b5336b` |
| Runtime image config digest | `sha256:75cca0a003a757719a3dd9389d7015f7471877178c6a0c2c7f653317e7e0e396` |
| Base image | `python:3.11.15-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba` |
| Source snapshot | Railway Build Logs (final deployment) |
| OCI descriptor | manifest v1, size 4471, platform linux/amd64, created 2026-07-21T07:05:17Z |

The prior rollback deployment is recorded separately for rehearsal:
`65cf93b697e44f08cd03e782aac9949d2336135a` / `feff19e0-69d6-42e9-818a-757d76090e2e`
with prior manifest `sha256:b379121959aa418ea27b03f7bd4a130f54f8277972e3e8509ea2398c5ffbe4a3`.
The final digest is bound to backend release-evidence run `29812847819`, job
`88577462455`, artifact `8488198995` (artifact digest
`sha256:ab884b891de35f688ceef49644cde6190de2705c00c5f512acf4336a4ebfbd12`),
protected `/versionz` (`git_sha=0684dc1…`), the active Railway deployment, the
release SBOM/audit artifact, and the rollback target above.

**Note (B1.A line 118):** Railway rebuilds from Git rather than promoting the
`Backend release evidence` image, so each deploy of the same SHA gets its own
digest and it will differ from the CI-attested image ID. Both are recorded; no
byte-identical promotion is claimed. The final deployment record above is the
definitive closure digest, bound to `/versionz`, the Railway deployment, the
release manifest, the SBOM/audit artifact, and the rollback target.
