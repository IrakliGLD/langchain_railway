# F10 B6 — Phase F3 Integrated Evidence Runbook — 2026-07-20

Gathers every closure-grade evidence artifact **at the single frozen identity**
from Phase F2. Assistant drafts this runbook and verifies/records each run;
operator dispatches the workflows and reads the Railway image digest.

## 0. PRECONDITION — hard gate (F2 freeze must be live)

F3 evidence is **only valid at the frozen identity**. Re-verify all three
surfaces before running anything below; if any reports a non-frozen SHA, **stop**
— you would just re-capture the REL-02 drift.

```bash
# browser (expect d52a97e3a3de34754444f5eeb02c2f3a9f1f5509)
curl -fsS https://dashboard.galdava.com/release-manifest.json | grep app_version
# edge (expect d52a97e… ; the deploy workflow already checks all nine)
curl -fsS https://qvmqmmcglqmhachqaezt.supabase.co/functions/v1/healthcheck
# backend (expect the F2.a merge SHA)
git fetch origin && git rev-parse --short origin/main
curl -H "X-App-Key: <gateway secret>" https://enai.galdava.com/versionz   # git_sha
```

**Gate status as of 2026-07-20: NOT MET** — Edge live `193896745…` (`1938967`),
backend `origin/main` `2f2a310` (F1 not merged). Do not start F3 until F2.a
(backend merge) and F2.b (Edge realign to `d52a97e`) have landed.

## 1. Frozen identities under test

- **Frontend** — `d52a97e3a3de34754444f5eeb02c2f3a9f1f5509`
- **Backend** — `<F2.a merge SHA>` + Railway runtime image digest `<sha256:…>`

## 2. Evidence items (all at the frozen identity)

| # | Item | How (all `workflow_dispatch`) | What it proves | Gate |
|---|---|---|---|---|
| 1 | **Frontend release evidence** | frontend Actions → *Build frontend release evidence* · `git_ref=d52a97e3a3de34754444f5eeb02c2f3a9f1f5509` · `environment=production` | immutable artifact (`artifact:verify`) + prod dep audit + CycloneDX SBOM at the frozen SHA; uploads `frontend-release-production-<sha>` | REL-02 / E2E-03 |
| 2 | **Backend release evidence** | backend Actions → *Backend release evidence* · `git_ref=<merge SHA>` · `environment=production` | reproducible image whose embedded **and** OCI-labeled revision == SHA, non-root runtime, excluded `.git/.env/tests/docs`, current lock, SBOM, `pip-audit`, release manifest; uploads `backend-production-<sha>` | REL-02 / E2E-03 |
| 3 | **Railway image-digest binding** | operator reads Railway → Deployments for the F2.a deploy | binds the **deployed runtime** image digest to the source SHA + records the rollback-target deployment ID | REL-02 |
| 4 | **Authenticated Post Deploy Smoke** | frontend Actions → *Post Deploy Smoke* · `environment=production` (dispatch on `main`, which is `d52a97e`) | live app serves `app_version==d52a97e` + matching artifact `sha256`, and the authenticated browser→Edge→backend path works (`SMOKE_REQUIRE_AUTH=true`, `SMOKE_EXPECT_VERSION=d52a97e`) | E2E-03 |
| 5 | **Live Browser Proof + axe (all three)** | frontend Actions → *Live Browser Proof* | authenticated dashboard + protected-RPC pipeline with no runtime errors (`test:e2e:live`); axe login/public + dashboard/chat + **admin** (`test:e2e:a11y` with `SMOKE_ADMIN_*`) | E2E-03 / A11Y-07 |
| 6 | **Full disposable-DB regression** | re-run exactly as B4.B (`f10_b4b_*` / `d218174`) against a throwaway Postgres, at the merge SHA — **never** against production `qvmqmmcglqmhachqaezt` | schema + migrations + RLS + SECURITY DEFINER RPCs regress clean on a disposable DB | E2E-03 |
| 7 | **`/versionz`** | `curl -H "X-App-Key: <secret>" https://enai.galdava.com/versionz` | `git_sha == <merge SHA>` (deployed backend identity) | REL-02 |
| 8 | **Manual a11y checklist (F1.5)** | authenticated session at `d52a97e`: visible-focus ring, NVDA name+role on tabs/chart/chat-input/toast, live-region announce, responsive/touch at 375 & 768 px | records the honest observations (browser+AT versions, pages, viewport/zoom, SHA) the B4 doc marked PENDING | A11Y-07 |

## 3. Assistant verification role (after each run)

Re-read the live identities, confirm the run/artifact targeted `fc44fd4`
(frontend) or `65cf93b` (backend), and record run IDs + artifact names into the
F3 evidence table below. **Any surface reporting a non-frozen SHA halts F3** and
re-opens F2. The **frozen identities are backend `65cf93b` / frontend `fc44fd4`**
(the pre-freeze `d52a97e` was superseded by the F1.4-follow-on edge-manifest fix;
see [`f10_b6_f2_freeze_evidence`](./f10_b6_f2_freeze_evidence_2026-07-20.md)).

Results (2026-07-20). The two release-evidence workflows and the edge deploy
each **reject mutable refs and assert `checkout HEAD == git_ref`**, so a green
run is itself proof the artifact was built at the exact frozen SHA.

| # | Item | Run / artifact ID | Target SHA | Status |
|---|---|---|---|---|
| 1 | Frontend release evidence | `frontend-release-evidence` run **#2** (artifact `frontend-release-production-<fc44fd4>`) | `fc44fd4` | ✅ green (44s) |
| 2 | Backend release evidence | run **`29761159168`** (artifact `backend-production-<65cf93b>`) | `65cf93b` | ✅ green — embedded **and** OCI-labeled revision == SHA, non-root, SBOM, pip-audit |
| 3 | Railway runtime image digest | current deploy `feff19e0` digest `sha256:b379121959…ffbe4a3` — see §8 (final digest captured at G5) | `65cf93b` | ✅ digest exposed + captured (REL-02 resolved; no amendment needed); final-deploy digest recorded at G5 |
| 4 | Post Deploy Smoke | run **#246** | `fc44fd4` | ✅ green (47s) |
| 5 | Live Browser Proof + axe ×3 | run **#8** | `fc44fd4` | ✅ green (1m19s) |
| 6 | Disposable-DB regression | carry-forward proof — see §7 (fresh CI stamp at G5 merge) | `fc44fd4` (via tree-identity to `a4c24a2`) | ✅ carry-forward proven; fresh CI `test:db` stamp lands at G5 |
| 7 | `/versionz` | `dashboard.galdava.com/versionz` (`X-App-Key`) | `65cf93b` | ✅ `git_sha == 65cf93b697…` |
| 8 | Manual a11y checklist | programmatic authenticated pass (2026-07-21, browser pane) — see §6 | `fc44fd4` | ⚠️ partial — structural pass green; SR-listen attestation remains operator-residual |

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
Frozen SHAs filled (backend `65cf93b` + Railway deployment `ffc9ec32`, frontend
`fc44fd4`); five compensating controls; remediation ticket to run the envelope
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

**Operator-residual (the only part not programmatically verifiable):** an actual
screen-reader *listening* pass (NVDA/VoiceOver) to confirm the announcements are
spoken correctly. The underlying structure (names, roles, `aria-live`) is
correct, so this is a confirmation formality.

Remaining (per the 2026-07-21 re-audit / closure-completion plan): the
disposable-DB fresh CI stamp (G5), the SR-listen attestation (G4), and the B1.A
runtime-digest control (G3), then the final merge + re-verify (G5) and re-audit
(G6).

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

**Fresh immutable CI stamp (definitive):** `TEST_DATABASE_URL` is now a frontend
GitHub secret, so `ci.yml`'s `test:db` step runs the pack against the throwaway
DB automatically. The **G5 final-merge CI run on `main`** (final SHA, identical
`database/` trees) produces the fresh immutable `test:db` run ID at the final
identity — recorded there to fully close E2E-03.

## 8. REL-02 — Railway runtime image digest (resolved 2026-07-21)

**Correction:** the earlier claim that Railway "has no separate Docker digest"
was wrong. Railway **does** expose the runtime image digest in the deployment's
**Build log** (`exporting to docker image format` step). So B1.A is satisfied by
**capturing** the digest — no control amendment is required.

Captured for the current backend `65cf93b` deployment (`feff19e0`, EU West, 1
replica; roll-forward redeploy from the F4 rollback rehearsal):

| Field | Value |
|---|---|
| Source SHA (embedded `ENAI_RELEASE_SHA`) | `65cf93b697e44f08cd03e782aac9949d2336135a` |
| Railway deployment ID | `feff19e0` |
| **Runtime image manifest digest** | `sha256:b379121959aa418ea27b03f7bd4a130f54f8277972e3e8509ea2398c5ffbe4a3` |
| Runtime image config digest | `sha256:ea41c441200d6419bbef863aea5ed6a902ff1bcd9194061c8da3c90dbb0e71fb` |
| Base image | `python:3.11.15-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba` |
| Source snapshot | `sha256:4aaa6f96c9edc944cf9acb329ef18559f8b8557e840378dd91f6dcf263ea915a` |
| OCI descriptor | manifest v1, size 4471, platform linux/amd64, created 2026-07-20T20:27:40Z |

**Note (B1.A line 118):** Railway rebuilds from Git rather than promoting the
`Backend release evidence` image, so each deploy of the same SHA gets its own
digest and it will differ from the CI-attested image ID. Both are recorded; no
byte-identical promotion is claimed. The **definitive closure digest is captured
at the G5 final-merge deploy** (final SHA), bound there to `/versionz`, the
Railway deployment, the release manifest, the SBOM/audit artifact, and the
rollback target.
