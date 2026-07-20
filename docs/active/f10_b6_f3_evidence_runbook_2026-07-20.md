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

Re-read the live identities, confirm the run/artifact targeted `d52a97e` (or the
merge SHA), and record run IDs + artifact names into the F3 evidence table
below. **Any surface reporting a non-frozen SHA halts F3** and re-opens F2.

| # | Item | Run / artifact ID | Target SHA observed | Status |
|---|---|---|---|---|
| 1 | Frontend release evidence | _fill_ | _fill_ | ⏳ |
| 2 | Backend release evidence | _fill_ | _fill_ | ⏳ |
| 3 | Railway image digest | _fill_ | _fill_ | ⏳ |
| 4 | Post Deploy Smoke | _fill_ | _fill_ | ⏳ |
| 5 | Live Browser Proof + axe | _fill_ | _fill_ | ⏳ |
| 6 | Disposable-DB regression | _fill_ | _fill_ | ⏳ |
| 7 | `/versionz` | _fill_ | _fill_ | ⏳ |
| 8 | Manual a11y checklist | _fill_ | _fill_ | ⏳ |

## 4. Exit

All eight items green at the **one** frozen identity, no surface reporting a
non-frozen SHA, and the table filled with run IDs + the Railway image digest.
Then Phase F4 (finalize the §8 load waiver with the now-known SHAs; rehearse
rollback) and Phase F5 (re-run the B6 closure matrix against the frozen
immutable identities).
