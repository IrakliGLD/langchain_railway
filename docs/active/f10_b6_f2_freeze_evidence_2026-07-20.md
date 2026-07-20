# F10 B6 — Phase F2 Freeze & Aligned Deployment — 2026-07-20

Closes the REL-02 gap by freezing **one** frontend identity across browser + all
nine Edge functions and binding the backend identity to its Railway runtime
image. Assistant verifies/records; operator (Irakli) performs the merge, the
Edge workflow dispatch, and reads the Railway image digest.

## 1. Pre-freeze live identities (verified 2026-07-20, read-only)

| Surface | Source of truth | Live value | State |
|---|---|---|---|
| Browser | `https://dashboard.galdava.com/release-manifest.json` → `app_version` | `d52a97e3a3de34754444f5eeb02c2f3a9f1f5509` | ✅ **already at the F1 final frontend SHA** (Railway auto-deployed on push; manifest is self-consistent with `aggregate_sha256`) |
| Edge | production `functions/v1/healthcheck` → `X-Enai-Edge-Version` / body `version` | `193896745b5fd985da6618c366e750b73887dd75` | ❌ **DRIFT** — stuck at the frozen `1938967`; must realign to `d52a97e` |
| Backend | `/versionz` `git_sha` (gateway-gated) | `2f2a31053dfa391fbb0958ae858141c6f3e26ff9` | ⏳ pre-F1; awaits the F1 merge to `main` |

This is the live confirmation of **F10-REL-02**: the browser (`d52a97e`) and the
Edge tier (`1938967`) are not one release identity. F2 closes it by realigning
Edge (the browser already moved).

## 2. Final F1 candidate SHAs to freeze

- **Frontend** — `d52a97e3a3de34754444f5eeb02c2f3a9f1f5509` (`main`; browser
  already live at it; Edge to be realigned to it).
- **Backend** — branch `refactor/review-phase-fixes` tip
  `4d0771c053a5db63bd285ff13e3522582e3f25fe` (last **code** change `e42833b`;
  `4d0771c` is the audit-log doc). Merges to `main`; the merge SHA becomes the
  frozen backend identity.

## 3. Operator runbook

### F2.a — Backend freeze (merge → deploy → bind image)

1. Open the PR (no `gh` here; use the web compare):
   `https://github.com/IrakliGLD/langchain_railway/compare/main...refactor/review-phase-fixes`
   Wait for CI green, then merge to `main`.
2. Railway (`Enai`/`enerbot`) auto-deploys from `main`. In **Railway →
   Deployments**, record for the new deployment: **deployment ID**, **runtime
   image digest** (`sha256:…`), and the **previous** deployment ID (the
   **rollback target**).
3. Confirm identity: `curl -H "X-App-Key: <gateway secret>"
   https://enai.galdava.com/versionz` → `git_sha` must equal the new `main`
   HEAD SHA. Paste `git_sha` + image digest back here.

### F2.b — Frontend Edge freeze (one SHA across all nine)

1. **GitHub → Actions → "Deploy Supabase edge functions" → Run workflow**, with:
   - `git_ref` = `d52a97e3a3de34754444f5eeb02c2f3a9f1f5509` (full 40-char SHA;
     the workflow rejects anything that isn't a full lowercase SHA)
   - `environment` = `production`
2. The workflow is self-verifying: it checks out that immutable SHA, sets
   `APP_VERSION = <SHA>` (→ `X-Enai-Edge-Version`), deploys all nine functions,
   then **fails unless every one** of `admin-delete-user, admin-get-users,
   admin-update-limit, admin-update-status, chat-with-enerbot, get-data,
   get-public-usage, healthcheck, is-admin` reports `X-Enai-Edge-Version ==
   d52a97e…` **and** a matching `X-Enai-Edge-Source` hash. It uploads an
   `edge-deployment-production-<sha>` evidence artifact (90-day retention).

### F2.c — Browser

✅ Already live at `d52a97e` (§1). No action.

## 4. Post-freeze verification (assistant re-runs once F2.a/F2.b land)

| Surface | Expected (frozen) | Observed | Status |
|---|---|---|---|
| Browser `app_version` | `d52a97e3a3de34754444f5eeb02c2f3a9f1f5509` | `d52a97e3a3de34754444f5eeb02c2f3a9f1f5509` | ✅ pre-verified |
| Edge `X-Enai-Edge-Version` (all 9) | `d52a97e3a3de34754444f5eeb02c2f3a9f1f5509` | _fill after F2.b_ | ⏳ |
| Backend `/versionz` `git_sha` | `<main merge SHA>` | _fill after F2.a_ | ⏳ |
| Backend Railway image digest | recorded + bound to SHA + rollback target | _fill after F2.a_ | ⏳ |

**Exit:** browser + all nine Edge functions report the single frontend SHA
`d52a97e…`; the backend runs the merge SHA bound to its Railway image digest and
a named rollback target. Only then does Phase F3 gather integrated evidence at
this frozen identity.
