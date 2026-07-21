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

## 4. Post-freeze verification — COMPLETE (2026-07-20)

**The frozen frontend identity moved `d52a97e` → `fc44fd4` during F2.** Running
the edge deploy at `d52a97e` failed `edge:verify`: the F1.4 `.gitattributes`
edit had staled the edge-source-manifest (which hashes `.gitattributes`).
Regenerating it produced frontend `main` `fc44fd40946bb0772ab4f178ac376196bec21498`,
the browser auto-redeployed to it, and Edge was realigned to it. The backend CI
was separately unblocked by regenerating the requirements lock (upstream
`langsmith 0.10.7` drift) before the PR #136 merge.

| Surface | Frozen identity | Observed | Status |
|---|---|---|---|
| Browser `app_version` | `fc44fd40946bb0772ab4f178ac376196bec21498` | same | ✅ live |
| Edge `X-Enai-Edge-Version` (all 9) | `fc44fd40946bb0772ab4f178ac376196bec21498` | same (deploy #20 self-verified all 9) | ✅ live |
| Backend `/versionz` `git_sha` | `65cf93b697e44f08cd03e782aac9949d2336135a` | same (via `dashboard.galdava.com/versionz`) | ✅ |
| Backend Railway deployment | `65cf93b` (Merge PR #136) | deployment `ffc9ec32` ACTIVE + healthy (`/readyz` 200) | ✅ |
| Backend rollback target | previous deploy | `2f2a310` (PR #135), REMOVED/redeployable | ✅ |
| Post Deploy Smoke | `fc44fd4` | run #246 green (47s) | ✅ |

Railway source-builds the backend (Nixpacks) and rebuilds per deploy, so there
is no stable Docker image digest in the captured console evidence. The source
is bound to `65cf93b` + Railway deployment id `ffc9ec32`, but that substitution
does **not** satisfy the unamended B1.A runtime-image-digest exit gate. Capture a
platform-issued immutable runtime digest after the final deploy, or formally
approve an amended control if Railway cannot expose one.

**Frontend/source freeze reached:** browser + all nine Edge functions report the
single frontend SHA `fc44fd4`; the backend runs `65cf93b` (deployment
`ffc9ec32`) with rollback target `2f2a310`. The B1.A runtime-digest control
remains open. Phase F3 evidence is gathered at these identities — see
[`f10_b6_f3_evidence_runbook`](./f10_b6_f3_evidence_runbook_2026-07-20.md) §3.
