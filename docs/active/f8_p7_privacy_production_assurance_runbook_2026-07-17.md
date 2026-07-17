# F8 / P7 Privacy and Production Assurance Runbook

Date: 2026-07-17

## Scope and safety boundary

This runbook closes the locally implementable F8 controls for the independent analytics backend and frontend/Supabase applications. It does not couple their repositories or artifacts. Their only runtime integration remains the authenticated HTTP chat contract.

Never treat a local test as production evidence. Never revoke `PUBLIC`, Supabase-managed owner roles, or an application credential until the consumer map, backup, replacement grants, validation, and rollback SQL have been reviewed for the specific production project. This project has no staging database, so global privilege changes require a scheduled maintenance window.

## Decisions already supported by the implementation

- Production routing-fixture capture remains `ENAI_FIXTURE_CAPTURE_MODE=off` and `ENAI_FIXTURE_CAPTURE_SAMPLE_RATE=0`. Raw capture is rejected outside development/test.
- Operational actor/session identifiers use keyed HMAC fingerprints. `ENAI_LOG_HASH_KEY` is a Railway-only secret and must remain stable within the intended log-correlation period; it must not equal a provider, database, JWT, or gateway secret.
- Deleted-account UUIDs remain only in the protected, append-only `admin_audit_log` for 24 monthly buckets. They are not converted to a new pseudonym column in F8. This preserves accountable admin-action correlation without a risky live migration. They are excluded from ordinary subject exports and age out through the approved retention operation.
- Support exports are AES-256-GCM encrypted, stored outside repositories, randomly named, default to seven-day expiry, and may never exceed 30 days.
- Keep exactly one Railway replica and one Uvicorn worker. Multi-replica operation remains unsupported until shared session, continuity, rate-limit, replay, and idempotency repositories are implemented and tested together.

The Privacy/Security owner must approve the operational retention/access settings below and record the evidence link and date. Code cannot approve a vendor-console policy.

## Storage and processor inventory

| Store/processor | Expected data | Required retention/access decision | Export/deletion/incident control | Evidence state |
| --- | --- | --- | --- | --- |
| Railway application/deploy logs | request/span IDs, typed events, HMAC actor/session fingerprints, aggregate timing/error data | restrict project access to current operators; select the shortest operational retention compatible with incidents, target no more than 30 days | export only into access-controlled incident storage; run the synthetic canary scan; rotate exposed secrets and contain logging on a leak | local sanitizers/scanner exist; console retention and access approval open |
| Supabase product database | profiles/access, chat history, usage, admin audit, privacy operation records | 180-day chat; 18 monthly usage buckets; 24 monthly audit buckets; three-year privacy-operation records, as documented in the frontend policy | authenticated subject export/delete; service-role retention RPC; protected audit exception | policy/code exist; latest live retention dry-run/apply evidence open |
| Supabase Auth | account and authentication/session data | governed by active account and Supabase project settings | admin deletion plus product-data reconciliation | implemented; dashboard access review open |
| Supabase Edge logs | function/request/status/typed error metadata only | restrict project log access; record actual Supabase retention | scan the exported canary window; incident response as for Railway | local allow-list exists; deployed scan open |
| Supabase Storage | no application Storage API use found in the 2026-07-17 code inventory | confirm there are no manually created buckets/integrations before declaring unused | delete obsolete objects only after owner inventory | code inventory complete; dashboard confirmation open |
| Active AI provider (OpenAI, Gemini, or NVIDIA) | user question plus system/evidence context required to produce the answer | record the active provider, account/project, training setting, data-retention setting, region if applicable, DPA/contract owner, and access administrators; prefer no training and the shortest supported retention | provider export/delete/incident process and credential rotation must be recorded | implementation supports all three; deployed provider/account policy open |
| GitHub repositories/Actions | source, tests, sanitized build logs, SBOMs, release manifests/artifacts | protected environment/reviewer access; backend evidence currently 30 days, frontend evidence 90 days | never upload raw production data/secrets; remove exposed artifacts and rotate credentials through an incident | immutable release workflows exist; exact deployed-SHA runs open |
| Monitoring | in-process aggregate metrics; no external monitoring exporter was found in backend source | confirm Railway/Supabase dashboards and any console-added drain/integration; approve access/retention | disable unknown drains and investigate unexpected destinations | code inventory complete; control-plane inventory open |
| Support-export storage | encrypted subject export envelopes | approved operator group only; seven days default, maximum 30 | daily `privacy:purge-exports`; record delivery and deletion | tool/tests exist; legacy locations and owner approval open |
| Local routing fixtures | raw questions only in explicit development/test capture | local-only, minimum necessary, delete after reviewed fixture derivation; never commit raw capture | production values off/zero; inspect developer machines and archives | production guard exists; workstation/archive attestation open |

## Privacy-owner approval record

Do not mark this section complete without named approval.

```text
Privacy/Security owner:
Approval date:
Railway retention/access evidence:
Supabase database/Auth/Edge/Storage retention/access evidence:
Active AI provider and account settings evidence:
GitHub environment/artifact retention evidence:
Monitoring/drain inventory evidence:
Support-export storage and deletion evidence:
Local-fixture/legacy-export disposition evidence:
Incident procedure owner and contact:
Next review date:
```

## Synthetic log-canary procedure

Use only a non-production test identity. Submit one chat question containing all five repository-owned canaries below; none is a real credential or person:

```text
ENAI_LOG_CANARY_PROMPT_20260717_DO_NOT_RETAIN
enai-log-canary+20260717@example.invalid
017f7110-4a77-4f88-9c21-000000000001
Bearer eyJsb2dfY2FuYXJ5.eyJzeW50aGV0aWM.signature01
ENAI_LOG_CANARY_SQL_20260717_DO_NOT_RETAIN
```

Export the bounded request window from Railway and Supabase Edge into local access-controlled files. Do not paste log lines into tickets. From the backend repository run:

```powershell
python scripts/scan_privacy_log_canaries.py D:\secure\enai-canary\railway.log D:\secure\enai-canary\edge.log
```

Exit `0` and `"clean": true` are required. Exit `1` means a canary leaked. Exit `2` means the scan was invalid and is not evidence. The scanner reports labels/counts only, never matching log content. Preserve its JSON output with date, request ID, deployed backend SHA, Edge source hash, and operator. Delete raw log exports after the approved incident/evidence period.

On a leak: stop rollout, preserve an access-controlled copy, identify the emitting boundary, disable optional diagnostics/fixture capture, rotate any potentially exposed real secret, notify the incident owner, fix and redeploy, then repeat the canary.

## Legacy privacy export inventory and disposition

The 2026-07-17 working-tree inventory found no tracked, untracked, or ignored privacy export payload in the backend and no frontend privacy export payload (the tracked `.env.example` is a reviewed template). This does not prove Git history, CI artifacts, synced archives, backups, operator machines, or support storage are clean.

The Privacy owner must inventory those locations by file name/metadata without copying payloads into the repository. For each discovered item choose and record exactly one disposition:

1. delete because the fulfilment/incident retention period expired;
2. move once to approved encrypted, access-controlled storage with an owner and expiry;
3. retain as incident/legal evidence with a documented basis, access list, and review date.

Never return a raw export to a repository, CI artifact, synced folder, or ordinary backup.

## Read-only PUBLIC and inherited-privilege inventory

In the Supabase SQL editor run `scripts/inventory_public_privileges.sql`. It changes no state. Export every result grid and map each privilege to these consumers:

- `enai_api_readonly`: backend reads only the explicit public/knowledge relations in `least_privilege_api_role.sql`;
- `anon`/`authenticated`: browser access only through approved RLS-protected tables and RPCs;
- `service_role`: Edge/admin/privacy operations that are explicitly source-mapped;
- `supabase_auth_admin`, `supabase_storage_admin`, `authenticator`, and object owners: Supabase-managed behavior; do not revoke blindly;
- `PUBLIC`: must have a documented reason or become a candidate for removal;
- external/manual consumers: dashboards, SQL clients, scheduled jobs, BI tools, or integrations not visible in source.

For every candidate removal, prepare two reviewed scripts before the maintenance window:

```sql
-- change.sql: explicit replacement first, then the narrow obsolete revoke
begin;
-- grant <exact privilege> on <exact object> to <legitimate role>;
-- revoke <exact privilege> on <exact object> from public;
-- run same-transaction verification where possible
commit;

-- rollback.sql: exact inverse, captured before execution
begin;
-- grant <removed privilege> on <exact object> to public;
-- revoke <temporary replacement only if it did not pre-exist>;
commit;
```

Do not execute placeholders. Record the original ACL rows so rollback restores the prior state exactly. Back up before the window; pause admin/privacy mutations; test login, Auth refresh, browser reads, chat, admin listing/mutation, privacy dry-run, Edge health, backend `/readyz`, and the runtime-role denial probes; then resume traffic. If any managed consumer fails, rollback immediately.

Network restriction is a separate control. Inventory Railway outbound addresses and every Supabase/Auth/Storage/Edge/operator consumer first. Apply an allow-list only if the current Supabase/Railway plan supports stable addresses without breaking managed services. Record the prior rule set and an emergency console-access path.

Revoke only obsolete application credentials after replacement is active and logs show no use. Never revoke Supabase-managed owner credentials or delete the last emergency path.

## Backend artifact assurance

The protected `Backend release evidence` workflow accepts only a full 40-character SHA, builds the pinned Dockerfile, verifies non-root UID and excluded paths, creates a CycloneDX SBOM, runs `pip-audit`, writes the release manifest and checksums, and preserves the image archive.

Run it for the exact deployed backend SHA:

```powershell
gh workflow run backend-release-evidence.yml --ref main -f environment=production -f git_ref=<FULL_BACKEND_SHA>
```

Review/download the artifact and require:

- successful exact checkout identity;
- zero unapproved `pip-audit` findings (document any approved exception with package, CVE, exposure analysis, owner, expiry);
- SBOM present and reviewable;
- non-root UID and excluded-file check passed;
- `backend-release-manifest.json`, `backend-image.tar.gz`, and `SHA256SUMS` present;
- the SHA equals the Railway deployment source SHA.

Because Railway currently rebuilds from source, also record the Railway deployment ID, deployment source SHA, build image digest/ID if Railway exposes it, deployment time, region, and successful smoke. Do not claim byte-identical promotion unless the platform exposes and matches the image identity.

## Frontend artifact assurance

The independent frontend `Build frontend release evidence` workflow is now pinned to immutable official action commits. Run it with the full frontend SHA, promote the preserved `dist/` without rebuilding where the host supports that, and record `dist/release-manifest.json`, `frontend-git-sha.txt`, the production SBOM, host deployment ID, Supabase Edge source manifest/hash, and smoke results.

The backend workflow never reads frontend files and the frontend workflow never reads backend files.

## Production smoke and rollback rehearsal

Record for both applications:

```text
Environment:
Backend full SHA / Railway deployment ID / image identity:
Frontend full SHA / host deployment ID / release-manifest aggregate:
Edge full SHA / source aggregate:
Database runtime role:
Replica count / autoscaling disabled evidence:
Smoke date/operator:
- backend /healthz and /readyz
- login/token refresh/logout
- normal chat and persisted/reloaded charts
- malformed/timeout/quota/paused-user paths
- admin bounded list and one reversible mutation
- privacy dry-run
- log canary
Last-known-good deployment IDs:
Rollback rehearsal date/result:
```

Rehearsal may use platform redeploy of the last-known-good immutable deployment without changing production traffic, if the platform cannot dry-run a rollback. Verify that the artifact still exists and the operator has permission. A real rollback must restore the preserved backend/frontend/Edge identities together only where their contract compatibility requires it, then repeat health/readiness/auth/chat/admin/privacy smoke.

## F8 exit gate

F8/P7 is production-complete only when all of the following have evidence links and owners:

- privacy inventory and policy approval, legacy-export disposition, and clean deployed log canary;
- dedicated runtime role plus reviewed PUBLIC/inherited privileges and network decision;
- no obsolete broad application credential remains;
- exact deployed-SHA backend/frontend release evidence, reviewed SBOM/audits, non-root/exclusion proof, and artifact/deployment identities;
- exactly one Railway replica, autoscaling disabled, and one Uvicorn worker;
- complete production smoke and a preserved/rehearsed rollback.

Local implementation completion must not be used to check these production-only items.