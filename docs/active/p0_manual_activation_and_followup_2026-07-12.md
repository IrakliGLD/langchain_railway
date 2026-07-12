# P0 Manual Activation and Follow-up Runbook

**Date:** 2026-07-12
**Applies to:** backend repository `D:/Enaiapp/langchain_railway` and frontend/Supabase repository linked to `D:/export_enai`
**Purpose:** turn the independently committed P0 code into active deployed behavior without assuming access to Railway, Supabase, GitHub branch protection, production traffic, or the live database

P0 code commits are independent. Do not wait to commit or deploy one repository because the other repository or a live service is unavailable. The chat integration becomes active only after the compatible backend and `chat-with-enerbot` edge versions are both deployed.

Do not paste real secrets, database URLs, access tokens, user payloads, or privacy exports into tickets, shell history, Git, or this document. Use the approved secret manager and operator environment.

## 1. Record the release identities

Before deployment, record:

- backend Git commit SHA and Railway service/environment;
- frontend Git commit SHA and Railway service/environment;
- Supabase project reference and database migration state;
- the SHA-256 of every edge source file being deployed;
- operator, reviewer, date/time, change/incident ID, and rollback owner;
- current Railway replica count and the previously deployed backend/frontend/edge versions.

The frontend source files named below are temporary deployment snapshots. P1.3 must replace them with standard deployable `supabase/functions/<function>/index.ts` sources and immutable CI deployment.

## 2. Backend/Railway activation (P0.A)

### 2.1 Required production variables

Set these in the backend Railway service. Generate independent high-entropy values for the three ENAI secrets; do not reuse the Supabase service-role key.

```text
ENAI_DEPLOYMENT_ENV=production
ENAI_AUTH_MODE=gateway_only
ENAI_GATEWAY_SECRET=<random shared secret also stored as Supabase CHAT_BACKEND_SECRET>
ENAI_SESSION_SIGNING_SECRET=<different random signing secret>
ENAI_EVALUATE_SECRET=<different random admin/evaluation secret>
SUPABASE_DB_URL=<least-privileged runtime PostgreSQL URL>
MAX_REQUEST_BODY_BYTES=262144
ENABLE_METRICS_ENDPOINT=false
ENABLE_EVALUATE_ENDPOINT=false
ALLOW_EVALUATE_ENDPOINT=false
ENABLE_EVIDENCE_REANALYSIS=false
```

Also set the selected model provider:

- Gemini: `MODEL_TYPE=gemini` and `GOOGLE_API_KEY`;
- OpenAI: `MODEL_TYPE=openai` and `OPENAI_API_KEY`;
- NVIDIA: `MODEL_TYPE=nvidia` and `NVIDIA_API_KEY` (plus any intentional model/base URL/timeout overrides).

Important constraints:

- Do not set `ENAI_AUTH_MODE=gateway_and_bearer` outside `ENAI_DEPLOYMENT_ENV=test`; production now fails startup by design.
- `MAX_REQUEST_BODY_BYTES` must be between 262144 and 1048576. Keep the P0 production value at 262144 unless a separately reviewed contract change requires more.
- Keep `/metrics` and `/evaluate` disabled in production. `ENAI_EVALUATE_SECRET` is still required at startup even while those endpoints are disabled.
- Keep Railway at one backend replica/worker while session state is in memory. Do not scale horizontally before P5.6/P7.4 shared-state acceptance passes.
- `SUPABASE_DB_URL` should eventually use the dedicated least-privilege role required by P7.2. If that role is not ready, record the current identity as a manual risk; do not claim least-privilege attestation.

### 2.2 Deploy and verify the backend alone

1. Deploy the recorded backend commit to its Railway service. The checked-in start command is `python main.py`.
2. Confirm Railway reports one healthy replica and did not silently override the start command or health probe.
3. Verify liveness:

   ```powershell
   Invoke-RestMethod -Method Get -Uri "https://<backend-host>/healthz"
   ```

4. Verify readiness:

   ```powershell
   Invoke-WebRequest -Method Get -Uri "https://<backend-host>/readyz"
   ```

   Expect HTTP 200 only when the database is reachable and all required reflected relations/columns are present. HTTP 503 is an honest deployment/database/schema failure; investigate it rather than changing the probe to `/healthz`.

5. From an approved operator environment, make one minimal signed-gateway request with `X-App-Key: <ENAI_GATEWAY_SECRET>`. Do not expose that header in logs. Confirm missing/incorrect keys are rejected.
6. If operationally safe in staging, temporarily break a required startup setting or point a staging deployment at an invalid required dependency, confirm the process exits/restarts nonzero, then restore the setting. Do not run this destructive check in production.
7. Record readiness latency and database reflection cost. A live database was not available during local P0 verification, so this is manual evidence, not a code-commit blocker.

### 2.3 Backend rollback

- Roll Railway back to the previous known-good backend deployment.
- Preserve `ENAI_AUTH_MODE=gateway_only`; never restore `auto` or production direct-bearer access.
- If the edge is already pointed at an incompatible backend, disable chat or restore the prior edge/backend pair. Do not weaken authentication to make mixed versions communicate.

## 3. Frontend/Railway variables and deploy (P0.B browser half)

Set these build-time variables in the frontend Railway service:

```text
VITE_SUPABASE_URL=https://<project-ref>.supabase.co
VITE_SUPABASE_ANON_KEY=<Supabase anon key; never the service-role key>
VITE_ALLOW_SELF_SIGNUP=false
APP_VERSION=<frontend commit SHA>
VITE_APP_VERSION=<frontend commit SHA>
```

`VITE_*` values are embedded at build time, so changing them requires a rebuild/redeploy. Deploy the recorded frontend commit only after confirming it is from the real frontend Git repository, not from the outer export directory or an uncommitted mirror.

After deployment:

1. Open `/`, `/login`, `/public`, `/chat`, and `/admin` as appropriate.
2. Confirm no unexpected console errors, React warnings, CSP violations, or failed lazy chunks.
3. Confirm an active user can load protected routes and paused/removed users are rejected.
4. Confirm signup visibility matches `VITE_ALLOW_SELF_SIGNUP`.
5. Confirm the HTML `app-version` metadata matches the deployed frontend commit.

## 4. Supabase database activation (P0.B database half)

### 4.1 Choose the correct path

- Brand-new database: apply the complete `database/baseline` in its documented order. Do not also apply the patch as though it were a migration.
- Existing database created before `admin_delete_user_data_txn` was added: apply `database/patches/2026-03-12_admin_delete_user_support.sql` once through the approved migration process.
- Never run files under `database/tests` as migrations.

### 4.2 Existing-database patch procedure

1. Take the normal database backup/snapshot and record its restore command and owner.
2. Inspect the target schema and confirm `public.admin_audit_log` exists and its append-only protection trigger is enabled.
3. Apply the exact committed patch through the Supabase SQL editor or the approved migration runner:

   `database/patches/2026-03-12_admin_delete_user_support.sql`

4. Verify the function signature exists and only `service_role` can execute it:

   ```sql
   select p.oid::regprocedure as function_signature,
          has_function_privilege('anon', p.oid, 'execute') as anon_execute,
          has_function_privilege('authenticated', p.oid, 'execute') as authenticated_execute,
          has_function_privilege('service_role', p.oid, 'execute') as service_role_execute
   from pg_proc p
   join pg_namespace n on n.oid = p.pronamespace
   where n.nspname = 'public'
     and p.proname = 'admin_delete_user_data_txn';
   ```

   Expect `anon_execute=false`, `authenticated_execute=false`, and `service_role_execute=true` for the five-argument function.

5. On a dedicated test database only, set `TEST_DATABASE_URL` and run:

   ```powershell
   Set-Location D:\export_enai
   $env:TEST_DATABASE_URL="postgresql://<dedicated-test-db>"
   npm run test:db
   ```

6. Perform a staged delete of a throwaway non-admin user and confirm:

   - user-owned application rows are removed;
   - historical `admin_audit_log` actor/subject rows remain;
   - a new redacted success audit row is appended;
   - the audit-protection trigger remains enabled after success and failure;
   - no code path disables that trigger.

If the live/test database cannot be accessed, record these items as `Manual verification pending` and proceed with the repository commit. Do not assume the migration has been deployed merely because its SQL is committed.

## 5. Supabase edge activation (temporary P0 snapshot procedure)

### 5.1 Set project secrets

In Supabase Project Settings / Edge Functions secrets, set or verify:

```text
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_ANON_KEY=<anon key>
SUPABASE_SERVICE_ROLE_KEY=<service-role key>
CHAT_BACKEND_URL=https://<backend-host>/ask
CHAT_BACKEND_SECRET=<exact same value as backend ENAI_GATEWAY_SECRET>
ENABLE_GET_DATA_ENDPOINT=false
APP_VERSION=<frontend/edge release commit or recorded release ID>
```

Server-only values must never have a `VITE_` prefix and must never be placed in the browser Railway service. Keep `ENABLE_GET_DATA_ENDPOINT=false` unless an active-admin-only data export is explicitly intended and reviewed.

### 5.2 Deploy the changed functions

P0 changes these seven function snapshots:

| Supabase function | Repository snapshot |
|---|---|
| `admin-delete-user` | `edge_functions/admin-delete-user.txt` |
| `admin-get-users` | `edge_functions/admin-get-users.txt` |
| `admin-update-limit` | `edge_functions/admin-update-limit.txt` |
| `admin-update-status` | `edge_functions/admin-update-status.txt` |
| `is-admin` | `edge_functions/is-admin.txt` |
| `get-data` | `edge_functions/get-data.txt` |
| `chat-with-enerbot` | `edge_functions/chat-with-enerbot.txt` |

Each snapshot imports `./cors.ts`, so deploy the matching committed `edge_functions/cors.ts` beside `index.ts` for every function. Do not paste only `index.ts` and leave an unknown older CORS helper.

Until P1.3 creates deployable sources, use the Supabase Dashboard function editor:

1. Open the correct project and environment; verify the project reference twice.
2. Open the named function and save its current source/hash as the rollback artifact.
3. Replace the function's `index.ts` with the entire matching committed `.txt` snapshot.
4. Replace/add `cors.ts` with the committed `edge_functions/cors.ts`.
5. Deploy the function and wait for a successful build.
6. Record the local SHA-256, deployed revision/time, operator, and rollback revision.
7. Repeat for all seven functions. Do not deploy only the admin functions and omit the chat body/history contract.

PowerShell hash command:

```powershell
Get-FileHash D:\export_enai\edge_functions\chat-with-enerbot.txt -Algorithm SHA256
```

Do not invent an ad-hoc CLI copy directory and commit it as permanent architecture. P1.3 must create the standard layout, pin dependencies/lockfiles, add executable tests, and deploy immutable artifacts from CI.

### 5.3 Edge verification

Using dedicated test identities:

1. Active admin: all four admin mutations/list functions and `is-admin` retain expected access.
2. Paused admin: `admin-delete-user`, `admin-get-users`, `admin-update-limit`, `admin-update-status`, `is-admin`, and enabled `get-data` return 403.
3. Demoted admin: replaying the old token does not restore admin access.
4. Chat: an authenticated active user can send a short Georgian and a non-BMP query; the response preserves `X-Request-Id`.
5. Chat: a request over 32 KiB returns 413 before the backend/model runs.
6. Chat: blank/malformed JSON, invalid UTF-8, more than three history turns after edge loading, or oversized fields fail with the documented safe response.
7. Integration: `CHAT_BACKEND_SECRET`/`ENAI_GATEWAY_SECRET` mismatch fails closed; after correcting it, the normal request succeeds.
8. `get-data` remains disabled unless explicitly enabled; if enabled, only an active admin can use it.

## 6. Privacy export activation and existing-file decision

Two pre-existing raw export artifacts were observed under `D:/export_enai/privacy_exports`. Their contents were not inspected. The Privacy owner must decide whether each is incident evidence, fulfilment evidence, or disposable data. Do not delete, move, decrypt, or copy them until that decision is recorded.

For all new exports, pre-create approved storage outside the repository, synced folders, and ordinary project backups. On Windows:

```powershell
New-Item -ItemType Directory -Force -Path "D:\secure\enai-privacy-exports"
icacls "D:\secure\enai-privacy-exports" /inheritance:r /grant:r "$($env:USERNAME):(OI)(CI)F"
$key = [Security.Cryptography.RandomNumberGenerator]::GetBytes(32)
[Convert]::ToBase64String($key)
```

Put the generated base64 value in the approved secret store, separately from the export directory. Do not leave `$key` or the printed value in a transcript; close the session after storing it.

Set the operator environment:

```text
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role key>
PRIVACY_ACTOR_EMAIL=<operator identity>
PRIVACY_EXPORT_DIR=D:\secure\enai-privacy-exports
PRIVACY_EXPORT_ENCRYPTION_KEY=<base64 32-byte key from secret store>
PRIVACY_EXPORT_RETENTION_DAYS=7
```

Then run from the frontend repository:

```powershell
npm run privacy:check
npm run privacy:report
npm run privacy:export -- --user <approved-user-uuid>
npm run privacy:purge-exports
```

Only after reviewing the dry-run counts and receiving approval:

```powershell
$env:PRIVACY_APPLY_CONFIRM="apply"
npm run privacy:apply
```

Schedule `npm run privacy:purge-exports` at least daily, retain its operator/scheduler log, verify expired encrypted files disappear, and never decrypt into the repository, CI artifacts, email, or a synced directory.

## 7. Integrated post-deploy checks

Set the documented `SMOKE_*` variables and run:

```powershell
npm run smoke:prod
```

Also verify manually:

- frontend and edge `APP_VERSION` values identify the intended deployed revisions;
- a normal browser chat crosses frontend → Supabase edge → backend with one request ID;
- the backend rejects calls without the edge shared secret;
- paused users/admins and malformed/oversized requests fail before model or privileged database work;
- `/readyz` reports the live database/schema state honestly;
- audit rows survive deletion and mutation attempts remain denied;
- production dependency, branch-protection, deployed-hash, RLS/grant, replica-count, and least-privilege checks are recorded as Passed or Manual verification pending, never silently Passed.

## 8. Later-phase complementary steps required for behavior activation

| Later track | Missing complementary step | Operational consequence until complete |
|---|---|---|
| P1.B / P1.1 | Make the actual frontend Git repository the only build/deploy source and retire the ambiguous outer/mirror arrangement. | A correct change can still be committed in one tree while Railway builds another. Record the deployed repo/commit every release. |
| P1.B / P1.2 | Port the guarded auth state machine into the canonical deployed tree. | Rapid auth events, timeouts, and stale profile/admin responses can still cause nondeterministic UI state. |
| P1.B / P1.3 | Create `supabase/functions/<name>/index.ts`, shared modules, pinned dependency/lock inputs, executable tests, CI deploy, and deployed-source hash verification. | P0 edge fixes require manual copy/paste and can drift from Git. This is the first follow-up needed before P3 edge/security cutovers. |
| P2.A + P2.B | Publish and consume one versioned metric/unit registry before canonical frames are enforced. | Enabling H1 first could make currently cold incorrect unit conversions user-visible. Keep canonical enforcement off. |
| P3.A + P3.B | Add the actor-bound entitlement ledger, server persistence, dashboard leases, active-user/last-admin invariants, request identity, and safe error contract. | P0 contains direct bearer/admin bypasses but does not provide final transactional billing, persistence, or concurrency guarantees. |
| P4.A + P4.B | Complete evidence finalization, enforceable plans, evidence-sourced charts, honest terminal outcomes, and full re-analysis transitions. | Keep `ENABLE_EVIDENCE_REANALYSIS=false`; do not claim the documented canonical pipeline is universal. |
| P5.A + P5.B | Establish one deadline/retry owner/idempotency identity and externalize actor-bound session state before scaling. | Keep one backend replica and avoid adding browser/edge retry layers that can duplicate model work or charges. |
| P6.A + P6.B | Generate/version the shared API contract and migrate all consumers before removing older fields/versions. | Hand-maintained body/history/chart limits and DTOs can drift across backend, edge, persistence, and UI. |
| P7.A + P7.B | Prove least-privilege DB identity, immutable production artifacts, logging/privacy controls, dependency scans, and live topology. | Local tests cannot establish production grants, deployed hashes, secrets, or replica count. These remain explicit manual attestations. |
| P8.A + P8.B | Refactor behind characterization tests and complete the deferred Vite/esbuild major upgrade. | Structural debt and development-only advisory risk remain, but no activation flag should depend on P8 alone. |

## 9. Behavior activation matrix

| Setting/cutover | Safe value now | Change only after | Required complementary evidence |
|---|---|---|---|
| `ENAI_AUTH_MODE` | `gateway_only` | P3.A/P3.B | Direct callers use the same active-status, entitlement, idempotency, and persistence authority. |
| `ENABLE_EVIDENCE_REANALYSIS` | `false` | P4.5 | Mode-transition suite proves all dependent state is recomputed once with no stale evidence. |
| Canonical evidence enforcement | off/shadow only | P2.A then P4.A | Correct unit/provenance corpus, frame finalization, chart parity, staged comparison telemetry. |
| Backend replicas/workers | one | P5.6/P7.4 | External actor-bound shared state and two-process failure tests pass. |
| `ENABLE_GET_DATA_ENDPOINT` | `false` | Reviewed P3/P6 authorization/data contract | Active-admin boundary, bounded output/pagination, audit, and deployed-source proof pass. |
| Browser authoritative quota/persistence | legacy compatibility only until cutover | P3.B staged cutover | Server reservation/persistence proven, browser recognizes it, reconciliation clean, then old grants revoked. |
| Manual edge snapshots | temporary P0 procedure | P1.3 | Standard source layout, pinned dependencies, CI deploy, immutable hashes, rollback by artifact. |
| Old API/JSONB compatibility readers | retained during migration | P3.6/P6.1/P6.2 soak | Generated contracts adopted, legacy data migrated/quarantined, consumer telemetry shows no old clients. |

## 10. Completion records

For each independently deployed track, attach:

- commit and deployed revision;
- exact test/build/audit results;
- environment-variable checklist with values redacted;
- migration/function/source hashes;
- smoke and manual attestation results;
- known `Manual verification pending` items and owner/date;
- rollback artifact/command and rollback owner.

Do not label all of P0 production-complete until both repository deployments and the integration checks are recorded. This does not prevent either repository's P0 code from being committed, reviewed, or deployed independently.
