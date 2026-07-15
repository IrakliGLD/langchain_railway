# P7.A Backend Privacy and Runtime Activation Runbook

**Date:** 2026-07-15  
**Repository:** `D:/Enaiapp/langchain_railway`  
**Scope:** backend only; this repository neither reads nor deploys files from the independent frontend/Supabase repository  
**Local state:** implementation can be committed independently; every live item below remains unverified until its evidence is recorded

## 1. What becomes active automatically

After the backend commit is deployed:

- public answer/chart metadata is projected through explicit allow-lists;
- raw fixture capture is off by default and cannot be enabled outside development/test;
- trace/session/private identifiers are reduced to keyed fingerprints while request/trace/span correlation remains available;
- readiness rejects a staging/production database connection whose `current_user` is not the configured runtime role or whose default transaction mode is not read-only;
- the Dockerfile is the only Railway build path, uses a pinned base-image digest, runs as UID/GID `10001`, and copies only runtime sources;
- Railway remains configured for one in-process state owner. Do not increase workers or replicas during P7.A activation.

The code does **not** create a live database login, change Railway variables, rotate credentials, configure vendor retention, prove a replica count, or deploy an image. Those are operator actions.

## 2. Required production variables

Set these in the matching Railway environment; never paste their values into this file, GitHub artifacts, tickets, or logs.

| Variable | Required value/policy |
|---|---|
| `ENAI_DEPLOYMENT_ENV` | `staging` or `production`, matching the target. |
| `SUPABASE_DB_URL` | Connection URL that authenticates as the dedicated `enai_api_readonly` login. |
| `ENAI_DB_RUNTIME_ROLE` | `enai_api_readonly`; set it explicitly even though this is the staging/production default. |
| `ENAI_LOG_HASH_KEY` | Independent random secret, at least 32 bytes. Rotation intentionally breaks fingerprint continuity. If absent, the session-signing secret is used as a compatibility fallback. |
| `ENAI_FIXTURE_CAPTURE_MODE` | `off` (default). Production rejects `raw`. |
| `ENAI_FIXTURE_CAPTURE_SAMPLE_RATE` | `0`. |

Keep the existing application/provider/authentication secrets. Rotate them under the normal secret-rotation procedure; do not combine every secret rotation with the database-role cutover unless rollback ownership is clear.

## 3. Create and canary the database role

Use a database-owner/admin session, not the backend runtime connection.

1. Inventory the deployed backend's reads. The committed allow-list is:
   - `public.dates_mv`
   - `public.entities_mv`
   - `public.price_with_usd`
   - `public.tariff_with_usd`
   - `public.tech_quantity_view`
   - `public.trade_derived_entities`
   - `public.monthly_cpi_mv`
   - `public.energy_balance_long_mv`
   - `public.mv_balancing_trade_with_tariff`
   - `knowledge.documents`
   - `knowledge.document_chunks`
2. Confirm no active backend flag or typed tool needs a relation outside that list. Knowledge ingestion is intentionally out of scope and must retain a separate offline write credential.
3. In the Supabase SQL editor, run [`scripts/least_privilege_api_role.sql`](../../scripts/least_privilege_api_role.sql).
4. Generate a strong password in an approved password manager, then run this separately:

   ```sql
   alter role enai_api_readonly login password '<generated value>';
   ```

5. In Supabase, open **Connect** and copy the appropriate backend connection string. Railway is a persistent service, so prefer direct connection where network support permits; otherwise use the session pooler. Replace the connection username with the custom role while preserving the project-reference suffix required by the pooler, percent-encode special password characters, and retain SSL requirements. Supabase connection guidance: <https://supabase.com/docs/guides/database/connecting-to-postgres> and role guidance: <https://supabase.com/docs/guides/database/postgres/roles>.
6. Set the staging `SUPABASE_DB_URL` and `ENAI_DB_RUNTIME_ROLE`, deploy, then run from a trusted shell with the same URL:

   ```powershell
   $env:SUPABASE_DB_URL = '<staging runtime URL>'
   $env:ENAI_DB_RUNTIME_ROLE = 'enai_api_readonly'
   python scripts/verify_runtime_database_role.py
   ```

7. The verifier must report `PASS`. It checks runtime identity/read-only default, required public/vector reads, and denial of `auth.users`, knowledge writes, DDL, and role escalation. All mutation/DDL probes are explicitly rolled back even if unexpectedly permitted.
8. Run representative analytics, fallback-SQL, typed-tool, vector-retrieval, reflection/startup, and readiness smoke. A missing grant is fixed by updating the reviewed allow-list and SQL together; do not restore a broad credential as the permanent fix.
9. Repeat the canary in production. Retain the previous credential only for the documented rollback window, then revoke it and record the revocation timestamp and owner.

### PUBLIC grants and network controls

PostgreSQL privileges inherited from `PUBLIC` cannot be denied only for this role. Before globally revoking or re-granting them:

1. Export current schema/function/table grants for `public`, `knowledge`, `auth`, and `storage`.
2. Map every grant to the backend, PostgREST, Supabase Auth/Storage, Edge Functions, migrations, and operator tooling.
3. Prepare additive grants for each legitimate consumer.
4. Canary global revocations in a test/staging project if available. If no test database exists, treat the change as a production maintenance operation with backup, owner, rollback SQL, and immediate smoke—not as locally verified work.
5. Restrict database/network access using the Supabase controls available to the project plan. Verify Railway egress remains allowed before revoking the previous path.

Do not blindly revoke `PUBLIC` privileges from this repository's role script: that could break the separately deployed frontend/Supabase application.

## 4. Build and deployment evidence

1. Push the exact backend commit to GitHub.
2. Create or use protected GitHub environments named `staging` and `production`; require approval for production.
3. Run **Backend release evidence** with:
   - `environment`: the target environment;
   - `git_ref`: the full 40-character backend commit SHA.
4. The workflow must:
   - check out exactly that SHA;
   - build the pinned Docker image;
   - prove the runtime UID is non-zero;
   - prove `.git`, `.env`, tests, docs, and development requirements are absent;
   - produce a CycloneDX SBOM, `pip-audit` JSON, release manifest, image archive, and checksums.
5. A failed/unavailable dependency audit is **Unverified/Failed**, never Passed. Review every advisory and document accepted risk with owner and expiry.
6. Deploy the same SHA through Railway. In deployment details, record source SHA, image/deployment ID, environment, approver, timestamp, and `/readyz` result.
7. Confirm the service was built from `Dockerfile`; do not configure a competing Nixpacks/build command.
8. Smoke `/healthz`, `/readyz`, authenticated `/ask`, gateway rejection, request-size limits, typed error envelopes, and authenticated `/metrics`.
9. Roll back once in staging to the previous known-good deployment, verify readiness/ask, then redeploy the candidate. Record both deployment IDs and the rollback owner.

## 5. One-replica and in-memory state gate

Session continuity and rate-limit state remain in process. Therefore:

1. In Railway deployment settings, set **exactly one replica** and retain one application worker.
2. Record the dashboard screenshot/export or `railway status --json` output showing one replica for staging and production.
3. Confirm no platform autoscaling setting can raise replicas.
4. Do not enable a second replica until shared session, continuity, rate-limit, and idempotency repositories have explicit TTL/failure policy and pass multi-process/failure tests.

This is a deliberate scaling block, not a claim that P7.4's future two-replica acceptance has passed.

## 6. Privacy/logging operational attestation

The privacy/security owner must inventory each of: Railway application/access logs, Supabase database/pooler/Auth/Storage/Edge logs, provider logs, GitHub Actions/artifacts, error monitoring, local fixtures, and support exports.

For every store, record:

- exact fields/content and whether raw user query/answer, email, token, or reusable identifier can appear;
- purpose and legal/operational basis;
- retention duration and automatic deletion mechanism;
- roles with read/export/delete access and access-review cadence;
- export/deletion request procedure and evidence location;
- incident owner, escalation path, and credential/log-hash rotation procedure;
- vendor setting or contract that enforces the retention claim.

Run automated canary scans against staging logs using synthetic email, UUID, JWT-like token, query, answer, SQL, and exception strings. Request/trace/span IDs may remain; the canary content and reusable credentials must not. Never use real personal data for this test.

## 7. Evidence ledger

| Gate | Staging evidence | Production evidence | Owner/status |
|---|---|---|---|
| Dedicated DB role and denial probes | Pending | Pending | Manual verification pending |
| Old broad credential revoked | N/A until canary | Pending | Manual verification pending |
| PUBLIC/network grant inventory | Pending | Pending | Manual verification pending |
| Exact-SHA workflow/SBOM/audit | Pending | Pending | Manual verification pending |
| Non-root/excluded-artifact inspection | Pending | Pending | Manual verification pending |
| `/healthz`, `/readyz`, `/ask`, `/metrics` smoke | Pending | Pending | Manual verification pending |
| One replica/one worker | Pending | Pending | Manual verification pending |
| Rollback rehearsal and owners | Pending | Pending | Manual verification pending |
| Vendor retention/access/export/deletion | Pending | Pending | Privacy-owner approval pending |
| Synthetic log canary scan | Pending | Pending | Manual verification pending |

P7.A is locally implementation-complete only after its commit and local gates pass. P7.A production attestation and the phase-wide P7 exit gate remain open until this ledger is populated with retained evidence.
