# P3.A backend activation runbook

Date: 2026-07-13  
Repository: `IrakliGLD/langchain_railway`  
Application boundary: backend only; the frontend/Supabase repository remains independently built, tested, deployed, and rolled back

## Outcome

P3.A makes the backend verify the actor/session/request assertion emitted by the independently deployed P3.B edge function. It also binds backend sessions to that actor, separates the end-to-end request ID from the backend span, treats every history source as untrusted, rejects unknown Ask v1 fields, and returns safe typed errors.

The only runtime link between applications remains HTTPS:

1. the frontend calls the Supabase `chat-with-enerbot` edge function;
2. the edge function reserves entitlement and calls backend `POST /ask` once;
3. the backend returns the answer to the edge;
4. the edge persists and returns the result to the frontend.

No backend source, build, test, or runtime path reads `D:\export_enai` or another frontend checkout. No frontend source reads this repository.

## Configuration

Add these backend environment variables in Railway:

- `ENAI_GATEWAY_ACTOR_ASSERTION_MODE=optional` for the first P3.A deployment;
- `ENAI_GATEWAY_ACTOR_ASSERTION_MAX_AGE_SECONDS=120` unless measured clock skew requires another value between 30 and 900;
- keep `ENAI_AUTH_MODE=gateway_only`;
- keep `ENAI_GATEWAY_SECRET` unchanged during the coordinated rollout.

The P3.B Supabase Edge runtime must have:

- `CHAT_BACKEND_URL`: the deployed backend `/ask` URL;
- `CHAT_BACKEND_SECRET`: exactly the same value as backend `ENAI_GATEWAY_SECRET`.

Do not copy either secret into browser variables, Git, logs, screenshots, tickets, or commands retained in shell history.

## Preconditions

1. Deploy the P3.B additive database patch from frontend commit `59b4d64` (or a reviewed descendant): `database/patches/2026-07-13_p3b_01_additive_authority.sql`.
2. Do **not** apply `2026-07-13_p3b_02_revoke_legacy_authority.sql` until the P3.B soak and reconciliation steps pass.
3. Deploy the tracked P3.B edge source through **GitHub → Actions → Deploy Supabase edge functions → Run workflow**. Do not manually paste an edge snapshot.
4. Confirm backend and edge clocks are synchronized. A difference greater than the configured assertion age will correctly fail authentication.
5. Keep the backend at one replica. The authoritative entitlement/idempotency ledger is in P3.B, but the additional backend replay cache and conversation memory remain process-local until P5/P7.

The user requested that unavailable live-database checks be skipped. Therefore local P3.A completion does not attest that the P3.B migrations, RLS/grants, functions, secrets, or live reconciliation are deployed; an operator must verify them using the P3.B runbook.

## Staged activation

### 1. Deploy backend in compatibility mode

1. Set `ENAI_GATEWAY_ACTOR_ASSERTION_MODE=optional` and the maximum age.
2. Deploy the P3.A backend commit to staging.
3. Confirm `/ready` succeeds and ordinary gateway chat still works.
4. Confirm responses preserve `X-Request-Id` and return a different `X-Trace-Id`/`X-Enai-Span-Id` beginning with `span-`.
5. Confirm error responses use `error.code`, `error.message`, `error.retryable`, and `error.request_id` and contain no stack, SQL, provider body, URL, or token.

Compatibility semantics are strict: optional mode permits **no assertion headers** from an older edge, but a partial, malformed, stale, future, replayed, or tampered assertion is rejected.

### 2. Verify the P3.B edge path

1. Deploy the P3.B edge function and frontend independently using the frontend repository runbook.
2. Send staging chat requests through the real UI.
3. In backend logs, find the same request ID emitted by browser/edge and confirm `Gateway actor assertion: verified=True`.
4. Confirm the backend span differs from the request ID and the edge span appears only as parent trace metadata.
5. Exercise two users and two concurrent tabs. Confirm no cross-user history appears and cross-actor replay is rejected.
6. Confirm the P3.B entitlement operation and persisted turns share the same request ID as the backend log.

If logs show `verified=False`, stop. The request is using a legacy edge artifact or missing headers; do not switch to required mode.

### 3. Enforce assertions

After staging evidence is clean:

1. Set `ENAI_GATEWAY_ACTOR_ASSERTION_MODE=required` in staging and restart/redeploy.
2. Repeat UI chat, quota, paused-user, error, idempotent replay, and persistence checks.
3. Verify a gateway request with only `X-App-Key` now returns HTTP 401 before pipeline execution.
4. Promote the same backend artifact and settings to production only after the P3.B production edge artifact is deployed.
5. Monitor 401/409/422/429/5xx rates and `verified=True` traffic during the production canary.

## Direct bearer remains disabled

Do not set `ENAI_AUTH_MODE=gateway_and_bearer` outside tests. P3.B supplies active-status, entitlement, idempotency, and persistence authority to the edge path, but the backend still has no trusted direct-bearer call into that authority. Local Supabase JWT verification alone would reopen the unmetered/paused-user bypass. A later change must define and deploy that shared authority before removing the test-only startup restriction.

## Rollback

- If the new edge has not reached all traffic, change only `ENAI_GATEWAY_ACTOR_ASSERTION_MODE` from `required` back to `optional`; do not disable verification of assertions that are present.
- Roll back the backend and frontend/Supabase artifacts independently. Do not make one deployment read files from the other repository.
- Keep `ENAI_AUTH_MODE=gateway_only` during rollback.
- Do not drop additive entitlement/audit data. Follow the P3.B runbook for database/edge rollback and reconciliation.
- Do not apply the legacy-grant revoke patch until upgraded consumers are proven. After revocation, prefer rolling forward; restoring grants requires explicit security approval.

## Complementary work still required

- Live deployment/reconciliation of P3.B database, RLS, RPC, edge, and frontend artifacts.
- Activation of `required` mode after verified traffic is observed.
- Final legacy-grant revocation after P3.B soak.
- A real shared authority for direct bearer before it can be enabled.
- P5/P7 external shared session/replay state and multi-replica tests before scaling the backend beyond one replica.
- P6 generated cross-repository API artifacts to replace the hand-maintained v1 bridge without creating a source-tree dependency.
