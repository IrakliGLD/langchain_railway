# P5.A Backend Deadline Foundation Activation Runbook

**Date:** 2026-07-14
**Scope:** Backend repository only: `D:/Enaiapp/langchain_railway`
**Status:** Local implementation and regression verification complete; deployment evidence pending

## Purpose

This change establishes one backend-owned monotonic request deadline for `POST /ask` without coupling the backend repository to frontend files. The Supabase Edge Function remains a separate deployable application and communicates only through the versioned HTTP contract.

The slice closes the backend request-boundary portion of P5.1. It does not yet claim complete provider/DB cancellation or end-to-end duplicate-charge acceptance.

## Behavior

- Legacy gateways may omit `X-Enai-Request-Budget-Ms`; the backend uses a 110,000 ms default.
- New gateways may send remaining milliseconds in `X-Enai-Request-Budget-Ms`.
- The backend caps the effective budget at 115,000 ms, preserving at least five seconds before the current 120-second Edge abort.
- The backend converts the effective budget once to a monotonic deadline.
- The pipeline checks the deadline before expensive analyzer, vector, evidence, SQL, enrichment, summary, and chart stages.
- Stage 0.8 receives the smaller of its local evidence budget and the request's remaining budget.
- Exhaustion returns HTTP 408 with `REQUEST_DEADLINE_EXCEEDED` and the existing safe error envelope.
- Successful responses expose `X-Enai-Request-Budget-Ms`, `X-Enai-Deadline-Remaining-Ms`, and `X-Enai-Retry-Owner: backend`.
- `backend` means the backend owns pipeline/model/provider retry decisions. Browser/Edge code may still handle explicitly typed entitlement or in-progress operation states using the same idempotency key; it must not replay backend/model timeout or ambiguous delivery.

## Backend environment variables

No new variable is required. The defaults are active after deployment.

Optional Railway variables:

- `ASK_DEFAULT_REQUEST_BUDGET_MS`: default budget when the gateway omits the header. Default `110000`; allowed `1000` through `300000`.
- `ASK_MAX_REQUEST_BUDGET_MS`: cap for a supplied gateway budget. Default `115000`; must be at least the configured default and no more than `300000`.

For the current Edge timeout, keep:

`ASK_DEFAULT_REQUEST_BUDGET_MS <= ASK_MAX_REQUEST_BUDGET_MS <= 115000`

Do not create these as `VITE_*` variables or Supabase browser variables. They are backend runtime settings only.

## Railway setup

1. Open the backend Railway project and select the production service.
2. Open **Variables**.
3. Leave both variables unset to use the reviewed defaults, or add the two values above if an explicit configuration record is required.
4. Do not raise the maximum above 115,000 ms while `CHAT_BACKEND_TIMEOUT_MS` remains 120,000 ms.
5. Deploy the exact backend commit produced by this phase.
6. Record the Railway deployment ID, backend Git SHA, effective variables, and deployment time.

## Safe rollout order

1. Deploy the backend first. The new header is optional, so the existing Edge Function remains compatible and immediately receives the 110-second backend default.
2. Verify `/healthz` and `/readyz`.
3. Send a normal gateway request without `X-Enai-Request-Budget-Ms`; verify a successful response includes all three deadline headers.
4. Send a staging request with `X-Enai-Request-Budget-Ms: 90000`; verify the effective budget header is `90000` and remaining time is lower but positive.
5. Send a staging request with `X-Enai-Request-Budget-Ms: 999999`; verify the effective budget is capped at `115000`.
6. Send a staging request with `X-Enai-Request-Budget-Ms: 0`; verify HTTP 408 and `REQUEST_DEADLINE_EXCEEDED` before pipeline/model work.
7. Deploy the separate frontend/Supabase complement only after the backend contract is live. That Edge release should send the remaining backend budget; it must not read backend repository files.
8. Repeat the smoke matrix in production and archive request IDs plus backend/Edge deployment identities.

## Failure injection

In staging, force a pipeline stage or provider call to consume the remaining budget. Verify:

- the next pipeline stage does not begin;
- the response is typed HTTP 408 when control returns to the pipeline;
- the request ID is preserved;
- the browser does not automatically replay the backend/model timeout;
- one idempotency identity produces no duplicate persisted assistant answer;
- model/provider work already in progress is measured separately, because this slice cannot forcibly interrupt every third-party SDK call.

## Rollback

1. Redeploy the previous backend commit.
2. Leave the Edge Function unchanged; the header is optional and ignored by the previous backend.
3. If only the configured budget is wrong, correct the Railway variables and redeploy the same immutable backend commit rather than disabling deadlines.
4. Do not roll back to an unbounded request path as a long-term fix.

## Remaining P5.A work

- Pass remaining time into every DB pool checkout/query and model/provider SDK call.
- Stop provider retries when insufficient budget remains.
- Add cooperative cancellation where the SDK/driver supports it.
- Prove end-to-end deadline tolerance and duplicate-charge behavior with deployed Edge and backend failure injection.
- Keep one backend replica until the separate session/shared-state acceptance gate passes.
