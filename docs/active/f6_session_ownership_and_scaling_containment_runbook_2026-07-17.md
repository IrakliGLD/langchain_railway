# F6 Session Ownership and Scaling Containment Runbook

**Date:** 2026-07-17  
**Repository:** `D:/Enaiapp/langchain_railway`  
**Scope:** backend process-local session continuity only

## Outcome and supported topology

F6 completes the local P5.6 containment path without adding a shared-state dependency. The supported production topology remains exactly one backend process and one Railway replica. Session continuity, rate limits, replay state, caches, and breakers remain process-local.

The implementation:

- preserves the P3 signed gateway actor/session assertion and actor-derived authoritative session identity;
- binds backend session tokens, stored history, and contract snapshots to both actor and authentication mode;
- serializes the complete `/ask` turn for one owned session with a request-budget-bounded wait;
- rejects a live session token presented by a different actor or authentication mode;
- issues a distinct identity when a valid backend session token outlives its expired, evicted, or restarted process-local record, instead of recreating the old empty record;
- prevents history or contract mutations from implicitly recreating missing sessions;
- applies idle expiry, a hard session-count limit, per-field history limits, and least-recently-used inactive eviction;
- never evicts or expires an active/waiting turn;
- exposes aggregate, content-free session metrics only through the existing protected `/metrics` route; and
- retains the existing startup failure for configured HTTP worker counts other than one.

No frontend source, Supabase Edge Function, or database migration is part of F6.

## Deployment behavior to expect

Backend actor-bound tokens now use the actor/auth-mode-bound token format. A deployment restarts the process-local store, so a previously issued backend token cannot retain in-memory continuity across that restart. It receives a new opaque session identity. This is intentional: an old token must not recreate an empty record under its old identity. Verified Edge-provided database history may seed the new process-local record through the existing P3 path.

The public chat contract is unchanged. Overlapping turns for the same session can now receive typed `409 SESSION_BUSY`; session-capacity saturation can receive typed `503 SESSION_CAPACITY_EXHAUSTED`; cross-owner replay receives typed `401 SESSION_OWNERSHIP_MISMATCH`.

## Backend environment

No new secret is required. Keep `ENAI_SESSION_SIGNING_SECRET` stable and secret.

The following settings have safe defaults and are optional unless an operator intentionally tunes them:

| Variable | Default | Allowed range | Meaning |
|---|---:|---:|---|
| `SESSION_HISTORY_MAX_TURNS` | `3` | `1..20` | Maximum stored Q/A turns per session. |
| `ENAI_SESSION_HISTORY_MAX_ITEM_CHARS` | `2000` | `256..20000` | Maximum stored characters in each question or answer field. |
| `SESSION_IDLE_TTL_SECONDS` | `3600` | `60..86400` | Idle expiry measured from the end of the latest active turn. |
| `ENAI_SESSION_MAX_ENTRIES` | `2048` | `1..100000` | Hard process-local session-record limit. |
| `ENAI_SESSION_TURN_WAIT_TIMEOUT_MS` | `5000` | `0..30000` | Maximum wait for the preceding same-session turn; the request's remaining budget can reduce it further. |

Set no worker variable above one. `ENAI_HTTP_WORKERS`, `WEB_CONCURRENCY`, and `UVICORN_WORKERS` must be absent or equal to `1`; any other value fails startup.

## Required Railway control-plane steps

Source code can reject multiple workers inside one process, but it cannot prove or prevent Railway from starting multiple replicas. For every deployed environment:

1. Open the backend service in Railway.
2. Open **Settings** and locate the deployment/scale controls.
3. Set the replica count to exactly `1`.
4. Disable horizontal autoscaling, or set both minimum and maximum replicas to `1` if the UI represents autoscaling as a range.
5. Confirm the start command remains `python main.py` and the healthcheck remains `/readyz` on port `3000`.
6. Save a screenshot or exported service configuration showing environment, service, one replica, and disabled autoscaling. Do not include secret values.
7. Repeat for production and for staging if staging exists.

A `1/1` healthcheck proves one healthy deployment allocation at that moment, but it is not sufficient evidence that autoscaling is disabled. Retain the settings export/screenshot as the P5.6/P7 topology attestation.

## Deployment and smoke order

1. Deploy the F6 backend commit to the single replica. Do not deploy frontend, Edge Functions, or SQL for this phase.
2. Confirm startup succeeds once, binds to `0.0.0.0:3000`, and `/readyz` returns `200`.
3. Run one authenticated chat turn and a second continuation turn. Confirm both succeed and the second receives the expected persisted/seeded conversation context.
4. Submit two near-simultaneous requests for the same actor/session. Confirm only one pipeline execution is active at a time; the second either runs after the first or receives `409 SESSION_BUSY` when its bounded wait expires.
5. Run simultaneous requests for different actor/session pairs. Confirm they are not serialized by the session layer.
6. In a controlled gateway test, present actor A's live backend session token with actor B's valid signed gateway context. Confirm `401 SESSION_OWNERSHIP_MISMATCH` and confirm the pipeline/provider is not called.
7. For expiry testing in staging, temporarily use `SESSION_IDLE_TTL_SECONDS=60`, obtain a backend session token, wait more than 60 seconds without activity, then reuse it. Confirm the response supplies a different `X-Session-Token` and does not restore the old in-memory history. Restore the intended TTL afterward.
8. Confirm an expired bearer/JWT is rejected by authentication before session creation. Session counters must not increase for that rejected request.
9. Query protected `/metrics` with the evaluation/admin key and inspect `session_memory`:
   - `storage` is `process_local`;
   - `supported_replicas` is `1`;
   - `current_sessions` never exceeds `max_sessions`;
   - `turn_participants` returns to `0` after the test;
   - `capacity_rejections`, `turn_timeouts`, `expired_sessions`, `stale_token_renewals`, and `capacity_evictions` change only when the corresponding test is executed; and
   - no actor ID, session ID, prompt, question, or answer appears in the payload.
10. Watch typed 401/409/503 rates and chat success/latency during the canary. Unexpected sustained `SESSION_BUSY` means the wait setting or request latency needs investigation; do not solve it by adding replicas.

## Rollback

Rollback the backend as one replica. A rollback or restart loses process-local sessions, just like a normal deployment. Confirm the existing Edge database-history seed path restores allowed continuity. Do not retain an unsafe old release merely to preserve in-memory sessions, and do not add replicas as a rollback response.

## Future multi-replica gate

Shared Redis/PostgreSQL state is deliberately deferred and is not required while the one-replica constraint is enforced and evidenced. If horizontal scaling becomes a product requirement, migrate these boundaries together rather than partially:

- session ownership, expiry, turn serialization, and contract/history continuity;
- rate limiting;
- assertion replay protection;
- request/provider idempotency and ambiguous-delivery reconciliation; and
- any cache/breaker behavior whose divergence changes correctness or cost.

The shared implementation must define TTL, fencing/lease semantics, atomic ownership, failure behavior, privacy/retention, and bounded capacity, then pass two-process tests covering process death, lease expiry, network partition, replay, duplicate delivery, and failover before the replica limit can be raised.

## Local verification evidence

- Scoped Ruff and Python compilation: passed.
- Session/config/main focused suite: `161` passed.
- Final complete backend suite: `1,623` passed.