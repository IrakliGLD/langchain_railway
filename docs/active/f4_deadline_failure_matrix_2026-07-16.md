# F4 deadline, retry, and ambiguous-delivery verification

**Date:** 2026-07-16
**Scope:** P5.1 backend completion plus the independent frontend/Supabase failure boundary
**Runtime constraint:** one backend worker and one Railway replica while the provider-attempt registry remains process-local

## Implemented contract

- The trusted backend request boundary owns one monotonic deadline. The pipeline binds that deadline, the safe request ID, and an HMAC-hashed actor binding into request-local context that is propagated into secondary worker threads.
- Every database acquisition checks that cleanup budget remains. Every transaction receives a transaction-local PostgreSQL `statement_timeout` equal to the smaller of `ENAI_DB_STATEMENT_TIMEOUT_MS` and remaining request time minus `ENAI_REQUEST_CLEANUP_ALLOWANCE_MS`.
- Read-only transactions are declared at gateway acquisition so PostgreSQL applies `SET TRANSACTION READ ONLY` before the timeout statement.
- OpenAI, Gemini, NVIDIA, and both embedding providers receive independent native timeouts bounded by the remaining request budget. SDK-owned retries are disabled (`0` for OpenAI-compatible clients and one total attempt for Google clients).
- Application fallback is permitted only after a proven pre-delivery rejection, currently a provider `429`, a factory failure, or an open circuit. Permanent client failures and ambiguous timeout/network/5xx delivery are not replayed.
- A bounded random delay may precede the one safe fallback. The fallback does not begin unless cleanup allowance plus the provider minimum-start budget still remains.
- Provider attempts are keyed by hashed actor binding, request ID, provider, and stage. Records contain state/timestamps only; raw prompts and responses are never stored. A duplicate completed, in-flight, or ambiguous stage is rejected rather than executed again.
- Secondary evidence uses the smaller of its configured budget and request remaining time, propagates request context into worker threads, cancels queued futures when the budget expires, and relies on the DB/provider native timeout to terminate already-running I/O.
- The durable Supabase chat-operation ledger remains the cross-process charging authority. A failure after Edge begins backend delivery is marked chargeable/ambiguous and therefore cannot be blindly replayed; a failure before delivery remains safely retryable under the existing typed policy.

## Failure-injection matrix

| Injection point | Required behavior | Automated evidence |
|---|---|---|
| Browser before Edge response | Local deadline wins, transport is aborted when cooperative, and an uncooperative transport still cannot hold the caller; outcome is non-retryable/ambiguous. | Frontend `src/lib/edgeFunctionClient.test.js`: local timeout, ignored abort, and caller-abort cases. |
| Browser navigation/caller abort | Parent cancellation is distinct from timeout and is never retried. | Frontend `src/lib/edgeFunctionClient.test.js` and `src/lib/requestControl.test.js`. |
| Edge before backend delivery | Failure remains non-chargeable and may follow the existing typed safe-retry policy with the same request ID. | Frontend `supabase/tests/chat_response_test.ts`: `chatFailureIsChargeable(false)`. |
| Edge after backend delivery | Failure becomes chargeable/ambiguous in the durable operation ledger; no blind replay. | Frontend `supabase/tests/chat_response_test.ts`: `chatFailureIsChargeable(true)`. |
| Backend before provider send | Factory/circuit rejection may use one bounded fallback only when sufficient budget remains. | Backend `tests/test_f4_deadline_semantics.py` plus provider breaker/registry tests. |
| Provider accepted request but response was lost | Attempt is ambiguous, recorded, and never sent to a fallback provider or retried. | Backend `test_lost_provider_response_is_ambiguous_and_never_falls_back`. |
| Permanent provider failure | Permanent 400/401/403/422 failures do not retry. | Backend parameterized permanent-failure test. |
| Provider rate-limit rejection | A proven 429 rejection may use one bounded fallback with the same actor-bound request identity. | Backend safe-fallback test. |
| Database statement timeout | Timeout is the smaller of configured timeout and remaining-minus-cleanup; insufficient cleanup budget prevents checkout. | Backend DB budget, no-checkout, and transaction-order tests. |
| Secondary evidence timeout | Configured evidence budget cannot extend the request; queued work is canceled and deep I/O receives the propagated request deadline. | Backend pipeline/evidence deadline and prefetch tests. |

## Exit-gate interpretation

Local automated gates establish the following:

- The application never intentionally starts an external call or safe fallback without enough remaining budget.
- Native external-call limits reserve `ENAI_REQUEST_CLEANUP_ALLOWANCE_MS` (default 3000 ms) for exception mapping, durable ledger completion, and response serialization.
- Scheduler/network teardown after the native timeout is allowed a documented 250 ms local tolerance; deployment smoke must measure the platform-specific tail before production attestation.
- One actor-bound request ID cannot execute the same provider/stage twice within the process. Cross-process replay remains prevented by the durable Edge entitlement/operation ledger and by maintaining the documented one-replica backend constraint.
- Provider APIs do not expose a portable post-hoc lookup for chat-completion delivery. Reconciliation is therefore conservative: ambiguous delivery is retained as chargeable/non-replayable instead of being guessed safe and reissued.

Production completion still requires the live failure-injection observations below. Local tests do not prove provider billing records or Railway/Supabase network teardown timing.

## Deployment and live canary

### Backend

Deploy the F4 backend commit through Railway. Defaults are safe and require no new variable, but explicitly recording these values is recommended:

```text
ENAI_REQUEST_CLEANUP_ALLOWANCE_MS=3000
ENAI_DB_STATEMENT_TIMEOUT_MS=30000
ENAI_DB_POOL_TIMEOUT_SECONDS=2
ENAI_DB_CONNECT_TIMEOUT_SECONDS=3
ENAI_PROVIDER_MINIMUM_START_BUDGET_MS=500
ENAI_PROVIDER_RETRY_JITTER_MAX_MS=250
OPENAI_TIMEOUT_SECONDS=120
GEMINI_TIMEOUT_SECONDS=120
NVIDIA_TIMEOUT_SECONDS=90
```

Do not increase Railway beyond one worker/replica. Canary a normal request, a deliberately short request budget, a DB `statement_timeout`, a provider timeout, and a 429 rejection. Correlate only safe request IDs and aggregate provider-attempt counters; do not capture prompts.

### Frontend/Supabase

After merging the independent frontend F4 commit, run the protected **Deploy Supabase edge functions** workflow with its full 40-character commit SHA and deploy all functions. No SQL patch and no browser application deployment are required for this F4 complement. Verify the deployed `X-Enai-Edge-Source` hash, then inject one pre-backend and one post-backend failure and confirm the durable operation state is respectively retryable/non-chargeable and ambiguous/chargeable.

### Final evidence to record

- backend and frontend/Edge deployed SHAs and Edge source-manifest hash;
- declared request budget, observed total latency, and tail tolerance for every injection point;
- provider execution count and charge record for the same safe request ID;
- durable chat-operation state after pre-delivery and ambiguous post-delivery failures;
- confirmation that permanent failures produced zero retry/fallback attempts; and
- confirmation that Railway still runs one worker and one replica.
