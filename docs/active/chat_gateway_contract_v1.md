# Chat gateway contract v1

The executable P1.A contract is [`contracts/chat_gateway_v1.json`](../../contracts/chat_gateway_v1.json). The Supabase `chat-with-enerbot` function is its production caller.

The gateway authenticates with `X-App-Key: <ENAI_GATEWAY_SECRET>`, sends at most 256 KiB of JSON to `POST /ask`, and declares `X-Enai-Contract-Version: chat-gateway-v1`. `MAX_REQUEST_BODY_BYTES` may configure a stricter platform-facing backend ceiling only within its validated range, but it must never be lower than the contract's 256 KiB gateway payload bound. The declaration header remains optional for independently deployed legacy gateways; an explicitly different value fails with HTTP 409 before pipeline execution.

`X-Request-Id` is preserved end to end when it matches the contract's safe 128-character syntax. Missing or unsafe values are replaced with a UUID. The backend returns the selected ID in `X-Request-Id`, allocates a distinct backend span in `X-Trace-Id` and `X-Enai-Span-Id`, and returns the supported contract in `X-Enai-Contract-Version`. An incoming `X-Enai-Span-Id` is upstream trace metadata, not authorization material.

P3.A extends v1 compatibly with an edge HMAC over five newline-delimited values: contract version, request ID, actor ID, Supabase session ID, and Unix issue time. The corresponding headers are `X-Enai-Actor-Id`, `X-Enai-Session-Id`, `X-Enai-Actor-Issued-At`, and `X-Enai-Actor-Signature`. `ENAI_GATEWAY_ACTOR_ASSERTION_MODE=optional` verifies any assertion presented but permits a request with no assertion headers during independent rollout; a partial, malformed, stale, future, or tampered assertion always fails. Set the mode to `required` only after the P3.B edge artifact is deployed and verified. `ENAI_GATEWAY_ACTOR_ASSERTION_MAX_AGE_SECONDS` defaults to 120 seconds and is bounded to 30–900.

All caller/database/session history is untrusted. Each question and answer is bounded, firewalled, sanitized, and rendered inside escaped `UNTRUSTED_CONVERSATION_HISTORY` prompt boundaries. Verified actor/session pairs select stable opaque backend sessions, and any backend session token issued for a known actor is cryptographically actor-bound.

Unknown request fields are rejected with 422. This includes the former `service_tier` compatibility field: tier labels are never backend authorization inputs. Errors use `{ "error": { "code", "message", "retryable", "request_id" } }` and never return exception, SQL, provider, token, or URL details. Direct bearer auth remains test-only until it can call the same deployed active-status, entitlement, idempotency, and persistence authority as the P3.B edge path; P3.A does not weaken that gate.

For every frontend edge integration run, record the backend Git SHA, deployment revision/environment, `chat-gateway-v1`, and the frontend Git SHA. Do not infer a deployed SHA from the application version or expose a source commit in public response headers.
