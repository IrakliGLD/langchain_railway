# Chat gateway contract v1

The executable P1.A contract is [`contracts/chat_gateway_v1.json`](../../contracts/chat_gateway_v1.json). The Supabase `chat-with-enerbot` function is its production caller.

The gateway authenticates with `X-App-Key: <ENAI_GATEWAY_SECRET>`, sends at most 256 KiB of JSON to `POST /ask`, and declares `X-Enai-Contract-Version: chat-gateway-v1`. `MAX_REQUEST_BODY_BYTES` may configure a stricter platform-facing backend ceiling only within its validated range, but it must never be lower than the contract's 256 KiB gateway payload bound. During the independent P1 rollout the declaration header is optional for compatibility; an explicitly different value fails with HTTP 409 before pipeline execution.

`X-Request-Id` is preserved end to end when it matches the contract's safe 128-character syntax. Missing or unsafe values are replaced with a UUID. The backend returns the selected ID in both `X-Request-Id` and the `/ask` trace context, and returns the supported contract in `X-Enai-Contract-Version`.

The contract records the current legacy `service_tier` field as accepted-but-ignored. The edge source should stop sending it in P1.B rather than treating browser or edge tier labels as backend authorization. P3 will introduce the final actor/entitlement authority; P6 will replace this hand-maintained bridge with generated shared API artifacts.

For every frontend edge integration run, record the backend Git SHA, deployment revision/environment, `chat-gateway-v1`, and the frontend Git SHA. Do not infer a deployed SHA from the application version or expose a source commit in public response headers.
