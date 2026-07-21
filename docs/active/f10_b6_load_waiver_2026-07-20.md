# F10 B6 — Representative Chat-Load Gate Waiver — APPROVED 2026-07-21

Per the F10 waiver template (remediation plan §11), satisfies the B6 **Load**
gate (finding F10-E2E-03) with a complete, named, expiring waiver rather than an
informal acceptance. Frozen artifact SHAs filled (F2 verified live). **Approved
by Irakli 2026-07-21** (operator risk sign-off), expiring 2026-08-20 — the
deferred load envelope must be run or the waiver renewed before then.

```text
Waiver ID:            F10-B6-LOAD-01
Finding:              F10-E2E-03 — Load gate: the approved representative chat-load
                      envelope (concurrency 2, ≤20 requests, ≤USD 2) was not run.
Affected exact
artifact SHA/digest:  backend 0684dc172eb2bb10a17a2e80941a6940b0882f2d (Railway
                      deployment e2e73849-c47d-4f4f-8073-76edd2e0df95;
                      runtime image manifest digest
                      sha256:319f6774f8197acd88941abae1b81a57bb10d19d98fa28a82e9d5b63c3b5336b)
                      and frontend
                      fc44fd40946bb0772ab4f178ac376196bec21498. Filled at the
                      F2 freeze; identities verified live (F2 evidence §4).
Reachable behavior:   No representative concurrent-chat-load observation exists
                      for the deployed artifacts. This is a MISSING-EVIDENCE gap,
                      not a demonstrated defect: no failing behavior is known.
Reason deferred:      The service runs one Railway replica at low traffic by
                      design. A formal load run would mostly re-confirm behavior
                      already covered by deterministic tests, at real provider
                      spend, without materially reducing risk before closure.
Compensating
controls:             (1) One-replica / autoscaling-off containment is the
                      deliberate topology (F5/B4). (2) Backpressure admission
                      gate, DB pool saturation, per-provider breaker, request
                      cancellation / no-orphan-work, and simultaneous
                      primary/secondary evidence are green in the 1,710-test
                      backend suite. (3) Idempotency: provider attempts finalize
                      exactly once keyed by actor+request+provider+stage, and an
                      exact signed-operation replay returns 409 — a duplicate
                      charge is structurally prevented (unit-tested + the durable
                      Supabase operation ledger cross-process). (4) A bounded
                      end-to-end request budget (ASK_MAX_REQUEST_BUDGET_MS) caps
                      per-request cost. (5) The live signed happy-path is proven
                      (B4 §6 operator chat log: verified assertion, /ask→200,
                      grounded answer, chart).
Named owner:          Irakli
Approver:             Irakli (sole operator of both repositories; this is an
                      evidence/load waiver, not a security exception — no
                      independent-reviewer requirement per B0 §6).
Remediation ticket
and target release:   Run the approved envelope (concurrency 2, ≤20 requests,
                      ≤USD 2, 15-min low-traffic window, abort on any duplicate
                      attempt/charge, error rate >10%, /readyz degradation, DB
                      saturation, or ceiling) within the waiver window and attach
                      request count, spend, p95, /readyz-under-load, and
                      zero-duplicate-charge evidence to the release record.
Approved at:          2026-07-21 — Irakli (operator risk sign-off; approved in
                      session on the confirmation "waiver approved").
Expires at:           2026-08-20 (Approved at + 30 days). Run the envelope or
                      renew before this date.
Rollback/disable
action:               None — this waiver enables no code path; it defers an
                      evidence-gathering load run. If the deferred load run later
                      surfaces a defect, the standing rollback path (Railway
                      previous deployment feff19e0-69d6-42e9-818a-757d76090e2e,
                      rehearsed in Phase F4) applies.
```

## Note

This waiver covers **only** the Load evidence row. It does not excuse any other
B6 gate. All other gates are being brought to a genuine Pass at the frozen
identity in Phases F2–F3 (aligned browser/Edge deploy, re-run authenticated
smoke / Live Browser Proof / axe / disposable-DB regression, Railway image
digest binding, and the completed manual accessibility checklist).
