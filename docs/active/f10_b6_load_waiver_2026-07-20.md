# F10 B6 — Representative Chat-Load Gate Waiver — 2026-07-20 (awaiting approval)

Per the F10 waiver template (remediation plan §11), satisfies the B6 **Load**
gate (finding F10-E2E-03) with a complete, named, expiring waiver rather than an
informal acceptance. The frozen artifact SHAs are now filled (F2 verified live).
**Only the operator's risk sign-off + approval date remain** — everything else
is complete. The assistant cannot self-approve this; approval is Irakli's named
acceptance of the deferred-load risk.

```text
Waiver ID:            F10-B6-LOAD-01
Finding:              F10-E2E-03 — Load gate: the approved representative chat-load
                      envelope (concurrency 2, ≤20 requests, ≤USD 2) was not run.
Affected exact
artifact SHA/digest:  backend 65cf93b697e44f08cd03e782aac9949d2336135a (Railway
                      deployment ffc9ec32; source-build, SHA-bound, no Docker
                      digest) and frontend
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
Approved at:          AWAITING Irakli's approval — the SHAs, controls, owner and
                      ticket are complete; only the operator's risk sign-off +
                      date remain. On approval, record the date here.
Expires at:           approval date + 30 days (e.g. 2026-08-20 if approved
                      2026-07-21).
Rollback/disable
action:               None — this waiver enables no code path; it defers an
                      evidence-gathering load run. If the deferred load run later
                      surfaces a defect, the standing rollback path (Railway
                      redeploy-previous, rehearsed in Phase F4) applies.
```

## Note

This waiver covers **only** the Load evidence row. It does not excuse any other
B6 gate. All other gates are being brought to a genuine Pass at the frozen
identity in Phases F2–F3 (aligned browser/Edge deploy, re-run authenticated
smoke / Live Browser Proof / axe / disposable-DB regression, Railway image
digest binding, and the completed manual accessibility checklist).
