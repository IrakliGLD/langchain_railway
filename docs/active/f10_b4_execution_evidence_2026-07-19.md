# F10 B4 — Production Execution Evidence & Operator Runbook — 2026-07-19

Records the B4 production-evidence items that can be gathered without secrets,
and hands the credential/spend/account/manual items to the operator (Irakli)
as an exact runbook. Deployed identities under test: backend
`2f2a31053dfa391fbb0958ae858141c6f3e26ff9`; frontend/Edge current `main` head.

## 1. B4.A §6 — safe production smoke (denial + readiness) — DONE

Gathered via `curl` against `https://enai.galdava.com` (denial cases only; no
gateway secret used, no provider spend):

| Probe | Result | Meaning |
|---|---|---|
| `POST /ask` no auth | **401** `AUTHENTICATION_REQUIRED` | gateway auth enforced |
| `POST /ask` wrong `X-App-Key` | **401** | bad gateway secret rejected |
| `POST /ask` invalid bearer | **401** | bad bearer rejected (bearer disabled in prod) |
| `POST /ask` malformed JSON | **422** | request validation enforced |
| `POST /ask` ~300 KB body | **413** `REQUEST_TOO_LARGE` | pre-parse body-size limit enforced |
| `GET /versionz` no key | **401** | protected identity endpoint gated |
| `GET /healthz` | **200** | live |
| `GET /readyz` | **200** | DB + schema ready |

Typed error envelope confirmed on denials, e.g.
`{"error":{"code":"AUTHENTICATION_REQUIRED","message":"Authentication required","retryable":false,"request_id":"<uuid>"}}`
— every denial carries a `request_id` and a stable typed `code`.

## 2. B4.A §6 — signed happy-path + idempotency — OPERATOR (needs gateway secret)

These send a *real* signed chat and therefore incur one provider call each;
run with the `ENAI_GATEWAY_SECRET` + actor-assertion signing (or simplest:
send one real chat through the deployed browser/Edge, which signs for you).

1. One real signed `/ask` → 200 with a grounded answer; conversation
   continuation works; quota decremented exactly once.
2. Resend the **same** signed operation (same request ID/actor) → **409**
   `Gateway assertion replayed` / one provider attempt in the ledger (no
   duplicate charge).
3. Ordinary browser abort mid-request → operation terminal state consistent,
   no orphan retry.

Fastest path: open `https://dashboard.galdava.com/chat`, sign in as the smoke
user, send one question, confirm the answer + that "Queries" usage
incremented by one.

## 3. B4.A §8 — bounded production chat load — OPERATOR (approved envelope)

Parameters approved 2026-07-19 (see `f10_b4a_backend_evidence_2026-07-19.md`
§8): concurrency 2, ≤20 requests, ≤$2 spend ceiling, 15-min window, abort on
any duplicate attempt/charge, error rate >10%, `/readyz` degradation, DB
saturation, or ceiling. Record: request count, provider spend, p95 latency,
`/readyz` during load, zero duplicate attempts, one-replica confirmation.
**Not run yet — no provider spend has occurred.**

## 4. B4.B — credentialed browser/accessibility — status + operator steps

| Item | Status |
|---|---|
| Authenticated Post Deploy Smoke (browser→Edge→backend) | ✅ green (#240) |
| Live Browser Proof (authenticated dashboard, protected RPC pipeline, no runtime errors) | ✅ green (#6) |
| Credentialed axe — login/public + authenticated dashboard/chat | ✅ green (#6) — **found & fixed a real critical WCAG 4.1.2 toast-close-button violation** (`04dd014`) |
| Credentialed axe — **admin** | ⏳ **skipped**: set `SMOKE_ADMIN_EMAIL`/`SMOKE_ADMIN_PASSWORD` repo secrets, then re-run Live Browser Proof |
| **Paused / quota-exhausted** denial states | ⏳ create two synthetic accounts (paused user; quota-exhausted user) and confirm the app denies them with the correct messaging |
| Manual keyboard/SR/touch/zoom checklist | ⏳ §6 below |

## 5. B4.B — production-safe read-only database checks — OPERATOR

Run [`f10_b4_prod_readonly_db_checks.sql`](../evidence/f10_b4/f10_b4_prod_readonly_db_checks.sql)
in the **production** Supabase SQL editor (project `qvmqmmcglqmhachqaezt`). It
is strictly read-only (SELECTs against catalogs) — object/constraint/index/
function existence, RLS-enabled inventory, and explicit-grant inventory.
Archive the output. The dedicated runtime-role allowed/denied SELECT probes
are exercised by `scripts/verify_runtime_database_role.py` against the runtime
connection.

## 6. B4.B — manual accessibility checklist — OPERATOR

At `https://dashboard.galdava.com`, signed in, record pass/fail + notes:

- [ ] **Keyboard order**: Tab through login → dashboard → a chart tab → chat;
      focus order is logical, nothing is unreachable.
- [ ] **Visible focus**: every interactive control shows a visible focus ring.
- [ ] **Screen reader**: with NVDA/VoiceOver, tab controls, the chart region,
      the chat input, and the toast close button all announce a name+role
      (toast close now announces "Close").
- [ ] **Live regions**: chat responses / toasts are announced.
- [ ] **Touch targets**: interactive targets ≥ 24×24 CSS px on mobile viewport.
- [ ] **Responsive**: 375-px and 768-px viewports have no clipped/overlapping
      controls.
- [ ] **200 % zoom**: at browser 200 % zoom, no horizontal scroll trap; content
      reflows and remains operable.

## 7. Exit status

Safe backend §6 denial/readiness evidence is complete and recorded (§1).
Everything else in B4 requires a production secret, a synthetic account,
provider spend, or human interaction, and is handed to the operator above.
None of the remaining items block each other; they can be executed in any
order and their evidence archived against the deployed identities.
