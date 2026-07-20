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
| Credentialed axe — **admin** | ✅ green (2026-07-20): operator set `SMOKE_ADMIN_EMAIL`/`SMOKE_ADMIN_PASSWORD` and re-ran Live Browser Proof; all three axe scans (login/public, dashboard/chat, admin) pass with no serious/critical violation. Must be re-confirmed at the frozen F2/F3 SHA. |
| **Paused / quota-exhausted** denial states | ⏳ create two synthetic accounts (paused user; quota-exhausted user) and confirm the app denies them with the correct messaging |
| Manual keyboard/SR/touch/zoom checklist | ⏳ §6 below |

## 5. B4.B — production-safe read-only database checks — DONE (2026-07-19)

Ran [`f10_b4_prod_readonly_db_checks.sql`](../evidence/f10_b4/f10_b4_prod_readonly_db_checks.sql)
(strictly read-only) in the production Supabase editor. Results:

| Check | Result |
|---|---|
| RLS enabled on protected tables | **true** on all 7: `user_profiles`, `chat_history`, `chat_usage`, `query_usage`, `user_access`, `user_queries`, `public_queries` |
| Direct INSERT/UPDATE/DELETE grants to `anon`/`authenticated` on protected tables | **none** (zero rows) — access mediated only by SECURITY DEFINER RPCs |
| Runtime role `enai_api_readonly` | `rolinherit=false` (**NOINHERIT**) and `rolbypassrls=false` (**does not bypass RLS**) — matches the least-privilege design |
| Legacy string rows in `chat_history` | **0** (operator cleared old rows; P6.B data hygiene met) |

Production database security posture attested clean.

## 6. B4.B — accessibility checklist — programmatic pass + operator confirmations

Automated credentialed axe (WCAG 2a/2aa/21aa/22aa) is green on login/public,
authenticated dashboard/chat, and admin (all three scans, after the toast-close
fix `04dd014`). A programmatic pass of the **public** surfaces was run through
the in-app browser (DOM/accessibility-tree inspection):

| Check | Login | Public dashboard | Method |
|---|---|---|---|
| Keyboard order logical (focusable in DOM order) | ✅ theme→tabs→email→password→forgot→Sign In→public | ✅ | focusable-element DOM order |
| Every interactive control has an accessible name | ✅ 10/10 | ✅ 35/35 (gated tabs announce "You need to register to access") | aria-label/label/text |
| Touch targets ≥ 24×24 px | ✅ none smaller | ✅ none smaller | getBoundingClientRect |
| No horizontal overflow (responsive) | ✅ 730 + 375 px | ✅ 375 px, chart renders | scrollWidth ≤ innerWidth |
| **200 % zoom** reflows without scroll trap | ✅ operator-confirmed | — | manual |

Remaining **operator confirmations** (need a real screen reader and/or an
authenticated session, which the programmatic pass can't fully substitute):

- [ ] **Visible focus ring**: eyeball that each focused control shows a ring
      (the CSS uses `focus:ring-*`; confirm it renders on your browser/theme).
- [ ] **Screen reader (NVDA)**: on the authenticated dashboard + chat, confirm
      tabs, the chart region, the chat input, and a toast all announce
      name+role (the toast close now announces "Close").
- [ ] **Live regions**: with NVDA on, confirm a chat response / toast is
      announced automatically (none present on the static public/login pages).
- [ ] **Authenticated dashboard/chat/admin** touch/overflow at 375 & 768 px —
      spot-check while signed in (axe already found no serious/critical issues).

## 7. Exit status (updated 2026-07-20 for the B6 remediation)

**Done (but see the SHA-realignment caveat):** §6 safe denial/readiness probes;
§6 signed happy-path (operator chat log — verified gateway assertion,
`/ask → 200`, grounded answer, chart); authenticated Post Deploy Smoke, Live
Browser Proof, and all three credentialed axe scans (login/public,
dashboard/chat, admin — after the toast-close fix); programmatic public-surface
a11y (§6, recorded) and 200% zoom (operator-confirmed); production read-only DB
security attestation (§5, clean); paused/quota denial states operator-checked
lightly.

**Honestly still pending (not done — do not mark complete by prose):**
- **Authenticated manual accessibility** — the visible-focus, NVDA/name-role,
  live-region, and authenticated responsive/touch checks are **not** performed;
  only the *public* surfaces were audited programmatically. To be completed at
  the frozen F2/F3 SHA with recorded browser/AT versions, pages, viewport/zoom,
  and evidence location (F10 B6 F1.5). The earlier "light manual" phrasing
  overstated this and is corrected here.
- **§8 bounded chat load** — **not run.** B6 decision (2026-07-20): satisfy the
  Load gate with a **formal F10-template waiver** (drafted in the B6 plan Phase
  F4), not an informal acceptance.
- **Rollback** — B6 decision (2026-07-20): **rehearse once** (Phase F4;
  redeploy previous, verify health, roll forward; record IDs). The earlier
  "deferred" note is superseded.

**SHA-realignment caveat (F10-REL-02/E2E-03):** the integrated evidence above
was gathered across drifting frontend SHAs (`1938967` smoke/Edge vs `04dd014`
browser). Per the B6 plan, browser + Edge are re-frozen at one SHA (Phase F2)
and the authenticated Post Deploy Smoke, Live Browser Proof, axe (all three),
and the full disposable DB regression are **re-run at that single identity**
(Phase F3). Only that re-run evidence counts toward closure.
