# F10 B6 Closure Remediation Plan — 2026-07-20

Response to the independent re-audit
[`f10_b6_final_independent_closure_2026-07-20.md`](./f10_b6_final_independent_closure_2026-07-20.md)
(decision: **F10 not closed**, overall **B-**). This plan is assessed and
accepted: every finding was verified against source and is valid. It sequences
the fixes so the production identity freezes **once** and all integrated
evidence is gathered at that single frozen identity.

## 1. Assessment of findings

| Finding | Severity | Verdict | Root cause |
|---|---|---|---|
| **F10-REL-02** browser/Edge no longer share one release identity; backend runtime image digest unrecorded | High | **Valid** | The toast a11y fix `04dd0143` changed browser source *after* the frozen `1938967`; Edge was never realigned. Railway image digest was never captured. |
| **F10-E2E-03** exact-current integrated / load / DB evidence incomplete | High | **Valid** | Post Deploy Smoke #240 is at `1938967`, not `04dd0143`; the §8 load run never happened and has no waiver; the disposable-DB run for the fixed source isn't unambiguously tied; rollback unrehearsed. |
| **F10-ARCH-04** expired Gemini fallback still present; bearer-surface B6 decision missing | Medium | **Valid** | Inert `google.generativeai` branch deferred to 2026-08-31 (B5 says expired ⇒ remove now); `gateway_and_bearer` decision was punted to B6 and never made. |
| **F10-DOC-06** architecture/registry/evidence docs contradict source | Medium | **Valid** | `query_pipeline_architecture.md:273` still claims `ENABLE_AGENT_LOOP` survives (deleted in B5.A); B5 registry says `PLAN_VALIDATION_MODE=shadow` (code is `warn`); B4 a11y §4 table vs §7 summary disagree on admin scan / manual checks. |
| **F10-A11Y-07** manual a11y closure row not evidenced | Medium | **Valid** | Programmatic public pass exists; authenticated visible-focus / NVDA name-role / live-region / responsive-touch checks are unrecorded; "light manual" lacks versions/pages/SHA/observations. |
| **F10-QA-08** generated SQL/hash checks not clean-checkout portable on Windows | Low | **Valid** | Vendored schema raw-byte hash + `db:f2:verify` fail under `autocrlf=true` (CRLF vs LF), same class as the requirements-lock digest bug already fixed. |

Nothing in the audit is a false positive or double-counts a known-open item as
new. Grade B- is fair; source quality is strong, production *assurance* is the
gap.

## 2. Correction to the audit's closure sequence

The audit's step 6 ("remove Gemini, decide bearer, reconcile docs") mutates
source and would move the frozen SHA **after** steps 1–2 re-gathered evidence.
This plan front-loads all source/doc changes (Phase F1) so the identity is
frozen **once** (Phase F2) and never re-invalidated before the evidence pass
(Phase F3).

## 3. Decisions required before execution

1. **Bearer surface (`gateway_and_bearer`)** — keep or remove?
   - *Remove* (recommended): delete the dormant `gateway_and_bearer` mode +
     `ENABLE_PUBLIC_BEARER_AUTH` + bearer verification path + its tests.
     Production is `gateway_only`; the surface was never shipped. Cleanest, and
     satisfies the B6 governance gate by deletion.
   - *Keep*: record a concrete registry classification (protocol option), named
     owner, evidence-based removal criterion, and future date.
2. **§8 chat-load** — run the bounded envelope (≤$2) or a formal waiver?
3. **Rollback rehearsal** — perform once, or a formal waiver?

(2) and (3) each need either execution evidence or a complete F10-template
waiver — an informal acceptance does not satisfy the gate.

**Decisions recorded (2026-07-20, Irakli):**
1. Bearer surface → **KEEP + DOCUMENT** (no code change). Removing it is an
   auth-core + rate-limit refactor (config, `utils/auth.py`, `main.py`
   rate-limiting, the B2.A.3 negative-JWT suite, and the now-unused PyJWT), and
   refactoring the production auth path during certification adds more risk than
   the dormant surface itself. Recorded in the B5 registry with a concrete
   protocol-compatibility classification, owner (Irakli), removal criterion, and
   2026-09-30 review date. F10-ARCH-04's governance gate is satisfied by the
   recorded decision.
2. §8 load → **FORMAL WAIVER** (assistant drafts the F10-template waiver).
3. Rollback → **REHEARSE ONCE** (operator, in Railway; assistant records IDs).

## 4. Phased plan

### Phase F1 — Repo-side code & doc remediation (assistant-led, per-phase gated)

Each sub-phase: implement → targeted suite green → adversarial audit → commit.

- **F1.1 (backend code, ARCH-04a — DONE `6769a08`):** deleted the inert
  `google.generativeai` fallback branch in `knowledge/vector_embeddings.py`
  (backend now collapses to the single `google_genai` path; a missing
  `google.genai` raises `RuntimeError`); updated
  `tests/test_vector_embeddings.py` to assert the requirement instead of the
  legacy fallback. Full backend suite 1,710 green. *(Did not touch the working
  embedding path or the API-key issue, which is operator-owned.)*
- **F1.2 (ARCH-04b, docs-only — DONE):** recorded the KEEP decision for the
  `gateway_and_bearer` surface in the B5 registry (protocol-compatibility
  classification, owner Irakli, removal criterion, 2026-09-30 review). No auth
  code changed; the production auth path is untouched during certification.
- **F1.3 (docs, DOC-06 — DONE `3aa6568`):** reconciled
  `query_pipeline_architecture.md` (`ENABLE_AGENT_LOOP`/
  `agent_loop_blocked_by_policy` are **removed entirely** in B5.A, not inert);
  fixed the B5 registry `PLAN_VALIDATION_MODE` default (`shadow`→`warn`);
  reconciled the B4 a11y §4 table with the §7 summary (admin axe **green** after
  `SMOKE_ADMIN_*` was set) and rewrote §7 to mark authenticated manual a11y
  **PENDING** at the frozen SHA, §8 load a **FORMAL WAIVER**, and rollback a
  **REHEARSE**, with the SHA-realignment caveat.
- **F1.4 (frontend, QA-08 — DONE `d52a97e`):** extended `.gitattributes` to
  force LF on `src/contracts/**` and `database/baseline/**` (the raw-byte-hashed
  contract/SQL sources); LF-normalized both the source read and the verify
  comparison in `tools/build-f2-chat-gateway-patch.js` (matching p3b/p6b/fb5);
  added `tools/line-ending-policy.test.js` pinning the required LF globs so the
  policy can't silently regress. All db verifiers green, `db:f2:verify` passes,
  lint exit 0, frontend suite 466 green.
- **F1.5 (a11y evidence, A11Y-07 — doc-honesty DONE in F1.3; evidence → F3):**
  the B4 evidence doc now honestly records authenticated manual a11y as pending
  (no prose "done" without observations). The actual authenticated pass
  (visible-focus / NVDA name-role / live-region / responsive-touch) is gathered
  **at the frozen SHA in Phase F3** — running it now, before F2 deploys the
  frozen identity, would only be superseded. No pre-freeze code/doc work remains.

**Exit — REACHED (2026-07-20):** both repos CI-green (backend 1,710 / frontend
466); final candidate SHAs — backend `3aa6568` (branch
`refactor/review-phase-fixes`, awaiting the F2 merge to `main`), frontend
`d52a97e` (already on `main`). The F4 §8 load waiver is pre-drafted:
[`f10_b6_load_waiver_2026-07-20.md`](./f10_b6_load_waiver_2026-07-20.md)
(SHA fields fill at F2 freeze).

### Phase F2 — Freeze & aligned deployment (operator; assistant drives/verifies)

- Merge/freeze the F1 final SHAs (backend to `main`; frontend `main` head).
- Deploy backend; **capture the Railway runtime image digest** and bind it to
  the protected source SHA, SBOM/audit artifact, deployment ID, and rollback
  target (REL-02).
- Deploy the browser **and all nine Edge functions at the same final frontend
  SHA** (REL-02 alignment).
- Verify `/versionz` `git_sha`, `X-Enai-Edge-Version`, and browser
  `release-manifest.json` `app_version` are the **one** frozen identity.

### Phase F3 — Integrated evidence at the frozen identity (operator dispatch; assistant verify/record)

At the single frozen SHA, re-run and archive exact run/artifact/deployment IDs:
Backend release evidence (with image digest), authenticated **Post Deploy
Smoke**, **Live Browser Proof**, credentialed **axe** (all three), the **full
disposable DB regression**, and `/versionz`. Complete the manual a11y checklist
(F1.5) against this SHA and make the table/checklist/summary agree (E2E-03,
A11Y-07).

### Phase F4 — Load & reliability closure (operator; assistant drafts waivers)

- **§8 load** — decision: **formal waiver** (not a run). Draft is complete:
  [`f10_b6_load_waiver_2026-07-20.md`](./f10_b6_load_waiver_2026-07-20.md)
  (owner/approver Irakli, five compensating controls, 30-day expiry, remediation
  ticket). Finalize at F3 by filling the frozen backend+frontend SHAs and the
  approval timestamp — a waiver cannot cover an unknown deployed SHA.
- **Rollback** — rehearse (redeploy previous, verify health, roll forward) and
  record IDs; **or** a complete waiver.

### Phase F5 — B6 re-audit (assistant-led)

Re-run the B6 closure matrix against the frozen immutable identities; confirm
every gate is Pass or covered by a complete waiver; assemble the final evidence
ledger. Mark F10 complete only when the matrix has no unwaived High/Critical.

## 5. Ownership summary

| Phase | Assistant | Operator |
|---|---|---|
| F1.1–F1.4 | implement + commit | approve, push |
| F1.5 | drive programmatic pass / draft record | log in (AT), attest |
| F2 | verify identities, record | merge, deploy, read Railway digest |
| F3 | verify + record run IDs | dispatch workflows, DB run |
| F4 | draft waivers | run load / rollback or approve waivers |
| F5 | re-run audit + matrix | final decision |

## 6. Effort note

F1 is small, well-scoped code/doc work (dead-branch deletion, one config/doc
default, three doc reconciliations, a line-ending fix). The critical path is
F2–F4 operator execution (one aligned deploy + one evidence pass + the load
decision). No new Critical/High risk was found; this is evidence-and-alignment
work, not defect remediation.
