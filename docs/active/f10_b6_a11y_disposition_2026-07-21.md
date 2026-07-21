# F10 B6 — A11Y-07 Screen-Reader Listening Disposition — APPROVED 2026-07-21

Per the F10 waiver/disposition template (remediation plan §11), closes the B6
**Accessibility** gate finding **F10-A11Y-07** — the residual assistive-technology
*listening* attestation — by a complete, named, reasoned disposition rather than
by performing the pass. Parallels the load-gate waiver `F10-B6-LOAD-01`. It
defers **only** the human listening confirmation; all structural accessibility
work is delivered and remains in force.

```text
Disposition ID:       F10-B6-A11Y-01
Finding:              F10-A11Y-07 — the residual NVDA/VoiceOver *listening*
                      attestation (confirming spoken announcements are
                      understandable and correctly ordered), on top of the
                      already-green structural pass.
Decision:             DEFER the listening attestation for this release.
Reachable behavior:   No demonstrated defect. Automated axe (WCAG 2a/2aa/21aa/
                      22aa) is green on login/public + dashboard/chat + admin
                      (Live Browser Proof #8), and the programmatic authenticated
                      structural pass is green (F3 runbook §6): 0 unnamed
                      controls, 0 sub-24px targets, no horizontal overflow at
                      375/730/768 px, visible keyboard focus, a semantic admin
                      table (th + caption), and a role=log / aria-live=polite chat
                      response region. Only the human *listening* confirmation is
                      not performed.
Rationale (owner):    (1) The dashboard is data-visualization-heavy; charts are
                      inherently limited for speech and are better served by their
                      underlying data than by a listening pass. (2) The admin
                      panel is restricted to named administrators, not general
                      users. (3) There is no current assistive-technology user
                      base and no WCAG-AA contractual/legal obligation for this
                      release. (4) The one text-first surface where screen-reader
                      support is most valuable — the ENAI Analyst chat — is
                      already structurally announcement-ready (aria-live), so it
                      is the priority surface to attest first if this is revisited.
Compensating          Automated-axe AA pass, the programmatic structural pass
evidence:             (F3 §6), named controls, semantic admin table, the aria-live
                      chat region, and visible keyboard focus — all green at the
                      frozen frontend identity fc44fd4.
Revisit trigger:      Perform the NVDA/VoiceOver listening pass (chat surface
                      first) if (a) an assistive-technology user is onboarded, or
                      (b) a WCAG-AA compliance requirement arises (procurement,
                      EAA / Section 508, enterprise contract).
Named owner:          Irakli
Approver:             Irakli (product owner and sole operator; this is an
                      accessibility-scope disposition, not a security exception).
Approved at:          2026-07-21
Review at:            2026-10-21 (3 months) or on the revisit trigger, whichever
                      comes first.
Rollback/disable:     None — this disposition enables no code path; it records a
                      scope decision. The structural accessibility remains in
                      force and is not weakened.
```

## Note

This disposition covers **only** the A11Y-07 listening attestation. It does not
excuse any other B6 gate, and it does not remove or weaken any accessibility that
is already implemented — named controls, the aria-live chat region, the semantic
table, and visible focus all stay. It records the product-owner decision that the
*listening* confirmation is deferred as out-of-scope for this release's user base
and obligations, with a defined trigger to revisit.
