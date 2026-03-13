---
name: developer-phased-audit
description: Use when implementing repository changes in explicit phases with an upfront plan, a short review or re-plan step, implementation, and an independent audit-plus-fix loop before moving to the next phase. Best for multi-step feature work, schema changes, routing changes, prompt changes, and safety-sensitive refactors.
---

# Developer Phased Audit

Use this skill when the user wants work to proceed phase by phase rather than as one uninterrupted coding pass.

Read [references/workflow.md](references/workflow.md) before starting the first phase.
Read [references/audit-checklist.md](references/audit-checklist.md) before auditing any phase.
If the task touches structured LLM outputs, routing, analytics, or evidence-based explanations, also read [references/llm-data-systems.md](references/llm-data-systems.md).

## Core rule

Do not move to the next phase until the current phase has been:

1. planned,
2. implemented,
3. audited independently,
4. fixed for any material findings.

## Operating mode

- Keep phases small and reviewable.
- Start each phase by briefly stress-testing the current plan and re-stating it if needed.
- Prefer contract/schema work before integration work.
- Prefer shadow mode before behavior cutover.
- Keep prompts, schemas, catalogs, and runtime code aligned.
- For classifier or routing changes, require explicit evaluation and disagreement review before cutover.
- For analytical explanation work, distinguish observed association from true causality.
- Treat tests as necessary but insufficient; always run an explicit audit pass.
- If an audit finds drift from plan, repair the drift before expanding scope.

## Per-phase output

At each phase, provide:

- short phase goal,
- implementation summary,
- audit findings first,
- fixes applied,
- remaining risks,
- next-phase recommendation.

## Guardrails

- Separate planning, implementation, and audit in your own reasoning.
- Audit from an adversarial perspective.
- Check correctness, regressions, contract drift, fallback behavior, and test coverage.
- Keep SKILL.md concise; move detail into reference files.
