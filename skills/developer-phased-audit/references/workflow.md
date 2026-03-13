# Workflow

Use this sequence for multi-phase work:

1. Define the current phase boundary.
2. Critique the current plan briefly and re-state the phase if needed.
3. State the expected artifact for the phase.
4. Implement only that artifact.
5. Verify mechanically where possible.
6. Audit independently:
   - compare against the phase goal,
   - look for regressions,
   - look for hidden scope expansion,
   - look for contract drift.
7. Fix findings before continuing.
8. Re-state the updated plan before the next phase.

## Extra rules for LLM and analytics tasks

If the phase changes prompts, schemas, routing, or analytical summaries:

1. Define what the model is allowed to decide.
2. Define what remains deterministic in code.
3. Add a contract or schema before integration.
4. Prefer shadow mode before cutover.
5. Review disagreement cases, not only pass cases.
6. Use observational language unless causality is actually established.

## Preferred phase order

For new capabilities, use this order unless there is a strong reason not to:

1. contract and schema,
2. catalogs or metadata,
3. LLM call or prompt wrapper,
4. pipeline integration in shadow mode,
5. behavior cutover,
6. threshold tuning and cleanup.

## When to stop and re-plan

Re-plan before continuing if:

- the contract changes materially,
- audit finds hidden coupling,
- tests require a different rollout order,
- fallback behavior is weaker than before,
- the phase grew beyond its original scope.
