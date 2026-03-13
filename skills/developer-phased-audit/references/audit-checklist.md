# Audit Checklist

Use this checklist after each implementation phase.

## Contract

- Are required fields clearly defined?
- Are enums and nullable fields consistent?
- Is the source of truth in code, not only in docs?
- Is naming precise enough to avoid ambiguous downstream behavior?

## Integration

- Does the new phase preserve current fallback behavior?
- Is the new logic additive before cutover, where possible?
- Are low-confidence or invalid outputs handled safely?
- Are logs and traces still useful for debugging?

## Quality

- Are there missing tests for positive, negative, and ambiguous cases?
- Did the implementation accidentally broaden scope?
- Are there obvious performance or token-cost issues?
- Is there duplicate logic that will drift later?

## LLM and data systems

- Is the LLM output schema strict enough for code to trust?
- Are enum choices, nullability, and confidence semantics coherent?
- Is there a shadow-mode or low-risk rollout path?
- Are disagreement cases reviewed, not just successful examples?
- Are observational findings clearly separated from causal claims?
- Are prompt instructions, examples, and runtime validation aligned?

## Skills and prompts

- Is SKILL.md concise?
- Are detailed rules moved to references?
- Does the prompt contract match the runtime schema?
- Are examples consistent with the allowed output schema?
