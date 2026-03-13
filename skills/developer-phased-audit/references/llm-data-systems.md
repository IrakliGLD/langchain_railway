# LLM and Data Systems

Use this reference when the task involves structured LLM outputs, routing, query classification, analytical summaries, or evidence-based explanations.

## Planning lens

For these tasks, the plan should answer:

1. What does the model decide?
2. What does code validate or override?
3. What is the contract shape?
4. What are the fallback paths?
5. How will success and disagreement be measured?

## Recommended sequence

1. Define the contract.
2. Define catalogs and enums.
3. Add the model call in shadow mode.
4. Compare against existing behavior.
5. Audit disagreement cases.
6. Cut over only when fallback behavior remains safe.

## Analytical claims

- Prefer `observed`, `associated`, `consistent with`, and `likely pressure` over strong causal wording.
- Use true causal wording only when the evidence actually supports it.
- If the dataset is observational, say so in your own reasoning and audit.

## Evaluation

For classifier or routing changes, include:

- clean positive cases,
- noisy or typo cases,
- multilingual cases,
- ambiguous cases,
- false-positive checks,
- fallback checks.
