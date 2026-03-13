# Workflow

Use this sequence:

1. Read the raw question.
2. Normalize typos and convert the meaning to clear English for `canonical_query_en`.
3. Detect input and answer language.
4. Classify query type and analysis mode.
5. Rank candidate knowledge topics.
6. Rank candidate tools.
7. Extract period, metric, entities, and dimensions if explicit or strongly implied.
8. Set chart-intent hints.
9. Return JSON only.

## Guardrails

- Do not answer the question.
- Do not infer causality.
- Do not force certainty when ambiguity is real.
- Prefer low confidence to fabricated precision.
