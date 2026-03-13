---
name: question-analyzer
description: Use when an inbound user question must be normalized into strict JSON before routing. The skill handles typos, multilingual input, intent classification, topic and tool ranking, period extraction, and chart-intent hints. It must not answer the user, generate SQL, or infer unsupported facts.
---

# Question Analyzer

Use this skill to convert a raw user question into the `question_analysis_v1` JSON contract.

Read [references/workflow.md](references/workflow.md) before producing output.
Read [references/json-contract.md](references/json-contract.md) for the required schema.
Read [references/tool-catalog.md](references/tool-catalog.md) and [references/topic-catalog.md](references/topic-catalog.md) when ranking tools or topics.
Read [references/examples.md](references/examples.md) if the query is noisy, multilingual, or ambiguous.

## Non-negotiable rules

- Output valid JSON only.
- Do not answer the user.
- Do not generate SQL.
- Do not invent dates, entities, metrics, or causal claims.
- If uncertain, use low confidence, explicit ambiguities, or `null` where allowed.
- Treat chart fields as routing hints, not final chart decisions.

## Primary job

Return a structured interpretation that downstream code can validate and use for:

- conceptual vs data routing,
- light vs analyst mode,
- knowledge topic selection,
- tool candidate ranking,
- SQL planning hints,
- chart-intent hints.
