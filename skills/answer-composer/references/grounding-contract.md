# Grounding Contract

## SummaryEnvelope JSON schema

The structured summarizer must return JSON matching this exact schema:

```json
{
  "answer": "string (min_length=1)",
  "claims": ["string", "..."],
  "citations": ["string", "..."],
  "confidence": 0.0
}
```

### Field definitions

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `answer` | string | min_length=1 | The complete user-facing answer text |
| `claims` | list[string] | | Factual claims made in the answer (each separately verifiable) |
| `citations` | list[string] | | Source anchors for the claims |
| `confidence` | float | 0.0 – 1.0 | Self-assessed confidence. Set below 0.5 if confidence is low |

## Valid citation anchors

- `data_preview` — numeric values from query results
- `statistics` — computed statistics (averages, CAGR, correlations)
- `domain_knowledge` — facts from domain knowledge base
- `conversation_history` — facts from prior conversation turns

## Grounding modes

### Normal grounding (default)

"Ground claims in provided DATA_PREVIEW and STATISTICS."

The grounding check extracts numeric tokens from the answer+claims and verifies ≥70% appear in the source data. If the check fails, a strict retry is triggered.

### Strict grounding (retry mode)

"STRICT GROUNDING: Every numeric value in answer/claims must appear verbatim in DATA_PREVIEW or STATISTICS. If unavailable, explicitly say that the value is not available in provided data."

### Grounding fallback

If both normal and strict grounding fail, the system returns a conservative answer:
"I could not fully ground a detailed narrative from the provided data preview. Please refine the query or narrow the period for a more precise grounded answer."
(confidence: 0.2, citations: ["guardrail_grounding_fallback"])

## Grounding check mechanics

1. Extract all numeric tokens from answer + claims text
2. Extract all numeric tokens from data_preview + stats_hint (including percentage aliases: 0.32 matches 32%)
3. Compute match_ratio = matched / total_answer_tokens
4. Pass if match_ratio ≥ 0.70

## Implications for answer generation

- Prefer citing exact numbers from the data preview rather than rounding or computing new values
- When the data shows ratio values (0.32), you can express them as percentages (32%) — both will pass grounding
- If a statistic is not in data_preview or stats_hint, state that it is not available rather than fabricating it
- Lower confidence score when the answer relies heavily on domain knowledge rather than query-specific data
