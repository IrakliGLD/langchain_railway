# Section Writing Rules

Write only the assigned section. Use its objective, word budget, evidence
packet, and verified chart descriptions.

- Ground every numeric statement in the assigned evidence references.
- Equivalent formatting and conventional display rounding are allowed for a
  directly cited value.
- New arithmetic values are allowed only through `derived_claims`. Select a
  supported operation, identify every operand by its table evidence reference,
  zero-based row index, and column, and provide the exact displayed value and
  unit. Code recomputes the result before accepting the paragraph.
- Use `mean`, `sum`, or `difference` only with compatible operand units. Use
  `percent_change` or `ratio` only with compatible units and a `%` result. Use
  `percentage_point_change` only for ratio/share or percent operands.
- Use `sum` only for additive energy, capacity, or monetary-amount units; never
  sum prices, tariffs, ratios, exchange rates, or other intensive values.
- Do not derive values from narrative evidence, missing rows, or columns without
  a declared unit. Do not include a `derived_claims` entry that is absent from
  the paragraph text.
- Write the exact `display_value` and canonical `unit` together in the paragraph
  text so the code-verified result is unambiguously tied to the prose.
- Include units and periods with values.
- Distinguish observation from explanation. Prefer “observed”, “associated
  with”, “consistent with”, or “likely pressure” unless causal evidence exists.
- Do not repeat the executive summary or another section's full argument.
- Do not cite a chart that is not assigned to the section.
- Do not introduce facts from model memory or independently retrieve evidence.
- State material uncertainty or missing evidence directly.
- Write headings and labels in the requested response language.
- Use every evidence reference assigned in `required_evidence_refs` at least
  once across the section paragraphs.
- Keep the section inside the exact inclusive word-count range supplied by the
  runtime validation rules.
- A rejected candidate may receive at most two evidence-scoped repairs. Each
  repair must address only the supplied typed validation errors.

Return only the structured section paragraphs and their evidence references
defined by the supplied output schema. Code derives chart placement and word
count separately before assembly.
