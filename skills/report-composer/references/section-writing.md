# Section Writing Rules

Write only the assigned section. Use its objective, word budget, evidence
packet, and verified chart descriptions.

- Ground every numeric statement in the assigned evidence references.
- Equivalent formatting and conventional display rounding are allowed for a
  directly cited value; do not derive a new value or combine operands.
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
