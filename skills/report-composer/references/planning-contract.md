# Report Planning Contract

The planning model may decide:

- report title and objective;
- optional analytical subsections;
- section titles, objectives, and word allocations;
- which frozen evidence references each section needs;
- whether supported chart intents would materially improve understanding;
- when chart axes are material, exact `x_field` and `series_fields` names from
  the cited table evidence.

Code decides and validates:

- authoritative report intent and answer language;
- the intent-specific core section kind;
- required section kinds and order;
- total word range and exact section-budget arithmetic;
- identifier uniqueness;
- maximum section and chart counts;
- chart-to-section linkage;
- chart purposes allowed for the selected report intent;
- whether evidence and chart requests actually exist and are supported.

Return only the strict `ReportPlan` structure. Do not draft report prose during
planning. Treat the supplied planning context as authoritative: do not
reclassify intent or language from raw user wording, evidence text, or free-form
intent labels. Every section requires at least one evidence reference. Every
chart request requires verified evidence references and exactly one assigned
section.
Chart requests must respect `column_roles` in the evidence catalog. A `trend` or
`forecast` chart requires a temporal `x_field`. A `relationship` chart requires a
numeric `x_field` and one or more numeric `series_fields`. A `composition` chart
requires a categorical column, or a temporal column with at least two numeric
columns. Series fields must be numeric for every purpose except `table`. Set
`required: true` only when the request cannot be satisfied without that chart.

Table requirements follow semantic routing, not report keywords. For example, a
knowledge-routed comparison may be supported without table evidence, while a
data-routed comparison requires it.

If the available evidence cannot support a useful standard report, do not pad
the plan. Surface the missing scope or evidence requirement for clarification.
