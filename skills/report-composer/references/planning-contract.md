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

- required section kinds and order;
- total word range and exact section-budget arithmetic;
- identifier uniqueness;
- maximum section and chart counts;
- chart-to-section linkage;
- whether evidence and chart requests actually exist and are supported.

Return only the strict `ReportPlan` structure. Do not draft report prose during
planning. Every section requires at least one evidence reference. Every chart
request requires verified evidence references and exactly one assigned section.
Relationship charts require an explicit numeric `x_field` and one or more
numeric `series_fields`. For other charts, provide explicit fields when the
evidence table has more than one plausible axis or series.

If the available evidence cannot support a useful standard report, do not pad
the plan. Surface the missing scope or evidence requirement for clarification.
