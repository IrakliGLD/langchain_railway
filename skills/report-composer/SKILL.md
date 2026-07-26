---
name: report-composer
description: Use when planning, drafting, charting, validating, or assembling a multi-section evidence-grounded analytical report. Not for brief or standard chat answers.
---

# Report Composer

Use this skill only for typed report-mode requests. The runtime contract in
`contracts/report.py` is authoritative for structure, identifiers, word
budgets, chart linkage, and evidence references.

Load the stage-specific reference required by the current report phase:

- [references/standard-structure.md](references/standard-structure.md)
- [references/planning-contract.md](references/planning-contract.md)
- [references/section-writing.md](references/section-writing.md)
- [references/chart-integration.md](references/chart-integration.md)
- [references/final-assembly.md](references/final-assembly.md)

## Non-negotiable rules

- Use one frozen evidence manifest across planning, section writing, and assembly.
- Do not invent calculations, values, sources, chart data, or causal claims.
- Assign every section and chart explicit evidence references.
- Treat charts as verified evidence exhibits, not decoration.
- Preserve validated section text during final assembly.
- If evidence cannot support the standard report, request clarification or report the limitation.
