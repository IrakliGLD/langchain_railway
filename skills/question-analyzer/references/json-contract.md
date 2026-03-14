# JSON Contract

The output must match `question_analysis_v1`.

The runtime source of truth is [contracts/question_analysis.py](../../../contracts/question_analysis.py).
The generated schema snapshot is [schemas/question_analysis.schema.json](../../../schemas/question_analysis.schema.json).

## Required top-level fields

- `version`
- `raw_query`
- `canonical_query_en`
- `language`
- `classification`
- `routing`
- `knowledge`
- `tooling`
- `sql_hints`
- `visualization`
- `analysis_requirements`

## Important rules

- Dates must use `YYYY-MM-DD`.
- `preferred_path` must be one of: `knowledge`, `tool`, `sql`, `clarify`, `reject`.
- `query_type` must be one of the runtime enum values.
- `candidate_topics` and `candidate_tools` are ranked lists, not final decisions.
- `analysis_requirements.derived_metrics` must choose only supported derived metric names from the runtime catalog.
- `analysis_requirements` specifies needed derived evidence only; it must never include computed values.
- `chart_requested_by_user` and `chart_recommended` are booleans.
- `preferred_chart_family` is a hint only.
