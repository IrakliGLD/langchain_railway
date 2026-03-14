# Examples

## Example 1

Input:

`what is genex?`

Expected shape:

- `version = question_analysis_v1`
- `classification.query_type = conceptual_definition`
- `routing.preferred_path = knowledge`
- `routing.needs_sql = false`
- `knowledge.candidate_topics[0].name = market_structure`

## Example 2

Input:

`why does balancing electricity price changed in november 2021?`

Expected shape:

- `version = question_analysis_v1`
- `classification.query_type = data_explanation`
- `classification.analysis_mode = analyst`
- `routing.preferred_path = sql`
- `knowledge.candidate_topics` should prioritize `balancing_price`
- `sql_hints.period.start_date = 2021-11-01`
- `sql_hints.period.end_date = 2021-11-30`
- `analysis_requirements.needs_driver_analysis = true`
- `analysis_requirements.derived_metrics` should include bounded requests such as `mom_absolute_change`, `mom_percent_change`, and `share_delta_mom`
- `visualization.chart_recommended = false`

## Example 3

Input:

`2024 enguri tariff in usd`

Expected shape:

- `classification.query_type = factual_lookup` or `data_retrieval`
- `routing.preferred_path = tool`
- `tooling.candidate_tools[0].name = get_tariffs`

## Example 4

Input:

`为什么2021年11月平衡电价变了？`

Expected shape:

- `language.input_language = zh`
- `canonical_query_en` should preserve the balancing-price meaning
- `classification.query_type = data_explanation`
- `routing.preferred_path = sql`
