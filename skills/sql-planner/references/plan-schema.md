# Plan Schema

The plan is a JSON object output before the `---SQL---` separator.

## Fields

```json
{
  "intent": "trend_analysis | general",
  "target": "<metric name>",
  "period": "YYYY-YYYY or YYYY-MM to YYYY-MM",
  "chart_strategy": "single | multiple",
  "chart_groups": [
    {
      "type": "line | bar | stacked_bar | stacked_area",
      "metrics": ["column_name1", "column_name2"],
      "title": "Chart title",
      "y_axis_label": "Unit (e.g., GEL/MWh, %, thousand MWh)"
    }
  ]
}
```

## Field rules

- **intent**: Use `trend_analysis` when analysis_mode is "analyst"; `general` otherwise.
- **target**: The primary metric name (e.g., `balancing_price_gel`, `generation_by_tech`, `import_dependence`).
- **period**: The time range from the query. If user says "trends" or "historical" with no dates, omit date bounds in SQL.
- **chart_strategy**: `single` when all metrics share the same unit; `multiple` when dimensions differ.
- **chart_groups**: One group per chart. Each group has homogeneous units. Max 5 metrics per group to avoid clutter.

## Output format

The complete output is a single string with two parts separated by `---SQL---`:

```
{plan_json}
---SQL---
SELECT ...
```

No markdown fences. No comments in SQL. Raw JSON + raw SELECT.
