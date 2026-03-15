# Chart Strategy Rules

## Dimension separation (CRITICAL)

NEVER mix different dimensions on the same chart:

- Don't mix % (shares) with prices (GEL/USD)
- Don't mix prices with quantities (MWh)
- Don't mix exchange rate (xrate) with prices or shares
- Don't mix different units (GEL vs USD vs % vs MWh)

If a query involves multiple dimensions → create separate chart groups.

## Examples

| Query asks for | chart_strategy | Groups |
|---------------|----------------|--------|
| "price and shares" | multiple | Group 1: price (GEL/MWh) line. Group 2: shares (%) stacked_area |
| "price and exchange rate" | multiple | Group 1: balancing_price_gel (GEL/MWh) line. Group 2: xrate (GEL/USD) line |
| "generation composition" | single | Group 1: share_hydro, share_thermal, share_wind (%) stacked_area |
| "summer vs winter price" | single | Group 1: avg_bal_price_gel (GEL/MWh) bar |

## Chart type selection

| Type | When to use |
|------|------------|
| `line` | Price trends, exchange rate trends, time series |
| `bar` | Entity comparisons, monthly comparisons, categorical data |
| `stacked_bar` | Composition/shares as discrete periods |
| `stacked_area` | Composition/shares as continuous time series, generation mix |

## Constraints

- Max 5 metrics per chart group to avoid clutter.
- y_axis_label must include the physical unit (GEL/MWh, %, thousand MWh, GEL/USD).
- Chart titles should be descriptive and in the user's query language.
