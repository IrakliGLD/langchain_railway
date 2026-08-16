# Plant fleet structure: capacity, vintage, utilisation, ownership

Applies to questions about the generating fleet itself — how much capacity
exists, when it was built, how hard it runs, who owns it — as opposed to how
much it generated. Served from four materialized views:

| View | Answers |
|---|---|
| `by_capacity` | how many plants and how much capacity, by size band |
| `by_commissioning` | fleet vintage — capacity by commissioning period |
| `capacity_factor` | utilisation — generation against installed capacity |
| `ownership_concentration` | HHI and top-owner shares |

## Units differ from everything else in this system — check before comparing

Three scales coexist and mixing them is a factor-of-1000 error:

- `capacity_factor.generation_mwh` is **plain MWh**.
- `quantity` columns elsewhere (`tech_quantity_view`, trade views) are
  **thousand MWh**.
- Prices are GEL/MWh; retail tariffs are GEL/kWh.

Never add or compare a `generation_mwh` figure to a `quantity_*` figure
without converting. State the unit in the answer.

`installed_capacity_mw` is **MW** (power), `generation_mwh` is **MWh**
(energy). They are not the same dimension and do not sum.

## Capacity factor is a ratio, and it is already computed

`capacity_factor` is a ratio in 0–1; `capacity_factor_percent` is the same
number ×100. Quote the column, do not recompute from generation and capacity —
a derived figure appears in no row and the grounding gate strips it.

A capacity factor above 1 is a data problem, not a record-breaking plant. Say
so rather than reporting it as a finding.

## Never pool across bands, technologies or owners

The same rule as retail tariffs, for the same reason. An "average capacity
factor" across hydro, thermal, wind and solar describes no plant: these
technologies have structurally different utilisation, and the mean sits
between them. Report per technology, or per capacity band, and say which.

Weighted and unweighted means differ sharply here because the bands are very
unequal in size. If a single figure is unavoidable, state the weighting.

## Vocabulary

- `by_capacity.entity` and `capacity_factor.capacity_category` share one
  eight-band vocabulary. They are the same bands under two column names.
- `segment` is **`total` only** in all four views. It is not a filter that
  selects anything; do not present results as though a segment had been chosen.
- `ownership_concentration.ownership` values are mixed case — `GIG` is
  uppercase, the others lowercase. Match case-insensitively.

## Seasonality

Capacity and commissioning are **stock** measures: they do not have a season,
and a month-over-month change means a plant was added or retired, not that
demand moved. Capacity factor *does* vary seasonally — strongly, for hydro —
and that variation is real and worth reporting.

## Close with a way to narrow

When an answer spans several technologies or bands, end by offering a targeted
follow-up naming real options — for example "hydro plants above 100 MW",
"capacity commissioned since 2020", or "capacity factor for wind in winter".
