# Plant fleet structure: capacity, vintage, utilisation, ownership

Applies to questions about the generating fleet itself — how much capacity
exists, when it was built, how hard it runs, who owns it — as opposed to how
much it generated. Served from four materialized views:

| View | Measures | Columns |
|---|---|---|
| `by_capacity` | **generation** and plant count, split by capacity band | `quantity`, `facility_count` |
| `by_commissioning` | **generation**, split by commissioning cohort | `quantity` only |
| `capacity_factor` | utilisation — generation against installed capacity | `capacity_factor`, `installed_capacity_mw`, `generation_mwh` |
| `ownership_concentration` | HHI and top-owner shares | `hhi`, `top1/3/5_share`, `owner_count`, `total_generation` |

**`by_capacity` does not contain installed capacity.** The band label is a MW
range; the measured value is energy. `installed_capacity_mw` exists only in
`capacity_factor`. Saying "installed capacity by band" of a `by_capacity`
result reports generation as though it were capacity.

`by_commissioning` has no plant count — only `quantity`. A zero is real
(`after 2020` is 0 throughout 2020), not a gap.

## The two partitions reconcile — use it as a check

`by_capacity` and `by_commissioning` split the **same monthly total**, and that
total is `ownership_concentration.total_generation` for the same month.
Verified on 2020-01: the eight capacity bands sum to 1000.976, the five
commissioning cohorts sum to 1000.976, and `total_generation` is 1000.976.

So the three columns carry the same unit. If a query's bands do not reconcile
to the month's total, a filter dropped rows — say so rather than reporting the
partial sum as the fleet.

## ownership_concentration mixes scales in one row

- `hhi` is on the standard **0–10000** scale, not 0–1 and not a percent
  (observed 1400–4900). Conventional reading: below 1500 unconcentrated,
  1500–2500 moderate, above 2500 highly concentrated. Georgian generation sits
  mostly at or above 2500 — do not call a value near 2800 low.
- `top1_share` / `top3_share` / `top5_share` are **ratios in 0–1** (0.42, 0.82,
  0.88). Multiply by 100 to state a percentage; never print the raw ratio with
  a `%` sign.
- `owner_count` changes over time (63 in 2020, 68 in 2021) — do not treat it as
  a constant.

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
