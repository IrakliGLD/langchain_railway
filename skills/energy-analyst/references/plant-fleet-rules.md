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

`capacity_factor.generation_mwh` is **plain MWh**. Every other quantity here —
`by_capacity.quantity`, `by_commissioning.quantity`,
`ownership_concentration.total_generation` — is **thousand MWh**. Exactly
1000× apart, confirmed against the database on 2020-01, where every capacity
band matched to the last decimal:

| band | `capacity_factor` (MWh) | `by_capacity` (thousand MWh) |
|---|---|---|
| 21-50 | 46,179 | 46.179 |
| 201-500 | 540,804 | 540.804 |
| 101-200 | 137,424 | 137.424 |
| more than 500 | 171,375 | 171.375 |

Never add or compare across that boundary without converting, and state the
unit in the answer.

`installed_capacity_mw` is **MW** (power); `generation_mwh` is **MWh**
(energy). Different dimensions — they do not sum.

## Capacity factor is a ratio, and it is already computed

`capacity_factor` = `generation_mwh / (installed_capacity_mw × hours_in_month)`,
verified to nine decimals against the published column. `hours_in_month` is
calendar hours — 744 in January.

So quote the column. Recomputing produces a figure that appears in no row, and
the grounding gate strips it — the failure that repeatedly gutted retail
answers in this same work.

`capacity_factor` is a ratio in 0–1; `capacity_factor_percent` is that ×100.
Pick one; never multiply the percent column again.

A capacity factor above 1 is a data problem, not a record-breaking plant. Say
so rather than reporting it as a finding.

## trade_by_ownership is trade, not generation

Its monthly total does **not** equal `ownership_concentration.total_generation`
— 993.063 against 1000.976 in 2020-01. They measure different things, so never
present one as a share of the other, and do not "reconcile" the gap.

It also has **no `segment` column**, unlike the other four views.

(Observed but unconfirmed: that 2020-01 gap of 7.913 equals wind output for the
same month exactly. One month is not a rule — do not state it as one.)

## Never pool across bands, technologies or owners

The same rule as retail tariffs, for the same reason. An "average capacity
factor" across hydro, thermal, wind and solar describes no plant: these
technologies have structurally different utilisation, and the mean sits
between them. Report per technology, or per capacity band, and say which.

Weighted and unweighted means differ sharply here because the bands are very
unequal in size. If a single figure is unavoidable, state the weighting.

## The eight capacity bands — all of them, in order

`by_capacity.entity` and `capacity_factor.capacity_category` share one
**eight**-band vocabulary. They are the same bands under two column names.

| order | band (MW) |
|---|---|
| 1 | `<=5` |
| 2 | `6-10` |
| 3 | `11-20` |
| 4 | `21-50` |
| 5 | `51-100` |
| 6 | `101-200` |
| 7 | `201-500` |
| 8 | `more than 500` |

**Never sort these alphabetically.** They are text, so `ORDER BY
capacity_category` produces `101-200, 11-20, 201-500, 21-50, 51-100, 6-10,
<=5, more than 500` — a sequence that reads as meaningless and makes any
"largest band" or "smallest band" claim wrong. Sort by
`capacity_category_order`, which exists for exactly this reason.

`by_capacity` has **no** order column, so apply the order above when
presenting its bands.

An answer covering "all bands" must account for all eight. Reporting a subset
without saying so understates the fleet — and if the bands do not sum to the
month's total, rows were dropped (see the reconciliation above).

Do not confuse these with the **five** commissioning cohorts in
`by_commissioning`: `<=1990`, `1991-2000`, `2001-2010`, `2011-2020`,
`after 2020`. Those are vintages, not sizes.

## Vocabulary
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
