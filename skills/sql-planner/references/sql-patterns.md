# SQL Patterns

Canonical SQL patterns per view. These serve as few-shot templates for the planner.

## Views available

| View | Key columns | Notes |
|------|-------------|-------|
| `price_with_usd` | date, p_bal_gel, p_bal_usd, xrate | Monthly balancing prices + exchange rate |
| `tariff_with_usd` | date, entity, tariff_gel, tariff_usd | Monthly entity tariffs |
| `trade_derived_entities` | date, segment, entity, quantity | Entity quantities. **Data from 2020 only**. Segment filter: `LOWER(REPLACE(segment, ' ', '_')) = 'balancing'` |
| `tech_quantity_view` | date, time_month, type_tech, quantity_tech | Generation by technology (thousand MWh) |
| `monthly_cpi_mv` | date, cpi_type, cpi | CPI indices. Filter: `cpi_type = 'electricity_gas_and_other_fuels'` |
| `energy_balance_long_mv` | year, category, value | Annual energy balance. Use yearly aggregation. |

## Pattern: Monthly balancing price

```sql
SELECT
  EXTRACT(YEAR FROM date) AS year,
  EXTRACT(MONTH FROM date) AS month,
  AVG(p_bal_usd) AS avg_balancing_usd
FROM price_with_usd
GROUP BY 1,2
ORDER BY 1,2
LIMIT 3750;
```

## Pattern: Share CTE (balancing composition)

CRITICAL: Always filter entities in denominator to only include the 7 relevant balancing entities.

```sql
WITH shares AS (
  SELECT
    t.date,
    SUM(t.quantity) AS total_qty,
    SUM(CASE WHEN t.entity = 'import' THEN t.quantity ELSE 0 END) AS qty_import,
    SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity ELSE 0 END) AS qty_dereg_hydro,
    SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity ELSE 0 END) AS qty_reg_hpp
  FROM trade_derived_entities t
  WHERE LOWER(REPLACE(t.segment, ' ', '_')) = 'balancing'
    AND t.entity IN ('import', 'deregulated_hydro', 'regulated_hpp',
                     'regulated_new_tpp', 'regulated_old_tpp',
                     'renewable_ppa', 'thermal_ppa')
  GROUP BY t.date
)
SELECT
  TO_CHAR(p.date, 'YYYY-MM') AS month,
  p.p_bal_gel,
  (s.qty_import / NULLIF(s.total_qty,0)) AS share_import,
  (s.qty_dereg_hydro / NULLIF(s.total_qty,0)) AS share_deregulated_hydro,
  (s.qty_reg_hpp / NULLIF(s.total_qty,0)) AS share_regulated_hpp
FROM price_with_usd p
LEFT JOIN shares s ON s.date = p.date
ORDER BY p.date
LIMIT 3750;
```

## Pattern: Season derivation

```sql
CASE WHEN EXTRACT(MONTH FROM date) IN (4,5,6,7) THEN 'summer' ELSE 'winter' END AS season
```

## Pattern: Combined price + tariffs + xrate

```sql
WITH tariffs AS (
  SELECT
    d.date,
    (SELECT t1.tariff_gel FROM tariff_with_usd t1 WHERE t1.date = d.date AND t1.entity = 'ltd "engurhesi"1' LIMIT 1) AS enguri_tariff_gel,
    (SELECT t2.tariff_gel FROM tariff_with_usd t2 WHERE t2.date = d.date AND t2.entity = 'ltd "gardabni thermal power plant"' LIMIT 1) AS gardabani_tpp_tariff_gel,
    (SELECT AVG(t3.tariff_gel) FROM tariff_with_usd t3 WHERE t3.date = d.date AND t3.entity IN ('ltd "mtkvari energy"', 'ltd "iec" (tbilresi)', 'ltd "g power" (capital turbines)')) AS grouped_old_tpp_tariff_gel
  FROM price_with_usd d
)
SELECT
  p.date,
  p.p_bal_gel,
  p.p_bal_usd,
  p.xrate,
  tr.enguri_tariff_gel,
  tr.gardabani_tpp_tariff_gel,
  tr.grouped_old_tpp_tariff_gel
FROM price_with_usd p
LEFT JOIN tariffs tr ON tr.date = p.date
ORDER BY p.date
LIMIT 3750;
```

## Pattern: Tariff comparison

```sql
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  entity,
  tariff_gel
FROM tariff_with_usd
WHERE entity IN ('ltd "engurhesi"1', 'ltd "gardabni thermal power plant"')
ORDER BY date, entity
LIMIT 3750;
```

## Pattern: Generation by technology

```sql
SELECT
  type_tech,
  SUM(quantity_tech) AS total_generation_thousand_mwh
FROM tech_quantity_view
WHERE EXTRACT(YEAR FROM date) = 2023
GROUP BY type_tech
ORDER BY total_generation_thousand_mwh DESC
LIMIT 3750;
```

## Pattern: Energy security (local vs import-dependent)

```sql
SELECT
  time_month,
  SUM(CASE WHEN type_tech IN ('hydro', 'wind', 'solar') THEN quantity ELSE 0 END) AS local_generation_thousand_mwh,
  SUM(CASE WHEN type_tech IN ('thermal', 'import') THEN quantity ELSE 0 END) AS import_dependent_thousand_mwh,
  ROUND(100.0 * SUM(CASE WHEN type_tech IN ('hydro', 'wind', 'solar') THEN quantity ELSE 0 END) /
        NULLIF(SUM(CASE WHEN type_tech IN ('hydro', 'wind', 'solar', 'thermal', 'import') THEN quantity ELSE 0 END), 0), 1) AS local_share_pct
FROM tech_quantity_view
GROUP BY time_month
ORDER BY time_month;
```

## Pattern: CPI trend

```sql
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  cpi AS electricity_fuels_cpi
FROM monthly_cpi_mv
WHERE cpi_type = 'electricity_gas_and_other_fuels'
ORDER BY date
LIMIT 3750;
```

## Aggregation disambiguation

| User says | GROUP BY? | Example |
|-----------|-----------|---------|
| "total generation" (single number) | No | `SELECT SUM(quantity_tech) * 1000 AS total_generation_mwh` |
| "generation by technology" (breakdown) | Yes | `GROUP BY type_tech` |
| "average price" (single number) | No | `SELECT AVG(p_bal_gel) AS average_balancing_price_gel` |
| "share of each technology" | CTE | Use CTE pattern to calculate shares properly |

## Forecasting

NEVER use SQL regression functions. For trend/forecast queries, return ONLY historical data. The Python visualization layer calculates trendlines and forecasts using scipy.stats.linregress.
