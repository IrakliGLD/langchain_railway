"""
Few-Shot Examples for LLM Plan+SQL Generation

These examples demonstrate correct patterns for common query types.
Used in llm_generate_plan_and_sql() to guide the model.

Categories:
1. Energy Security
2. Balancing Price (drivers, trends, summer/winter)
3. Support Schemes (PPA, CfD)
4. Stakeholders
5. Demand & Drivers
6. Generation & Mix
7. Ownership Structure
8. Tariffs (regulated entities, comparisons)
9. Exchange Rate & Currency
10. Seasonal Analysis
"""

# =============================================================================
# CATEGORY 1: ENERGY SECURITY
# =============================================================================

ENERGY_SECURITY_EXAMPLES = """
EXAMPLE 1.1 - Energy Security Assessment (Georgian):
Query: "როგორ შეაფასებ საქართველოს ენერგეტიკულ უსაფრთხოებას?"
Plan:
{
  "intent": "general",
  "target": "energy_security",
  "period": "recent_years"
}
---SQL---
SELECT
    time_month,
    SUM(CASE WHEN type_tech IN ('hydro', 'wind', 'solar') THEN quantity ELSE 0 END) as local_generation_thousand_mwh,
    SUM(CASE WHEN type_tech IN ('thermal', 'import') THEN quantity ELSE 0 END) as import_dependent_thousand_mwh,
    SUM(CASE WHEN type_tech IN ('hydro', 'wind', 'solar', 'thermal', 'import') THEN quantity ELSE 0 END) as total_generation_thousand_mwh,
    ROUND(100.0 * SUM(CASE WHEN type_tech IN ('hydro', 'wind', 'solar') THEN quantity ELSE 0 END) /
          NULLIF(SUM(CASE WHEN type_tech IN ('hydro', 'wind', 'solar', 'thermal', 'import') THEN quantity ELSE 0 END), 0), 1) as local_share_pct
FROM tech_quantity_view
WHERE time_month >= '2020-01'
GROUP BY time_month
ORDER BY time_month;

EXAMPLE 1.2 - Import Dependence Trends:
Query: "Show me Georgia's import dependence trends from 2020 to 2024"
Plan:
{
  "intent": "trend_analysis",
  "target": "import_dependence",
  "period": "2020-2024"
}
---SQL---
SELECT
    time_month,
    SUM(CASE WHEN type_tech = 'import' THEN quantity ELSE 0 END) as direct_import_thousand_mwh,
    SUM(CASE WHEN type_tech = 'thermal' THEN quantity ELSE 0 END) as thermal_thousand_mwh,
    SUM(CASE WHEN type_tech IN ('hydro', 'wind', 'solar') THEN quantity ELSE 0 END) as local_renewables_thousand_mwh
FROM tech_quantity_view
WHERE time_month >= '2020-01' AND time_month <= '2024-12'
GROUP BY time_month
ORDER BY time_month;

EXAMPLE 1.3 - Winter Vulnerability:
Query: "How vulnerable is Georgia in winter months?"
Plan:
{
  "intent": "general",
  "target": "winter_vulnerability",
  "period": "recent_winters"
}
---SQL---
SELECT
    EXTRACT(YEAR FROM time_month::date) as year,
    AVG(CASE WHEN type_tech = 'import' THEN quantity ELSE 0 END) as avg_import_thousand_mwh,
    AVG(CASE WHEN type_tech = 'thermal' THEN quantity ELSE 0 END) as avg_thermal_thousand_mwh,
    AVG(CASE WHEN type_tech = 'hydro' THEN quantity ELSE 0 END) as avg_hydro_thousand_mwh
FROM tech_quantity_view
WHERE EXTRACT(MONTH FROM time_month::date) IN (1, 2, 3, 10, 11, 12)
  AND time_month >= '2020-01'
GROUP BY EXTRACT(YEAR FROM time_month::date)
ORDER BY year;
"""

# =============================================================================
# CATEGORY 2: BALANCING PRICE (DRIVERS, TRENDS, SUMMER/WINTER)
# =============================================================================

BALANCING_PRICE_EXAMPLES = """
EXAMPLE 2.1 - Balancing Price Drivers (Georgian):
Query: "რატომ გაიზარდა საბალანსო ფასი 2024 წელს?"
Plan:
{
  "intent": "correlation",
  "target": "balancing_price",
  "period": "2024",
  "chart_strategy": "single",
  "chart_groups": [{
    "type": "line",
    "metrics": ["p_bal_gel", "xrate", "share_import", "share_renewable_ppa"],
    "title": "Balancing Price Drivers (2024)",
    "y_axis_label": "Mixed units"
  }]
}
---SQL---
SELECT
    p.time_month,
    p.p_bal_gel,
    p.p_bal_usd,
    p.xrate,
    MAX(CASE WHEN t.entity = 'import' THEN t.share ELSE 0 END) as share_import,
    MAX(CASE WHEN t.entity = 'renewable_ppa' THEN t.share ELSE 0 END) as share_renewable_ppa,
    MAX(CASE WHEN t.entity = 'deregulated_hydro' THEN t.share ELSE 0 END) as share_deregulated_hydro,
    MAX(CASE WHEN t.entity = 'thermal_ppa' THEN t.share ELSE 0 END) as share_thermal_ppa
FROM price_with_usd p
LEFT JOIN (
    SELECT
        time_month,
        entity,
        SUM(quantity) as entity_qty,
        SUM(SUM(quantity)) OVER (PARTITION BY time_month) as total_qty,
        ROUND(100.0 * SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY time_month), 0), 2) as share
    FROM trade_derived_entities
    WHERE segment = 'Balancing Electricity'
    GROUP BY time_month, entity
) t ON p.time_month = t.time_month
WHERE p.time_month >= '2024-01'
GROUP BY p.time_month, p.p_bal_gel, p.p_bal_usd, p.xrate
ORDER BY p.time_month;

EXAMPLE 2.2 - Summer vs Winter Price Comparison:
Query: "Compare summer and winter balancing prices from 2020 to 2023"
Plan:
{
  "intent": "comparison",
  "target": "balancing_price",
  "period": "2020-2023",
  "comparison_type": "seasonal"
}
---SQL---
SELECT
    EXTRACT(YEAR FROM time_month::date) as year,
    CASE
        WHEN EXTRACT(MONTH FROM time_month::date) IN (4, 5, 6, 7) THEN 'Summer'
        ELSE 'Winter'
    END as season,
    ROUND(AVG(p_bal_gel), 2) as avg_price_gel_per_mwh,
    ROUND(AVG(p_bal_usd), 2) as avg_price_usd_per_mwh,
    ROUND(AVG(xrate), 4) as avg_exchange_rate
FROM price_with_usd
WHERE time_month >= '2020-01' AND time_month <= '2023-12'
GROUP BY EXTRACT(YEAR FROM time_month::date),
         CASE WHEN EXTRACT(MONTH FROM time_month::date) IN (4, 5, 6, 7) THEN 'Summer' ELSE 'Winter' END
ORDER BY year, season;

EXAMPLE 2.3 - Balancing Price Trend with Decomposition:
Query: "Show me balancing price trends and composition changes 2022-2024"
Plan:
{
  "intent": "trend_analysis",
  "target": "balancing_price_composition",
  "period": "2022-2024"
}
---SQL---
SELECT
    p.time_month,
    p.p_bal_gel,
    p.xrate,
    MAX(CASE WHEN t.entity = 'regulated_hpp' THEN t.share ELSE 0 END) as share_regulated_hpp,
    MAX(CASE WHEN t.entity = 'deregulated_hydro' THEN t.share ELSE 0 END) as share_deregulated_hydro,
    MAX(CASE WHEN t.entity = 'thermal_ppa' THEN t.share ELSE 0 END) as share_thermal_ppa,
    MAX(CASE WHEN t.entity = 'renewable_ppa' THEN t.share ELSE 0 END) as share_renewable_ppa,
    MAX(CASE WHEN t.entity = 'import' THEN t.share ELSE 0 END) as share_import
FROM price_with_usd p
LEFT JOIN (
    SELECT
        time_month,
        entity,
        ROUND(100.0 * SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY time_month), 0), 2) as share
    FROM trade_derived_entities
    WHERE segment = 'Balancing Electricity'
    GROUP BY time_month, entity
) t ON p.time_month = t.time_month
WHERE p.time_month >= '2022-01' AND p.time_month <= '2024-12'
GROUP BY p.time_month, p.p_bal_gel, p.xrate
ORDER BY p.time_month;

EXAMPLE 2.4 - Price Prediction Query (forward-looking):
Query: "What factors will affect balancing price in 2025?"
Plan:
{
  "intent": "general",
  "target": "price_forecast_factors",
  "period": "2024-recent"
}
---SQL---
SELECT
    p.time_month,
    p.p_bal_gel,
    p.xrate,
    MAX(CASE WHEN t.entity = 'renewable_ppa' THEN t.share ELSE 0 END) as share_renewable_ppa,
    MAX(CASE WHEN t.entity = 'deregulated_hydro' THEN t.share ELSE 0 END) as share_deregulated_hydro
FROM price_with_usd p
LEFT JOIN (
    SELECT time_month, entity,
           ROUND(100.0 * SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY time_month), 0), 2) as share
    FROM trade_derived_entities
    WHERE segment = 'Balancing Electricity'
    GROUP BY time_month, entity
) t ON p.time_month = t.time_month
WHERE p.time_month >= '2024-01'
GROUP BY p.time_month, p.p_bal_gel, p.xrate
ORDER BY p.time_month DESC
LIMIT 12;
"""

# =============================================================================
# CATEGORY 3: SUPPORT SCHEMES (PPA, CFD)
# =============================================================================

SUPPORT_SCHEMES_EXAMPLES = """
EXAMPLE 3.1 - CfD Explanation (Georgian):
Query: "რა არის CfD სქემა საქართველოში?"
Plan:
{
  "intent": "general",
  "target": "cfd_explanation",
  "period": null
}
---SQL---
SELECT 1 as conceptual_query;
-- Note: This is a conceptual question, no data query needed

EXAMPLE 3.2 - Renewable PPA Share Trends:
Query: "Show me renewable PPA share in balancing market from 2020 to 2024"
Plan:
{
  "intent": "trend_analysis",
  "target": "renewable_ppa_share",
  "period": "2020-2024"
}
---SQL---
SELECT
    time_month,
    ROUND(100.0 * SUM(CASE WHEN entity = 'renewable_ppa' THEN quantity ELSE 0 END) /
          NULLIF(SUM(quantity), 0), 2) as renewable_ppa_share_pct,
    SUM(CASE WHEN entity = 'renewable_ppa' THEN quantity ELSE 0 END) as renewable_ppa_quantity_thousand_mwh,
    SUM(quantity) as total_balancing_quantity_thousand_mwh
FROM trade_derived_entities
WHERE segment = 'Balancing Electricity'
  AND time_month >= '2020-01' AND time_month <= '2024-12'
GROUP BY time_month
ORDER BY time_month;

EXAMPLE 3.3 - PPA vs Deregulated Hydro Comparison:
Query: "Compare renewable PPA and deregulated hydro shares in balancing electricity"
Plan:
{
  "intent": "comparison",
  "target": "ppa_vs_deregulated",
  "period": "2022-2024"
}
---SQL---
SELECT
    time_month,
    MAX(CASE WHEN entity = 'renewable_ppa' THEN share ELSE 0 END) as renewable_ppa_share_pct,
    MAX(CASE WHEN entity = 'deregulated_hydro' THEN share ELSE 0 END) as deregulated_hydro_share_pct,
    MAX(CASE WHEN entity = 'renewable_ppa' THEN quantity ELSE 0 END) as renewable_ppa_thousand_mwh,
    MAX(CASE WHEN entity = 'deregulated_hydro' THEN quantity ELSE 0 END) as deregulated_hydro_thousand_mwh
FROM (
    SELECT
        time_month,
        entity,
        SUM(quantity) as quantity,
        ROUND(100.0 * SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY time_month), 0), 2) as share
    FROM trade_derived_entities
    WHERE segment = 'Balancing Electricity'
      AND time_month >= '2022-01'
    GROUP BY time_month, entity
) sub
WHERE entity IN ('renewable_ppa', 'deregulated_hydro')
GROUP BY time_month
ORDER BY time_month;
"""

# =============================================================================
# CATEGORY 4: STAKEHOLDERS & MARKET PARTICIPANTS
# =============================================================================

STAKEHOLDERS_EXAMPLES = """
EXAMPLE 4.1 - Market Participants List (Georgian):
Query: "ვინ არიან ბაზრის მთავარი მონაწილეები?"
Plan:
{
  "intent": "list",
  "target": "market_participants",
  "period": null
}
---SQL---
SELECT 1 as conceptual_query;
-- Note: This query requires domain knowledge (GNERC, ESCO, GSE, GENEX)

EXAMPLE 4.2 - Entity Trading Volumes:
Query: "Show me which entities trade the most on balancing market in 2024"
Plan:
{
  "intent": "list",
  "target": "entity_volumes",
  "period": "2024"
}
---SQL---
SELECT
    entity,
    SUM(quantity) as total_quantity_thousand_mwh,
    ROUND(AVG(share), 2) as avg_share_pct,
    COUNT(DISTINCT time_month) as months_active
FROM (
    SELECT
        time_month,
        entity,
        SUM(quantity) as quantity,
        ROUND(100.0 * SUM(quantity) / NULLIF(SUM(SUM(quantity)) OVER (PARTITION BY time_month), 0), 2) as share
    FROM trade_derived_entities
    WHERE segment = 'Balancing Electricity'
      AND time_month >= '2024-01'
    GROUP BY time_month, entity
) sub
GROUP BY entity
ORDER BY total_quantity_thousand_mwh DESC;

EXAMPLE 4.3 - GNERC Role Explanation:
Query: "What is GNERC's role in the electricity market?"
Plan:
{
  "intent": "general",
  "target": "gnerc_role",
  "period": null
}
---SQL---
SELECT 1 as conceptual_query;
-- Note: Domain knowledge query about regulatory body
"""

# =============================================================================
# CATEGORY 5: DEMAND & DRIVERS
# =============================================================================

DEMAND_EXAMPLES = """
EXAMPLE 5.1 - Total Demand Trends:
Query: "Show me total electricity demand from 2020 to 2023"
Plan:
{
  "intent": "trend_analysis",
  "target": "total_demand",
  "period": "2020-2023"
}
---SQL---
SELECT
    time_month,
    SUM(CASE WHEN type_tech IN ('abkhazeti', 'supply-distribution', 'direct customers', 'losses', 'export')
        THEN quantity ELSE 0 END) as total_demand_thousand_mwh
FROM tech_quantity_view
WHERE time_month >= '2020-01' AND time_month <= '2023-12'
GROUP BY time_month
ORDER BY time_month;

EXAMPLE 5.2 - Demand by Segment (Georgian):
Query: "როგორია მოთხოვნის სტრუქტურა სეგმენტების მიხედვით?"
Plan:
{
  "intent": "general",
  "target": "demand_structure",
  "period": "2023"
}
---SQL---
SELECT
    time_month,
    SUM(CASE WHEN type_tech = 'supply-distribution' THEN quantity ELSE 0 END) as supply_distribution_thousand_mwh,
    SUM(CASE WHEN type_tech = 'direct customers' THEN quantity ELSE 0 END) as direct_customers_thousand_mwh,
    SUM(CASE WHEN type_tech = 'losses' THEN quantity ELSE 0 END) as losses_thousand_mwh,
    SUM(CASE WHEN type_tech = 'export' THEN quantity ELSE 0 END) as export_thousand_mwh,
    SUM(CASE WHEN type_tech = 'abkhazeti' THEN quantity ELSE 0 END) as abkhazeti_thousand_mwh
FROM tech_quantity_view
WHERE time_month >= '2023-01' AND time_month <= '2023-12'
  AND type_tech IN ('supply-distribution', 'direct customers', 'losses', 'export', 'abkhazeti')
GROUP BY time_month
ORDER BY time_month;

EXAMPLE 5.3 - Demand Growth Rate:
Query: "What was the demand growth rate from 2015 to 2023?"
Plan:
{
  "intent": "trend_analysis",
  "target": "demand_growth",
  "period": "2015-2023"
}
---SQL---
SELECT
    time_month,
    SUM(CASE WHEN type_tech IN ('abkhazeti', 'supply-distribution', 'direct customers', 'losses', 'export')
        THEN quantity ELSE 0 END) as total_demand_thousand_mwh
FROM tech_quantity_view
WHERE time_month >= '2015-01' AND time_month <= '2023-12'
GROUP BY time_month
ORDER BY time_month;

EXAMPLE 5.4 - Seasonal Demand Patterns:
Query: "Compare summer and winter demand patterns"
Plan:
{
  "intent": "comparison",
  "target": "seasonal_demand",
  "period": "2020-2023"
}
---SQL---
SELECT
    EXTRACT(YEAR FROM time_month::date) as year,
    CASE WHEN EXTRACT(MONTH FROM time_month::date) IN (4, 5, 6, 7) THEN 'Summer' ELSE 'Winter' END as season,
    ROUND(AVG(total_demand), 2) as avg_demand_thousand_mwh
FROM (
    SELECT
        time_month,
        SUM(CASE WHEN type_tech IN ('abkhazeti', 'supply-distribution', 'direct customers', 'losses', 'export')
            THEN quantity ELSE 0 END) as total_demand
    FROM tech_quantity_view
    WHERE time_month >= '2020-01' AND time_month <= '2023-12'
    GROUP BY time_month
) sub
GROUP BY year, season
ORDER BY year, season;
"""

# =============================================================================
# CATEGORY 6: GENERATION & MIX
# =============================================================================

GENERATION_EXAMPLES = """
EXAMPLE 6.1 - Generation Mix Trends:
Query: "Show me generation mix evolution from 2020 to 2024"
Plan:
{
  "intent": "trend_analysis",
  "target": "generation_mix",
  "period": "2020-2024"
}
---SQL---
SELECT
    time_month,
    SUM(CASE WHEN type_tech = 'hydro' THEN quantity ELSE 0 END) as hydro_thousand_mwh,
    SUM(CASE WHEN type_tech = 'thermal' THEN quantity ELSE 0 END) as thermal_thousand_mwh,
    SUM(CASE WHEN type_tech = 'wind' THEN quantity ELSE 0 END) as wind_thousand_mwh,
    SUM(CASE WHEN type_tech = 'solar' THEN quantity ELSE 0 END) as solar_thousand_mwh,
    SUM(CASE WHEN type_tech = 'import' THEN quantity ELSE 0 END) as import_thousand_mwh
FROM tech_quantity_view
WHERE time_month >= '2020-01' AND time_month <= '2024-12'
  AND type_tech IN ('hydro', 'thermal', 'wind', 'solar', 'import')
GROUP BY time_month
ORDER BY time_month;

EXAMPLE 6.2 - Hydro vs Thermal Comparison (Georgian):
Query: "შეადარე ჰიდრო და თერმული გენერაცია 2023 წელს"
Plan:
{
  "intent": "comparison",
  "target": "hydro_vs_thermal",
  "period": "2023"
}
---SQL---
SELECT
    time_month,
    SUM(CASE WHEN type_tech = 'hydro' THEN quantity ELSE 0 END) as hydro_thousand_mwh,
    SUM(CASE WHEN type_tech = 'thermal' THEN quantity ELSE 0 END) as thermal_thousand_mwh,
    ROUND(100.0 * SUM(CASE WHEN type_tech = 'hydro' THEN quantity ELSE 0 END) /
          NULLIF(SUM(CASE WHEN type_tech IN ('hydro', 'thermal') THEN quantity ELSE 0 END), 0), 2) as hydro_share_pct
FROM tech_quantity_view
WHERE time_month >= '2023-01' AND time_month <= '2023-12'
  AND type_tech IN ('hydro', 'thermal')
GROUP BY time_month
ORDER BY time_month;

EXAMPLE 6.3 - Renewable Share:
Query: "What is the share of renewables in total generation?"
Plan:
{
  "intent": "general",
  "target": "renewable_share",
  "period": "2024"
}
---SQL---
SELECT
    time_month,
    SUM(CASE WHEN type_tech IN ('hydro', 'wind', 'solar') THEN quantity ELSE 0 END) as renewable_thousand_mwh,
    SUM(quantity) as total_generation_thousand_mwh,
    ROUND(100.0 * SUM(CASE WHEN type_tech IN ('hydro', 'wind', 'solar') THEN quantity ELSE 0 END) /
          NULLIF(SUM(quantity), 0), 2) as renewable_share_pct
FROM tech_quantity_view
WHERE time_month >= '2024-01'
  AND type_tech IN ('hydro', 'thermal', 'wind', 'solar', 'import')
GROUP BY time_month
ORDER BY time_month;
"""

# =============================================================================
# CATEGORY 7: OWNERSHIP STRUCTURE
# =============================================================================

OWNERSHIP_EXAMPLES = """
EXAMPLE 7.1 - Hydro Plant Ownership:
Query: "Who owns the main hydro plants in Georgia?"
Plan:
{
  "intent": "list",
  "target": "hydro_ownership",
  "period": null
}
---SQL---
SELECT 1 as conceptual_query;
-- Note: Domain knowledge query about tariff_entities ownership

EXAMPLE 7.2 - Thermal Plant List:
Query: "List all thermal power plants and their owners"
Plan:
{
  "intent": "list",
  "target": "thermal_plants",
  "period": null
}
---SQL---
SELECT 1 as conceptual_query;
-- Note: Domain knowledge query about thermal entities (Gardabani, Mtkvari, etc.)

EXAMPLE 7.3 - Entity Generation Volumes:
Query: "Show me generation volumes by major producers in 2024"
Plan:
{
  "intent": "list",
  "target": "producer_volumes",
  "period": "2024"
}
---SQL---
SELECT
    entity,
    SUM(quantity) as total_quantity_thousand_mwh,
    COUNT(DISTINCT time_month) as months_active
FROM trade_derived_entities
WHERE time_month >= '2024-01'
  AND segment IN ('Balancing Electricity', 'Bilateral Contracts & Exchange')
GROUP BY entity
ORDER BY total_quantity_thousand_mwh DESC
LIMIT 15;
"""

# =============================================================================
# CATEGORY 8: TARIFFS (REGULATED ENTITIES, COMPARISONS)
# =============================================================================

TARIFF_EXAMPLES = """
EXAMPLE 8.1 - Regulated Tariff Trends:
Query: "Show me Enguri and Gardabani tariff trends from 2020 to 2024"
Plan:
{
  "intent": "trend_analysis",
  "target": "regulated_tariffs",
  "period": "2020-2024"
}
---SQL---
SELECT
    time_month,
    MAX(CASE WHEN entity = 'ltd "engurhesi"1' THEN tariff_gel ELSE NULL END) as enguri_tariff_gel,
    MAX(CASE WHEN entity = 'ltd "gardabni thermal power plant"' THEN tariff_gel ELSE NULL END) as gardabani_tariff_gel,
    MAX(CASE WHEN entity = 'ltd "engurhesi"1' THEN tariff_usd ELSE NULL END) as enguri_tariff_usd,
    MAX(CASE WHEN entity = 'ltd "gardabni thermal power plant"' THEN tariff_usd ELSE NULL END) as gardabani_tariff_usd
FROM tariff_with_usd
WHERE time_month >= '2020-01' AND time_month <= '2024-12'
  AND entity IN ('ltd "engurhesi"1', 'ltd "gardabni thermal power plant"')
GROUP BY time_month
ORDER BY time_month;

EXAMPLE 8.2 - Hydro vs Thermal Tariff Comparison (Georgian):
Query: "შეადარე ჰიდრო და თერმული ტარიფები 2024 წელს"
Plan:
{
  "intent": "comparison",
  "target": "tariff_comparison_hydro_thermal",
  "period": "2024"
}
---SQL---
SELECT
    time_month,
    AVG(CASE WHEN entity IN (
        'ltd "engurhesi"1',
        'jsc "energo-pro georgia genration" (dzevrulhesi)',
        'jsc "energo-pro georgia genration" (gumathesi)',
        'jsc "georgian water & power" (zhinvalhesi)',
        'ltd "vardnili hpp cascade"'
    ) THEN tariff_gel ELSE NULL END) as avg_hydro_tariff_gel,
    AVG(CASE WHEN entity IN (
        'ltd "gardabni thermal power plant"',
        'ltd "mtkvari energy"',
        'ltd "iec" (tbilresi)',
        'ltd "g power" (capital turbines)'
    ) THEN tariff_gel ELSE NULL END) as avg_thermal_tariff_gel
FROM tariff_with_usd
WHERE time_month >= '2024-01' AND time_month <= '2024-12'
GROUP BY time_month
ORDER BY time_month;

EXAMPLE 8.3 - Number of Regulated Entities Over Time:
Query: "How many regulated entities are there and how has this changed?"
Plan:
{
  "intent": "trend_analysis",
  "target": "regulated_entity_count",
  "period": "2015-2024"
}
---SQL---
SELECT
    EXTRACT(YEAR FROM time_month::date) as year,
    COUNT(DISTINCT entity) as num_regulated_entities,
    COUNT(DISTINCT CASE WHEN entity LIKE '%hesi%' OR entity LIKE '%hpp%' OR entity LIKE '%hydro%'
                        THEN entity END) as num_hydro_entities,
    COUNT(DISTINCT CASE WHEN entity LIKE '%thermal%' OR entity LIKE '%tpp%' OR entity LIKE '%energy%'
                        OR entity LIKE '%tbilresi%' OR entity LIKE '%power%'
                        THEN entity END) as num_thermal_entities
FROM tariff_with_usd
WHERE time_month >= '2015-01' AND time_month <= '2024-12'
GROUP BY EXTRACT(YEAR FROM time_month::date)
ORDER BY year;

EXAMPLE 8.4 - Tariff Increases Analysis:
Query: "Which entities had the largest tariff increases in 2024?"
Plan:
{
  "intent": "comparison",
  "target": "tariff_increases",
  "period": "2023-2024"
}
---SQL---
WITH tariff_change AS (
    SELECT
        entity,
        MAX(CASE WHEN time_month BETWEEN '2023-01' AND '2023-12' THEN tariff_gel END) as tariff_2023_gel,
        MAX(CASE WHEN time_month BETWEEN '2024-01' AND '2024-12' THEN tariff_gel END) as tariff_2024_gel
    FROM tariff_with_usd
    WHERE time_month BETWEEN '2023-01' AND '2024-12'
    GROUP BY entity
)
SELECT
    entity,
    tariff_2023_gel,
    tariff_2024_gel,
    ROUND(tariff_2024_gel - tariff_2023_gel, 2) as increase_gel,
    ROUND(100.0 * (tariff_2024_gel - tariff_2023_gel) / NULLIF(tariff_2023_gel, 0), 1) as increase_pct
FROM tariff_change
WHERE tariff_2023_gel IS NOT NULL AND tariff_2024_gel IS NOT NULL
ORDER BY increase_pct DESC;

EXAMPLE 8.5 - Relative Tariff Comparison:
Query: "Compare Enguri tariff to other hydro plants"
Plan:
{
  "intent": "comparison",
  "target": "relative_tariff",
  "period": "2024"
}
---SQL---
SELECT
    entity,
    AVG(tariff_gel) as avg_tariff_gel_per_mwh,
    AVG(tariff_gel) / NULLIF(
        (SELECT AVG(tariff_gel) FROM tariff_with_usd
         WHERE entity = 'ltd "engurhesi"1' AND time_month >= '2024-01'), 0
    ) as ratio_to_enguri
FROM tariff_with_usd
WHERE time_month >= '2024-01'
  AND (entity LIKE '%hesi%' OR entity LIKE '%hpp%')
GROUP BY entity
ORDER BY avg_tariff_gel_per_mwh;
"""

# =============================================================================
# CATEGORY 9: EXCHANGE RATE & CURRENCY EFFECTS
# =============================================================================

EXCHANGE_RATE_EXAMPLES = """
EXAMPLE 9.1 - Exchange Rate Impact on Prices:
Query: "How does exchange rate affect electricity prices?"
Plan:
{
  "intent": "correlation",
  "target": "xrate_price_correlation",
  "period": "2020-2024"
}
---SQL---
SELECT
    time_month,
    xrate,
    p_bal_gel,
    p_bal_usd,
    ROUND(p_bal_gel / NULLIF(xrate, 0), 2) as implied_usd_price
FROM price_with_usd
WHERE time_month >= '2020-01' AND time_month <= '2024-12'
ORDER BY time_month;

EXAMPLE 9.2 - GEL vs USD Price Divergence (Georgian):
Query: "როგორ განსხვავდება ფასები ლარსა და დოლარში?"
Plan:
{
  "intent": "comparison",
  "target": "gel_vs_usd_prices",
  "period": "2022-2024"
}
---SQL---
SELECT
    time_month,
    p_bal_gel,
    p_bal_usd,
    xrate,
    ROUND(100.0 * (p_bal_gel / NULLIF(LAG(p_bal_gel) OVER (ORDER BY time_month), 0) - 1), 2) as gel_growth_pct,
    ROUND(100.0 * (p_bal_usd / NULLIF(LAG(p_bal_usd) OVER (ORDER BY time_month), 0) - 1), 2) as usd_growth_pct
FROM price_with_usd
WHERE time_month >= '2022-01' AND time_month <= '2024-12'
ORDER BY time_month;
"""

# =============================================================================
# CATEGORY 10: SEASONAL & TEMPORAL ANALYSIS
# =============================================================================

SEASONAL_EXAMPLES = """
EXAMPLE 10.1 - Monthly Patterns Analysis:
Query: "What are the typical monthly patterns in demand and generation?"
Plan:
{
  "intent": "general",
  "target": "monthly_patterns",
  "period": "2020-2023"
}
---SQL---
SELECT
    EXTRACT(MONTH FROM time_month::date) as month_num,
    TO_CHAR(time_month::date, 'Month') as month_name,
    ROUND(AVG(CASE WHEN type_tech IN ('abkhazeti', 'supply-distribution', 'direct customers', 'losses', 'export')
              THEN quantity END), 2) as avg_demand_thousand_mwh,
    ROUND(AVG(CASE WHEN type_tech = 'hydro' THEN quantity END), 2) as avg_hydro_thousand_mwh,
    ROUND(AVG(CASE WHEN type_tech = 'thermal' THEN quantity END), 2) as avg_thermal_thousand_mwh
FROM tech_quantity_view
WHERE time_month >= '2020-01' AND time_month <= '2023-12'
GROUP BY EXTRACT(MONTH FROM time_month::date), TO_CHAR(time_month::date, 'Month')
ORDER BY month_num;

EXAMPLE 10.2 - Yearly Aggregates:
Query: "Show me yearly totals for demand and generation 2015-2023"
Plan:
{
  "intent": "trend_analysis",
  "target": "yearly_aggregates",
  "period": "2015-2023"
}
---SQL---
SELECT
    EXTRACT(YEAR FROM time_month::date) as year,
    SUM(CASE WHEN type_tech IN ('abkhazeti', 'supply-distribution', 'direct customers', 'losses', 'export')
        THEN quantity ELSE 0 END) as total_demand_thousand_mwh,
    SUM(CASE WHEN type_tech IN ('hydro', 'thermal', 'wind', 'solar', 'import')
        THEN quantity ELSE 0 END) as total_supply_thousand_mwh,
    COUNT(DISTINCT time_month) as months_in_year
FROM tech_quantity_view
WHERE time_month >= '2015-01' AND time_month <= '2023-12'
GROUP BY EXTRACT(YEAR FROM time_month::date)
ORDER BY year;
"""

# =============================================================================
# COMBINED EXAMPLES STRING FOR PROMPT
# =============================================================================

ALL_EXAMPLES = f"""
{ENERGY_SECURITY_EXAMPLES}

{BALANCING_PRICE_EXAMPLES}

{SUPPORT_SCHEMES_EXAMPLES}

{STAKEHOLDERS_EXAMPLES}

{DEMAND_EXAMPLES}

{GENERATION_EXAMPLES}

{OWNERSHIP_EXAMPLES}

{TARIFF_EXAMPLES}

{EXCHANGE_RATE_EXAMPLES}

{SEASONAL_EXAMPLES}
"""

# =============================================================================
# EXPORT - SELECTIVE EXAMPLE LOADING (Phase 1 Fix)
# =============================================================================

def get_relevant_examples(user_query: str, max_categories: int = 2) -> str:
    """
    Load only relevant examples based on query content.

    Reduces token usage from ~5,800 (all examples) to ~800-1,500 (selective),
    keeping domain knowledge and guidance prominent for better answer quality.

    Args:
        user_query: The user's query text
        max_categories: Maximum number of example categories to include (default: 2)

    Returns:
        String with selected examples only (much smaller than ALL_EXAMPLES)
    """
    query_lower = user_query.lower()

    # Category detection with keywords
    category_keywords = {
        "energy_security": ["energy security", "უსაფრთხოება", "import dependence",
                           "დამოკიდებულება", "self-sufficiency", "vulnerability",
                           "independence", "თვითკმარობა"],
        "balancing_price": ["balancing price", "საბალანსო ფასი", "price driver",
                           "why price", "რატომ გაიზარდა", "ფასის ზრდა", "p_bal",
                           "price increase", "price decrease"],
        "tariff": ["tariff", "ტარიფი", "regulated", "enguri", "gardabani",
                  "gnerc approval"],
        "demand": ["demand", "consumption", "მოთხოვნა", "growth"],
        "generation": ["generation", "გენერაცია", "produce", "capacity", "mix"],
        "seasonal": ["seasonal", "summer", "winter", "სეზონური", "ზაფხულ", "ზამთარ"]
    }

    # Category example mapping
    category_examples_map = {
        "energy_security": ENERGY_SECURITY_EXAMPLES,
        "balancing_price": BALANCING_PRICE_EXAMPLES,
        "tariff": TARIFF_EXAMPLES,
        "demand": DEMAND_EXAMPLES,
        "generation": GENERATION_EXAMPLES,
        "seasonal": SEASONAL_EXAMPLES
    }

    # Find matching categories
    matched = []
    for category, keywords in category_keywords.items():
        if any(kw in query_lower for kw in keywords):
            matched.append(category)

    # Load examples for matched categories (limit to max_categories)
    selected_examples = []
    for category in matched[:max_categories]:
        if category in category_examples_map:
            selected_examples.append(category_examples_map[category])

    # If no matches, use minimal core set (balancing price only - most common)
    if not selected_examples:
        # Return only first 2 balancing price examples (~600 tokens)
        return BALANCING_PRICE_EXAMPLES[:600] + "\n\n(Core examples loaded)"

    result = "\n\n".join(selected_examples)
    return result + f"\n\n(Loaded {len(matched[:max_categories])} relevant example categories)"


def get_examples_for_prompt(category: str = "all") -> str:
    """
    Get examples for inclusion in LLM prompt (legacy function).

    Args:
        category: "all" or specific category name

    Returns:
        String of examples
    """
    if category == "all":
        return ALL_EXAMPLES

    category_map = {
        "energy_security": ENERGY_SECURITY_EXAMPLES,
        "balancing_price": BALANCING_PRICE_EXAMPLES,
        "support_schemes": SUPPORT_SCHEMES_EXAMPLES,
        "stakeholders": STAKEHOLDERS_EXAMPLES,
        "demand": DEMAND_EXAMPLES,
        "generation": GENERATION_EXAMPLES,
        "ownership": OWNERSHIP_EXAMPLES,
        "tariffs": TARIFF_EXAMPLES,
        "exchange_rate": EXCHANGE_RATE_EXAMPLES,
        "seasonal": SEASONAL_EXAMPLES
    }

    return category_map.get(category, ALL_EXAMPLES)
