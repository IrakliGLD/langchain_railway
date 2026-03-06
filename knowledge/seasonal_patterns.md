# Seasonal Patterns and Forecasting

## Season Definition
- **Summer Months:** April, May, June, July (months 4-7)
- **Winter Months:** January, February, March, August, September, October, November, December (months 1-3, 8-12)

**Description:** Summer: hydro-dominant, low prices. Winter: thermal/import-dominant, high prices.

## Analytical Use
- Compare prices and generation composition between seasons
- Hydro share typically >60% in summer, <30% in winter
- Use SUM for quantities, AVG for prices when comparing seasons

## Seasonal Trends
**Definition:** Balancing electricity prices, generation, and demand exhibit structurally different behaviors across seasons due to shifts in supply composition and consumption patterns.

**Rule:** Always compute and compare seasonal averages and CAGRs for April–July (Summer) and August–March (Winter).

### Interpretation
- Summer prices rise faster as cheap hydro shares decline and more output moves to contracts
- Winter prices increase moderately due to higher gas costs and import reliance

## Price Comparison Rules

**CRITICAL:** ALWAYS mention summer and winter averages separately when comparing prices — never use annual averages only.

### Reasoning
- Summer and winter prices are structurally different due to generation mix
- Summer: hydro-dominant, lower prices (~40-60 GEL/MWh)
- Winter: thermal/import-dominant, higher prices (~80-120 GEL/MWh)
- Annual averages obscure these critical seasonal differences

### Trend Divergence

**Summer Trend:**
- Rising share of renewable PPA and CfD scheme generation
- In absence of liquid markets, investors require government support schemes
- New renewable development depends almost entirely on PPA/CfD support
- As renewable PPA/CfD share increases in balancing electricity, summer prices converge toward average support scheme price
- This pushes summer prices UP over time despite high hydro availability

**Winter Trend:**
- Prices follow gas prices and exchange rate (thermal/import dominant)
- Less affected by renewable PPA/CfD trend
- More volatile due to import price variations and gas market dynamics

### Mandatory Format
When comparing prices across years:
1. State summer average for period A
2. State winter average for period A
3. State summer average for period B
4. State winter average for period B
5. Explain the different drivers for each season's trend

**Example:**
- ✅ CORRECT: "Balancing prices increased from 2020 to 2024. In summer, average prices rose from 45 GEL/MWh to 62 GEL/MWh due to growing renewable PPA share. In winter, prices increased from 95 GEL/MWh to 118 GEL/MWh due to gas price increases and GEL depreciation."
- ❌ WRONG: "Balancing prices increased from 70 GEL/MWh in 2020 to 90 GEL/MWh in 2024." (No seasonal breakdown!)

## Default Time Period Rules

**CRITICAL RULE:** If user does NOT specify a time period, use ALL available data.

**DO NOT:**
- Add WHERE clauses filtering by year/month unless user explicitly requests
- Assume "recent" means last year
- Default to any specific year like 2023 or 2024
- Add date filters for trend/historical queries without specific time mentions

**ONLY add date filter if user explicitly mentions:**
- Specific year (e.g., "2023", "last year", "this year")
- Specific month (e.g., "June", "last month", "June 2024")
- Date range (e.g., "from 2020 to 2024", "since 2022")
- Period (e.g., "recent 2 years", "last 6 months", "past year")

## Balancing Price Forecasting Challenges

**CRITICAL WARNING:** Balancing electricity price prediction is difficult because many factors beyond supply/demand affect price formation. Use linear regression trendlines with extreme caution and ALWAYS mention limitations.

### Price Formation Complexity

**PPA Dominance:**
- Significant share of balancing electricity comes from PPA-contracted generation
- PPA prices are NOT market-driven — they are fixed by contract
- Growing PPA share means growing portion of balancing price is administratively determined

**Regulated Thermal Uncertainty:**
- Thermal plant tariffs based on gas prices set through bilateral negotiations (Georgia-Azerbaijan)
- Gas prices are often speculated to be under-priced due to political considerations
- Tariff adjustments depend on unpredictable regulatory decisions by GNERC

**Import Price Unpredictability:**
- Depends on situation in neighboring markets (non-transparent)
- Import prices vary significantly based on regional hydrology, gas availability, and political factors

**Market Rule Changes:**
- Past market rule changes had significant impact on prices
- Future rule changes (exchange expansion, balancing market reform) are planned but timing uncertain
- Past price patterns may not hold if market design changes

### Forecasting Guidance

**For Historical Analysis:**
- Historical price changes CAN be explained by analyzing composition changes, xrate movements, tariff adjustments
- Decompose price changes into: composition effect, exchange rate effect, tariff/PPA price changes, import price changes
- Cite R² values — strong correlation with historical drivers does NOT guarantee future predictability

**For Forecasting:**
- MANDATORY DISCLAIMER: Always state that forecasts have high uncertainty
- Linear regression shows HISTORICAL patterns only
- Required caveats:
  - "This forecast assumes current market structure, PPA contracts, and regulatory framework remain unchanged"
  - "Actual prices may differ significantly due to: gas price negotiations, new PPA capacity, market rule changes"
  - If R² < 0.5: "Low reliability — historical prices show high variability"
  - If R² > 0.7: "Statistically strong but may not continue if conditions change"

**Short-term (1-2 years):** Linear trendlines reasonable with caveats
**Medium-term (3-5 years):** Emphasize scenario-based thinking
**Long-term (5+ years):** Linear extrapolation is unreliable — focus on structural drivers

**ALWAYS separate summer and winter forecasts.**

### What CAN Be Forecasted
- Seasonal patterns (summer lower than winter) — structurally driven
- Direction of composition shift (growing PPA/CfD share) — follows announced capacity auctions
- Qualitative impact of known events

### What CANNOT Be Forecasted
- Absolute price levels 3+ years ahead
- Sudden market rule changes
- Import price levels
- Exact PPA/CfD capacity additions

## Generation Adequacy and Forecast (TYNDP)

### Current State
- As of 2023, total installed capacity: ~4,621 MW (73% hydro, 23% thermal, 0.5% wind)
- Highly seasonal: summer hydro exceeds demand (exports), winter thermal covers ~28% of supply

### Demand Scenarios
- Three scenarios: L1 (1% growth), L2 (3% base case), L3 (5%)
- Base case (L2G3): 3% annual consumption growth with on-time project integration
- Georgia expected to maintain energy adequacy through 2034

### Adequacy Analysis
- PLEXOS simulations confirm sufficient capacity under all scenarios
- Wind and solar require reserve margins and flexible hydropower
- Scenario G2 (with reservoir hydro) provides higher flexibility

**Source:** All figures from GSE Ten-Year Network Development Plan 2024-2034 (TYNDP).
