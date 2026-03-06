# Market Structure

## Balancing Market Design

### Current Design
Despite being formally called a "balancing market", Georgia's current system functions as an imbalance settlement mechanism rather than a real-time balancing market in the European sense.

- Balancing responsibility is not defined on an hourly basis — there is no concept of Balance Responsible Parties (BRPs) with continuous imbalance settlement
- The current balancing period is one month, and the imbalance is calculated as the difference between the total electricity consumed or generated and the electricity sold or purchased during that same month

### Price Determination
- The balancing electricity price is calculated as a weighted average price of electricity sold as balancing energy during the month
- This price formation principle aggregates transactions across generation entities based on their quantities and individual tariffs or market values
- Therefore, the current "balancing price" represents a settlement value for deviations over a month, not an hourly marginal price

### Comparison with EU Practice
- In the European electricity market model, the balancing market includes trading of balancing products such as FCR, aFRR, and mFRR
- Those products are activated on a sub-hourly basis to maintain system frequency and resolve imbalances in real time
- Georgia's system does not yet have such hourly or product-based balancing — it performs monthly imbalance settlement after-the-fact

### Future Direction
- Full transition toward an EU-style balancing market is expected in future market reforms, with the introduction of BRPs, hourly metering, and separate balancing product procurement
- GSE is licensed operator of a balancing market; real balancing market and hourly imbalance responsibility was set to launch on July 2027
- Until that transition, interpret any reference to the "balancing market" as meaning "monthly imbalance settlement"

## Trade Data

### Data Availability
- `trade_derived_entities` has reliable data ONLY from 2020 onwards
- No entity-level balancing composition data exists before 2020
- NULL share values mean DATA IS NOT AVAILABLE — never interpret NULL as 0% share
- Includes transactions across exchange and balancing segments; the Exchange was introduced in July 2024
- Trade volumes determine the weights used in calculating the balancing electricity price

## Market Participants and Data Sources

### GNERC (Georgian National Energy and Water Supply Regulatory Commission)
**Role:** Independent energy regulator, tariff authority, energy market monitoring and licensing body.
- Approves electricity generation, transmission, and distribution tariffs
- Issues, modifies, and revokes licenses for generation, transmission, distribution, market operator
- Approves and enforces the Grid Code, network connection rules
- Oversees cost audits, tariff reviews, guaranteed capacity payments
- **Data source for:** `tariff_with_usd`, `price_with_usd`, `tech_quantity_view`

### ESCO (Electricity System Commercial Operator)
**Role:** Responsible for buying and selling balancing electricity.
- Administers the balancing and guaranteed capacity settlement processes
- Registers wholesale market participants, manages direct contracts
- Handles import/export settlements and acts as counterparty for CfD and guaranteed capacity contracts
- **Data source for:** `trade_derived_entities` and other balancing-related views

### GSE (Georgian State Electrosystem)
**Role:** Transmission System Operator (TSO), system dispatcher, and transmission network owner.
- Owns and operates Georgia's transmission infrastructure
- Performs real-time system dispatch, grid stability control
- Manages cross-border interconnections with neighboring systems (Turkey, Azerbaijan, Armenia, Russia)
- Plans transmission development and publishes the Ten-Year Network Development Plan (TYNDP)

### GENEX (Georgian Energy Exchange)
**Role:** Electricity Exchange Operator for electricity day-ahead and intraday markets.
- Operates day-ahead and intraday markets
- Publishes market prices, traded volumes, and clearing results
- Established jointly by GSE and ESCO in 2019; current shareholders are GSE, ESCO, GGTC and GOGC

### GEOSTAT (National Statistics Office of Georgia)
**Role:** Official statistical agency providing macroeconomic and energy data.
- Publishes national energy balances, sectoral demand indicators, and inflation indices
- Maintains the CPI series including "electricity, gas, and other fuels" category
- **Data source for:** `monthly_cpi_mv` and `energy_balance_long_mv`

## Import Dependence
- Georgia imports in winter, exports in summer; import exposure sets upper bound on domestic prices
- Imports are USD-denominated and follow Turkish/Azeri prices, transmitting regional volatility
- Higher import share + weaker GEL → higher balancing prices
- Hydro shortfall or Enguri/Vardnili outages trigger import reliance and winter spikes

## Transmission Interconnections

### Available Data
- **Import volumes:** `tech_quantity_view` where `type_tech = 'import'` — thousand MWh per month, from 2014 onwards
- **Export volumes:** `tech_quantity_view` where `type_tech = 'export'` — thousand MWh per month, from 2014 onwards
- **IMPORTANT:** Export volumes exist in the data — ALWAYS check both import AND export when analyzing cross-border flows

### NOT Available
- Interconnection capacity (MW) for cross-border connections
- Technical specifications of interconnection infrastructure
- Simultaneous import/export limits
- For this data, recommend: GSE technical documentation or TYNDP

### Correct Interpretation
When user asks about "interconnections", they likely want capacity data (MW):
- ❌ WRONG: Analyze only import/export volumes and claim this answers the question
- ✅ CORRECT: Acknowledge that capacity data is not available, suggest sources, then provide what IS available (volume trends)

## Transmission Network Development (TYNDP 2024–2034)

### Main Objectives
- Ensure security of supply through meeting N-1, G-1, and N-G-1 criteria
- Address west–east transmission imbalance (hydro generation in west, consumption in east)
- Eliminate critical bottlenecks along the Enguri–Zestaponi–Imereti 500/220 kV corridor
- Modernize substations for growing urban demand (Tbilisi and Batumi)

### Renewable Integration
- Up to 750 MW new wind and 500 MW new solar by 2028 under existing balancing resources
- Integration contingent on new flexible capacity (CCGT) and reservoir HPPs
- West–east imbalance intensifies during high-hydro summer periods

### Cross-Border Projects
- Georgia–Romania Black Sea Submarine Cable (HVDC) — direct electricity trade with EU markets
- Georgia–Russia–Azerbaijan Power System Connection — synchronized or coordinated operation
- Additional Turkey and Armenia interconnections planned

**Source:** All data must be attributed to "GSE Ten-Year Network Development Plan 2024–2034 (TYNDP)".

## Direct Customers
**CRITICAL:** Direct customers are NOT a specific industry sector — they are a MARKET PARTICIPANT CATEGORY.

- Large electricity consumers who purchase directly on the wholesale market
- Include MULTIPLE different industries: metallurgy, mining, manufacturing, commercial centers
- Cannot determine which specific industry's consumption from aggregate data
- Number of direct customers changes over time as consumers switch between retail and wholesale
- When consumers switch from retail to wholesale: "supply-distribution" decreases, "direct customers" increases (market structure shift, not necessarily total consumption change)

**Data Source:** `tech_quantity_view` where `type_tech = 'direct customers'`

### Data Limitations
- No sectoral breakdown available
- Cannot answer: "How much do metallurgical enterprises consume?"
- Can answer: Total direct customers consumption trends, seasonal patterns, comparison with other market segments

## Abkhazeti Consumption
- Measured separately in `tech_quantity_view` (`type_tech = 'abkhazeti'`)
- Strong seasonal variation: winter consumption roughly doubles compared to summer (electric heating)
- Long-term growth drivers: electric heating adoption, cryptocurrency mining, general economic activity
- From May 2025: Enguri and Vardnili tariffs increased to recover costs of electricity supplied to Abkhazia

## Table Selection Guidance

### tech_quantity_view
Use for ONLY technical generation/consumption data:
- Contains: quantity by type_tech (hydro, thermal, wind, solar, import, demand-side types)
- Use for: demand trends, supply trends, generation mix by technology
- Examples: "Show me demand trends", "Hydro generation over time", "Import quantities"

### trade_derived_entities
Use for trade information, market prices, or entity-level analysis:
- Contains: traded quantities by entity and segment (use canonical normalized token `balancing` for balancing-segment trade; also covers bilateral contract/exchange activity)
- Contains: entity shares (share_import, share_renewable_ppa, share_deregulated_hydro, etc.)
- Use for: balancing price analysis, composition changes, entity market behavior
- Examples: "Explain balancing price variations", "Entity shares in balancing market"

### CRITICAL DISTINCTION
- `tech_quantity_view` = Technical data (generation/demand by technology)
- `trade_derived_entities` = Market/trade data (prices, segments, entities, shares)
- Default: simple quantity queries → tech_quantity_view; price explanation → trade_derived_entities

## Data Evidence Integration
- Every analytical or causal statement should be justified by trends or values from materialized views
- Never include raw database column names in narrative text — use descriptive terms
- Prioritize causal storytelling supported by numeric evidence
- When comparing across currencies, units, or dimensions, explicitly reflect the measurement unit
