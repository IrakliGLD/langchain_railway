# Market Structure

## 1. Market Model Overview

Georgia’s electricity market currently operates under a **transitional market model**, which differs significantly from the EU target model.

### Core Characteristics (Current)
- **Monthly balancing model**
- **Centralized dispatch by GSE**
- **Limited role of competitive exchange**
- **Balancing settlement is retrospective (not real-time)**

### Transitional Nature (CRITICAL)
- The current model is an **interim / limited design**, not a fully developed market
- It represents a **phased transition** toward the EU target model
- Market opening is ongoing and incomplete

---

## 2. Transitional Market Model (Current)

### 2.1 Core Design

Despite being formally called a "balancing market", the current system functions as a **monthly imbalance settlement mechanism**, not a real-time balancing market.

- No hourly balancing responsibility
- No Balance Responsible Parties (BRPs)
- Imbalance is calculated **monthly**, not hourly
- System is **centrally dispatched by GSE**
- Generators do not self-dispatch

---

### 2.2 Settlement Logic

- Balancing period = **one month**
- Imbalance is calculated as:

> total electricity consumed or generated  
> MINUS  
> electricity sold or purchased during the same month

- Settlement is done **after the month ends**

---

### 2.3 Price Determination

- Balancing electricity price is a **weighted average price (WAP)** of electricity sold as balancing energy during the month
- Price reflects:
  - generation composition
  - regulated tariffs
  - contract-based prices (CfD, PPA)
  - market-based sources

**Important:**
- This is NOT a marginal price
- It is an **average settlement price**

---

### 2.4 Role of Exchange (from July 2024)

- Day-ahead exchange (GENEX) introduced in **July 2024**
- Trade on exchange is **hourly**
- However:
  - exchange is layered on top of the existing monthly balancing model
  - does NOT replace balancing settlement

**Implication:**
- Market is a **hybrid structure**:
  - monthly balancing settlement remains dominant
  - exchange provides partial price signals

---

### 2.5 Structural Characteristics

- Market still has **limited competition and participation**
- A large share of electricity is settled through balancing rather than competitive trading
- Balancing electricity represents a **significant share of total system volume**
- Price formation is therefore:
  - not purely market-based
  - strongly influenced by regulated and contract-based components

---

## 3. Target Market Model (Future)

### 3.1 Planned Design

Georgia aims to transition to an **EU-style electricity market model**.

Key features:
- **Self-dispatch**
- **Balance Responsible Parties (BRPs)**
- **Hourly imbalance settlement**
- **Real balancing market with products (FCR, aFRR, mFRR)**

---

### 3.2 Timeline

- Planned launch: **July 2027**
- There is a **significant risk of delay**, as previous reform steps have already been postponed

---

### 3.3 Expected Changes

- Shift from:
  - monthly settlement → hourly settlement
- Shift from:
  - centralized dispatch → decentralized (self-dispatch)
- Introduction of:
  - real-time balancing products
  - marginal price formation

---

### 3.4 Price Formation Changes

- Balancing price:
  - will become **marginal and time-dependent**
- Exchange:
  - will play a **central role in price formation**
- CfD/PPA:
  - will interact with market prices rather than fully shaping balancing price

---

## 4. Comparison with EU Practice

### Current Georgian Model
- Monthly imbalance settlement
- No balancing products
- No hourly imbalance responsibility
- Central dispatch
- Average price (WAP)

---

### EU Model
- Real-time balancing markets
- Products: FCR, aFRR, mFRR
- Hourly or sub-hourly settlement
- Self-dispatch
- Marginal pricing

---

## 5. Reform Context and Legal Framework

- Georgia has already adopted a **legal and regulatory framework aligned with the EU electricity market model**
- Market rules include:
  - day-ahead market
  - intraday market
  - balancing market
  - ancillary services

**Important:**
- The full market design is already defined
- The main gap is **implementation and operational transition**, not conceptual design

---

## 6. International Context

- Georgia is a member of the **Energy Community (since 2017)**
- Market reform is driven by:
  - alignment with EU electricity market rules
  - integration into regional and European electricity markets

**Implication:**
- Market evolution is **externally anchored**
- Transition toward EU model is a **regulatory obligation**, not optional

---

## 7. Trade Data

### Data Availability
- `trade_derived_entities`:
  - reliable from 2020 onwards
  - contains balancing and exchange segments
- No entity-level balancing composition data before 2020
- NULL values mean **data not available**, not zero

### Important
- Trade volumes determine weights used in balancing price calculation
- Exchange segment appears only from **July 2024**

---

## 8. Market Participants and Roles

### GNERC
- Regulator and tariff authority
- Approves tariffs and licenses
- Data source for:
  - `tariff_with_usd`
  - `price_with_usd`
  - `tech_quantity_view`

---

### ESCO
- Balancing electricity buyer/seller
- Counterparty for:
  - CfD
  - guaranteed capacity
- Handles settlement processes
- Data source for:
  - `trade_derived_entities`

---

### GSE
- Transmission System Operator (TSO)
- Central dispatcher (current model)
- Responsible for:
  - system stability
  - cross-border interconnections

---

### GENEX
- Exchange operator
- Runs:
  - day-ahead market
  - intraday market
- Exchange introduced in July 2024

---

### GEOSTAT
- National statistics office
- Provides:
  - energy balance data
  - macroeconomic indicators
- Data source:
  - `energy_balance_long_mv`

---

## 9. Key Events

- **2006:** Balancing market established
- **Jul 2024:** Exchange launched (GENEX) — hybrid model begins
- **Jul 2027 (planned):** Target model launch (EU-style market) — delay risk exists

---

## 10. Import Dependence

- Georgia imports in winter, exports in summer
- Imports are USD-denominated
- Import prices follow regional markets (Turkey, Azerbaijan)
- Higher import share + weaker GEL → higher prices
- Hydro shortages increase import reliance and winter price pressure

---

## 11. Transmission Interconnections

### Available Data
- Import: `tech_quantity_view`, `type_tech = 'import'`
- Export: `tech_quantity_view`, `type_tech = 'export'`

### Not Available
- Interconnection capacity (MW)
- Technical limits

### Correct Interpretation
- Do not confuse volume with capacity
- For capacity → refer to GSE / TYNDP

---

## 12. Transmission Network Development (TYNDP)

### Objectives
- Ensure system reliability (N-1, G-1, N-G-1)
- Address west–east imbalance
- Remove transmission bottlenecks
- Modernize substations

---

### Renewable Integration
- Up to:
  - 750 MW wind
  - 500 MW solar (by 2028)
- Requires:
  - flexible generation (CCGT)
  - reservoir HPPs

---

### Cross-Border Projects
- Black Sea cable (Georgia–Romania)
- Georgia–Russia–Azerbaijan link
- Expansion with Turkey and Armenia

---

## 13. Direct Customers

- Market participant category (not sector)
- Large consumers buying directly in wholesale market
- Includes multiple industries
- Structure changes over time

### Data Source
- `tech_quantity_view`, `type_tech = 'direct customers'`

---

## 14. Abkhazeti Consumption

- Separate category in `tech_quantity_view`
- Strong seasonality (winter peak)
- Growth drivers:
  - electric heating
  - crypto mining
  - economic activity

- From May 2025:
  - Enguri/Vardnili tariffs increased to cover supply costs

---

## 15. Table Selection Guidance

### tech_quantity_view
- Technical data (generation, demand)

---

### trade_derived_entities
- Market/trade data
- Includes:
  - balancing segment
  - exchange
  - entity shares

---

### Critical Distinction
- `tech_quantity_view` → physical system
- `trade_derived_entities` → market behavior

---

## 16. Data Evidence Integration

- Always support analysis with data trends
- Avoid raw column names in narrative explanations
- Use clear units and consistent interpretation
- Separate:
  - currency effects
  - composition effects
  - structural effects