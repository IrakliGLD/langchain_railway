# General Definitions

Basic energy sector terminology for answering general conceptual questions.  
Provides universal definitions with Georgia-specific context.

---

## 1. Core Market Concepts

### Electricity Market
**Definition:** A system where electricity is bought and sold between generators, traders, suppliers, and consumers. Markets can include bilateral contracts, day-ahead trading, intraday trading, and balancing mechanisms.

**Georgia Context:** Georgia operates a hybrid market with:
- bilateral contracts between generators and suppliers
- balancing mechanism (since 2006) as the dominant settlement layer
- day-ahead exchange (since July 2024)

The market is currently in a **transitional phase** and is not yet fully competitive or EU-aligned.

---

### Market Model
**Definition:** The overall structure and rules governing how electricity is dispatched, traded, and settled.

**Georgia Context:**
- **Transitional Model (current):**
  - monthly balancing (imbalance settlement)
  - centralized dispatch by GSE
  - no Balance Responsible Parties (BRPs)
  - exchange plays limited role
- **Target Model (planned, ~July 2027):**
  - self-dispatch
  - hourly imbalance settlement
  - BRPs introduced
  - EU-style balancing market with marginal pricing  
  - implementation delay risk exists

---

### Balancing Market
**Definition:** A mechanism to balance supply and demand when actual generation differs from scheduled or contracted amounts.

**Georgia Context:**  
Georgia’s balancing market is not a real-time balancing market. It functions as a **monthly imbalance settlement system**, where deviations are calculated and settled after the month ends.

---

### Balancing Price (WAP)
**Definition:** The weighted average price of electricity used for imbalance settlement.

**Georgia Context:**
- Calculated as weighted average of all electricity sold as balancing energy during the month
- Includes:
  - regulated tariffs
  - deregulated hydro
  - thermal generation
  - PPA/CfD electricity
  - imports
- It is:
  - **NOT a marginal price**
  - **NOT an hourly market price**
- Strongly influenced by generation composition and exchange rate

---

### Dispatch
**Definition:** The process of determining which power plants generate electricity and when.

**Types:**
- **Central Dispatch:** System operator (TSO) decides generation (current Georgian model)
- **Self-Dispatch:** Generators decide production based on market signals (EU model)

**Georgia Context:**
- Current model → central dispatch by GSE  
- Target model → transition to self-dispatch

---

### Imbalance and BRP
**Definition:**
- **Imbalance:** Difference between scheduled and actual electricity generation or consumption
- **BRP (Balance Responsible Party):** Entity responsible for managing and settling imbalances

**Georgia Context:**
- No BRPs in current model
- Imbalance calculated monthly
- BRPs and hourly imbalance responsibility planned in target model

---

## 2. Pricing Concepts

### Tariff
**Definition:** A regulated price for electricity set by authorities.

**Georgia Context:**  
GNERC sets cost-plus tariffs for:
- regulated HPPs (Enguri, Vardnili)
- thermal plants (Gardabani TPP, old TPPs)

- Tariffs are GEL-denominated
- Thermal tariffs are effectively **USD-driven** (via gas cost)
- Hydro tariffs have delayed FX adjustment

---

### Price Types (Important Distinction)

- **Tariff:** regulated cost-based price  
- **Balancing Price:** weighted average settlement price (WAP)  
- **Market Price (Exchange):** competitive price (day-ahead/intraday)  
- **Contract Price (PPA/CfD):** pre-agreed or strike-based price  

**Georgia Context:**  
These price types coexist and must not be confused.

---

### Exchange Rate Impact
**Definition:** When costs or contracts are denominated in foreign currency, exchange rate movements affect electricity prices.

**Georgia Context:**  
Electricity prices are highly sensitive to GEL/USD because:
- gas is USD-priced
- imports are USD-priced
- PPAs and CfDs are USD-linked

GEL depreciation → higher electricity prices in GEL terms.

---

## 3. Contract and Support Concepts

### Support Scheme
**Definition:** Mechanism that provides revenue stability for electricity generators.

**Georgia Context:**  
Only two support schemes exist:
- PPA
- CfD

Regulated tariffs are **NOT support schemes**.

---

### PPA (Power Purchase Agreement)
**Definition:** Long-term contract guaranteeing purchase of electricity at agreed price.

**Georgia Context:**
- Used for renewable and thermal projects
- Typically USD-indexed
- Government often acts as counterparty
- Provides revenue certainty but reduces market exposure

---

### CfD (Contract for Difference)
**Definition:** Mechanism where generator receives difference between strike price and market price.

**Georgia Context:**
- Introduced in 2023 via capacity auctions (~170 MW)
- Renewable-only projects (hydro, wind, solar)
- Transitional model:
  - centrally dispatched
  - electricity sold as balancing electricity
  - no price risk
  - quantity (curtailment) risk
- Target model:
  - self-dispatch
  - reduced curtailment exposure
  - imbalance risk emerges

**Disambiguation:** The Electricity Market Concept Design regulation also uses "CfD" to describe a transitional compensation mechanism for existing regulated generators — compensating the difference between regulated tariff and market price during market reform. This is NOT the same as the renewable support scheme CfD. Context determines which meaning applies. See `cfd_ppa.md` Section 3.5 for details.

---

## 4. Physical System Concepts

### Generation Mix
**Definition:** Composition of electricity supply by source.

**Georgia Context:**
- Hydro-dominated (~80% in good years)
- Thermal fills winter gaps
- Wind/solar growing
- No nuclear
- Import dependence increases in winter

---

### Demand
**Definition:** Electricity consumption by end-users.

**Georgia Context:**
- Strong seasonal variation:
  - winter peaks (heating)
  - summer peaks increasing (cooling)
- Annual consumption ~13–14 TWh
- Large consumers may act as direct customers

---

### Capacity vs Energy
**Definition:**
- Capacity (MW): maximum output
- Energy (MWh): output over time

**Georgia Context:**
- Installed capacity ~4,500 MW
- Actual generation depends on hydrology and dispatch
- Capacity factors:
  - hydro: 30–50%
  - thermal: 20–40%
  - wind/solar: 20–30%

---

## 5. Technology Concepts

### Hydropower
**Definition:** Electricity from flowing or stored water.

**Georgia Context:**
- Major plants: Enguri (1,300 MW), Vardnili, Zhinvali, Khrami
- Mostly run-of-river (limited storage)
- Strong seasonality (high in summer, low in winter)

---

### Thermal Power
**Definition:** Electricity generated by burning fuel (mainly natural gas).

**Georgia Context:**
- Uses imported gas (USD-priced)
- Key plants:
  - Gardabani TPP (new, efficient)
  - old TPPs (Tbilisi TPP, Mtkvari, G-Power)
- Critical in winter
- Strongly linked to exchange rate

---

## 6. System Interaction Concepts

### Import/Export
**Definition:** Cross-border electricity trade.

**Georgia Context:**
- Imports in winter
- Exports in summer (hydro surplus)
- Main partners: Turkey, Azerbaijan, Armenia, Russia
- Prices are USD-based

---

### Regulated vs Deregulated
**Definition:**
- Regulated: government-set tariffs
- Deregulated: market-based prices

**Georgia Context:**
- Regulated:
  - Enguri, Vardnili
  - thermal plants
- Deregulated:
  - small hydro
- PPAs:
  - hybrid (contract-based, not pure market)

---

## 7. Usage Instructions

- For "What is X?" questions:
  1. Provide general definition
  2. Add Georgia-specific context

- Keep definitions:
  - accurate
  - consistent across documents

- Do not mix:
  - tariff vs market vs contract prices
  - physical vs market concepts

- If term is not defined:
  - provide general explanation
  - clarify limitations