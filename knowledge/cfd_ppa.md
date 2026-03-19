# CfD Contracts and PPA Support Schemes

## 1. Definitions
- **Support Scheme:** Government-backed mechanism to ensure revenue stability for power generation projects.
- **PPA (Power Purchase Agreement):** Contract where a buyer commits to purchase electricity at a predefined price.
- **CfD (Contract for Difference):** Contract where the generator receives a fixed strike price via compensation against a reference market price.
- **Strike Price:** Agreed price per MWh under CfD.
- **Reference Market Price:** Market-based benchmark used only for settlement (does not affect generator revenue).
- **Quantity Risk:** Risk of not being able to generate or sell electricity.
- **Price Risk:** Risk of market price fluctuations.
- **Imbalance Risk:** Risk of deviations between scheduled and actual generation in a self-dispatch system.

---

## 2. Support Schemes Clarification

**CRITICAL:** In Georgia, support schemes are ONLY:
- PPA (Power Purchase Agreements)
- CfD (Contracts for Difference)

Support schemes are essential for building power plants, particularly large-capacity projects.

To make a project bankable, a stable revenue stream is required, especially protection against price risk.

- Typically, long-term price hedging is achieved through PPAs.
- In Georgia, there is a lack of reliable private offtakers acceptable to financial institutions.
- Therefore, the Government of Georgia acts as a counterparty:
  - Under PPA → obligation to purchase electricity
  - Under CfD → obligation to compensate price differences

These contracts define:
- predetermined price
- predetermined months of support (when buy or compensate)
- predetermined contract duration

Additionally:
- Due to the lack of a transparent and liquid market providing strong price signals, support schemes play a critical role in generation development.

---

## 3. CfD (Contracts for Difference)

Contracts for Difference are introduced for new renewable power plant projects under Georgia’s capacity auction scheme.

### 3.1 Key Facts
- Georgia conducted two capacity auctions to support development of new hydro, solar, and wind power plants
- All winning projects are renewable and represent the first batch of CfD-based investments
- Under the transitional market model:
  - CfD plants are not allowed to sell electricity on the competitive exchange
  - Their generation during the support period is sold to ESCO
  - Plants are centrally dispatched by GSE (no self-dispatch)
- CfD ensures a fixed payment:
  - revenue = strike price × accepted generation
- The reference market price is used only for accounting and settlement between the offtaker and the market operator
- The reference market price does NOT affect the generator’s final income
- CfD owners are fully insulated from market price risk

---

## 3.2 Market Model Distinction (CRITICAL)

### Transitional Market Model (Current)
- Dispatch: **central dispatch by GSE**
- Market access: **no  participation in exchange**
- Sales: electricity sold to ESCO as balancing electricity. generation quantity = quantity sold to ESCO.
- Risk profile:
  - **No price risk**
  - **Quantity risk (curtailment not compensated)**
- Revenue:
  - strictly based on **accepted generation**

---

### Target Market Model (Future)
- Dispatch: **self-dispatch**
- Market access: **participation in electricity market (exchange)**
- Sales: electricity sold in market
- Risk profile:
  - **No price risk (still protected via CfD strike price)**
  - **Reduced curtailment exposure** (can optimize dispatch and sales)
  - **Imbalance risk emerges**
- Revenue:
  - based on **sold electricity**
  - CfD compensates difference to strike price

---

## 3.3 Market Implications

- Under the transitional market model:
  - CfD plants face **quantity risk**, not **price risk**
  - Curtailment by GSE leads to direct revenue loss (not compensated)

- As CfD capacity increases:
  - central dispatch complexity for GSE increases


- CfD electricity is effectively treated as balancing electricity:
  - contributes to balancing volumes
  - affects reference price formation under transitional model

- Under the target model:
  - producers can optimize market participation
  - quantity risk from unsold electricity decreases
  - imbalance risk becomes the dominant operational risk

- Price effect is uncertain,but generally:
  - in summer, higher share of CfD/PPA lead to higher market price
  - in winter, the CfD/PPA price may appear lower than the alternatives import/old regulated thermals, and its higher share may contribute to lower balancing electricity price.

---

## 3.4 Analytical Notes

- Treat CfD generators as:
  - renewable
  - fully price-insulated
  - operationally constrained

- Risk differentiation:
  - Transitional model → **quantity risk (curtailment)**
  - Target model → **imbalance risk** - still in quantity risk group



---

## 4. PPA (Power Purchase Agreements)

- Used for both renewable and thermal projects
- Typically long-term agreements
- All USD-indexed in Georgian case
- Provide price stability and bankability

### Market Model Distinction

#### Transitional Market Model
- Often linked with ESCO or government-backed arrangements
- Limited exposure to market mechanisms
- No or limited price risk

#### Target Market Model
- Can coexist with market trading
- May include structured contracts (e.g. partial hedging)
- Still provides price stability but interacts with market outcomes

---

## 5. Renewable Integration

- Renewable PPAs are fixed-price and USD-indexed
- This reduces residual balancing liquidity
- Increasing renewable share:
  - reduces hydro flexibility
  - increases balancing volatility

---

## 6. Key Events

- **2023:** First CfD capacity auction conducted (~170 MW awarded across hydro, solar, wind)
- **2024–2025:** First CfD projects entering operation and appearing as `CfD_scheme` in balancing composition

---

## 7. Support Scheme Terminology

**CRITICAL:** Use correct terminology.

### NOT Support Schemes
- Regulated tariffs (regulated_hpp, regulated_old_tpp, regulated_new_tpp) are NOT support schemes
- These are cost-plus tariffs approved by GNERC for cost recovery

---

### Additional Support Mechanism (Not a Support Scheme)

**Guaranteed Capacity Payments:**
- Applies to regulated thermal plants:
  - Gardabani TPP
  - Mtkvari Energy
  - Tbilisi TPP
  - G-POWER
- Purpose:
  - ensures cost recovery even at low generation
- Scope:
  - NOT applicable to PPA/CfD plants
- Nature:
  - internal tariff-based mechanism
  - not a market-based support scheme
  - not direct government contractual support
  - fully compensated through regulated tariffs

---

## 8. Correct Usage Examples

- ✅ "Georgia has two main support schemes: PPA and CfD. Additionally, guaranteed capacity payments provide support for old thermal power plants."
- ✅ "საქართველოში მოქმედებს ორი ძირითადი წახალისების სქემა: PPA და CfD."

- ❌ WRONG: "Two support schemes: renewable PPA and thermal PPA"
- ❌ WRONG: "Support schemes include regulated tariffs for HPPs"

---

## 9. Data Interaction Notes (for Analytics)

Support schemes are not directly stored as tariffs but can be observed indirectly in system data.

### Where CfD and PPA appear
- CfD and PPA generation can be observed in system trade / balancing structures
- CfD projects may appear as `CfD_scheme` (or similar classification)
- These volumes are typically outside exchange trading

---

### Implications for Analysis

#### Transitional Market Model (Current)

- CfD and PPA electricity is:
  - fully sold as **balancing electricity** during the support period
  - not traded in the competitive exchange

- Therefore, CfD/PPA forms a **core component of balancing supply** and acts as a **reference base for balancing price formation**
  - See `balancing_price.md` → "Core Pricing Logic" for how composition shifts affect balancing price

- Pricing characteristics:
  - CfD/PPA prices are typically **USD-linked**
  - Exchange rate (xrate) has a strong impact on GEL-denominated balancing prices

- Analytical treatment:
  - CfD/PPA volumes must be INCLUDED in balancing price analysis
  - They should be treated as **price-forming supply**, not external or residual
  - Changes in their volume or price directly affect system-wide price signals

---

#### Target Market Model (Future)

- CfD/PPA electricity:
  - ESCO, on behaldof the state, still has obligation to buy electricity form PPA plant at PPA price. it will become ESCO responsibility to market the procured electricity as it does not need itself. So, ESCO should act as a trader in this regard. w
  - CfD electricity will be sold on the Exchange and only price difference will be compensated under support scheme.

- Risk shift:
  - reduced curtailment exposure
  - increased imbalance risk



---

#### Important Considerations

- Do not treat CfD/PPA as neutral or excluded volumes
- Always assess:
  - their share in balancing supply
  - their price level (USD-linked)
  - exchange rate impact

