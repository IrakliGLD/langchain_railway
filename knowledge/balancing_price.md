# Balancing Price Formation and Drivers

## Definition
The balancing price is the weighted-average price of electricity sold on the BALANCING MARKET (not the general market or exchange), operated by ESCO monthly. ESCO is sole buyer of all balancing electricity. All non-contracted electricity is sold to ESCO as balancing electricity. "renewable_ppa" and "thermal_ppa" mandatorily sell only on balancing segment, they are not allowed to sell with bilateral contract or exchange.

**CRITICAL:** This definition and all analytical logic in this document apply to the CURRENT TRANSITIONAL MARKET MODEL (monthly balancing model with central dispatch).  
For all data analysis, ALWAYS use the current model logic.

**Target Model Note (Brief):**
In the future target model (planned ~July 2027), balancing price will no longer be a monthly weighted average. It will shift to hourly, marginal pricing under self-dispatch.  
However, this document applies ONLY to the current model.

---

## ESCO Buy vs. Sell Asymmetry (CRITICAL)

The "balancing price" defined above is the price ESCO **SELLS** balancing
electricity at — a single weighted-average price applied to all buyers of
balancing electricity in a given month.

ESCO **PAYS DIFFERENT PRICES** to each seller. The procurement (buy-side)
price is **per seller category**, not a single value. The two sides are
governed by different regulatory paragraphs and use different formulas.

The regulatory source is `transitory_market_rules.md`, **Article 14** —
*"მუხლი 14. სისტემის კომერციული ოპერატორის მიერ შესყიდული და გაყიდული
საბალანსო ელექტროენერგიის ფასის ფორმირება"* (formation of the price of
balancing electricity **purchased and sold** by the system commercial
operator). Article 14 is structurally dependent on **Article 13** (which
defines what counts as balancing electricity and who sells/buys it) and
references **Article 36(1)** and **Article 36(2)** for special seller
categories.

### A. What counts as balancing electricity (Article 13)

ESCO is the **sole wholesale buyer** of balancing electricity (Art 13 §1).
Trade happens via direct contracts OR under standard conditions defined by
the rules.

**Mandatory purchase categories** (Art 13 §4.ა–ლ):
- Generators (busbar output minus direct-contract sales)
- Importers (imports minus direct-contract sales)
- Concept Art 17(4) §7 enterprises (busbar output minus exchange + direct-contract sales)
- Direct customers / USP / free supplier / public-service supplier /
  last-resort supplier (positive difference between exchange purchases and actual consumption)
- Dispatch licensee (transmission losses + transit losses)
- Distribution licensee (distribution losses)

**Also counts as balancing electricity** (Art 13 §5–§5(8)):
- §5: **Mandatory direct contracts** under government/legal acts —
  this is the **PPA/CfD bridge** (`renewable_ppa`, `thermal_ppa`,
  `CfD_scheme` are all sold to ESCO via such contracts)
- §5(1): Other direct contracts with producers/importers
- §5(2): Transit-deviation import contracts under Art 14(8)
- §5(4): Transit-deviation import direct contracts under Art 14(8)
- §5(5): Capacity-auction-selected producer during support period
- §5(6): Premium-tariff producer during support period
- §5(7): Plant purchase from exchange (mandatory resale to ESCO)
- §5(8): Exchange operator (Art 8(1) §2(3) case)

**NOT balancing electricity** (Art 13 §7): export-direct-contracted
electricity and import-direct-contracted electricity sold downstream.

**Buyers of balancing electricity from ESCO** (Art 13 §9–§14):
direct customers, USP, free/public/last-resort suppliers, dispatch
licensee (transmission + transit losses), distribution licensee
(distribution losses), exporter qualified enterprises, Concept Art 17(4)
qualified enterprises, exchange operator.

### B. Per-seller buy-side prices (Article 14 — buy side)

ESCO's buy-side price is determined per seller category. Article 14 has
multiple buy-side paragraphs. Most of the activity is in §1; the rest
cover special cases.

#### B.1. Article 14 §1 — Standard-conditions per-seller rules

The main per-category enumeration. Sub-letter codes (ა, ბ, ..., კ.ბ)
are the original Georgian sub-points — preserve them when citing.

| Sub-point | Seller category | Price ESCO pays |
|---|---|---|
| **ა** | Qualified enterprise owning a regulated **fixed-tariff** regulating power plant | The Commission-set generation tariff for that plant |
| **ბ** | Qualified enterprise owning a regulated **upper-cap-tariff** power plant (excluding cases გ, დ, ე) | The Commission-set upper-cap generation tariff for that plant |
| **გ** | Thermal power plant qualified enterprise that is **NOT** designated by the Government of Georgia as a guaranteed-capacity source for the period | The Commission-set upper-cap generation tariff for that plant |
| **დ** | Qualified enterprise owning a **guaranteed-capacity source** (excluding case ე) | The upper-cap generation tariff for that guaranteed-capacity source |
| **ე** | Guaranteed-capacity source during **testing**, on the enterprise's request, for electricity supplied to the busbar | Weighted-average price of balancing electricity purchased that period under §§1, 1(1), 1(2), 1(4), 2, 3(6), 3(7) — but **not exceeding** the Commission-set upper-cap generation tariff for that source |
| **ვ** | Qualified enterprise owning a **deregulated small-capacity** power plant | Per **Article 36(1) §2** of these Rules — see section D |
| **ზ.ა** | **Importer** (other than ESCO), Sep 1 – May 1 (winter) | The **highest** generation tariff among balancing-electricity sellers to ESCO under standard or direct-contract (Art 13 §5) terms — capped at the Commission-set import upper-cap tariff |
| **ზ.ბ** | **Importer** (other than ESCO), May 1 – Sep 1 (summer) | The tariff of the regulated fixed-tariff plant with the **LOWEST** Commission-set generation tariff |
| **თ** | Qualified enterprise owning a **newly built** power plant | Per **Article 36(2)** of these Rules (with exception in §1(1)) — see section D |
| **ი** | Qualified enterprise per Art 3 §2 "b.z" | The tariff of the regulated fixed-tariff plant with the **LOWEST** Commission-set generation tariff — unless the government MoU on plant construction specifies otherwise |
| **კ.ა** | **Deregulated** plant (excluding ვ, თ), Sep 1 – May 1 (winter) | The tariff of the post-2010-built guaranteed-capacity source with the **HIGHEST** Commission-set generation tariff (testing periods excluded). Fallback: if no current-period tariff exists, use the previous reporting period's highest tariff (excluding May 1 – Sep 1) |
| **კ.ბ** | **Deregulated** plant (excluding ვ, თ), May 1 – Sep 1 (summer) | The tariff of the regulated fixed-tariff regulating power plant with the **LOWEST** Commission-set generation tariff |

#### B.2. Article 14 §1(1) — Direct contracts (THE PPA/CfD BRIDGE)

For electricity covered by Art 13 §5 (mandatory direct contracts) and
Art 13 §5(1) (other direct contracts), ESCO purchases at the **price
defined in the direct contract**.

This is how `renewable_ppa`, `thermal_ppa`, and `CfD_scheme` producers
get paid — at their **actual USD support-scheme contract price**.
Contract rates are confidential; the system-level inferred PPA/CfD
benchmark is roughly **55–57 USD/MWh** per the analytical estimation
methodology elsewhere in this document.

#### B.3. Article 14 §1(2), §1(4), §2 — Import buy-side variants

- **§1(2)** — Art 13 §5(2) imports: priced at the Commission-set
  **import upper-cap tariff**.
- **§1(4)** — Art 13 §5(4) transit-deviation imports: priced at the rate
  defined in **Article 14(9) §1**.
- **§2** — Non-direct-contract imports including **emergency imports**:
  priced at the Commission-set **import upper-cap tariff**.

#### B.4. Article 14 §2(1) and §2(2) — Special producer settlements

These are weighted-average settlements for producers in special support
windows (capacity auction, premium tariff). The structure is symmetric:
each excludes the other's quantities to avoid double counting.

- **§2(1)** — Art 13 §5(5) **capacity-auction** producer during support
  period: priced at the **weighted average** of §1, §1(1), §1(2), §1(4),
  §2 balancing electricity in that period — excluding §2(2), §4, §5, §6
  quantities and costs.
- **§2(2)** — Art 13 §5(6) **premium-tariff** producer during support
  period: priced at the **weighted average** of §1, §1(1), §1(2), §1(4),
  §2 — excluding §2(1), §4, §5, §6 quantities and costs.

Final settlement for these producers also involves compensation amounts:
- §5(5) producer: settled via §2(1) + §3(2) (per §3(3))
- §5(6) producer: settled via §2(2) + §3(4) (per §3(5))

#### B.5. Article 14 §3(6), §3(7) — Cheapest-HPP-priced cases

- **§3(6)** — Art 13 §4.დ–ლ losses (dispatch licensee, distribution
  licensee, exchange-side suppliers) plus Art 13 §5(7) (plant exchange
  purchase): settled at the tariff of the regulated fixed-tariff HPP
  with the **LOWEST** Commission-set generation tariff.
- **§3(7)** — Art 13 §5(8) exchange operator (Art 8(1) §2(3) case):
  settled at the **cheapest fixed-tariff regulated HPP** tariff.

#### B.6. Article 14 §7 — Documentation default

If a qualified enterprise fails to submit pricing documentation within
**2 working days**, the price defaults to the **cheapest fixed-tariff
regulated HPP** tariff. (Penalty/default rule.)

#### B.7. Mapping Article 14 paragraphs to internal entity categories

For `trade_derived_entities WHERE segment='balancing'`:

| Internal entity | Article 14 rule(s) | Currency basis |
|---|---|---|
| `regulated_hpp` (most fixed-tariff regulated HPPs, e.g. Enguri, Khrami, Vardnili) | §1.ა | GEL (per Commission tariff) |
| `regulated_new_tpp` (Gardabani — designated guaranteed-capacity source) | §1.დ | GEL (per upper-cap tariff, gas-cost linked) |
| `regulated_old_tpp` (older thermal, not guaranteed-capacity) | §1.გ or §1.ბ | GEL (per upper-cap tariff, gas-cost linked) |
| `deregulated_hydro` (mid-large deregulated plants) | §1.კ.ა (winter) / §1.კ.ბ (summer) | GEL (winter: indirect USD link via post-2010 GC source; summer: cheapest fixed-HPP tariff) |
| Deregulated **small-capacity** plants | §1.ვ → **Art 36(1) §2** | GEL (same winter/summer pattern as §1.კ) |
| Newly built plants | §1.თ → **Art 36(2)** | GEL or contract (winter post-2010 GC tariff capped by GEP contract; see section D) |
| `renewable_ppa`, `thermal_ppa`, `CfD_scheme` | **§1(1)** via Art 13 §5 mandatory direct contracts | USD (contract, confidential, ~55–57 USD/MWh inferred benchmark) |
| `import` (other importers, non-ESCO) | §1.ზ.ა (winter) / §1.ზ.ბ (summer) | USD/GEL (winter: highest seller tariff capped at import upper-cap; summer: cheapest fixed-HPP tariff) |
| Special Art 13 §5(2) imports | §1(2) | Import upper-cap tariff |
| ESCO's own non-direct-contract / emergency imports | §2 | Import upper-cap tariff |
| Capacity-auction producer in support period | §2(1) + §3(2) + §3(3) | (weighted-average derived) |
| Premium-tariff producer in support period | §2(2) + §3(4) + §3(5) | (weighted-average derived) |

### C. Sell-side balancing price formation (Article 14 §3)

ESCO sells balancing electricity to buyers (defined in Art 13 §9, §10,
§10(1), §11, §14) at a **single weighted-average price** computed per
reporting period.

**Article 14 §3 — Sell-side weighted-average price formula:**

The §3 weighted-average is what `p_bal_gel` and `p_bal_usd` in
`price_with_usd` represent.

Paragraphs that **ENTER** the §3 weighted-average:
- §1 (the per-seller-category buys — ა through კ.ბ)
- §1(1) (direct contracts including PPA/CfD)
- §1(2) (Art 13 §5(2) imports)
- §1(4) (Art 13 §5(4) transit-deviation imports at Art 14(9) §1 price)
- §2 (non-direct-contract imports)
- §2(1) (capacity-auction producer settlement)
- §2(2) (premium-tariff producer settlement)
- §3(6) (Art 13 §4.დ–ლ losses, §5(7) plant exchange — cheapest-HPP-priced)
- §3(7) (Art 13 §5(8) exchange operator — cheapest-HPP-priced)

Paragraphs that **DO NOT ENTER** (export-related, separate regime):
- §4 (export-buyer purchases from ESCO)
- §5 (ESCO's export rights)
- §6 (ESCO's export sales)

**Article 14 §3(1) — FX adjustment:**
The import-FX gain or loss recognized in the period is added to the §3
weighted-average. Adjusts the sell-side price for currency movement on
imports.

**Article 14 §3(2), §3(4) — Compensation amounts:**
For specific sell-side cases involving Art 13 §5(5) and §5(6) producers,
§3(2) defines a compensation amount = (bid_tariff − §2(1)_price) ×
§5(5)_volume; §3(4) defines a compensation amount = max(0, max_price −
§2(2)_price) × §5(6)_volume. These adjust the §3 sell-side price for
those producer transactions.

### D. Special seller categories (Articles 36(1) and 36(2))

Article 14 §1.ვ and §1.თ delegate pricing to Articles 36(1) and 36(2)
respectively. Both follow the same seasonal pattern as §1.კ but are
written as standalone articles.

#### D.1. Article 36(1) — Deregulated small-capacity power plants

**Triggering condition (§1):** if a small-capacity plant's electricity
isn't pre-sold via direct contract or exchange, the plant is deemed to
be selling that electricity to ESCO at standard conditions.

**Pricing (§2):**
- **§2.ა** (Sep 1 – May 1, winter): tariff of the post-2010-built
  guaranteed-capacity source with the **HIGHEST** Commission-set tariff
  (excluding testing periods). Same fallback rule as Art 14 §1.კ.ა if
  current-period tariff is undefined.
- **§2.ბ** (May 1 – Sep 1, summer): tariff of the regulated **fixed-tariff
  HPP** with the **LOWEST** Commission-set tariff.

#### D.2. Article 36(2) — Newly built power plants

**Triggering condition (§1):** same as Art 36(1) — not pre-sold ⇒ deemed
sold to ESCO at standard conditions.

**Pricing (§2):**
- **§2.ა** (Sep 1 – May 1, winter): tariff of the post-2010-built
  guaranteed-capacity source with the **HIGHEST** Commission-set tariff,
  but **NOT EXCEEDING** the price specified in the **"Guaranteed
  Electricity Purchase Agreement"** between ESCO and the plant (excluding
  testing periods).

The GEP contract price cap is the **key difference** from Art 36(1) —
newly built plants typically have such a government-guaranteed purchase
agreement that caps ESCO's procurement cost.

### E. Export-related pricing (Article 14 §4–§6)

These rules govern export transactions through ESCO and **do NOT enter
the sell-side weighted-average** (§3 explicitly excludes them).

- **§4** — Exporter purchases from ESCO (also Art 13 §12, §13
  enterprises): the exporter purchases the **highest-tariff** balancing
  electricity for that period. If insufficient quantity exists at the
  highest tariff, ESCO fills by tariff descending order.
- **§5** — ESCO's right to export both direct-contract-sourced and
  balancing-electricity-sourced electricity.
- **§6** — When ESCO exports balancing electricity, it must export the
  **highest-tariff** balancing electricity for that period (excluding
  §4-route exports) within the export contract price (minus
  export-related expenses).

### F. Disambiguation rules + completeness for LLM answers

**Question routing (which side of the asymmetry):**

- *"what is THE balancing electricity price"* / *"what does ESCO sell
  at"* / *"what is `p_bal_gel`"* →
  Answer the **single weighted-average sell-side price** from
  `price_with_usd.p_bal_gel` / `p_bal_usd`. Cite **Article 14 §3** as
  the regulatory source. The number is the §3 weighted-average of the
  buy-side paragraphs listed in section C.

- *"what price does ESCO PAY to sellers"* / *"how much do producers
  RECEIVE for balancing electricity"* / *"price of balancing electricity
  ESCO is paying"* →
  Answer is **per seller category** from section B. Enumerate the
  relevant §1 sub-points (ა, ბ, ..., კ.ბ) AND any relevant non-§1
  buy-side paragraphs (§1(1) for PPA/CfD, §1(2)/§2 for imports,
  §2(1)/§2(2) for capacity-auction / premium-tariff producers).
  Cite Article 14 with the specific sub-letter or paragraph
  (e.g. "per Article 14 §1.ე" or "per Article 14 §1(1)"). Do NOT quote
  a single number for `p_bal_gel` — that is the sell side, not the buy
  side.

- *"how is the balancing price formed"* / *"how is `p_bal_gel` computed"*
  / *"what determines the balancing price"* →
  Answer with the **§3 weighted-average formula** from section C: list
  the paragraphs that enter, the paragraphs that are excluded, and
  mention §3(1) FX adjustment.

- *"what does ESCO pay to PPA / CfD / renewable PPA producers"* →
  Answer **§1(1) of Article 14** — purchased at contract price, USD
  support-scheme rate. Note PPA/CfD contract prices are confidential;
  the inferred system-level benchmark is ~55–57 USD/MWh.

- *"what does ESCO pay deregulated small plants"* / *"what does ESCO pay
  newly built plants"* →
  Answer via §1.ვ → **Article 36(1) §2** (small) or §1.თ →
  **Article 36(2) §2** (newly built). Both have summer/winter rules
  matching §1.კ; Art 36(2) has the additional GEP-contract-price cap.

- Ambiguous phrasing (e.g. *"price of balancing electricity"* with no
  buy/sell anchor) → present both sides briefly and ask the user which
  they meant, OR lead with the per-category buy-side list (more
  informative) and then note the aggregate sell-side number.

**Mandatory completeness for buy-side answers:**

When asked about ESCO's procurement prices, the answer **MUST**:

1. State the asymmetry: ESCO pays per-seller-category, sells at a single
   §3 weighted-average — these are different prices, different paragraphs.
2. Enumerate the relevant §1 sub-points (ა, ბ, ..., კ.ბ) and any other
   relevant buy-side paragraphs (§1(1) for PPA/CfD, §1(2) for special
   imports, etc.).
3. State the **seasonal split** explicitly for §1.ზ (import), §1.კ
   (deregulated), Art 36(1) §2, Art 36(2) §2 — never collapse winter
   and summer rules into one sentence.
4. Map to internal entity categories (the table in section B.7) if the
   user asked in terms of `regulated_hpp`, `thermal_ppa`, etc.
5. Cite the specific paragraph (e.g. "per Article 14 §1.კ.ა" or "per
   Article 14 §1(1)").

A buy-side answer that does NOT enumerate the relevant paragraphs is
incomplete — refuse to short-cut into a high-level summary when the
regulatory detail is the whole point of the question.

**Citation style:**

- Use `§` for paragraphs of an article: "Article 14 §1.ე", "Article 14 §3"
- Preserve Georgian sub-letter codes for §1 sub-points: ა, ბ, გ, ..., კ.ბ
- Use parens form for sub-articles: "Article 14(9)", "Article 36(1)",
  "Article 36(2)" — NOT "Article 14.9" or "Article 36¹" (the Georgian
  regulatory standard renders the superscript as parens in markdown)

---

**CRITICAL TERMINOLOGY:** when talking about balancing electricity price or balancing electricity, ALWAYS say "balancing market" or "balancing segment" — NEVER shorten to just "market". "market" is general term which includes "balancing market", "bilateral market" and "exchange" segments.

### Terminology in All Languages
- **English:** balancing market / balancing electricity / balancing segment
- **Georgian:** საბალანსო ბაზარი / საბალანსო ელექტროენერგია / საბალანსო სეგმენტი
- **Russian:** балансирующий рынок / балансирующая электроэнергия / балансовый сегмент

**RULE:** NEVER omit the word "balancing" when discussing balancing electricity price.

Examples:
- ✅ CORRECT: "საბალანსო ბაზარზე გაყიდული ელექტროენერგიის საშუალო შეწონილი ფასი"
- ❌ WRONG: "ბაზარზე გაყიდული ელექტროენერგიის საშუალო შეწონილი ფასი"
- ✅ CORRECT: "weighted average price of electricity sold on the balancing market"
- ❌ WRONG: "weighted average price of electricity sold on the market"

---

## Core Pricing Logic (System Level)

**CRITICAL:** In the current transitional model:
- The balancing market is the **core electricity settlement layer**
- A **significant share of total system electricity** is sold as balancing electricity
- Therefore, the balancing price acts as the **reference price for the system**

**Causal Chain:**
Central dispatch (GSE) → determines generation mix → determines balancing composition → determines balancing price (WAP)

There is:
- no marginal pricing
- no hourly clearing
- no merit-order price formation

Hence, price is driven by **composition**, not marginal cost.

**Structural Role of PPA/CfD:**
- CfD and PPA generation form a **core component of balancing supply**
- They act as a **reference base for balancing price formation**

Therefore:
- If lower-cost generation (e.g. deregulated or regulated hydro) is added to a system where CfD/PPA dominates → balancing price decreases  
- If higher-cost sources (e.g. import, regulated_old_tpp) are added → balancing price increases  

This reflects the weighted-average nature of the balancing price.

**IMPORTANT:**  
Not all generated electricity affects balancing price.  
Only electricity **actually sold on the balancing segment** enters price formation.

Therefore:
- High hydro generation does NOT reduce balancing price if that hydro is not sold as balancing electricity  
- Price depends on **balancing composition**, not total generation mix

---

## Data Source
Table: `price_with_usd` — columns: `p_bal_gel` (GEL/MWh), `p_bal_usd` (USD/MWh).  
Coverage: January 2015–present, monthly granularity.

**Coverage distinction (important):**
- **Price history** (`price_with_usd`): available from January 2015. Price trends, levels, and year-over-year comparisons can be shown for any period from 2015 onward.
- **Composition and driver evidence** (`trade_derived_entities`): reliable from 2020 onward only. Entity shares, balancing composition analysis, and driver explanations based on who sold what cannot be produced for pre-2020 periods.
- For pre-2020 periods: quote prices freely, but do NOT claim composition-based driver evidence. Say "composition data is not available before 2020."

---

## Weighting Entities
The following entities participate in the balancing price calculation:

- `deregulated_hydro` — Deregulated hydropower plants, one of the cheapest sources of balancing electricity  
- `import` — Direct electricity imports, USD-priced  
- `regulated_hpp` — Regulated hydro power plants, GEL tariffs, mostly the cheapest source  
- `regulated_new_tpp` — Regulated new thermal power plant (Gardabani), GEL tariff reflecting current xrate  
- `regulated_old_tpp` — Regulated old thermal power plants, GEL tariffs reflecting current xrate  
- `renewable_ppa` — Renewable PPA projects (hydro, solar, wind), USD-priced  
- `thermal_ppa` — Thermal PPA projects, USD-priced  
- `CfD_scheme` — Renewable projects under CfD support schemes, USD-priced  

**Important Structural Note:**
- PPA and CfD electricity are **mandatorily sold on balancing segment**
- They represent a **structural (non-optional) component** of balancing electricity

---

**Calculation Rule:**  
Weights are based on electricity sold as balancing energy by each entity.  
Total balancing quantity = sum of all listed entities from `trade_derived_entities` WHERE `LOWER(REPLACE(segment, ' ', '_')) = 'balancing'`.

---

## Primary Drivers

### 1. Market Composition (Most Important)
**Importance:** PRIMARY DRIVER — Most important for BOTH GEL and USD prices.

Balancing electricity price = weighted average of all categories selling on the BALANCING SEGMENT.

**Reference Role of PPA/CfD in Composition:**
- CfD and PPA volumes are structurally present in balancing electricity
- They act as a baseline level of price in the system

Price changes are driven by deviations from this baseline:
- Adding cheaper sources → lowers average price  
- Adding more expensive sources → increases average price  

- Higher share of cheap sources (`regulated_hpp`, `deregulated_hydro`) → lower price  
- Higher share of always-expensive sources (`import`, `regulated_old_tpp`, `thermal_ppa`) → higher price  
- **`renewable_ppa` and `CfD_scheme` — SEASON-DEPENDENT direction (CRITICAL):**
  - Their price is approximately 55–57 USD/MWh (fixed, USD-indexed contract)
  - **Summer:** this is ABOVE the typical hydro-dominant summer price level → higher share pushes balancing price UP
  - **Winter:** this is BELOW the most expensive winter source (import, position #7 in the price hierarchy); when PPA/CfD displaces import, their higher share pushes balancing price DOWN. The effect versus regulated thermals is ambiguous and gas-price-dependent.
  - **RULE:** Never state a general directional claim for renewable PPA/CfD without specifying the season. A general "impact on prices" question MUST be answered separately for summer and winter.

**Seasonal Effect:**
- Summer → free hydro (`regulated_hpp`, `deregulated_hydro`) abundant → lower price; BUT `renewable_ppa`/`CfD_scheme` push price UP relative to free-hydro baseline
- Winter → balancing mix dominated by PPA/CfD (baseline), regulated thermal, and imports → higher price; `renewable_ppa`/`CfD_scheme` are relatively cheaper than import/old thermals → they moderate (lower) winter price
  - Higher-cost sources have limited ability to displace this mix because buyers prefer the cheapest available balancing supply

**Balancing vs Total Generation (CRITICAL DISTINCTION):**
- Only electricity sold on the balancing segment affects the balancing price  
- Cheap generation reduces price ONLY if present in balancing electricity mix  
- If cheap hydro is not sold on balancing → it does NOT push price down  

**Structural Evolution:**
- Demand growth increasingly absorbs cheap generation outside balancing  
- As a result:
  - share of cheap hydro in balancing declines  
  - balancing price does not decrease as strongly as in the past, even in high-hydro periods  

**Deregulated Hydro Pricing (CRITICAL):**
- Summer (Apr–Jul): price is low (~<50 GEL/MWh), referenced to cheapest regulated HPP
- Winter (Aug–Mar): price increases (>100 GEL/MWh), referenced to thermal tariffs
→ In winter months, deregulated hydro becomes **indirectly USD-linked**

**Note on month boundaries:** the canonical summer/winter split used in code (`config.SUMMER_MONTHS` = months 4–7; `config.WINTER_MONTHS` = months 1–3 and 8–12) is the source of truth for all seasonal calculations and forecasts. The Aug–Mar deregulated-hydro reference window aligns with this split.

---

**Structural Trends:**
- Declining share of `regulated_hpp` and `deregulated_hydro`
- Increasing role of:
  - `renewable_ppa`
  - `import`
  - `thermal_ppa`
  - thermal generation  

→ Structural upward pressure on balancing price

---

**Data Source:**  
`trade_derived_entities` WHERE `LOWER(REPLACE(segment, ' ', '_')) = 'balancing'`

---

**Analysis Requirements:**
- MANDATORY: For long-term trends → analyze by season (summer vs winter)
- MANDATORY: Cite actual share changes
- MANDATORY: Identify cheap vs expensive sources

---

### Transmission Constraints

Even if cheap generation exists (e.g. Enguri HPP), it may NOT affect balancing price if:
- transmission constraints prevent delivery to demand centers

Therefore:
- balancing composition depends on **deliverable energy**, not total generation

---

### Export Effect

If hydro is exported:
- it is NOT available for balancing
- does NOT reduce balancing price

Export acts as competing sink for cheap generation

---

### Exchange Rate
**Importance:** CRITICAL for GEL price, SMALL impact on USD price.

**Variable:** `xrate` (GEL/USD) from `price_with_usd`.

---

**Entity Pricing Structure:**

- **Direct USD-priced:**
  - `renewable_ppa`
  - `thermal_ppa`
  - `import`

- **USD-linked (via gas cost):**
  - `regulated_old_tpp`
  - `regulated_new_tpp`

- **Partially USD-linked:**
  - `deregulated_hydro` (winter months)

- **GEL-priced:**
  - `regulated_hpp`

---

**Mechanism:**
- GEL depreciation → strong increase in GEL balancing price
- Majority of balancing electricity is **directly or indirectly USD-linked**
- See `currency_influence.md` for the full FX transmission logic and the detailed treatment of seasonally USD-linked deregulated hydro

---

**Analysis Requirements:**
- GEL analysis → composition + xrate
- USD analysis → composition only
- USD removes FX noise

---

## Price Hierarchy
From cheapest to most expensive (approximate):

1. Regulated HPP — ~below 50 GEL/MWh  
2. Deregulated Hydro — seasonal (low in summer, high in winter)  
3. Regulated Thermal (old/new TPP) — gas + xrate driven  
4. Renewable PPA — USD-priced (CONFIDENTIAL)  
5. CfD_scheme — USD-priced (CONFIDENTIAL)  
6. Thermal PPA — USD-priced (CONFIDENTIAL)  
7. Import — USD-priced, variable (CONFIDENTIAL)

---

**CONFIDENTIALITY RULE:**  
Do not disclose specific PPA/import prices.

**Allowed analytical inference:**  
- Exact PPA and CfD contract prices are not public and must not be presented as disclosed tariffs.  
- However, an approximate system-level benchmark may be inferred from balancing-market data in months where PPA/CfD volumes together with entities that have known prices account for more than roughly 99% of balancing electricity.  
- In those months, the implied weighted-average price of the remaining support-scheme block is approximately **55-57 USD/MWh**.  
- This should be treated as an **analytical estimate**, not as an official public contract price, not as a plant-specific tariff, and not as a guarantee that the same value applies in every month.  
- If imports or other residual components are still material in the unexplained bucket, the inferred value may reflect a blended residual benchmark rather than a pure PPA-only price.

---

## Key Events (Policy & Structural)

This is the canonical timeline of policy decisions that have shifted balancing-price formation. Other files reference back here:
- `market_structure.md` §9 covers the regulatory/market-design timeline (2006 balancing market, 2024 GENEX, 2027 target model).
- `cfd_ppa.md` §6 covers CfD auction rounds.
- `tariffs.md` covers individual-plant deregulation dates.

### Composition-affecting policy events

- **Jan 2024:** Gas price increase for regulated thermals → thermal tariffs increased → higher balancing price when regulated thermals are sold as balancing electricity.
- **Jan–Mar 2024 (temporary rule):** the Electricity (Capacity) Market Rules were temporarily changed for three months. During Jan–Mar 2024, the reference price for deregulated hydropower plants was linked to the more expensive **regulated old thermal power plant** benchmark, increasing the procurement price of balancing electricity from deregulated hydropower plants.
- **Jul 2024:** Exchange launched (GENEX) — added a new trading segment alongside balancing.
- **May–June 2025 (allocation pattern):** Enguri HPP, the largest state-owned hydropower plant and one of the cheapest electricity sources in the system, sold exactly **16.698 thousand MWh** as balancing electricity in **both May 2025 and June 2025**. Identical monthly balancing volumes do not look like random residual sales. This pattern suggests that cheap state-owned hydro may have been deliberately allocated to the balancing segment in those months to help push the balancing price downward. Treat this as a strong analytical inference, not a proven policy fact unless supported by direct operational evidence.
- **2020 onwards:** Entity-level data available (no composition-based driver evidence before 2020).
- **From May 2025:** Enguri/Vardnili tariffs increased to cover Abkhazia supply costs (see `market_structure.md` §14).
- **Jul 2027 (planned):** target market model launch (hourly marginal pricing under self-dispatch) — see `market_structure.md` §3 for the regime-break detail. Forecasts crossing this horizon must flag the structural break.

### Why this matters for forecasting

Balancing-price forecasts that ignore the policy lever set above will systematically miss inflection points. The dominant levers are: (1) gas-price decisions, (2) deregulated-hydro reference rules, (3) state-owned-hydro allocation choices, (4) CfD/PPA capacity additions (see `cfd_ppa.md` §6), (5) the 2027 regime break. Any 3+ year forecast narrative MUST mention which of these are assumed unchanged.

---

## Analysis Guidelines

### For Correlation Analysis
1. Focus on composition  
2. Include xrate for GEL  
3. Use balancing segment data  
4. Support with numeric evidence  

---

### For Price Explanation
1. Start with share changes  
2. Identify drivers (cheap vs expensive)  
3. Add xrate (if GEL)  
4. Separate seasonal vs structural effects  

---

### For Seasonal Analysis
- Summer → free hydro (`regulated_hpp`, `deregulated_hydro`) → lower price  
- Summer → `renewable_ppa` / `CfD_scheme` → HIGHER price (fixed ~55–57 USD/MWh exceeds cheap summer hydro levels)
- Winter → thermal/import → higher price  
- Winter → `renewable_ppa` / `CfD_scheme` → LOWER price when displacing import; ambiguous when displacing regulated thermal (depends on gas prices)
- NEVER state a single directional claim for renewable PPA/CfD without seasonal qualification
- Use MoM and YoY comparisons  

---

### For Forecasting

**Mandatory output shape:**
- ALWAYS produce a **separate forecast for summer and winter** — never a single annual figure. Drivers differ across seasons (hydro share, PPA/CfD displacement, import reliance), so a single number masks the real dynamics.
- ALWAYS produce **both GEL and USD forecasts** when both series are available. They diverge because GEL is exposed to xrate while USD is not; users analysing affordability or contract economics need both.
- For a 5-year horizon, the minimum output is 4 forecast figures: (summer × GEL), (summer × USD), (winter × GEL), (winter × USD).

**Core rule:** forecasting the balancing electricity price is inherently difficult because it is **not directly determined by supply-demand clearing**. Under the current transitional model, it is a **weighted-average price** of electricity actually sold on the balancing segment.

Therefore, a serious forecast of balancing price requires forecasting **both**:
- the future **price** of each component
- the future **share/composition** of each component in balancing electricity

**Main forecasting uncertainties:**
- **Imports:** prices depend on regional market conditions and external electricity prices, so they are highly uncertain and difficult to forecast.
- **Regulated hydro:** its share in balancing electricity is currently modest and is expected to decline as hydropower deregulation expands. However, **state-owned hydropower plants remain regulated** and can still influence balancing prices when their cheap electricity is directed into the balancing segment. The **May-June 2025 Enguri HPP case** is an important example of this possibility.
- **PPA and CfD:** these support schemes are the main component and key driver of the reference balancing price. Their future effect depends on how quickly new projects are commissioned and what their contract prices are.
- **Regulated thermal power plants:** their cost is driven mainly by **gas prices** and the **exchange rate**. Gas prices are influenced by state decisions and negotiations, so future costs are uncertain.
- **Exchange rate risk:** most important components are directly or indirectly **USD-linked**, so GEL balancing price is highly sensitive to FX movements.

**Forecast interpretation rule:**
- Trend-based forecasts can be constructed from historical price series, but their reliability is limited because imports, gas prices, exchange rate, policy decisions, and market reform can all change the future balancing mix.
- For simple trend-based forecasting, it is better to **avoid separately forecasting the exchange rate**. If an assumption is needed, assume the exchange rate remains broadly stable around recent/current levels, and state that assumption explicitly.
- Forecasts extending toward or beyond **July 2027** must be treated with extra caution because the planned target market model would fundamentally change balancing price formation.

---

### Disclosure Rules
- Do NOT disclose PPA/import prices  
- Clarify support schemes = PPA + CfD only  

---

## Price Decomposition

### compute_entity_price_contributions

Calculates monthly entity contributions.

**Output columns:**
- `balancing_price_gel`
- `share_[entity]`
- `price_[entity]`
- `contribution_[entity]`
- `total_known_contributions`
- `residual_contribution_ppa_import`

---

**Available prices:**
- `deregulated_hydro` → `p_dereg_gel`  
- `regulated_hpp` → quantity-weighted tariff from `mv_balancing_trade_with_tariff`  
- `regulated_new_tpp` → Gardabani tariff from `mv_balancing_trade_with_tariff`  
- `regulated_old_tpp` → quantity-weighted tariff from `mv_balancing_trade_with_tariff` (all non-Gardabani TPPs with regulated tariffs)

Tariffs are weighted by each entity's actual balancing quantity. NULL when a regulated group had no balancing sales in a month.  

---

**Unavailable prices:**
- Renewable PPA  
- CfD_scheme  
- Thermal PPA  
- Import  

---

## Analytical Workflow
1. Identify share changes  
2. Estimate contribution impact  
3. Include xrate  
4. Adjust for seasonality  
5. Interpret residual  

---

## Interpretation Guidelines
- Contribution increase → higher price impact  
- Large residual → PPA/import influence  
- Always validate with:
  - seasonality
  - tariffs
  - composition  

