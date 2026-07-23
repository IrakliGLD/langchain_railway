# Cross-Border Trade

## Source Scope

Primary source: final **Electricity (Capacity) Market Rules** (Ministerial Order No. 77, August 30, 2006), Chapter IV^1:
- Article 14^1: general conditions for import and export
- Article 14^2: import/export during emergency situations
- Articles 14^3-14^14: import, export, transit, applications, capacity allocation, priorities, and capacity concepts

Related source:
- Article 16^1: Dispatch Licensee / TSO powers to restrict cross-border flows and production in deficit, surplus, or emergency situations
- Article 60: temporary 2026 surplus and production-restriction rule
- Electricity Market Model Concept Article 17^4: exchange trading transition from July 1, 2024 to July 1, 2027

---

## 1. Core Concept

Cross-border trade covers:
- import
- export
- transit
- cross-border capacity allocation
- emergency import/export
- curtailment or restriction of cross-border flows for system security

It is not the same as domestic balancing electricity, though imports can enter balancing-price formation when sold as balancing electricity under the Electricity (Capacity) Market Rules.

---

## 2. General Rule

Under Article 14^1, import/export is generally performed within the electricity/capacity balance and the relevant period. In emergency situations, import/export may be allowed outside the normal balance treatment under Article 14^2.

The Dispatch Licensee / TSO has a system-security role. It may restrict, reduce, or exclude import/export requests when needed for:
- system stability
- deficit management
- surplus management
- emergency operation
- transmission constraints
- balancing internal resources against cross-border schedules

---

## 3. Registration and Applications

Cross-border transactions normally require:
- import/export or transit application
- direct contract registration where the transaction is contract-based
- Dispatch Licensee review
- capacity availability check where cross-border transfer capacity is limited

Import/export direct contracts have an additional registration route under Article 9^1. Use `direct_contracts.md` for the contract-registration side and this file for cross-border operational treatment.

---

## 4. Capacity Concepts

The rules use several capacity-allocation concepts. Use these carefully:

- **NTC (Net Transfer Capacity):** transfer capacity available on an interconnection after technical/security limits.
- **NAC (Non-Available Capacity):** capacity not available because of technical or security constraints.
- **ATC (Available Transfer Capacity):** capacity still available for commercial use after reductions and allocations.
- **AAC / ALC:** already allocated or allocated capacity.
- **CAA:** capacity available for allocation by auction.

Do not confuse:
- physical import/export volume in MWh
- transfer capacity in MW
- auctioned capacity rights
- contractual energy schedules

In the app data, `tech_quantity_view` contains physical import/export quantities, not interconnection capacity.

For the **physical interconnection layer** — per-interconnection Total Transfer Capacity (TTC), operating modes, HVDC/transformer limits, the Russia–Georgia–Azerbaijan synchronous-ring constraint, and planned interconnection projects — see `cross_border_capacity.md`. This file governs the *rules and allocation* of cross-border flows; `cross_border_capacity.md` describes *the physical capacity and configuration* that bound them.

---

## 5. Export Priority and Auctions

The final rules include priority logic for export and cross-border capacity allocation. In general:
- some export quantities may receive priority based on the rules
- renewable or contract-backed export cases can have special treatment where explicitly named
- if available capacity is not enough for priority or requested volumes, the remaining allocation can move to auction procedures

Do not state that export access is always first-come-first-served. It depends on the article, the interconnection, the application type, and whether priority capacity rules apply.

---

## 6. Surplus, Curtailment, and TSO Powers

Article 16^1 is critical for surplus/curtailment questions.

When the system faces surplus, forecast surplus, deficit, emergency, or a real risk of these conditions, the Dispatch Licensee / TSO may:
- stop or restrict cross-border inflow
- reduce import/export volumes
- exclude certain cross-border applications from the balance
- restrict domestic generation if needed for system security

Production restriction is generally applied proportionally against expected daily load-balance quantities, with important exceptions.

Important exceptions:
- curtailment does not apply to electricity produced for export
- annual-regulation reservoir HPPs under PSO can be restricted or spilled only for hydrotechnical safety reasons
- Article 60 creates a temporary May-July 2026 rule for surplus and production restrictions

**Price effect of surplus (pointer):** under the current transitional model, a surplus does NOT lower electricity prices — contracts are not firm and curtailed electricity is not compensated, so no producer gains by offering a low price; the surplus is absorbed by curtailment while the balancing price stays high or rises (observed May–June 2020 and May–June 2026). For the full mechanism and the "reference price minus a discount" pricing behavior, see *"Why the Balancing Price Does NOT Fall During Surplus"* in `balancing_price.md`.

---

## 7. Article 60 Temporary 2026 Rule

Article 60 is a temporary rule for May, June, and July 2026.

High-level treatment:
- if PSO plants export generation during surplus, an equivalent volume from other plants may be treated through the balancing/direct-contract settlement logic
- the equivalent volume is generally allocated proportionally by actual busbar generation
- exceptions include PSO plants themselves, plants under GEP/PPA/CfD or mandatory ESCO-sale/CfD-compensation obligations for that period, and exported portions

Use this only for the temporary 2026 surplus situation. Do not generalize it as the normal balancing-price rule.

---

## 8. Price and Data Interpretation

Import affects electricity prices through several channels:
- import volumes can be USD-priced and raise GEL balancing prices through exchange-rate exposure
- imports sold as balancing electricity may enter the Article 14 weighted-average balancing price
- direct-contracted import or export quantities may be excluded from the balancing price depending on the Article 13 route
- export can remove cheap hydro from domestic balancing availability, increasing balancing-price pressure

For analytics:
- use `tech_quantity_view` for physical import/export volume
- use `trade_derived_entities` for market-segment trade where available
- use `balancing_price.md` for import pricing inside the balancing-price formula
- use `market_structure.md` for import dependence and transmission interpretation

---

## 9. Answering Guidance

Use `cross_border_trade.md` when the user asks about:
- import/export rules
- transit
- interconnection capacity
- cross-border capacity allocation
- export priority
- emergency import/export
- surplus, curtailment, or production restrictions linked to exports/imports

When answering:
1. Separate physical flow, contract, capacity right, and balancing settlement.
2. State whether the question is about normal operation, emergency, or surplus.
3. If the question asks about price impact, connect imports/exports to `balancing_price.md`.
4. If the question asks about data, warn that MWh volumes are not MW capacity.
