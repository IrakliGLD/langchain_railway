# Cross-Border Capacity

## Source Scope

Primary source: **GSE Ten-Year Network Development Plan of Georgia 2024–2034 (TYNDP)**, approved November 2024 — Table 1.3, Section 1.11, and the related project chapters.

This file covers the **physical cross-border interconnection layer**: interconnection lines, Total Transfer Capacity (TTC), operating modes, HVDC/transformer constraints, the regional synchronous-ring limitation, and planned interconnection projects.

Companion file — use `cross_border_trade.md` for the **rules/rights side** (import/export procedures, capacity-allocation concepts NTC/ATC, export priority, emergency trade, curtailment, TSO restriction powers). This file is about *how much can physically flow and in what configuration*; `cross_border_trade.md` is about *the rules that govern and allocate those flows*.

Georgia is interconnected with **Russia, Azerbaijan, Armenia, and Turkey**. These interconnections support imports, exports, transit, frequency support / system stability, evacuation of prospective Georgian generation, and regional electricity-market integration.

### Current-status and freshness rule
- This document is a **knowledge snapshot of a planning document**, not a live registry of built infrastructure.
- Project dates and statuses are the plan's **planning targets as stated in November 2024** — they are **not** confirmation that a project was completed by that date.
- Dates described as **planned** or given as a target year (e.g. *planned for 2025*, *planned for 2030*) **must not be treated as confirmation that the interconnection is operational**. Report them as *planned; completion unverified*.
- When asked whether a specific interconnection (e.g. Marneuli–Ayrum, Tao / Akhaltsikhe–Tortum, Ksani–Stepantsminda–Mozdok) is **in service now**, state this limitation and verify against an authoritative, up-to-date source before asserting completion.

---

## 1. TTC Is Configuration-Dependent, Not a Line Rating

**TTC (Total Transfer Capacity)** is the transfer ceiling of an interconnection *under a given operating configuration and after security requirements*. It is distinct from:
- the **thermal / physical rating** of an individual line;
- the **nameplate rating** of a converter or transformer;
- the **physical energy flow** (MWh) actually exchanged.

Usable exchange capability is limited not only by line ratings but also by: the permitted regional operating configuration; synchronous-stability requirements; HVDC back-to-back station capacity; transformers and autotransformers; internal Georgian transmission constraints; and security criteria, including the simultaneous availability of other interconnections.

**Relationship to the allocation concepts in `cross_border_trade.md`:** TTC is the *planning / operational ceiling* for a corridor or interconnection. The market-allocation concepts (NTC, ATC, already-allocated capacity) sit *below* TTC, after security margins and prior allocations are removed. Do not equate TTC with ATC, and do not treat TTC as a value that exists as a series in the app's data.

---

## 2. Interconnections and Total Transfer Capacity

TTC values below are from Table 1.3 of the TYNDP. **Summer** and **winter** TTC differ because the permissible operating conditions differ by season.

| Neighbor | Interconnection | Summer TTC (MW) | Winter TTC (MW) | Operating mode / status |
|---|---|---:|---:|---|
| Russia | Kavkasioni | 570 | 650 | Synchronous (S) |
| Russia | Stepantsminda — Ksani–Stepantsminda–Mozdok | 1,000 | 1,000 | Synchronous (S); **planned for 2030** |
| Russia | Salkhino | 264 | 300 | Isolated (I) |
| Azerbaijan | Mukhranis Veli | 1,300 | 1,500 | Synchronous (S) |
| Azerbaijan | Gardabani 1,2 | 700 | 700 | Synchronous (S) |
| Armenia | Alaverdi | 150 / 100 | 150 / 100 | 150 MW synchronous / 100 MW isolated |
| Armenia | Marneuli–Ayrum | 700 | 700 | Back-to-back (B); **planned for 2025** |
| Turkey | Meskheti | 1,050\* | 1,050\* | Back-to-back (B); export route in Table 1.3 |
| Turkey | Tao — Akhaltsikhe–Tortum | 1,050\* | 1,050\* | Back-to-back (B); **planned**; import route in Table 1.3 |
| Turkey | Adjara | 150 | 150 | Isolated / reserve (I/R) |

**Operating-mode legend**
- **S — Synchronous:** the interconnected AC systems operate in parallel at the same frequency.
- **I — Isolated:** the tie supplies a separated / islanded part of the system rather than joining the complete systems synchronously.
- **B — Back-to-back:** electricity is exchanged through an HVDC converter station while the AC systems remain asynchronous.
- **R — Reserve:** the line is retained as a reserve connection rather than being continuously used.

**\* The Turkey values are NOT additive.** The 1,050 MW shown for both Meskheti and Tao is the transfer ceiling of the **Georgia–Turkey corridor through the shared Akhaltsikhe HVDC station** — not 1,050 MW simultaneously available on each line. See §3.

The Russia, Azerbaijan, and Armenia values appear in Table 1.3 for both import and export. Turkey is presented differently: Table 1.3 associates **Meskheti with export** and the planned **Tao line with import**, both sharing one 1,050 MW TTC supported by the Akhaltsikhe back-to-back station.

---

## 3. Georgia–Turkey Constraint: Akhaltsikhe HVDC

Each 400 kV line (Meskheti and Tao) can technically transfer up to ~1,500 MW, but the usable exchange capacity with Turkey is capped by the **Akhaltsikhe HVDC back-to-back station**:
- existing station: two 350 MW converter units ≈ 700 MW conversion capacity;
- plan: build the Tao / Akhaltsikhe–Tortum line, install a **third 350 MW unit**, and raise total HVDC conversion capacity to **1,050 MW**.

The project chapter gives a 2024–2026 implementation window while Table 1.3 marks Tao as planned for 2025 — treat both as indicative planning targets (see the freshness rule).

> **The transmission lines are not the principal Turkey-corridor bottleneck; the shared HVDC conversion capacity is.**

---

## 4. Georgia–Azerbaijan Constraint: Gardabani Substation

The combined operating limit of the 330 kV **Gardabani 1 and Gardabani 2** lines is ~1,400 MW, but the section is constrained by the **330/220 kV autotransformers at Gardabani Substation** — two units, 400 MVA each, 800 MVA total. Table 1.3 therefore gives a conservative **700 MW TTC** for Gardabani 1,2.

The 800 MVA equipment rating and the 700 MW TTC are **not interchangeable**: TTC is an active-power transfer limit set after operating and security requirements.

The plan also states total Georgia–Azerbaijan exchange capability of ~**2,000 MW from 2023** — lower than the arithmetic sum of the line-level winter TTCs, confirming that individual interconnection capacities must not be mechanically added.

> Note: "Gardabani" also names the Gardabani thermal power plant in `tariffs.md`. Here it refers to the **Gardabani interconnection substation** on the Azerbaijan corridor.

---

## 5. Georgia–Russia–Azerbaijan Synchronous-Ring Limitation

Precise formulation (do not overstate it):

> The Georgia–Russia–Azerbaijan systems **currently cannot operate as one fully closed synchronous ring**. At least one interconnection in that ring must remain open.

So the synchronous TTC values listed for Russia and Azerbaijan are **not necessarily all simultaneously usable** under a closed regional-ring configuration.

**Why the ring cannot currently be closed:** a fully closed ring would create uncontrolled **circulating power flows** (the plan expects them to move clockwise through the Georgia–Russia–Azerbaijan ring). Without dedicated cross-border flow-control equipment this would: reduce operational reliability; raise the risk of Georgia and/or Azerbaijan losing synchronism with the wider IPS/UPS system; reduce effective inertia; weaken frequency-containment capability; worsen security of supply and power quality; and restrict simultaneous exchange with both neighboring systems. The current solution is a weakened / open-ring scheme in which at least one tie is disconnected, or part of the exchange runs in isolated mode.

---

## 6. Planned Solution: Russia–Georgia–Azerbaijan Feasibility Study

The TYNDP includes a **Georgia–Russia–Azerbaijan Power-System Connection Project — Feasibility Study** (plan timeframe 2024–2025, with preparatory work beginning). Its stated aims are to identify the optimal infrastructure and the appropriate **flow-control technology** for joint operation (regulating cross-border and circulating flows), define the equipment's technical characteristics and installation locations, establish the required operating philosophy, and increase regional reliability and electricity-exchange potential. The study is meant to *identify* the flow-control solution rather than presuppose a particular technology.

---

## 7. Planned Regional Expansion (2025–2034)

For the 2025–2034 horizon the plan states exchange capability would increase by **+1,600 MW with Russia**, **+1,050 MW with Turkey**, and **+700 MW with Armenia**. Principal supporting projects:
- **Ksani–Stepantsminda–Mozdok** — a second high-capacity synchronous route to Russia, backing up Kavkasioni. Planned for 2030; designed for ~**700 MW in normal operation and 1,000 MW in emergency / maximum-transfer conditions**, improving reliability and access to frequency support from the Russian system.
- **Marneuli–Ayrum** — connects Georgia to the Armenian back-to-back infrastructure, supporting transit toward Armenia and Iran. Planned for 2025 in Table 1.3.
- **Akhaltsikhe–Tortum / Tao** — a second 400 kV route toward Turkey (see §3).
- **Third 350 MW Akhaltsikhe HVDC converter unit** — raises Turkey conversion capacity to 1,050 MW.

(All subject to the freshness rule — planning targets, completion unverified.)

---

## 8. Georgia–Romania Black Sea Interconnection

A **Georgia–Romania Black Sea Submarine Interconnection** was under feasibility study when the plan was prepared. Preliminary concept:
- a double-circuit 500 kV AC line from Jvari to a new **Anaklia** substation;
- a bipolar 500 kV DC submarine cable from **Anaklia to Constanța**;
- a 500/500 kV converter station at Anaklia;
- preliminary converter capacity of **2 × 500 MW**.

It would electrically connect Georgia and the South Caucasus with Continental Europe, supporting electricity trading, renewable-energy development, energy security, and regional transit.

**Technical note:** because it is planned as an **HVDC** connection, it would provide a *controllable, asynchronous* link — it would **not** automatically place the Georgian and Continental European AC systems in synchronous operation. It is similar in principle to a back-to-back converter, but over a long submarine cable.

---

## 9. Data Interpretation and Answering Guidance

**Data caveat:** TTC (MW) is a planning / operational ceiling and is **not** a series in the app's data. The app's `tech_quantity_view` holds **physical import/export volumes (MWh)** — do not present MWh flow as MW capacity, and do not present a TTC figure as a measured quantity.

**Capacities are scenario- and configuration-dependent — do not mechanically add them:**
1. **Turkey:** Meskheti and Tao share the Akhaltsikhe HVDC conversion limit — their capacities are not additive.
2. **Azerbaijan:** the sum of individual line ratings exceeds the plan's stated ~2,000 MW overall Georgia–Azerbaijan capability.
3. **Russia–Azerbaijan:** the synchronous interconnections cannot presently be combined into one fully closed three-system ring (circulating-flow and stability limits).

**Use this file when the user asks about:** interconnection capacity, TTC / total transfer capacity, operating modes (synchronous / isolated / back-to-back / reserve), a specific interconnection (Kavkasioni, Stepantsminda, Salkhino, Mukhranis Veli, Gardabani 1,2, Alaverdi, Marneuli–Ayrum, Meskheti, Tao, Adjara), the Akhaltsikhe HVDC or Gardabani transformer limits, the synchronous-ring question, planned interconnection projects, or the Black Sea / Romania interconnection.

**When answering:**
1. Separate **line rating**, **TTC**, **converter / transformer limit**, and **physical MWh flow** — they are different quantities.
2. State the **operating mode** (synchronous / isolated / back-to-back) where relevant.
3. Flag **plan status** for any planned project (planning target, completion unverified) per the freshness rule.
4. For the *rules* governing these flows (allocation, export priority, emergency, curtailment), use `cross_border_trade.md`. For import price impact, use `balancing_price.md`. For import dependence and transmission context, use `market_structure.md`.
