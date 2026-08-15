# Network and End-User Supply Tariffs

Scope: the **retail** side of the Georgian electricity price — network (distribution and
transmission) tariffs and the regulated end-user price paid by households and commercial
customers.

This is the counterpart to **Tariff Structure** (`tariffs.md`), which covers the
**generation** side: GNERC cost-plus tariffs for regulated hydro and thermal plants. Use
`tariffs.md` for questions about what a *power plant* is paid. Use this document for questions
about what a *consumer* pays.

---

## Definitions

- **End-user (retail) tariff:** The regulated price a final consumer pays per kWh consumed.
  Expressed in **GEL/kWh** — note this differs from generation tariffs and wholesale prices,
  which are quoted in GEL/MWh.
- **Distribution tariff:** Regulated charge for delivering electricity over the low- and
  medium-voltage distribution network. A natural monopoly, so it is a PSO activity and GNERC
  sets it.
- **Transmission tariff:** Regulated charge for the high-voltage transmission network operated
  by GSE (Georgian State Electrosystem). Also a natural monopoly and PSO activity.
- **Supply tariff:** The charge for the supply service itself — procuring electricity on the
  wholesale market and selling it to the end customer, plus the pass-through costs that come
  with it.
- **Universal service:** Regulated supply to households and small commercial customers who
  have not chosen a competitive supplier.
- **Public service:** Regulated supply to the public-service commercial customer category.
- **SOLR (supplier of last resort):** Backstop supply when a customer's supplier fails or
  exits. It is **not** part of the standard end-user price build-up.

---

## Structure of the end-user price

The regulated end-user price for one month and one consumer category is the **sum of three
components**:

| # | Component | Companies | Role |
| - | ------------- | ---------------- | ---- |
| 1 | Distribution | `telasi`, `epg` | Distribution network |
| 2 | Supply | `telmico`, `eps` | Supply service (universal or public) |
| 3 | Transmission | `gse` | Transmission network |

### Supplier to distributor pairing

Each supplier operates in the service territory of one distribution company, so the two are
always paired:

| Supplier | Distribution company |
| ---------- | -------------------- |
| `telmico` | `telasi` |
| `eps` | `epg` |

`telasi` is the Tbilisi distribution network; `epg` (Energo-Pro Georgia) covers most of the
rest of the country.

Transmission is the **same single row for every category** — `gse`, with blank voltage and
blank consumer classes — because the transmission charge does not vary by consumer class.

---

## What the supply tariff includes

The supply component is not merely a retail margin. It carries the supplier's cost of
procuring electricity plus regulated pass-through elements, most importantly:

- **Wholesale electricity procurement cost** — what the supplier pays to buy the energy.
- **Guaranteed capacity fee pass-through.** The guaranteed capacity fee is a fixed availability
  payment made to regulated thermal power plants for standing ready to supply capacity (see
  `tariffs.md`, where it is defined on the generation side and expressed in GEL/day). That cost
  is recovered from end consumers in proportion to their consumption, and it reaches them
  **through the supply component of the end-user tariff**. This is the link between the two
  documents: a change in the guaranteed capacity fee paid to thermal plants shows up on the
  retail side as movement in the supply component, not in distribution or transmission.
- Other regulated pass-through and system-service costs.

**This is why supply and network components move for different reasons.** Distribution and
transmission tariffs are set for a regulatory period and change on regulatory decisions.
The supply component tracks wholesale procurement cost and pass-through elements, so it is
the component that responds to gas prices, exchange rates, and the generation-side cost
changes described in `tariffs.md`.

---

## Consumer categories

A category is identified by three dimensions: **voltage level**, **consumer class**
(`level_1_cat`), and **consumer sub-class** (`level_2_cat`).

### Voltage levels

Stored verbatim as `220/380`, `3.3-6-10`, and `35-110`. Report them exactly as stored; do not
rename them to LV/MV/HV.

### Household consumption bands

Households are billed on a tiered scale by monthly consumption:

| Class | Band |
| ------ | ------------------------- |
| `cat1` | up to and including 101 kWh |
| `cat2` | 101–301 kWh (301 included) |
| `cat3` | above 301 kWh |

Both major distribution territories use the same 101 / 301 kWh thresholds.

### The eight categories

| # | volate | level_1_cat | level_2_cat | Supply activity | Meaning |
| - | ---------- | ----- | ------- | ----------- | ------- |
| 1 | `220/380` | `com` | `other` | `public` | Commercial, other |
| 2 | `220/380` | `com` | `small` | `universal` | Commercial, small |
| 3 | `220/380` | `hh` | `cat1` | `universal` | Household ≤101 kWh |
| 4 | `220/380` | `hh` | `cat2` | `universal` | Household 101–301 kWh |
| 5 | `220/380` | `hh` | `cat3` | `universal` | Household >301 kWh |
| 6 | `3.3-6-10` | `com` | `other` | `public` | Commercial, other |
| 7 | `3.3-6-10` | `hh` | `` (blank) | `universal` | Household |
| 8 | `35-110` | `com` | `other` | `public` | Commercial, other |

Each category exists for both suppliers, giving 16 published end-user prices per month.

**Load-bearing irregularity.** In categories 6 and 8 the supply component is filed under
`level_2_cat = 'other'` while the matching distribution component uses a **blank**
`level_2_cat`. This mismatch is real, not a data error. Resolving these two categories by
matching `level_2_cat` across all three components will silently drop the distribution row and
produce an incomplete price.

---

## Relationship to the wholesale market

- Generation tariffs (`tariffs.md`) and wholesale prices are **GEL/MWh**. End-user tariffs are
  **GEL/kWh**. A factor of 1000 separates them. Never compare or combine them without
  converting.
- The end-user price is not a marked-up balancing price. It is a regulated build-up of three
  separately approved components, only one of which (supply) tracks procurement cost.
- Consequently a movement in the balancing price does **not** translate directly or
  immediately into a movement in the end-user price.

---

## Analytical implications

- Decomposing the end-user price into its three components shows *where* a retail price change
  came from: a network decision (distribution or transmission) or procurement and
  pass-through cost (supply).
- Comparing the same consumer category across the two suppliers isolates territory-specific
  network cost, since the transmission component is identical for both.
- Comparing `cat1` / `cat2` / `cat3` within one supplier shows the tiered structure's
  progressivity.
- Component **shares** (each component as a percentage of the total) are usually more
  informative across time than absolute GEL/kWh levels, because they show whether network or
  supply cost is driving the change.

---

## Data Mapping

### End-user tariff components

- View: `demand_tariff_mv`
- Grain: one row per `(date, company, activity, volate, level_1_cat, level_2_cat)`
- Columns:
  - `value` — the component tariff in **GEL/kWh**
  - `company` — `telasi`, `epg`, `telmico`, `eps`, `gse`
  - `activity` — `distribution`, `universal`, `public`, `transmission`, `final_price`, `solr`
  - `volate` — voltage level, blank for transmission
  - `level_1_cat` / `level_2_cat` — consumer class and sub-class, blank for transmission
- Coverage: from 2021-07 onwards.

### Reporting rules

- **Report in GEL/kWh, as stored.** Do not convert tariffs up to GEL/MWh. The converted figure
  appears in no row of the view, so the grounding gate strips it and the answer ships
  truncated. When a comparison to wholesale prices is needed, convert the *price* down instead.
- **Quote `final_price` for the total.** A summed three-component total exists in no row. Use
  the components for the breakdown and `final_price` for the headline number; if a computed sum
  and the published total disagree, report the discrepancy rather than either figure.
- **`value` and `final_price` are net of VAT.** VAT is 18% and is levied on top, so a consumer
  pays `final_price × 1.18`. Report the net figure by default and say it is net of VAT; give
  the gross total only when the question asks what a consumer actually pays.

**Critical usage notes:**

- **Blank dimensions are empty strings (`''`), never NULL.** Filter with `= ''`; `IS NULL`
  matches nothing and returns an empty result.
- **`final_price` rows are the regulator's own pre-summed total.** Use them to cross-check a
  computed three-component sum, not to build one. If a computed sum and the `final_price` row
  disagree, the component resolution is wrong — report the discrepancy rather than presenting
  either number as correct.
- **`solr` rows are not part of the standard end-user price.** Exclude them.
- **Usable range is shorter than the data range.** Distribution tariffs are published years
  further ahead than the supply and transmission rows they must be combined with. Rows extend
  to 2030 while complete end-user prices only exist through the latest `final_price` month.
  Bound end-user price questions with
  `(SELECT MAX(date) FROM demand_tariff_mv WHERE activity = 'final_price')`, and never report
  `MAX(date)` on this view as "the latest end-user price".
- USD conversion: `value / xrate`, joining `price_with_usd` on `date`. Use a LEFT join — a
  month with no FX row should yield a null USD value, not drop the GEL value.

---

## Verification status

This document was drafted from general knowledge plus public sources and **needs review by the
maintainer**. The following points are the ones most worth confirming:

1. ~~**VAT treatment.**~~ **Settled 2026-08-15.** The view stores tariffs **net of VAT**; VAT
   of 18% is levied on top of the published `final_price`. An earlier draft of this file
   recorded the treatment as undeterminable — that was wrong. See "Reporting rules" above.
2. **Voltage-to-customer-type association.** Public GNERC material associates voltage levels
   with customer types, but the data contains household categories at both `220/380` and
   `3.3-6-10`. This document therefore reports voltages verbatim and asserts no mapping.
3. **Composition of the supply tariff.** The guaranteed capacity fee pass-through and wholesale
   procurement cost are stated here as the principal elements. The complete regulated cost
   stack, and the relative weight of each element, should be confirmed against GNERC's supply
   tariff methodology.
4. **`public` vs `universal` scope.** The mapping of `public` to public-service commercial
   customers and `universal` to households and small commercial customers follows the category
   table above; confirm the precise regulatory definitions.

Sources consulted: GNERC end-user tariff pages and tariff methodology documents, GNERC tariff
resolutions, and the CEER Georgian tariff model presentation. Items 1–4 above are **not**
settled by those sources.
