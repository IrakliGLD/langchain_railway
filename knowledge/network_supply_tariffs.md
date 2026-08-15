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

| Component | Code | Full name | Role | Territory |
| ------------- | --------- | --------- | ---- | --------- |
| Transmission | `gse` | Georgian State Electrosystem | TSO, transmission network | national |
| Distribution | `telasi` | Telasi | distribution network | Tbilisi |
| Supply | `telmico` | Tbilisi Electricity Supply Company | supply service | Tbilisi |
| Distribution | `epg` | Energo-Pro Georgia | distribution network | outside Tbilisi, plus some Tbilisi suburbs |
| Supply | `eps` | EP Georgia Supply | supply service | outside Tbilisi, plus some Tbilisi suburbs |

The distribution company and the supplier on the same network are **different legal
entities**. Never use one in place of the other when assembling a price.

### Supplier to distributor pairing

Each supplier operates in the service territory of one distribution company, so the two are
always paired:

| Supplier | Distribution company |
| ---------- | -------------------- |
| `telmico` | `telasi` |
| `eps` | `epg` |

Telasi and Telmico operate in Tbilisi, the capital. Energo-Pro Georgia and EP Georgia Supply
operate across the rest of the country **and also cover some suburbs of Tbilisi** — so knowing
a customer is "in Tbilisi" does not by itself determine which supplier serves them. If a
question names only the city, say which pair is the usual one and note the suburb exception
rather than asserting a single supplier.

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

**Never mix categories.** A final end-user price is assembled from three components that all
belong to the *same* `(supplier, volate, level_1_cat, level_2_cat)` category, plus the single
national transmission row. Taking the distribution component from one category and the supply
component from another produces a number corresponding to no real tariff. Each of the 16
published prices is self-contained: resolve one category completely, or report that it cannot
be resolved — never assemble a price from parts of two.

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

## Comparing to the wholesale price

The regulated supply tariff already bundles the guaranteed capacity fee, so a bare balancing
price is **not** comparable to an end-user price. The capacity charge is **added to the
wholesale side** rather than subtracted from the tariff — that way the regulated figure stays
equal to what is actually charged and the adjustment sits on one series.

Benchmark, per month, from `public.price_with_usd`:

```
(p_bal_gel + p_gcap_gel) / 1000     -- GEL/kWh
```

Both prices are published in GEL/MWh, so divide by 1000 to reach the tariff's unit. Never
multiply the tariff by 1000 instead: that figure appears in no row and the grounding gate will
strip it from the answer.

Compare against the **net** `final_price`, not the VAT-inclusive figure — the wholesale price
is itself net of VAT, so comparing gross to net overstates the spread by 18%.

The spread between the two is the combined distribution, transmission and retail-supply margin
plus any regulated cost not present in the wholesale price. It is not a profit measure.

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
3. **Composition of the supply tariff — partly settled.** That the supply tariff *contains* the
   guaranteed capacity fee is established: it is precisely why the fee is added to the wholesale
   side when the two are compared. Wholesale procurement cost is likewise certain. What is *not*
   established is the **complete** list of regulated elements and their relative weights.
   Answers should therefore say "principally procurement cost and the guaranteed capacity fee
   pass-through" and must not present that as an exhaustive breakdown, or quote a share for any
   element. Closing this needs GNERC's supply-tariff methodology.
4. **`public` vs `universal` — operationally settled.** Which activity to query is fixed by the
   category table and needs no further confirmation: `com|other` uses `public`; `com|small` and
   every `hh` category use `universal`. GNERC's own tariff pages list exactly these two provider
   types. What remains open is only the regulatory **eligibility rule** — which customers
   qualify as public-service rather than universal-service. That affects how the categories are
   *described*, never which rows are selected.

Sources consulted: GNERC end-user tariff pages and tariff methodology documents, GNERC tariff
resolutions, and the CEER Georgian tariff model presentation. Item 1 is settled; items 3 and 4
are partly settled as described; item 2 is open.
