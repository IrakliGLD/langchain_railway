# Regulated end-user (retail) tariff rules

Applies whenever the data carries `final_price_net_gel_kwh`.

## What the final price is made of

The regulated end-user price is a stack of three separately regulated
components, all in **GEL/kWh**:

```
final price (net of VAT) = transmission + distribution + supply
```

- **Transmission** — charged by GSE (Georgian State Electrosystem), the
  transmission system operator. One national rate; it does not vary by
  voltage or customer class.
- **Distribution** — charged by the distribution company whose network serves
  the customer: **Telasi** in Tbilisi, **Energo-Pro Georgia (EPG)** elsewhere.
  Varies by voltage level and customer class.
- **Supply** — charged by the supply company: **Telmico** (Tbilisi Electricity
  Supply Company) on the Telasi network, **EP Georgia Supply (EPS)** on the
  EPG network. **Includes the guaranteed capacity fee.**

State this composition when the answer reports a final price. A reader seeing
one number cannot otherwise tell what it contains.

VAT of 18% is levied **on top** of this stack. `final_price_net_gel_kwh` is
net; quote `total_gross_gel_kwh` when it is present rather than computing a
gross figure yourself.

## Name companies in full, not by their database code

The data carries short codes. Write the company name the first time it appears
and the short form afterwards:

- `telmico` → **Telmico (Tbilisi Electricity Supply Company)**
- `eps` → **EPS (EP Georgia Supply)**
- `telasi` → **Telasi**, the Tbilisi distribution company
- `epg` → **Energo-Pro Georgia (EPG)**, the distribution company elsewhere

The `supply_company` column already holds the full name — quote it rather than
the `supplier` code. Writing "eps" at a reader who asked about companies is
quoting a database key back at them.

## Never mix companies or categories

There are eight customer categories across two company pairs — sixteen
distinct prices, and they differ. An average across them is a number no
customer pays.

- Report figures **per (supplier, category)**. Never average across suppliers,
  across categories, or across both.
- The statistics section is already grouped per series. Quote those values;
  do not pool them.
- When the answer covers several categories, say so explicitly and say which.

## Growth rates must name their basis

A CAGR or percentage change is meaningless without stating what it was
measured over. Any growth figure must name the supplier, the category and the
period it describes. Pre-computed per-series growth appears in the statistics
section — cite it rather than deriving your own, which will not match the data
and will be rejected as ungrounded.

## Comparing against the wholesale price

Wholesale prices are **GEL/MWh**; these tariffs are **GEL/kWh**. Divide the
wholesale figure by 1000 before any comparison, and add the guaranteed
capacity fee (`p_gcap_gel`) to the balancing price, because the retail supply
component already contains it. Comparing without both adjustments overstates
the retail margin.

## The eight categories

By voltage level, customer class and consumption band:

| Voltage | Class | Band / type | Supply activity |
|---|---|---|---|
| 220/380 V | Household | up to 101 kWh | universal |
| 220/380 V | Household | 101–301 kWh | universal |
| 220/380 V | Household | above 301 kWh | universal |
| 220/380 V | Commercial | small | universal |
| 220/380 V | Commercial | other | public |
| 3.3–6–10 kV | Household | — | universal |
| 3.3–6–10 kV | Commercial | other | public |
| 35–110 kV | Commercial | other | public |

## Close with a way to narrow

When an answer covers more than one category, end by telling the reader how to
get a sharper answer — naming two or three real options they can ask for, for
example "Telmico, 220/380 V, household 101–301 kWh" or "EPS, 3.3–6–10 kV,
commercial (public supply)". Do not ask them to pick from a vocabulary they
have not been shown.
