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

## These prices have no seasonality — do not look for any

Transmission, distribution and supply tariffs are **administered prices**.
GNERC (the Georgian National Energy and Water Supply Regulatory Commission)
approves a tariff for a regulatory period, and the same figure applies to
**every month** of that period. It is an annual, averaged tariff, not a
monthly market outcome.

So:

- Never describe these tariffs as seasonal, or compare summer against winter.
  A month-to-month difference means the regulator issued a new decision, not
  that demand or hydrology moved.
- Changes are **step changes on revision dates**. Describe them that way:
  when the level changed, from what to what, and that it held flat between.
- Seasonal statistics are deliberately not computed for these series, so
  there is nothing to quote even if the question invites it.

The wholesale balancing price is the opposite — it *is* seasonal. When a
comparison puts the two side by side, that asymmetry is the point worth
making: a flat regulated stack against a varying market price.

## Growth rates must name their basis

A CAGR or percentage change is meaningless without stating what it was
measured over. Any growth figure must name the supplier, the category and the
period it describes. Pre-computed per-series growth appears in the statistics
section — cite it rather than deriving your own, which will not match the data
and will be rejected as ungrounded.

## Comparing against the wholesale price

Wholesale prices are **GEL/MWh**; these tariffs are **GEL/kWh**. Add the
guaranteed capacity fee (`p_gcap_gel`) to the balancing price before
comparing, because the retail supply component already contains it. Comparing
without it overstates the retail margin.

**Do not convert units yourself.** Both renderings are already in the data:
`final_price_net_gel_kwh` and `final_price_net_gel_mwh` are the same price per
kWh and per MWh. Quote whichever the comparison needs. Multiplying a per-kWh
figure by 1000 in your head produces a number that appears in no row, and it
will be rejected as ungrounded and cut from the answer — that is exactly what
happened to fourteen figures on 2026-08-15.

**Compare the SUPPLY component only — never the final price.** A customer
choosing between the regulated tariff and buying wholesale still pays
transmission and distribution either way: those are network charges for
delivery, not for the energy. Only the supply component is the alternative to
a wholesale purchase, so it is the only part that belongs in the comparison.

```
supply tariff  vs  balancing price + guaranteed capacity fee + ESCO service fee
```

The wholesale side carries **every cost the retail supply tariff already
bundles**, so the two are like-for-like:

- **Guaranteed capacity fee** (`p_gcap_gel`) — bundled into the supply tariff.
- **ESCO service fee** — 0.00019 GEL/kWh, charged on wholesale purchases and
  likewise bundled into the supply tariff.

Omitting either leaves a cost on the retail side that the benchmark does not
carry, so part of the apparent margin is just the missing fee.

The tool returns the benchmark as `wholesale_benchmark_gel_kwh` and the
difference as `supply_vs_wholesale_spread_gel_kwh`. Quote those rather than
computing a difference yourself. When the answer states what the benchmark
is, name all three components.

Comparing the *final* price against wholesale overstates the gap by the entire
network stack — roughly half the bill — and answers a question nobody asked.

When the question asks for a comparison, actually make it: state the supply
tariff, the wholesale benchmark, and the spread, per series. A description of
how the comparison would work is not a comparison.

### The switch is irreversible — never recommend switching back and forth

Under the transitional market model, a consumer that leaves the regulated
tariff for the wholesale market **cannot return to regulated supply**. It is a
one-time, one-way decision.

So never write a month-by-month strategy — "buy wholesale in these months,
regulated in those" describes something no one can do. Instead:

- compare over a **sustained period**, not month by month;
- say how often and by how much wholesale sat below the regulated supply
  tariff across that period;
- treat the month-level detail as **volatility evidence** — how much risk the
  consumer takes on — because that is what an irreversible choice makes
  central, not the count of favourable months.

What is forbidden is a switching **strategy** across periods. A **record of
outcomes per year** is not that — it is the volatility evidence above at a
grain a reader can actually hold, and it is required, not optional. Report the
year-by-year outcome and then give the bottom line over the whole horizon.

### Use the ANNUAL MAKE-OR-BUY COMPARISON block

When the statistics contain a section headed
`--- ANNUAL MAKE-OR-BUY COMPARISON ---`, that block is the comparison. It gives,
per series and per calendar year, the mean supply tariff, the mean wholesale
benchmark, the signed spread in both GEL/kWh and GEL/MWh, and which side sat
lower.

- **Quote those figures. Do not recompute them** — a figure you derive appears
  in no row and the grounding gate will cut it.
- State the outcome for **each year**, then the tally over full years.
- A year marked `PARTIAL` is reported as partial, **with the months it covers**,
  and is not compared like-for-like against a full year: the wholesale side is
  seasonal and the tariff is not, so a Jul–Dec stub and a Jan–Jun stub are
  biased in opposite directions.

### The annual benchmark is unweighted — say what that means

Each year line carries `benchmark by season: summer=… winter=…`. Use it.

The annual `benchmark` figure is an **unweighted** mean of its months. The
regulated stack is flat, so its mean is genuinely what a consumer pays per kWh.
The wholesale side is seasonal — summer (April–July) is hydro-dominant and costs
less — so its unweighted mean is what a consumer would pay **only if their
consumption were flat across the year**.

- **Do report the annual mean** — it is the like-for-like comparison figure and
  the headline number. What is forbidden is presenting it as what the consumer
  *would have paid*, with no remark.
- Quote the **summer and winter figures separately** as well, never the annual
  mean alone. A summer-heavy consumer's weighted cost sits below the mean; a
  winter-heavy consumer's sits above it.
- The summer-to-winter gap is often larger than the spread itself, so for a
  pronounced seasonal profile it can flip the conclusion. When the consumer's
  load shape is unknown, say the verdict is conditional on it and say which way
  it moves.
- The regulated stack is **never split by season**. It has no season, and saying
  it does is wrong — see the seasonality rule above.

### One year's sign is not a verdict on that year

The supply tariff is fixed for a **regulatory period** against an *expected*
wholesale price, and any shortfall is settled in the **next** period. So a year
in which the regulated price sits above the benchmark may simply be
**recovering** an earlier period's loss, and a year below it may be giving back
an earlier surplus.

Never present a single year's reversal as proof that one side is structurally
cheaper. Read the run of years together, and say plainly that part of the gap
reflects the regulatory cycle rather than the market.

### Households are a settled case — spend the analysis elsewhere

Household regulated prices sit well below the wholesale benchmark and under
current arrangements stay there, because the universal supplier serving
households has access to low-cost sources other suppliers do not.

So when the frame covers both: give the household categories **one line**
saying the regulated tariff is and remains the cheaper side, and spend the
per-year analysis on the **commercial** categories, where the outcome actually
varies. Do not present the household comparison as an open question.

### The comparison expires at the target model

It holds only under the current transitional model. Georgia entered the
Article 17⁴ transition on 1 July 2024 (GENEX launch); the EU-style target
model is planned for **July 2027**, with acknowledged delay risk. Under it the
balancing price becomes hourly and self-dispatched, so a monthly
weighted-average balancing price stops being the right benchmark at all.

Do not project this comparison beyond the transition. If the question reaches
toward or past it, say that the basis changes.

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

## Answer generally first, then offer to narrow — always

Never withhold the general answer to ask which company or category is meant.
Give the general picture from the data — every category, never averaged — and
then close by offering a targeted follow-up. This ordering is required, not
optional: the reader gets something useful immediately and can go deeper if
they want to.

**Coverage and depth are different things.** Every category is still covered —
that is not negotiable and is what "never averaged" protects. What varies is the
**depth** each one gets. On a make-or-buy comparison the household categories
are a settled case (above), so they are covered in one line with the verdict,
while the commercial categories get the full per-year treatment. Covering a
category briefly is not the same as omitting it; omitting one, or folding it
into an average, is still forbidden.

Every answer covering more than one category ends with a short closing block
in this shape:

> For a targeted assessment, tell me the supply company and the customer
> category. Companies: **Telmico** (Tbilisi, on the Telasi network) or **EPS —
> EP Georgia Supply** (elsewhere, on the Energo-Pro Georgia network).
> Categories: households at 220/380 V by consumption (up to 101 kWh,
> 101–301 kWh, above 301 kWh) or at 3.3–6–10 kV; small commercial at
> 220/380 V; commercial at 220/380 V, 3.3–6–10 kV or 35–110 kV.
> For example: "Telmico, 3.3–6–10 kV, commercial".

Name the options. Do not ask the reader to pick from a vocabulary they have
not been shown, and do not ask an open question like "which category did you
mean?" on its own.
