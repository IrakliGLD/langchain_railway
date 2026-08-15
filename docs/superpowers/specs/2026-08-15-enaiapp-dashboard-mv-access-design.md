# Give enaiapp access to the dashboard's Network & Supply and Plant Analytics materialized views

**Date:** 2026-08-15
**Status:** Implemented 2026-08-15 across four audited phases. Suite green (2,619 passed,
0 failures), `ruff check .` clean. **One outstanding DB action:** the `GRANT` in §4 for
`trade_by_ownership`, which is the only one of the six `enai_api_readonly` did not already hold.

## 1. Summary

The dashboard (separate repo, read-only reconnaissance copy at `D:\export_enai\_repo_sync`)
added two tabs — **Network & Supply** and **Plant Analytics** — that read six materialized
views enaiapp cannot see. This spec makes those six views first-class enaiapp views by the
same mechanism as the existing nine: allowlist, role grant, and the `context.py` registries.

No new access layer, no RPC, no new service.

## 2. The six views

Column lists and vocabularies below are **verified against the live database on 2026-08-15**,
not inferred. All six are materialized views in `public`, owned by `postgres`.

### 2.1 Plant Analytics

| View | Columns | Rows | Coverage |
| --- | --- | --- | --- |
| `trade_by_ownership` | `date, ownership, quantity` | 546 | 2020-01-01 → 2026-06-01 |
| `ownership_concentration` | `date, segment, total_generation, owner_count, hhi, top1_share, top3_share, top5_share` | 78 | 2020-01-01 → 2026-06-01 |
| `by_capacity` | `date, entity, segment, quantity, facility_count` | 624 | 2020-01-01 → 2026-06-01 |
| `by_commissioning` | `date, entity, segment, quantity` | 390 | 2020-01-01 → 2026-06-01 |
| `capacity_factor` | `date, technology, capacity_category, capacity_category_order, segment, facility_count, generation_mwh, installed_capacity_mw, hours_in_month, capacity_factor, capacity_factor_percent` | 882 | 2020-01-01 → 2026-06-01 |

### 2.2 Network & Supply

| View | Columns | Rows | Coverage |
| --- | --- | --- | --- |
| `demand_tariff_mv` | `demand_tariff_id, date, company, activity, volate, level_1_cat, level_2_cat, value` | 4,134 | 2021-07-01 → **2030-12-01** |

`demand_tariff_id` is deliberately **excluded** from `DB_SCHEMA_DICT`. It is null on every
calculated row, the dashboard never emits it, and `REQUIRED_SCHEMA_COLUMNS` checks
*required ⊆ reflected* — so omitting it is safe and stops the model joining on a null key.

### 2.3 Verified value vocabularies

Exact, case-sensitive, as stored:

- `by_capacity.entity` **and** `capacity_factor.capacity_category` (identical 8-band vocabulary):
  `<=5`, `6-10`, `11-20`, `21-50`, `51-100`, `101-200`, `201-500`, `more than 500` (MW)
- `by_commissioning.entity`: `<=1990`, `1991-2000`, `2001-2010`, `2011-2020`, `after 2020`
- `capacity_factor.technology`: `hpp`, `tpp`, `wpp`, `solar`
- `trade_by_ownership.ownership`: `energo-pro group`, `georgian water and power jcs`, `GIG`,
  `inter-rao`, `other`, `state`, `vartsikhe 2005 jsc` — **note `GIG` is uppercase** while
  every other value is lowercase
- `segment` (in `by_capacity`, `by_commissioning`, `capacity_factor`, `ownership_concentration`):
  `total` only
- `demand_tariff_mv.company`: `epg`, `eps`, `gse`, `telasi`, `telmico`
- `demand_tariff_mv.activity`: `distribution`, `final_price`, `public`, `solr`, `transmission`, `universal`
- `demand_tariff_mv.volate`: `` (empty), `220/380`, `3.3-6-10`, `35-110`
- `demand_tariff_mv.level_1_cat`: `` (empty), `com`, `hh`
- `demand_tariff_mv.level_2_cat`: `` (empty), `cat1`, `cat2`, `cat3`, `other`, `small`

Grain cross-checks that confirm the vocabularies are complete: 78 months × 8 bands = 624
(`by_capacity`); × 5 cohorts = 390 (`by_commissioning`); × 7 owners = 546
(`trade_by_ownership`); 1 row/month = 78 (`ownership_concentration`); 66 months × 16
(8 categories × 2 suppliers) = 1,056 `final_price` rows.

`capacity_factor` at 882 rows is **not** a full technology × band grid — the combinations are
sparse (there is no 500 MW solar). The model must not assume every pair exists.

## 3. Risks the live data confirmed, and two it falsified

### 3.1 Confirmed

**Unit collision.** Three scales now coexist: `demand_tariff_mv.value` is **GEL/kWh**;
`capacity_factor.generation_mwh` / `installed_capacity_mw` are **plain MWh**; every
pre-existing view is GEL/MWh and thousand MWh.

**Usable-range overhang.** `demand_tariff_mv` runs to 2030-12-01, but `final_price` rows stop
at 2026-12-01 — a four-year tail of distribution-only rows. `max(date)` is a trap; the usable
horizon is `where activity = 'final_price'`.

**Blank dimensions are empty strings, not NULL.** Transmission rows carry `volate = ''`,
`level_1_cat = ''`, `level_2_cat = ''`. `IS NULL` matches nothing.

**Two scales for one quantity.** `capacity_factor` (ratio) and `capacity_factor_percent`
coexist. Never multiply the percent column by 100.

**Mixed casing in `trade_by_ownership.ownership`.** `GIG` against six lowercase values.

### 3.2 Falsified — do not encode these

**`segment` double-counting.** Predicted `total` alongside `balancing`; the live value set is
`total` only. The dashboard's filter is defensive, not load-bearing. Documented as "currently
constant", not as a mandatory filter — a rule to filter on `total` would silently break if a
second segment is ever added.

**`demand_tariff_mv` normalization asymmetry.** The design spec's `lower(btrim(...))` implied
raw casing/whitespace variants. The live dump is uniformly clean. Dropped as a rule.

## 4. Access

Verified 2026-08-15: `enai_api_readonly` already holds `SELECT` on five of the six.
**`trade_by_ownership` is the exception** and needs:

```sql
GRANT SELECT ON public.trade_by_ownership TO enai_api_readonly;
```

The grant script edit is mandatory regardless of live DB state, for two reasons:

1. `scripts/least_privilege_api_role.sql` opens with
   `REVOKE ALL ON ALL TABLES IN SCHEMA public FROM enai_api_readonly` and then grants an
   explicit list. It is a *convergence* script — its next run would strip the five existing
   grants.
2. `tests/test_config.py::test_runtime_role_script_matches_allowed_tables` and
   `tests/test_p7_runtime_hardening.py` assert the script's granted set equals
   `STATIC_ALLOWED_TABLES` byte-for-byte.

Adding the six to the allowlist without the script edit turns the suite red.

**Readiness will not catch a missing grant.** Schema reflection reads `pg_matviews` /
`pg_attribute`, which ignores privileges, so `/readyz` stays green while queries fail with
permission-denied. Grants land before code.

## 5. The scrubber conflict, and its general fix

`test_context.py::test_schema_dict_columns_have_labels` requires every column in
`DB_SCHEMA_DICT` to appear in `COLUMN_LABELS`. But `COLUMN_LABELS` also drives
`scrub_schema_mentions`, a case-insensitive `\b{key}\b` substitution applied to **narrative
answer text**.

Four of the new columns are ordinary English words: **`value`**, **`activity`**, **`company`**,
**`technology`**. Labelling them would mangle prose — "the value of imports rose" becomes
"the Tariff Component Value (GEL/kWh) of imports rose".

This is the identical failure already documented for `VALUE_LABELS` at `context.py:148-166`,
where bare common-English keys were deliberately excluded after production incidents
(`hydro` → "Hydro Generation generation"; `balancing` → "the Balancing Electricity price").
`COLUMN_LABELS` simply had not hit the case yet.

**Fix:** split label lookup from scrub eligibility. Keep the coverage test — every column still
gets a label, which readiness and display need — but exclude common-English keys from the
scrubber via a documented exemption set carrying the same rationale. This is the root-cause
fix; special-casing `value` alone would leave the trap armed for the next view added.

## 6. Knowledge

`tariffs.md` (184 lines) is entirely generation-side: GNERC cost-plus for plants, hydro and
thermal tariff entities, and a Data Mapping section pointing at `tariff_with_usd`. A grep of
all 14 knowledge files found **zero** coverage of distribution or transmission network tariffs
and **zero** coverage of the final end-user price. The `end-user` / `retail` hits that exist
are about wholesale *eligibility*, not price.

The guaranteed capacity fee is half-covered: `tariffs.md:7,45,48` defines it as a fixed GEL/day
payment *to* thermal plants and notes end-consumers pay it proportionally. The retail-side
mechanism — that it reaches consumers through the **supply component** of the end-user tariff —
is stated nowhere.

### 6.1 One new topic: `network_supply_tariffs`

Not two. Network tariffs and the end-user price are the same subject from two angles: the
end-user price *is* the sum of the network and supply components. Splitting them would let
retrieval pick one and answer half the question.

Contents: the three-component structure (distribution via `telasi`/`epg`; supply via
`universal`/`public` from `telmico`/`eps`; transmission via `gse`); the supplier→distributor
pairing; `solr` as supplier-of-last-resort and why the dashboard excludes it; what the supply
tariff contains, **including the guaranteed capacity fee pass-through with an explicit
cross-link to `tariffs.md`**; the 8 consumer categories with voltages verbatim and household
bands (cat1 ≤101 kWh, cat2 101–301, cat3 >301); the GEL/kWh unit warning; and the data mapping
to `demand_tariff_mv` with the `final_price` cross-check and usable-range rule.

Drafted from general knowledge plus web research, with online-sourced claims marked for the
maintainer's correction.

### 6.2 The routing trap

`TOPIC_MAP` currently maps `"tariff" → ["tariffs"]`. Every end-user tariff question would hit
that keyword and route to the generation-side file — the new topic would exist and never be
retrieved. `"tariff"` must fan out to both files, and both catalog entries need `use_for` text
that disambiguates.

Precedent to copy verbatim: `cross_border_trade` ↔ `cross_border_capacity`
(`question_analysis_catalogs.py:139-152`), two adjacent topics whose entries each name the
other and state when to prefer it.

### 6.3 Plant Analytics gets no topic file

The capacity bands and commissioning cohorts are self-describing. Two short rules go into
`generation_mix.md` instead: the `segment` note and the `capacity_factor` vs
`capacity_factor_percent` trap.

### 6.4 No re-embedding

`load_knowledge()` globs `knowledge/*.md` into memory at startup. Vector ingestion reads
`docs_to_ingest/` and is a separate path. A new topic file needs no embedding run.

## 7. Registries touched

| Layer | File |
| --- | --- |
| SQL allowlist | `config.py` `STATIC_ALLOWED_TABLES` |
| DB grant | `scripts/least_privilege_api_role.sql` |
| Structured schema | `context.py` `DB_SCHEMA_DICT` (drives readiness) |
| Prompt schema text | `context.py` `DB_SCHEMA_DOC` |
| Labels | `context.py` `VIEW_LABELS`, `COLUMN_LABELS` + scrub exemption |
| Joins | `context.py` `DB_JOINS` |
| Evidence topics | `utils/query_validation.py` `extract_sql_topics` |
| Knowledge topic enum | `contracts/question_analysis.py` `KnowledgeTopicName` |
| Topic catalog | `contracts/question_analysis_catalogs.py` |
| Knowledge files | `knowledge/network_supply_tariffs.md`, `knowledge/tariffs.md`, `knowledge/generation_mix.md` |
| Keyword routing | `knowledge/__init__.py` `TOPIC_MAP` |
| Schema snapshot | `schemas/question_analysis.schema.json` |
| Analyzer skill mirror | `skills/question-analyzer/references/topic-catalog.md` |
| Retrieval tier | `agent/pipeline.py` `_MARKET_STRUCTURE_TOPICS` |
| Few-shot SQL | `knowledge/sql_example_selector.py` |

## 8. Phases

Following `skills/developer-phased-audit` — each phase is planned, implemented, verified
against the full targeted suite, audited adversarially, and fixed before the next begins.

1. **Contract & schema** — allowlist, grant script, `DB_SCHEMA_DICT`, labels, scrubber
   exemption, joins.
2. **Prompt schema doc** — signatures, vocabularies, and the rules in §3.1.
3. **Knowledge topic** — `network_supply_tariffs` across its seven registries, plus the
   `generation_mix.md` additions.
4. **Few-shot SQL & topic routing** — `end_user_price` and `plant_fleet` example categories,
   `extract_sql_topics` table map.

**Baseline before any change:** 2,583 passed, 2 pre-existing failures in `test_guardrails.py`
(`test_cross_check_warns_chart_with_no_date_params`,
`test_cross_check_warns_trend_goal_with_no_date_params`). Unrelated to this work; held as
known-red.

## 9. Out of scope

New vector-knowledge documents under `docs_to_ingest/`. The remaining MVs the dashboard can
see and enaiapp cannot: `trade_by_source`, `trade_by_type`, `tech_quantity_pivot`,
`support_scheme_changes`, `cross_border_mv`, `hourlyfacts`.
