# P2 execution ledger — 2026-07-13

**Status:** P2.A backend locally complete and independently verified; P2.B frontend pending

**Release state:** The backend analytical-truth prerequisites are ready for an independent commit/deploy. Do not enable H1 canonical-frame enforcement. Overall P2 closes only after the frontend consumes the same versioned unit contract and passes its consumer gates.

## Repository state

| Track | Repository | Branch | State |
|---|---|---|---|
| P2.A backend | `D:/Enaiapp/langchain_railway` | `refactor/review-phase-fixes` | Locally complete; this ledger ships in the P2.A implementation commit |
| P2.B frontend | `D:/Enaiapp/p0_frontend_commit_repo` | `fix/p0-remediation` | Pending |

## P2.A delivered scope

| Item | Result | Primary artifacts |
|---|---|---|
| P2.1 metric/unit registry | Versioned registry/schema/golden corpus with concrete source aliases/patterns for P2.B consumers; canonical adapters convert price, energy quantity, and ratios before filters; tariffs remain per-MWh | `contracts/metric_units_v1.json`, `contracts/metric_units_v1.schema.json`, `agent/metric_units.py`, `evaluation/metric_unit_golden_v1.json` |
| P2.2 chronological statistics | Parsed/sorted/deduplicated periods, unique-period completeness, elapsed-year CAGR, chronological recent windows, per-metric quick stats, stable equal-value trend | `analysis/stats.py`, `analysis/seasonal_stats.py`, `agent/metric_registry.py` |
| P2.3 provenance | Removed dead `ctx.provenance`; query and exact-source hashes flow through tool, merged, scenario, forecast, answer, and chart surfaces | `agent/provenance.py`, `agent/pipeline.py`, `agent/evidence_planner.py`, `agent/summarizer.py`, `agent/chart_pipeline.py` |
| P2.4 scope/horizon/cache | Correlation fallback scoped to requested period with pairwise N/period/uncertainty; structured horizon authoritative; vector caches fully versioned | `agent/analyzer.py`, `utils/forecasting.py`, `knowledge/vector_embeddings.py`, `knowledge/vector_retrieval.py` |

## Intentional numeric/behavior changes

| Input/storage value | Previous canonical behavior | P2.A behavior | Reason |
|---|---|---|---|
| `150 GEL/MWh` price | Relabeled as `150 tetri/kWh` | `15 tetri/kWh` | Correct energy/currency conversion |
| `100 USD/MWh` price | Relabeled as `100 cents/kWh` | `10 USD cents/kWh` | Correct per-MWh to per-kWh conversion |
| `1.5 thousand MWh` | Relabeled as `1.5 MWh` | `1500 MWh` | Correct storage scale |
| `0.10` ratio/share | Relabeled as `0.10%` | `10%` | Correct ratio-to-percent conversion |
| `150 GEL/MWh` tariff | `150 GEL/MWh` | unchanged | Tariff contract is already correct |
| Equal endpoints | `decreasing (0.0%)` | `stable (0.0%)` | Correct classification |
| CAGR across 2020 and 2024 only | One observed interval | Four elapsed calendar years, missing years disclosed | Honest elapsed-time exponent |
| Correlation fallback | Could use all available history | Structured requested period only; pairwise sample disclosed | Contract scope and uncertainty |

## Verification evidence

- P2 focused consolidated gate: **74 passed**.
- Compatibility rerun after independent-audit fixes: **119 passed**.
- Full backend suite with workspace-local pytest temp root: **1,385 passed**, 3 dependency/cache warnings, 0 failures.
- Python compilation passed for `agent`, `analysis`, `contracts`, `knowledge`, `utils`, `main.py`, and `models.py`.
- `git diff --check` passed; line-ending notices are repository/platform normalization warnings.
- No live database was required. Per operator instruction, live-data/deployment verification is deferred.

## Manual deployment and activation instructions

1. Deploy the P2.A backend commit (or a reviewed descendant) through the normal Railway/backend pipeline.
2. Keep canonical evidence enforcement/H1 in its current off or non-authoritative state. P2.A corrects its prerequisites but does not authorize the P4 cutover.
3. Add explicit backend environment versions:
   - `VECTOR_KNOWLEDGE_NORMALIZATION_VERSION=v1`
   - `VECTOR_KNOWLEDGE_CORPUS_VERSION=v1`
   The code defaults both to `v1`, so omission does not break startup; explicit values make cache and ingestion identity auditable.
4. Whenever either version, embedding provider, exact model, or dimension changes:
   - update the environment value;
   - re-ingest/rebuild the vector corpus using that same configuration;
   - deploy the backend;
   - run a known-query retrieval smoke test before production traffic.
5. Do not copy files to Supabase for P2.A. This track changes only the backend repository.

## Complementary work still required

- P2.B must consume registry version `1.0.0` for UI labels, axes, tables, exports, filters, precision, and validation. It must not create an independent conversion table.
- Frontend deployment must reject or safely fall back on an unsupported registry version.
- P4 entry review must confirm both P2 tracks, production shadow comparisons, and rollback readiness before enabling H1.
- If older clients assume raw price numbers while displaying tetri/cents labels, deploy P2.B compatibility handling before making canonical-frame output authoritative.
