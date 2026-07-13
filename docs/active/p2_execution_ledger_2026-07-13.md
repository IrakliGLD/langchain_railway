# P2 execution ledger — 2026-07-13

**Status:** P2.A backend and P2.B frontend locally complete, independently verified, and independently committed

**Release state:** The local P2 implementation gate is closed. Both applications remain independently deployable and communicate only through the deployed HTTP/API response contract. Do not enable H1 canonical-frame enforcement; production smoke/shadow evidence and the P4 entry review remain required.

## Repository state

| Track | Repository | Branch | State |
|---|---|---|---|
| P2.A backend | `D:/Enaiapp/langchain_railway` | `refactor/review-phase-fixes` | Complete: `19a369b`, `37129e1`, `bcdb56a` |
| P2.B frontend | frontend repository (`EnaiDashboard.git`) | `fix/p0-remediation` | Complete: `fa20c98` |

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

## P2.B delivered scope and verification

- Vendored registry version `1.0.0` is consumed by dashboard labels, builder axes/tariffs, chat-chart axes/tooltips, precision, compatibility validation, and filter conversion helpers.
- The frontend build verifies the canonical hash and its browser-compatible generated artifact entirely inside the frontend repository. It never reads a backend checkout or filesystem path.
- Historical responses without a registry version remain compatible while H1 is off. Responses declaring an unsupported version keep the answer text but suppress the chart with an explicit warning.
- Frontend contract gate passed; ESLint passed; production Vite build passed; **342 tests passed**, 0 failed.
- A repository-path scan found no `langchain_railway`, `D:/Enaiapp`, `D:/export_enai`, or sibling-backend references in frontend runtime/build/test inputs.

## Manual deployment and activation instructions

1. Deploy backend commit `bcdb56a` (or a reviewed descendant containing `19a369b` and `37129e1`) through the normal Railway/backend pipeline.
2. Deploy frontend commit `fa20c98` (or a reviewed descendant) through the frontend's own pipeline. Do not copy files between repositories or make either deployment mount/read the other checkout.
3. The two P2 deployments are order-tolerant: the new frontend accepts historical unversioned responses and registry `1.0.0`; an unknown declared version fails closed for charts while preserving answer text. Deploying backend first gives the clearest smoke-test evidence.
4. Keep canonical evidence enforcement/H1 in its current off or non-authoritative state. P2 corrects its prerequisites but does not authorize the P4 cutover.
5. Add explicit backend environment versions:
   - `VECTOR_KNOWLEDGE_NORMALIZATION_VERSION=v1`
   - `VECTOR_KNOWLEDGE_CORPUS_VERSION=v1`
   The code defaults both to `v1`, so omission does not break startup; explicit values make cache and ingestion identity auditable.
6. Whenever either version, embedding provider, exact model, or dimension changes:
   - update the environment value;
   - re-ingest/rebuild the vector corpus using that same configuration;
   - deploy the backend;
   - run a known-query retrieval smoke test before production traffic.
7. Run a production chat smoke query that returns one price chart and one quantity/share chart. Confirm the response declares `chart_metadata.metric_unit_registry_version = "1.0.0"`, answer text remains visible, axes/tooltips show the declared units, and the browser console has no contract warning.
8. No Supabase database migration, edge-function copy/deploy, new frontend environment variable, or direct repository-file integration is required for P2.

## Complementary work still required

- P4 entry review must confirm both P2 tracks, production shadow comparisons, and rollback readiness before enabling H1.
- Production smoke/shadow comparisons must confirm intentional unit changes and historical-client behavior before making canonical-frame output authoritative.
