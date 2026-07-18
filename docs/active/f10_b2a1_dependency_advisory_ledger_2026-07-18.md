# F10 B2.A.1 — Dependency Advisory and Reachability Ledger — 2026-07-18

Closes the inventory step of `F10-SEC-01` in
[`f10_blocker_remediation_plan_2026-07-18.md`](./f10_blocker_remediation_plan_2026-07-18.md).
This phase changes no production code or pins; it records what is vulnerable,
what is reachable, and the exact remediation sequence for B2.A.2–B2.A.6.

## 1. Method

| Step | How it was produced |
|---|---|
| Production closure | `pip install --dry-run --report` of `requirements.txt` resolved for **cp311 / manylinux2014, wheels-only** — the container interpreter (`python:3.11.15-slim-bookworm`). 101 packages. [`resolved-cp311-freeze.txt`](../evidence/f10_b2/resolved-cp311-freeze.txt) |
| Forward/reverse graph | Built from the resolution report's `requires_dist` metadata, restricted to the resolved set. [`dependency-graph-cp311.json`](../evidence/f10_b2/dependency-graph-cp311.json) |
| Advisory query | OSV.dev `querybatch` + per-vulnerability details for all 101 exact pins. [`osv-advisories-pinned-closure.json`](../evidence/f10_b2/osv-advisories-pinned-closure.json) |
| Reachability | ripgrep of production imports (tests excluded), with lazy/guarded imports read in context. |
| Local-env comparison | `pip freeze` of the workstation interpreter that produced the recent 1,685-test green runs, OSV-audited the same way. [`local-env-divergence.md`](../evidence/f10_b2/local-env-divergence.md) |

**Scope:** `requirements.txt` only — the production Docker image installs
nothing from `requirements-dev.txt` (pytest/ruff/pip-audit/cyclonedx-bom are
CI-only tools), so dev pins are outside F10-SEC-01. The release-evidence
workflow already consumes `pip-audit==2.10.1`/`cyclonedx-bom==7.3.0` from the
dev file; B2.A.6 builds its gate on those existing hooks.

**Deviation from the plan text:** `pip-audit -r` could not run on this
workstation because its resolver executes under the local Python 3.14, which
cannot resolve the cp311-only pins (`numpy==1.26.4`, `pandas==2.1.4`, …).
OSV.dev is the same advisory database pip-audit queries by default; the exact
pinned closure was audited record-for-record. The authoritative `pip-audit`
JSON + SBOM for the B2 candidate SHA is still produced by the protected
**Backend release evidence** workflow (Python 3.11) and must be attached at
B3 promotion. `Manual verification pending` applies only to that workflow
artifact, not to this inventory.

## 2. Headline findings

1. **94 OSV advisory records (~34 unique CVEs) across 11 packages** in the
   pinned production closure. (The final F10 audit's pip-audit count of 59 is
   the same set; OSV lists GHSA and PYSEC aliases as separate records.)
2. **The local test environment is not the production pin set.** 76 of 101
   closure packages differ (or are absent) locally — e.g. fastapi 0.135.1 vs
   pinned 0.109.2, langchain-core 1.2.17 vs 0.1.52, PyJWT 2.10.1 vs 2.9.0 —
   and `litellm`, `google-generativeai`, `statsmodels` are not installed
   locally at all while the full suite passes. Local green runs therefore
   already characterize most of the modern target stack, but they are
   **diagnostic, not release evidence**: the pinned cp311 set is what deploys,
   and the local set carries its own residual advisories
   (starlette 0.52.1, PyJWT 2.10.1, protobuf 6.33.1, langchain-core 1.2.17).
3. **42 of 94 records are closed by deleting pins that nothing imports**
   (B2.A.2), with zero runtime behavior change.
4. **No waiver is required.** Every advisory either has a fixed version
   reachable through the target stack or exits the closure entirely. The only
   no-fix record, `PYSEC-2025-183` (CVE-2025-45768, PyJWT "weak encryption"),
   is disputed by the maintainer and marks `last_affected: 2.10.1` — the
   B2.A.3 target 2.13.0 is outside its range.

## 3. Direct-pin inventory (requirements.txt, 22 pins)

| Pin | Production import | Advisories | Action (phase) |
|---|---|---:|---|
| sqlalchemy==2.0.44 | core/db.py + tools | 0 | keep |
| psycopg[binary]==3.2.1 | via SQLAlchemy driver coercion | 0 | keep |
| fastapi==0.109.2 | main.py | 0 (starlette carries them) | upgrade → 0.139.2 (B2.A.4) |
| uvicorn==0.27.1 | process entrypoint | 0 | keep (optional refresh later) |
| slowapi==0.1.9 | `main.py:48` (`get_remote_address` only) | 0 | keep |
| langchain==0.1.20 | **none** | 9 | **remove (B2.A.2)** |
| langchain-core==0.1.52 | none direct (transitive of wrappers) | 14 | upgrade via wrappers → ≥1.4.9 (B2.A.5) |
| langchain-community==0.0.38 | **none** | 8 | **remove (B2.A.2)** |
| langchain-google-genai==0.0.11 | `core/llm.py:29`, `core/llm_runtime.py:20` | 0 | upgrade → 4.2.7 (B2.A.5) |
| langchain-openai==0.1.1 | `core/llm*.py`, `knowledge/vector_embeddings.py:123` | 2 | upgrade → 1.3.5 (B2.A.5) |
| pydantic==2.9.2 | models/contracts everywhere | 0 | keep; raise only if resolver requires (B2.A.4/5) |
| pandas==2.1.4 | pipeline/tools | 0 | keep (cp311-bound pin) |
| numpy==1.26.4 | stats/tools | 0 | keep (cp311-bound pin) |
| statsmodels==0.14.1 | **none** | 0 | **remove (B2.A.2)** |
| scipy==1.11.4 | `visualization/chart_builder.py:17` | 0 | keep |
| PyJWT==2.9.0 | `utils/auth.py:12` | 13 | upgrade → 2.13.0 (B2.A.3) |
| tenacity==8.2.3 | **none direct** (transitive: google-genai, langchain-core) | 0 | **drop direct pin (B2.A.2)**; stays transitively |
| python-dotenv==1.0.1 | `main.py:40`, `config.py:11` | 2 | upgrade → 1.2.2 (B2.A.3) |
| google-generativeai==0.4.1 | guarded legacy fallback only (`knowledge/vector_embeddings.py:197`, inside try/except) | 0 (but pins protobuf<5 chain) | **remove (B2.A.5)** — held until then by `langchain-google-genai==0.0.11` (`requires google-generativeai>=0.4.1,<0.5`) |
| google-genai==1.65.0 | `knowledge/vector_embeddings.py:193` (lazy) | 0 | keep (compatible with wrapper 4.2.7: `>=1.65.0,<3`) |
| litellm==1.44.10 | **none** | 21 | **remove (B2.A.2)** |
| sqlglot==25.25.0 | `core/sql_generator.py:16` | 0 | keep |

## 4. Advisory-bearing packages (all 94 records accounted for)

| Package (pinned) | Records | Worst severity | Reachable today? | Full-fix version | Closed by |
|---|---:|---|---|---|---|
| litellm==1.44.10 | 21 | CRITICAL ×2 | No — zero imports | n/a | B2.A.2 removal |
| langchain==0.1.20 | 9 | HIGH | No — zero imports | n/a | B2.A.2 removal |
| langchain-community==0.0.38 | 8 | HIGH | No — zero imports | n/a | B2.A.2 removal |
| langchain-text-splitters==0.0.2 | 4 | HIGH | No — transitive of umbrella only | n/a | B2.A.2 (falls out with `langchain`) |
| PyJWT==2.9.0 | 13 | HIGH | **Yes** — `utils/auth.py` verifies gateway/session tokens | 2.13.0 (disputed no-fix record ends at 2.10.1) | B2.A.3 |
| python-dotenv==1.0.1 | 2 | MODERATE | Yes — startup env loading | 1.2.2 | B2.A.3 |
| starlette==0.36.3 | 14 | HIGH | **Yes** — network-facing request handling under FastAPI | 1.3.1 | B2.A.4 |
| langchain-core==0.1.52 | 14 | CRITICAL (CVE-2025-68664) | Yes — transitive runtime of both provider wrappers | 1.3.3+ → target 1.4.9 | B2.A.5 |
| langchain-openai==0.1.1 | 2 | LOW | Yes — OpenAI/embeddings client | 1.1.14+ → target 1.3.5 | B2.A.5 |
| langsmith==0.1.147 | 5 | HIGH | Marginal — imported by langchain-core, no tracing configured | 0.8.18+ → target 0.10.x | B2.A.5 (follows langchain-core) |
| protobuf==4.25.9 | 2 | HIGH | Marginal — legacy Gemini gRPC chain only | 6.33.5 / exits closure | B2.A.5 (chain removed with google-generativeai) |

Burn-down: B2.A.2 −42 · B2.A.3 −15 · B2.A.4 −14 · B2.A.5 −23 → **0 remaining, 0 waivers**.

## 5. Reverse-graph proofs for removals

From [`dependency-graph-cp311.json`](../evidence/f10_b2/dependency-graph-cp311.json):

- `litellm` ← nothing. Removal also drops `jinja2`, `jsonschema` (+`referencing`, `rpds-py`, `jsonschema-specifications`), `tokenizers` → `huggingface-hub` → `hf-xet`/`filelock`/`fsspec`/`tqdm`.
- `langchain` ← nothing; `langchain-community` ← `langchain` only; `langchain-text-splitters` ← `langchain` only; `dataclasses-json` (+`marshmallow`, `typing-inspect`, `mypy-extensions`) ← umbrella/community only.
- `aiohttp` (+`aiohappyeyeballs`, `aiosignal`, `frozenlist`, `multidict`, `propcache`, `yarl`, `attrs`) ← `langchain`, `langchain-community`, `litellm` only — exits entirely at B2.A.2.
- `statsmodels` ← nothing (+`patsy` falls).
- `tenacity` ← `google-genai`, `langchain-core` (and the packages being removed) — direct pin is redundant, package remains.
- `protobuf` ← `google-ai-generativelanguage`, `google-api-core`, `google-generativeai`, `googleapis-common-protos`, `grpcio-status`, `proto-plus` — the entire chain exists only for the **legacy** Gemini SDK, which is held by `langchain-google-genai==0.0.11`. The new `google-genai` SDK requires **no protobuf** (httpx/pydantic/websockets). At B2.A.5 the whole gRPC/protobuf chain leaves the closure.
- `openai` and `tiktoken` ← `langchain-openai` (and `litellm`) — retained, upgraded with the wrapper at B2.A.5 (`langchain-openai 1.3.5` requires `openai>=2.45.0,<3`).

## 6. Compatibility constraints for the target stack (verified against PyPI metadata)

- `fastapi 0.139.2` requires `starlette>=0.46.0` (**no upper cap**) and `pydantic>=2.9.0` — current `pydantic==2.9.2` satisfies it; `starlette==1.3.1` is resolvable.
- `langchain-openai 1.3.5` requires `langchain-core>=1.4.9,<2` and `openai>=2.45.0,<3`.
- `langchain-google-genai 4.2.7` requires `langchain-core>=1.4.7,<2`, `google-genai>=1.65.0,<3`, `pydantic>=2.0,<3` — the existing `google-genai==1.65.0` pin remains valid.
- `google-genai 1.65.0` requires `tenacity>=8.2.3,<9.2.0` — tenacity persists transitively after the direct pin is dropped.
- Local (cp314) characterization already exercises langchain-core 1.2.x / fastapi 0.135.x / starlette 0.52.x semantics against the full suite; the remaining jumps (core 1.2.17→1.4.9, starlette 0.52.1→1.3.1) still require the B2.A.4/B2.A.5 characterization gates in the plan — local green does not substitute for them.

## 7. Phase sequence confirmed for implementation

| Phase | Change | Records closed | Behavior risk |
|---|---|---:|---|
| B2.A.2 | Delete pins: `litellm`, `langchain`, `langchain-community`, `statsmodels`, `tenacity` (direct) — one commit per removal group | 42 | None (zero imports; resolver-verified) |
| B2.A.3 | `PyJWT==2.13.0`, `python-dotenv==1.2.2` + negative JWT tests | 15 | Low (auth policy tests added first) |
| B2.A.4 | `fastapi==0.139.2` (+ resolved `starlette==1.3.1`) after HTTP characterization | 14 | Medium (middleware/exception/CORS/limits contracts) |
| B2.A.5 | `langchain-openai==1.3.5`, `langchain-google-genai==4.2.7`, drop `google-generativeai` (+ transitive `langchain-core 1.4.9`, `langsmith 0.10.x`, protobuf chain exits) | 23 | Medium-high (provider runtime; adapt behind `ProviderInvocationRuntime`) |
| B2.A.6 | Reproducible lock + required CI Critical/High audit gate | — | None |

## 8. Manual verification pending

1. **Backend release evidence** workflow at the B2 candidate SHA: authoritative
   `pip-audit` JSON + CycloneDX SBOM on Python 3.11 (this workstation cannot
   execute that toolchain against cp311 pins).
2. CI (Python 3.11) green at each B2 slice — local 3.14 runs are diagnostic.
3. Protected evidence archive selection is still an open B0 item; until then,
   repository-tracked copies live in `docs/evidence/f10_b2/`.
