# Vector Knowledge — Ingestion Runbook

How to add a new regulatory or analytical document to the vector knowledge store. The rollout itself (initial pilot, schema apply, shadow-mode evaluation) is complete; this file is now a forward-looking runbook for ingesting additional documents.

For the runtime side (retrieval, three-tier policy, packing) see [`query_pipeline_architecture.md`](query_pipeline_architecture.md) §3.3 and [`knowledge/vector_retrieval.py`](../../knowledge/vector_retrieval.py).

## Storage Model

- Postgres schema: `knowledge`
- Tables: `knowledge.documents`, `knowledge.document_chunks`
- Vector column: `knowledge.document_chunks.embedding vector(1536)`
- Vector index on `document_chunks.embedding`
- Optional Supabase Storage bucket `knowledge-documents` for original files (not read at retrieval time)

The schema is created by `schemas/knowledge_vector.sql`. Required Postgres extensions: `vector`, `pgcrypto`.

## Embedding Providers

| Env | OpenAI (default) | Gemini |
|---|---|---|
| `VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER` | `openai` | `gemini` |
| `VECTOR_KNOWLEDGE_EMBEDDING_MODEL` | `text-embedding-3-small` | `gemini-embedding-001` |
| `VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION` | `1536` | `1536` |

The dimension must match the SQL `vector(...)` column width. Switching providers without matching dimensions corrupts ingestion.

## Runtime Flags

- `ENABLE_VECTOR_KNOWLEDGE_SHADOW=true` — retrieval runs and is traced, but retrieved passages do not enter prompts.
- `ENABLE_VECTOR_KNOWLEDGE_HINTS=true` — retrieved passages enter planner/summarizer prompts. Currently the production default.

Other env vars relevant to retrieval:

- `VECTOR_KNOWLEDGE_TOP_K=6` — top-K at retrieval tier FULL.
- `VECTOR_KNOWLEDGE_MAX_CHARS=9000` — packing cap on the prompt section.
- `VECTOR_KNOWLEDGE_MIN_SIMILARITY=0.2` — base similarity threshold.

### Adjacency expansion (Phase A of the cross-reference rollout)

- `VECTOR_ADJACENCY_MODE` — three-state env controlling whether neighbouring chunks (preceding and following by `chunk_index` within the same document) are pulled at retrieval time. Defaults to `off`.
  - `off` (default) — no adjacency fetch, no trace event, no prompt change. The safe state for rollback at any time.
  - `shadow` — adjacency IS fetched and attached to the bundle's `adjacent_chunks` field. A `stage_0_3_vector_knowledge_adjacency` trace event is emitted with `adjacency_mode`, `adjacent_chunk_count`, `adjacent_sections`, and `would_be_packed_chars`. **Pack output remains byte-identical to `off`** — adjacency does NOT enter the prompt. Use this state to observe hit rate before cutover.
  - `on` — adjacency packs into the prompt after the primary chunks, within the same `VECTOR_KNOWLEDGE_MAX_CHARS` budget. Adjacency entries are tagged `| adjacent` in their header so the LLM can tell direct matches from contextual siblings. Primary chunks always pack first; adjacency only fills remaining budget. If primary truncates, adjacency is not attempted.
  - **Unknown values** (typos like `true`, `1`, `enabled`) are treated as `off` — defensive default to prevent silent rollouts on operator error.

#### Safe rollout path

1. Deploy with `VECTOR_ADJACENCY_MODE` unset (default `off`). No behaviour change.
2. Flip to `shadow` in staging. Inspect `stage_0_3_vector_knowledge_adjacency` trace events on a representative query set. Expected signal: ≥25% of regulation queries pull at least one meaningful adjacent chunk.
3. Flip to `shadow` in production. Observe for ≥1 week. Pack output is unchanged, so no answer-quality regression risk during this window.
4. Flip to `on`. Adjacency now packs. Inspect trace events for `packed_truncated=true` rate — if it climbs noticeably, the score multiplier in `_sort_adjacent_by_parent_score` is too generous or `VECTOR_KNOWLEDGE_MAX_CHARS` needs to grow.
5. Rollback at any step: set `VECTOR_ADJACENCY_MODE=off`. Pack output reverts to byte-identical to pre-rollout.

Phase B of the cross-reference plan (explicit `მუხლი`-style reference extraction + one-hop resolution) is independent of this knob and tracked separately.

### Reference expansion (Phase B of the cross-reference rollout)

Phase B follows each top-K chunk's parsed cross-references (`მე-14 მუხლის მე-7 პუნქტი` and similar morphological variants — see `knowledge/vector_reference_parser.py` for the full catalog) and resolves them to the actual referenced chunks via the canonical `(document_id, article_number)` lookup. Same-document only; external-code references (`კოდექსი`) are rejected at parse time.

**Schema prerequisite.** Phase B requires the additive columns from `schemas/knowledge_vector.sql` (`article_number`, `chapter_number`, `parent_chapter`, `section_kind`, `outgoing_refs jsonb`) plus their indexes (`idx_knowledge_chunks_article`, `idx_knowledge_chunks_outgoing_refs`). The migration is idempotent (`add column if not exists`) and defaults to safe empties for legacy rows; apply it before deploying Phase B code:

```powershell
psql "$env:SUPABASE_DB_URL" -f schemas/knowledge_vector.sql
```

**Ingestion-side dependency.** The structural fields and `outgoing_refs` are populated by the chunker at ingest time. **Re-ingest any document that needs reference resolution** — legacy chunks pre-dating Phase B have empty `outgoing_refs` and won't participate in expansion (they still serve as primary matches). Re-ingestion uses the standard runbook above; nothing special.

**Env knob.** `VECTOR_REFERENCE_EXPANSION_MODE` — same three-state shape as `VECTOR_ADJACENCY_MODE`, independent default:
- `off` (default) — no resolution, no trace event, no prompt change. The safe state for rollback.
- `shadow` — references resolved and exposed on `bundle.reference_chunks`. A `stage_0_3_vector_knowledge_references` trace event is emitted with `reference_mode`, `reference_chunk_count`, `reference_sections`, `attempted_article_numbers`, and `would_be_packed_chars`. **Pack output remains byte-identical to `off`.** Use this state to observe hit rate before cutover.
- `on` — resolved reference chunks pack into the prompt after the primary chunks, **before** any adjacency. Reference entries are tagged `| referenced` in their header. Reference chunks earn priority over adjacency under budget pressure because the citing chunk explicitly cited them.
- **Unknown values** treated as `off` defensively.

**Pack order under any combination of modes:**

```
primary chunks (always packed first; budget exhaustion truncates here)
  ↓ if VECTOR_REFERENCE_EXPANSION_MODE == "on" and budget remains
references (tagged | referenced)
  ↓ if VECTOR_ADJACENCY_MODE == "on" and budget remains
adjacency (tagged | adjacent)
```

Either expansion list is independently env-gated. Setting only references to `on` (with adjacency `off`) packs references but no adjacency, and vice versa.

**Expansion budgets** (in `knowledge/vector_retrieval.py`):

- `REFERENCE_EXPANSION_PER_CHUNK_BUDGET = 3` — a single primary chunk with many cross-references cannot pull more than 3 of them. Caps the impact of long enumeration sections that cite a dozen articles.
- `REFERENCE_EXPANSION_TOTAL_BUDGET = 10` — across all primaries in a single request, at most 10 article lookups are issued. Caps the worst-case fan-out.

The resolver respects the `VECTOR_KNOWLEDGE_MAX_CHARS` pack budget on top of these — even if the request budget allows 10 lookups, the pack function will drop individual resolved chunks that don't fit.

#### Safe rollout path for reference expansion

1. **Schema migration.** Apply `schemas/knowledge_vector.sql` to the target DB. Idempotent; safe to re-run.
2. **Re-ingest documents.** Run the existing ingestion runbook (`ingest_one_document.py`) for each document in `docs_to_ingest/` that needs reference resolution. New chunks carry `outgoing_refs`; legacy chunks keep empty refs and behave like pre-Phase-B chunks.
3. **Deploy code with `VECTOR_REFERENCE_EXPANSION_MODE` unset.** Default `off`. No behaviour change.
4. **Flip to `shadow` in staging.** Inspect `stage_0_3_vector_knowledge_references` trace events. The `attempted_article_numbers` field shows which refs the primary set carried; `reference_chunk_count` shows how many resolved. A large gap means the targets are not in the corpus (expected for cross-document refs to uningested external codes — they should already be filtered as `კოდექსი` rejections, but plain mismatches surface here).
5. **Flip to `shadow` in production.** Observe ≥1 week. Pack output unchanged.
6. **Flip to `on`.** References now pack. Inspect `packed_truncated=true` rate. If it climbs noticeably, either narrow the `REFERENCE_EXPANSION_PER_CHUNK_BUDGET` / `REFERENCE_EXPANSION_TOTAL_BUDGET`, raise `VECTOR_KNOWLEDGE_MAX_CHARS`, or examine the top-K query type — `LIGHT` tier already does only top-K=2 and may not need expansion at all.
7. **Rollback at any step.** Set `VECTOR_REFERENCE_EXPANSION_MODE=off`. Pack output reverts to byte-identical to pre-rollout. The `outgoing_refs` data stays in the DB — no destructive change.

#### Known limitations (acceptable miss rate, will revisit if data warrants)

- **Chapter references are not yet resolved.** The parser captures Roman-numeral chapter refs (`თავი XI`) and emits `ChunkReferenceKind.chapter`, but the resolver currently skips them — only article-kind refs go through expansion. The 13 chapter references across the live corpus are surfaced via trace `attempted_article_numbers` not being populated for them; if production data shows chapter resolution would help, add a `fetch_chunks_by_chapter` companion to the store.
- **Plural-conjoined refs** like `მე-3 და მე-4 პუნქტებში` ("paragraphs 3 AND 4") may emit only the first paragraph number, not both. Estimated ~5–10% of compound refs in the corpus.
- **Decimal-article paragraph refs are not bound** — `14.7 მუხლის მე-2 პუნქტი` emits a reference to article `14.7` but loses the paragraph qualifier. Decimal articles are short in practice; the body of the resolved chunk usually contains the paragraph text anyway.
- **External-document references with quoted full titles** (e.g., `„ენერგეტიკისა და წყალმომარაგების შესახებ" საქართველოს კანონის X-ე მუხლის`) are NOT resolved cross-document — Phase B is strictly same-document. Per the audit, these are rare (<5 corpus-wide) and the citing chunk's body already references the right context.

## Step-By-Step: Ingest One Document

### 1. Prepare extracted text

Place a clean `.md` or `.txt` file in `docs_to_ingest/`. Plain text or markdown only — the ingestion path does not parse PDFs. If you only have a `.docx` or `.pdf`, convert/extract to markdown first; OCR'd scans must be cleaned before ingestion.

```text
docs_to_ingest/
  gnerc_regulation_2024.md
```

### 2. Set local environment

These env vars must be present **locally** because ingestion runs as a local Python script (the deployed app does not ingest at runtime).

```powershell
# OpenAI (recommended default)
$env:SUPABASE_DB_URL = "postgresql://YOUR_USER:YOUR_PASSWORD@YOUR_HOST:5432/postgres"
$env:OPENAI_API_KEY = "..."
$env:VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER = "openai"
$env:VECTOR_KNOWLEDGE_EMBEDDING_MODEL = "text-embedding-3-small"
$env:VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION = "1536"
```

```powershell
# Gemini (alternative)
$env:SUPABASE_DB_URL = "postgresql://YOUR_USER:YOUR_PASSWORD@YOUR_HOST:5432/postgres"
$env:GOOGLE_API_KEY = "..."
$env:VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER = "gemini"
$env:VECTOR_KNOWLEDGE_EMBEDDING_MODEL = "gemini-embedding-001"
$env:VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION = "1536"
```

`SUPABASE_DB_URL` must point to the same Postgres database where `knowledge_vector.sql` has been applied.

**CRITICAL — provider must match deployment.** The query path calls
`get_embedding_provider()` at runtime against the **deployed app's** env,
not the local ingest env. If Railway has
`VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER=gemini` and you ingest locally with
`openai` (or vice versa), every search returns noise — the two providers
produce vectors in different latent spaces. Before re-ingesting:
1. Check Railway env vars match what you'll use locally for ingest.
2. If you change provider, truncate `knowledge.document_chunks` and
   re-ingest everything. Mixed providers in the same table are a silent
   correctness bug.

### 3. Write the registration script

The repo includes [`ingest_one_document.py`](../../ingest_one_document.py) as a starter. Adapt it per document — change `text_path`, `source_key`, `title`, `document_type`, `issuer`, `language`, and topics:

```python
from pathlib import Path
from contracts.vector_knowledge import DocumentRegistration
from knowledge.vector_ingestion import VectorKnowledgeIngestor

text_path = Path(r"docs_to_ingest/gnerc_regulation_2024.md")
text_content = text_path.read_text(encoding="utf-8")

document = DocumentRegistration(
    source_key="gnerc-regulation-2024",          # stable unique id
    title="GNERC Regulation 2024",
    document_type="regulation",                  # regulation | law | report | order | ...
    issuer="GNERC",
    language="ka",                               # ka | en | ru | ...
    published_date="2024-07-01",
    effective_date=None,
    version_label="2024-07",
    metadata={"country": "georgia"},
)

ingestor = VectorKnowledgeIngestor()
result = ingestor.ingest_text_document(
    document=document,
    text_content=text_content,
    topics=["market_structure", "balancing_price"],   # see knowledge/vector_catalogs.py
)
print(result.model_dump())
```

### 4. Run

```powershell
python ingest_one_document.py
```

Expected output:

```text
{'document_id': '...', 'chunk_count': 12, 'embedding_dimension': 1536, 'source_key': 'gnerc-regulation-2024'}
```

### 5. Verify in Supabase

```sql
-- Document landed
select id, source_key, title, document_type, issuer, language
from knowledge.documents
where source_key = 'gnerc-regulation-2024';

-- Chunks landed
select chunk_index, section_title, left(text_content, 200)
from knowledge.document_chunks c
join knowledge.documents d on d.id = c.document_id
where d.source_key = 'gnerc-regulation-2024'
order by chunk_index;
```

### 6. Sanity-check retrieval

Send a query that should hit the new document and inspect the `stage_0_3_vector_knowledge` trace event:

- `top_sources` should include the new document title.
- `chunk_count > 0`.
- `error` should be empty.
- The packed section text in `packed_sections` should contain readable chunks (not split mid-clause).

## Quality Checks Before Trusting a New Document

- Chunk text is readable, not broken mid-sentence too often.
- Section headings became meaningful `section_title` values.
- Chunk count is reasonable for the document size (~1 chunk per 650 tokens).
- `language`, `document_type`, `issuer` are correct.
- Topic tags are useful and not too broad — see [`knowledge/vector_catalogs.py`](../../knowledge/vector_catalogs.py) for the catalog.
- Retrieval for known test questions returns the expected document.

## Common Questions

**Do I need Jupyter?** No. Plain Python script, run from PowerShell.

**Do I put the `.md` file in `knowledge/`?** No. Keep ingestion sources in `docs_to_ingest/`. The `knowledge/` folder is for curated runtime markdown consumed directly by the LLM.

**Do I need Railway env vars for local ingestion?** No. Local ingestion uses local env vars; Railway env vars matter only for the deployed app's retrieval.

**Do I need `source_url` / `storage_path`?** Optional. Use `source_url` to record the official link; use `storage_path` only if you keep originals in a Storage bucket. Vector search never reads from the bucket.

**What if dimension mismatches?** Stop and resolve before continuing. Mismatches must be aligned across the embedding-provider output, `VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION` env, and the SQL `vector(...)` column width.

## Guardrails

- Curated markdown knowledge in `knowledge/*.md` remains the canonical explanation layer; vector chunks are additive external-source passages.
- Prompt usage is bounded by `VECTOR_KNOWLEDGE_MAX_CHARS` (default 9000).
- Retrieval failures degrade safely to non-vector behaviour — the pipeline does not abort the request.
- No cross-encoder reranker today; retrieval quality depends on chunking + embeddings + topic/keyword boosts. See the architecture doc §5.x for the open improvement items.

## Appendix: Batch Re-ingest All Documents

When you need to re-ingest every file in `docs_to_ingest/` (for example after
applying the Phase B schema migration so that `outgoing_refs` populates), use
the script below.  One `VectorKnowledgeIngestor` instance is reused across
all documents — the embedding client (Gemini or OpenAI) initialises once.

**Prerequisites** (same as single-document ingest):
1. `schemas/knowledge_vector.sql` applied to the target DB.
2. Local env vars set per the Gemini or OpenAI block in [§2](#2-set-local-environment).
   The provider you choose here MUST match `VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER`
   on the deployed app (see the CRITICAL note in that section).
3. (Optional) `truncate table knowledge.document_chunks;` if you want a clean
   slate.  Otherwise the per-document `delete-then-insert` is enough — but
   any documents you don't re-ingest will still carry their old rows.

**Before running:** review each `DocSpec` entry below.  `source_key` is the
stable identity used to dedupe across re-ingests — keep it unchanged, or you
will create duplicate documents.  `title`, `published_date`, `effective_date`,
`version_label`, `issuer`, `topics`, and `metadata` should match what you set
during the previous ingest (or your best current understanding); a fresh
re-ingest is a good moment to correct any wrong values.

Save this as `ingest_all_documents.py` (a copy is already checked in at the
repo root for convenience), edit the metadata, then `python ingest_all_documents.py`.

```python
"""Batch ingestion driver: re-embed every file in ``docs_to_ingest/``."""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# --- Import-time config validation ---
# The deployed app validates auth/session/evaluate secrets and a default
# model type at module import.  Ingest doesn't use any of these values,
# but the imports below trigger the same validation.  Use ``setdefault``
# so real values you export in PowerShell ($env:SUPABASE_DB_URL etc.)
# win; placeholders only fill the gaps.  Must run BEFORE the
# ``from contracts...`` / ``from knowledge...`` imports below.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "ingest-script-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "ingest-script-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "ingest-script-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "ingest-script-openai-placeholder")

from contracts.vector_knowledge import DocumentRegistration  # noqa: E402
from knowledge.vector_ingestion import VectorKnowledgeIngestor  # noqa: E402


DOCS_DIR = Path(__file__).parent / "docs_to_ingest"


@dataclass
class DocSpec:
    """Per-document metadata + topics, paired with the source filename."""

    filename: str
    source_key: str
    title: str
    issuer: str
    language: str
    published_date: str | None
    effective_date: str | None
    version_label: str | None
    topics: List[str]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-document metadata.  REVIEW these before running — title / dates /
# issuer / topics must match what was set during the previous ingest
# (otherwise downstream traces and filters drift silently).  source_key
# is the stable identity used by ``replace_document_chunks``; keep it
# unchanged across re-ingests or you will create duplicate documents.
# ---------------------------------------------------------------------------

DOCUMENTS: List[DocSpec] = [
    DocSpec(
        filename="electricity_retail_market_rules.md",
        source_key=(
            "Electricity_retail_market_runes_______only_net_metering_"
            "and_net_billing_sections_20250731"
        ),
        title=(
            "Exctract about net metering and net billing from the "
            "'Electricity Retail Market Rules'"
        ),
        issuer="GNERC",
        language="ka",
        published_date="2020-08-14",
        effective_date="2020-08-14",
        version_label="2025-07-31",
        topics=[
            "net_metering",
            "net_billing",
            "microgeneration",
            "consumer_engagement",
            "renewable_energy_integration",
            "renewable_energy_communities",
        ],
        metadata={
            "country": "georgia",
            "city": "tbilisi",
            "resolution_number": "47",
            "annex_number": "NA",
            "notes": "Excerpt focused on net metering and net billing",
        },
    ),
    DocSpec(
        filename="gnerc_electricity_market_rules_excerpt_ka.md",
        source_key="GNERC_electricity_market_rules_excerpt_ka",
        # TODO: confirm title/dates from prior ingest record.
        title="GNERC Electricity Market Rules (excerpt, Georgian)",
        issuer="GNERC",
        language="ka",
        published_date=None,
        effective_date=None,
        version_label=None,
        topics=["market_structure", "balancing_price"],
        metadata={"country": "georgia"},
    ),
    DocSpec(
        filename="law_on_energy_and_water_supply.md",
        source_key="law_on_energy_and_water_supply",
        title="Law of Georgia on Energy and Water Supply",
        issuer="Parliament of Georgia",
        language="ka",
        published_date=None,
        effective_date=None,
        version_label=None,
        topics=["market_structure", "general_definitions"],
        metadata={"country": "georgia"},
    ),
    DocSpec(
        filename="market_concept_design.md",
        source_key="market_concept_design",
        title="Electricity Market Concept Design",
        issuer="MEPA",  # TODO: confirm issuer
        language="ka",
        published_date=None,
        effective_date=None,
        version_label=None,
        topics=["market_structure", "general_definitions"],
        metadata={"country": "georgia"},
    ),
    DocSpec(
        filename="transitory_market_rules.md",
        source_key="transitory_market_rules",
        title="Transitory Electricity Market Rules",
        issuer="GNERC",
        language="ka",
        published_date=None,
        effective_date=None,
        version_label=None,
        topics=["market_structure", "balancing_price", "general_definitions"],
        metadata={"country": "georgia"},
    ),
]


def _build_registration(spec: DocSpec) -> DocumentRegistration:
    return DocumentRegistration(
        source_key=spec.source_key,
        title=spec.title,
        issuer=spec.issuer,
        language=spec.language,
        source_url=None,
        storage_path=None,
        logical_key=spec.source_key,
        published_date=spec.published_date,
        effective_date=spec.effective_date,
        effective_end_date=None,
        version_label=spec.version_label,
        is_latest=True,
        is_active=True,
        abolished=False,
        supersedes_document_id=None,
        metadata=spec.metadata,
    )


def main() -> None:
    ingestor = VectorKnowledgeIngestor()

    successes: list[tuple[str, dict]] = []
    failures: list[tuple[str, str]] = []

    for spec in DOCUMENTS:
        path = DOCS_DIR / spec.filename
        if not path.exists():
            failures.append((spec.filename, f"file not found: {path}"))
            continue
        print(f"\n>>> Ingesting {spec.filename} (source_key={spec.source_key})")
        try:
            text_content = path.read_text(encoding="utf-8")
            result = ingestor.ingest_text_document(
                document=_build_registration(spec),
                text_content=text_content,
                topics=spec.topics,
            )
            payload = result.model_dump()
            successes.append((spec.filename, payload))
            print(f"    ok -> {payload}")
        except Exception as exc:  # batch driver, report-and-continue
            tb = traceback.format_exc()
            failures.append((spec.filename, f"{exc.__class__.__name__}: {exc}"))
            print(f"    FAILED: {exc.__class__.__name__}: {exc}")
            print(tb)

    print("\n" + "=" * 70)
    print(f"Done. {len(successes)} succeeded, {len(failures)} failed.")
    for fname, payload in successes:
        print(f"  ok  {fname}: chunks={payload.get('chunk_count')}")
    for fname, msg in failures:
        print(f"  ERR {fname}: {msg}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

**Expected output** (one block per document, then a summary):

```text
>>> Ingesting electricity_retail_market_rules.md (source_key=Electricity_retail_market_runes_...)
    ok -> {'document_id': '...', 'chunk_count': 12, 'embedding_dimension': 1536, 'source_key': '...'}
>>> Ingesting gnerc_electricity_market_rules_excerpt_ka.md (source_key=GNERC_electricity_market_rules_excerpt_ka)
    ok -> {'document_id': '...', 'chunk_count': 25, 'embedding_dimension': 1536, 'source_key': '...'}
...
======================================================================
Done. 6 succeeded, 0 failed.
  ok  electricity_retail_market_rules.md: chunks=12
  ok  gnerc_electricity_market_rules_excerpt_ka.md: chunks=25
  ...
```

**Post-ingest verification** (run in Supabase SQL editor):

```sql
-- (a) Phase B structural fields populated?
select count(*) filter (where article_number <> '') as with_article,
       count(*) filter (where outgoing_refs <> '[]'::jsonb) as with_refs,
       count(*) as total
from knowledge.document_chunks;

-- (b) Article 14 of transitory_market_rules indexed and resolvable?
select c.id, c.section_title, c.article_number, c.parent_chapter,
       jsonb_array_length(c.outgoing_refs) as ref_count
from knowledge.document_chunks c
join knowledge.documents d on d.id = c.document_id
where c.article_number = '14'
  and d.source_key = 'transitory_market_rules'
order by c.chunk_index
limit 5;

-- (c) Chunks that cite article 14 (these pull article 14 in via expansion):
select c.id, d.title, c.section_title
from knowledge.document_chunks c
join knowledge.documents d on d.id = c.document_id
where c.outgoing_refs @> '[{"kind":"article","number":"14"}]'::jsonb
limit 10;

-- (d) Confirm embedding dimension is uniform (1536):
select vector_dims(embedding) as dim, count(*)
from knowledge.document_chunks
group by 1;
```

If (a) shows `with_refs = 0`, the chunker ran against unmigrated schema or with stale code — investigate before retrying the affected queries.  If (d) returns more than one row, embedding-provider drift happened somewhere; truncate and re-ingest the affected documents under a single provider.
