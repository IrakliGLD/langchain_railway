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

`SUPABASE_DB_URL` must point to the same Postgres database where `knowledge_vector.sql` has been applied.

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
