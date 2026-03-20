# Vector Knowledge Rollout

## Purpose

This rollout adds Supabase-backed vector retrieval as a hybrid layer alongside the existing curated markdown knowledge.

## Storage Model

- Postgres schema: `knowledge`
- Source files: optional Supabase Storage bucket `knowledge-documents`
- Searchable rows:
  - `knowledge.documents`
  - `knowledge.document_chunks`

## Embeddings

- Model target: `text-embedding-3-small`
- Embedding dimension: `1536`

Example provider choices:

- OpenAI
  - `VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER=openai`
  - `VECTOR_KNOWLEDGE_EMBEDDING_MODEL=text-embedding-3-small`
  - `VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION=1536`

- Gemini
  - `VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER=gemini`
  - `VECTOR_KNOWLEDGE_EMBEDDING_MODEL=gemini-embedding-001`
  - `VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION=1536`

## Runtime Flags

- `ENABLE_VECTOR_KNOWLEDGE_SHADOW`
  - run retrieval and traces only
  - no prompt impact

- `ENABLE_VECTOR_KNOWLEDGE_HINTS`
  - run retrieval and pass retrieved passages into planner/summarizer prompts

## Preconditions

Before rollout, make sure these are available in the running environment:

- `SUPABASE_DB_URL`
  - used by the app to connect to Postgres
- `OPENAI_API_KEY`
  - required for embedding generation
- `VECTOR_KNOWLEDGE_EMBEDDING_MODEL`
  - depends on provider
- `VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER`
  - `openai` or `gemini`
  - current default: `openai`
- `VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION`
  - must match the table schema
  - current expected value: `1536`

Important:

- the current implementation stores vectors in Postgres tables, not in Supabase Storage
- Supabase Storage is optional and can be used only to keep original files such as PDFs
- the current ingestion path does not parse PDFs itself; it expects already-extracted text
- if you switch embedding provider or model, the produced vector dimension must still match `knowledge.document_chunks.embedding`

## Rollout Order

1. Apply `schemas/knowledge_vector.sql` in Supabase.
2. Ingest a small pilot corpus.
3. Enable `ENABLE_VECTOR_KNOWLEDGE_SHADOW=true`.
4. Inspect traces for chunk quality, noise, and latency.
5. Enable `ENABLE_VECTOR_KNOWLEDGE_HINTS=true` only after shadow results are acceptable.

## What Each Step Means

### 1. Apply `schemas/knowledge_vector.sql` in Supabase

This creates the database objects needed for vector retrieval:

- extension `vector`
- extension `pgcrypto`
- schema `knowledge`
- table `knowledge.documents`
- table `knowledge.document_chunks`
- vector index on `knowledge.document_chunks.embedding`

After running it, verify that:

- both tables exist
- `knowledge.document_chunks.embedding` is `vector(1536)`
- inserts into both tables work

Recommended verification queries:

```sql
select count(*) from knowledge.documents;
select count(*) from knowledge.document_chunks;
```

```sql
select column_name, data_type
from information_schema.columns
where table_schema = 'knowledge'
  and table_name = 'document_chunks';
```

To inspect the embedding column definition more directly:

```sql
select
  a.attname as column_name,
  format_type(a.atttypid, a.atttypmod) as formatted_type
from pg_attribute a
join pg_class c on c.oid = a.attrelid
join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'knowledge'
  and c.relname = 'document_chunks'
  and a.attname = 'embedding'
  and a.attnum > 0
  and not a.attisdropped;
```

### 2. Ingest a small pilot corpus

This means:

- do not load all documents immediately
- start with a small, representative set
- use documents that are important and easy to judge for retrieval quality

Recommended pilot size:

- `3-10` documents first

Good pilot choices:

- one regulation
- one methodology/order
- one market rules document
- one analytical report

Avoid in the first batch:

- image-only scanned PDFs
- very noisy OCR text
- hundreds of pages at once

The goal of the pilot is not volume. The goal is to verify:

- text extraction quality
- chunking quality
- retrieval relevance
- metadata usefulness
- latency impact

### 3. Enable `ENABLE_VECTOR_KNOWLEDGE_SHADOW=true`

This turns on vector retrieval in the pipeline, but only in observation mode.

What happens in shadow mode:

- the app embeds the query
- retrieves vector chunks from Supabase
- stores them on `QueryContext`
- logs the retrieval result
- does **not** pass the retrieved passages into the planner/summarizer prompts

So in shadow mode:

- answers should stay behaviorally the same
- only logs and traces should change

This is the safe stage to evaluate whether retrieval is useful before it influences answers.

### 4. Inspect traces for chunk quality, noise, and latency

In Railway logs, inspect `stage_0_3_vector_knowledge`.

Useful fields:

- `chunk_count`
  - how many chunks were returned
- `top_sources`
  - which documents were retrieved
- `preferred_topics`
  - what topic filters were applied
- `strategy`
  - currently `hybrid`
- `error`
  - should usually be empty

What you want to see:

- the right source documents appear for the question
- chunks are semantically relevant, not random
- retrieval works for both conceptual and document-heavy questions
- latency increase is acceptable

What bad shadow results look like:

- `chunk_count=0` for clearly relevant questions
- irrelevant documents appearing in `top_sources`
- same document always appearing regardless of question
- large error rate
- noticeable latency spike on most requests

Recommended shadow test questions:

- questions clearly answered by a specific regulation
- questions answered by a report subsection
- paraphrased questions
- multilingual variants if documents are mainly English

### 5. Enable `ENABLE_VECTOR_KNOWLEDGE_HINTS=true`

Only do this after shadow results look good.

What changes in active mode:

- retrieved external passages are added to planner and summarizer prompts
- vector retrieval can influence answers
- current curated markdown knowledge still remains the main canonical layer

Recommended activation sequence:

1. keep the pilot corpus small
2. enable hints
3. test a small set of representative questions
4. confirm answer quality improves rather than becoming noisier
5. only then ingest more documents

## How Vectorization Works In This Implementation

Vectorization here means:

1. take extracted document text
2. split it into chunks
3. create an embedding vector for each chunk
4. store document metadata in `knowledge.documents`
5. store chunk text + metadata + embedding in `knowledge.document_chunks`

Current chunking behavior:

- heading-aware splitting for markdown-like text
- target chunk size: about `650` tokens
- overlap: about `100` tokens

Current ingestion classes:

- [vector_ingestion.py](/d:/Enaiapp/langchain_railway/knowledge/vector_ingestion.py)
- [vector_chunking.py](/d:/Enaiapp/langchain_railway/knowledge/vector_chunking.py)
- [vector_embeddings.py](/d:/Enaiapp/langchain_railway/knowledge/vector_embeddings.py)
- [vector_store.py](/d:/Enaiapp/langchain_railway/knowledge/vector_store.py)

## How To Add Vectorized Data Into The Tables

The database is not filled directly by SQL inserts of raw vectors by hand. The intended path is Python ingestion through `VectorKnowledgeIngestor`.

Important:

- this ingestion flow runs as a **local Python script**
- it is **not** part of the app UI
- it is **not** Jupyter-specific
- if you run it locally, the required env vars must also exist **locally**
- the script can still write into the same remote Supabase database used by Railway

## Step-By-Step: Ingest One Document Locally

This is the most practical first workflow for `10-20` documents.

### Step 0: Open terminal in the project folder

Use PowerShell and go to the repo root:

```powershell
cd D:\Enaiapp\langchain_railway
```

### Step 1: Prepare one extracted document as `.md` or `.txt`

Current ingestion expects plain text, markdown, or very clean extracted text.

Example folder:

```text
D:\Enaiapp\langchain_railway\docs_to_ingest\
  gnerc_regulation_2024.md
```

If your original document is a Word file like:

```text
gnerc_regulation_2024.docx
```

then first convert or copy its contents into:

```text
gnerc_regulation_2024.md
```

For the first run, keep it simple:

- one document only
- clean text only
- avoid scanned PDFs and messy OCR

### Step 2: Set local environment variables in PowerShell

These are local placeholders. Replace the values with your real secrets and connection string.

If using Gemini embeddings locally:

```powershell
$env:SUPABASE_DB_URL = "postgresql://YOUR_USER:YOUR_PASSWORD@YOUR_HOST:5432/postgres"
$env:GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
$env:VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER = "gemini"
$env:VECTOR_KNOWLEDGE_EMBEDDING_MODEL = "gemini-embedding-001"
$env:VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION = "1536"
```

If using OpenAI embeddings locally:

```powershell
$env:SUPABASE_DB_URL = "postgresql://YOUR_USER:YOUR_PASSWORD@YOUR_HOST:5432/postgres"
$env:OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
$env:VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER = "openai"
$env:VECTOR_KNOWLEDGE_EMBEDDING_MODEL = "text-embedding-3-small"
$env:VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION = "1536"
```

Notes:

- `SUPABASE_DB_URL` should point to the same Supabase Postgres database where you already applied `knowledge_vector.sql`
- these env vars are needed **locally** because the script runs locally
- `source_url` is optional; if you only have a local file, you can leave it empty
- `storage_path` is optional; it matters only if you keep originals in Supabase Storage or another bucket

### Step 3: Create a temporary Python ingestion script

Create a file in the repo root, for example:

```text
D:\Enaiapp\langchain_railway\ingest_one_document.py
```

Put this code inside:

```python
from pathlib import Path

from contracts.vector_knowledge import DocumentRegistration
from knowledge.vector_ingestion import VectorKnowledgeIngestor


def main() -> None:
    text_path = Path(r"D:\Enaiapp\langchain_railway\docs_to_ingest\gnerc_regulation_2024.md")
    text_content = text_path.read_text(encoding="utf-8")

    document = DocumentRegistration(
        source_key="gnerc-regulation-2024",
        title="GNERC Regulation 2024",
        document_type="regulation",
        issuer="GNERC",
        language="ka",
        source_url=None,
        storage_path=None,
        published_date="2024-07-01",
        effective_date=None,
        version_label="2024-07",
        metadata={
            "country": "georgia",
            "notes": "first pilot ingestion",
        },
    )

    ingestor = VectorKnowledgeIngestor()
    result = ingestor.ingest_text_document(
        document=document,
        text_content=text_content,
        topics=["market_structure", "balancing_price"],
    )

    print("Ingestion result:")
    print(result.model_dump())


if __name__ == "__main__":
    main()
```

What you should replace in that file:

- `text_path`
  - path to your `.md` or `.txt` file
- `source_key`
  - stable unique id for the document
- `title`
- `document_type`
  - such as `regulation`, `law`, `report`, `order`
- `issuer`
- `language`
  - `ka`, `en`, `ru`, etc.
- `published_date`, `effective_date`, `version_label` if known
- `topics`
  - the most relevant topic tags for retrieval

### Step 4: Run the script

From PowerShell, in the repo root:

```powershell
python ingest_one_document.py
```

If `python` does not work on your machine, try:

```powershell
py ingest_one_document.py
```

### Step 5: Understand what the script does

When you run it, the code does this:

1. reads your local `.md` file
2. creates a `DocumentRegistration` object
3. normalizes the document type
4. inserts or updates one row in `knowledge.documents`
5. splits the text into chunks
6. creates embeddings for those chunks
7. inserts chunk rows into `knowledge.document_chunks`

### Step 6: Check the printed result

If the script succeeds, it should print something like:

```text
Ingestion result:
{'document_id': '...', 'chunk_count': 12, 'embedding_dimension': 1536, 'source_key': 'gnerc-regulation-2024'}
```

What this means:

- `document_id`
  - database id of the document row
- `chunk_count`
  - how many chunks were created from the text
- `embedding_dimension`
  - should match your env var and SQL schema
- `source_key`
  - your document identifier

### Step 7: Verify data in Supabase

Run these SQL queries in Supabase:

```sql
select id, source_key, title, document_type, issuer, language
from knowledge.documents
order by created_at desc
limit 20;
```

```sql
select document_id, chunk_index, section_title, language, topics
from knowledge.document_chunks
order by created_at desc
limit 50;
```

To inspect just your document:

```sql
select d.source_key, d.title, c.chunk_index, c.section_title, left(c.text_content, 200)
from knowledge.document_chunks c
join knowledge.documents d on d.id = c.document_id
where d.source_key = 'gnerc-regulation-2024'
order by c.chunk_index;
```

If you see rows there, ingestion worked.

## Recommended First Ingestion Workflow

1. choose `3-5` important documents
2. extract clean text into local `.md` or `.txt` files
3. assign `source_key`, `document_type`, `issuer`, `language`, dates
4. ingest one document at a time
5. verify rows in Supabase after each ingestion
6. only then enable shadow mode

This is better than bulk-loading everything first, because errors are easier to see and fix.

## Common Beginner Questions

### Do I need Jupyter?

No.

Use a normal Python file and run it from PowerShell.

### Do I need to put the `.md` file inside `knowledge/`?

No.

Keep source documents in a separate folder such as:

```text
docs_to_ingest/
```

The `knowledge/` folder is for curated app knowledge, not raw ingestion input.

### Do I need `source_url`?

No.

It is optional.

Use it only if you want to remember the official source URL for the document.

### Do I need Railway env vars for local ingestion?

No.

If the script runs locally, the env vars must be set locally.

Railway env vars matter only when the deployed app itself performs retrieval or ingestion.

## What To Observe After Ingestion

Before enabling hints, confirm:

- chunk text is readable and not broken mid-sentence too often
- headings became meaningful `section_title` values
- chunk count is reasonable for the document size
- `language`, `document_type`, and `issuer` are correct
- topic tags are useful and not too broad
- retrieval for known test questions returns the expected documents

## Notes On Source Files And Storage

- storing original PDFs in Supabase Storage is optional
- vector search does not read from the bucket directly
- vector search reads from `knowledge.documents` and `knowledge.document_chunks`
- if you do use Storage, save the bucket path in `storage_path`

## Practical Next Step If SQL Is Already Applied

Since you already ran the SQL, your next real step is:

1. prepare a small set of extracted document texts
2. ingest them with `VectorKnowledgeIngestor`
3. verify rows in `knowledge.documents` and `knowledge.document_chunks`
4. turn on `ENABLE_VECTOR_KNOWLEDGE_SHADOW=true`
5. test real questions and inspect `stage_0_3_vector_knowledge` logs

That is the actual start of rollout from here.

## Gemini Example For Railway

If you want to use Gemini embeddings in Railway, set:

```text
VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER=gemini
VECTOR_KNOWLEDGE_EMBEDDING_MODEL=gemini-embedding-001
VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION=1536
GOOGLE_API_KEY=...
```

Then:

1. redeploy
2. ingest one pilot document
3. confirm one row lands in `knowledge.documents`
4. confirm chunk rows land in `knowledge.document_chunks`
5. only then continue with the rest of the pilot corpus

If ingestion fails with a dimension mismatch, do not continue. Resolve the mismatch between:

- Gemini output dimension
- `VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION`
- SQL `vector(...)` dimension

## Current Guardrails

- curated markdown knowledge remains the canonical explanation layer
- vector chunks are additive external source passages
- active prompt usage is bounded by `VECTOR_KNOWLEDGE_MAX_CHARS`
- retrieval failures degrade safely to existing non-vector behavior

## Remaining Risks

- no reranker yet; retrieval quality depends on chunking + embeddings + metadata filters
- ingestion currently assumes extracted text is already available; PDF parsing is not part of this implementation
- citations use source metadata in prompt context, but answer formatting does not yet expose first-class chunk citations to the UI
