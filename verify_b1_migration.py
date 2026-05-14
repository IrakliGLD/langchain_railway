"""Verify Phase B.1 schema migration against the live staging DB.

Usage::

    # 1. Apply the migration first — idempotent, safe to re-run:
    psql "$SUPABASE_DB_URL" -f schemas/knowledge_vector.sql

    # 2. Then run this verifier locally:
    $env:SUPABASE_DB_URL = "postgresql://..."
    python verify_b1_migration.py

The script:
    * checks that the five new columns exist on ``knowledge.document_chunks``,
    * checks that the two new indexes exist,
    * round-trips a smoke INSERT/SELECT through the actual store code so
      ``_row_to_chunk_record`` and the JSONB serialiser see the real DB,
    * cleans up the test rows it creates.

It does NOT mutate any existing rows. The smoke document is created with a
unique ``source_key`` prefixed ``__b1_verify_`` and removed at the end (and
on failure via try/finally).
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime

# Project bootstrap — same env baseline the test suite uses.
os.environ.setdefault("ENAI_GATEWAY_SECRET", "verify")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "verify")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "verify")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "verify")

if not os.getenv("SUPABASE_DB_URL"):
    print("ERROR: SUPABASE_DB_URL is not set in the environment.")
    print("Set it to the staging Postgres URL and re-run.")
    sys.exit(2)

from sqlalchemy import text  # noqa: E402

from contracts.vector_knowledge import (  # noqa: E402
    ChunkIngestRecord,
    ChunkReference,
    ChunkReferenceKind,
    DocumentRegistration,
)
from knowledge.vector_store import KnowledgeVectorStore, _resolve_engine  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Verify columns
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "article_number": "text",
    "chapter_number": "text",
    "parent_chapter": "text",
    "section_kind": "text",
    "outgoing_refs": "jsonb",
}

EXPECTED_INDEXES = {
    "idx_knowledge_chunks_article",
    "idx_knowledge_chunks_outgoing_refs",
}


def verify_columns() -> bool:
    sql = text(
        """
        select column_name, data_type, is_nullable, column_default
        from information_schema.columns
        where table_schema = 'knowledge'
          and table_name = 'document_chunks'
          and column_name = any(:names)
        """
    )
    with _resolve_engine().begin() as conn:
        rows = conn.execute(sql, {"names": list(EXPECTED_COLUMNS)}).mappings().all()

    found = {row["column_name"]: dict(row) for row in rows}
    print("\n[1/4] Column check")
    ok = True
    for name, expected_type in EXPECTED_COLUMNS.items():
        if name not in found:
            print(f"  ✗ MISSING: {name}")
            ok = False
            continue
        actual_type = found[name]["data_type"]
        if actual_type != expected_type:
            print(f"  ✗ TYPE MISMATCH: {name} expected={expected_type} actual={actual_type}")
            ok = False
            continue
        print(f"  ✓ {name}: {actual_type} (nullable={found[name]['is_nullable']}, default={found[name]['column_default']})")
    return ok


def verify_indexes() -> bool:
    sql = text(
        """
        select indexname
        from pg_indexes
        where schemaname = 'knowledge'
          and tablename = 'document_chunks'
          and indexname = any(:names)
        """
    )
    with _resolve_engine().begin() as conn:
        rows = conn.execute(sql, {"names": list(EXPECTED_INDEXES)}).mappings().all()
    found = {row["indexname"] for row in rows}
    print("\n[2/4] Index check")
    ok = True
    for name in EXPECTED_INDEXES:
        if name in found:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ MISSING: {name}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# 2. Smoke INSERT + readback round-trip
# ---------------------------------------------------------------------------

def smoke_insert_and_readback() -> bool:
    """Insert one document with one chunk that exercises every new B.1 field,
    then read it back via the store's normal SELECT path.  Cleans up
    afterwards.  Pinpoints whether INSERT writes/SELECT reads the columns
    correctly end-to-end (not just at the contract layer)."""

    store = KnowledgeVectorStore()
    unique_suffix = uuid.uuid4().hex[:12]
    source_key = f"__b1_verify_{unique_suffix}"

    document = DocumentRegistration(
        source_key=source_key,
        title=f"B.1 Verification Doc {unique_suffix}",
        document_type="regulation",
        issuer="verifier",
        language="ka",
        published_date=datetime.utcnow().date().isoformat(),
        metadata={"phase": "B.1", "purpose": "schema_verification"},
    )

    print("\n[3/4] Smoke INSERT")
    document_id: str | None = None
    try:
        document_id = store.upsert_document(document)
        print(f"  ✓ Upserted document {document_id}")

        chunk = ChunkIngestRecord(
            chunk_index=0,
            text_content="Smoke chunk body for B.1 verification.",
            section_title="მუხლი 14",
            section_path="თავი IV / მუხლი 14",
            article_number="14",
            chapter_number="IV",
            parent_chapter="IV",
            section_kind="article",
            outgoing_refs=[
                ChunkReference(
                    kind=ChunkReferenceKind.article,
                    number="8.1",
                    raw_text="მე-8.1 მუხლი",
                ),
                ChunkReference(
                    kind=ChunkReferenceKind.article,
                    number="14",
                    sub_kind="paragraph",
                    sub_number="7",
                    raw_text="მე-14 მუხლის მე-7 პუნქტი",
                ),
                ChunkReference(
                    kind=ChunkReferenceKind.self_article,
                    number="14",
                    raw_text="ამ მუხლის",
                ),
            ],
            language="ka",
            topics=["balancing_price"],
            metadata={"verify": True},
        )

        # Embedding dimension must match the configured dimension (1536 by default).
        from config import VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION

        result = store.replace_document_chunks(
            document_id=document_id,
            source_key=source_key,
            chunks=[chunk],
            embeddings=[[0.0] * VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION],
        )
        print(f"  ✓ Inserted {result.chunk_count} chunk(s)")

        # ----- Readback via fetch_chunks_by_index (uses the new SELECT path) -----
        print("\n[4/4] Readback via fetch_chunks_by_index")
        records = store.fetch_chunks_by_index([(document_id, 0)])
        if not records:
            print("  ✗ Readback returned nothing")
            return False
        rec = records[0]

        ok = True
        for field, expected in (
            ("article_number", "14"),
            ("chapter_number", "IV"),
            ("parent_chapter", "IV"),
            ("section_kind", "article"),
        ):
            actual = getattr(rec, field)
            if actual != expected:
                print(f"  ✗ {field}: expected={expected!r} actual={actual!r}")
                ok = False
            else:
                print(f"  ✓ {field} = {actual!r}")

        if len(rec.outgoing_refs) != 3:
            print(f"  ✗ outgoing_refs length: expected=3 actual={len(rec.outgoing_refs)}")
            ok = False
        else:
            kinds = {r.kind for r in rec.outgoing_refs}
            if kinds != {ChunkReferenceKind.article, ChunkReferenceKind.self_article}:
                print(f"  ✗ outgoing_refs kinds: {kinds}")
                ok = False
            else:
                print(f"  ✓ outgoing_refs: 3 entries, kinds={sorted(k.value for k in kinds)}")
                for ref in rec.outgoing_refs:
                    sub = f" / {ref.sub_kind}={ref.sub_number}" if ref.sub_kind else ""
                    print(f"     - {ref.kind.value} {ref.number}{sub} ('{ref.raw_text}')")

        return ok
    finally:
        if document_id is not None:
            with _resolve_engine().begin() as conn:
                conn.execute(
                    text("delete from knowledge.documents where id = :id"),
                    {"id": document_id},
                )
            print(f"\nCleanup: removed document {document_id} (cascade deletes chunks).")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Phase B.1 migration verification")
    print("=" * 60)

    all_ok = True
    all_ok &= verify_columns()
    all_ok &= verify_indexes()
    all_ok &= smoke_insert_and_readback()

    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: PASS — Phase B.1 schema verified end-to-end.")
        return 0
    print("RESULT: FAIL — see ✗ lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
