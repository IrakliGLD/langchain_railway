"""Offline validation for Phase B.2 — chunker + reference parser.

Runs the production chunker (``knowledge.vector_chunking.chunk_markdown_text``)
over every ``*.md`` file in ``docs_to_ingest/`` and reports:

* per-document chunk counts by ``section_kind``;
* total outbound references captured, broken down by kind and by whether
  they carry a paragraph sub-reference;
* the first N detailed examples per document (citing section →
  resolved tuple → raw matched text) for eyeballing accuracy.

Cross-doc totals at the bottom show the morphological coverage we
actually achieve against the live corpus — compare against the
expected counts from the audit (suffix-ordinal: 151, prefix-ordinal:
116, decimal: 116, Roman chapter: 52, etc.).

No DB needed, no env vars needed beyond the chunker's own. Run from
the repo root::

    python scripts/validate_b2_corpus.py
    python scripts/validate_b2_corpus.py --details 10   # 10 examples per doc
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

# Make this script self-bootstrapping when invoked from anywhere — the
# chunker module reads project config that requires these env vars.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "validate")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "validate")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "validate")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "validate")

# Add repo root to sys.path so `python scripts/validate_b2_corpus.py` from
# either the root or the scripts/ dir resolves project imports.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from contracts.vector_knowledge import ChunkIngestRecord  # noqa: E402
from knowledge.vector_chunking import chunk_markdown_text  # noqa: E402

CORPUS_DIR = _ROOT / "docs_to_ingest"


def _summarise_chunks(chunks: list[ChunkIngestRecord]) -> dict:
    """Return per-document counts: chunks by section_kind, refs by kind/sub."""
    kind_counter = Counter[str]()
    ref_kind_counter = Counter[str]()
    refs_with_sub = 0
    refs_without_sub = 0
    chunks_with_refs = 0
    total_refs = 0

    article_numbers_seen: set[str] = set()
    chapter_numbers_seen: set[str] = set()

    for chunk in chunks:
        kind_counter[chunk.section_kind or "(none)"] += 1
        if chunk.article_number:
            article_numbers_seen.add(chunk.article_number)
        if chunk.chapter_number:
            chapter_numbers_seen.add(chunk.chapter_number)
        if chunk.outgoing_refs:
            chunks_with_refs += 1
        for ref in chunk.outgoing_refs:
            total_refs += 1
            ref_kind_counter[ref.kind.value] += 1
            if ref.sub_kind:
                refs_with_sub += 1
            else:
                refs_without_sub += 1

    return {
        "chunks": len(chunks),
        "by_section_kind": dict(kind_counter),
        "articles_distinct": len(article_numbers_seen),
        "chapters_distinct": len(chapter_numbers_seen),
        "total_refs": total_refs,
        "by_ref_kind": dict(ref_kind_counter),
        "refs_with_sub": refs_with_sub,
        "refs_without_sub": refs_without_sub,
        "chunks_with_refs": chunks_with_refs,
    }


def _iter_detailed_examples(
    chunks: Iterable[ChunkIngestRecord],
    *,
    limit: int,
) -> Iterable[tuple[ChunkIngestRecord, object]]:
    """Yield (chunk, ref) pairs up to ``limit`` for human inspection."""
    seen = 0
    for chunk in chunks:
        for ref in chunk.outgoing_refs:
            yield chunk, ref
            seen += 1
            if seen >= limit:
                return


def _format_ref(ref) -> str:
    sub = f" / {ref.sub_kind}={ref.sub_number}" if ref.sub_kind else ""
    return f"{ref.kind.value} {ref.number}{sub}"


def _format_citing(chunk: ChunkIngestRecord) -> str:
    label = chunk.section_title or chunk.section_path or f"chunk_{chunk.chunk_index}"
    if chunk.article_number:
        label += f" [article={chunk.article_number}]"
    elif chunk.chapter_number:
        label += f" [chapter={chunk.chapter_number}]"
    return label


def _run_one_document(path: Path, *, details: int) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        print(f"\n=== {path.name}: EMPTY ===")
        return {"chunks": 0}

    # Document language hint — used by the chunker but doesn't affect
    # the reference parser. Default to "ka" for the Georgian corpus.
    chunks = chunk_markdown_text(text, language="ka", topics=[])

    summary = _summarise_chunks(chunks)
    print(f"\n=== {path.name} ===")
    print(f"  Chunks:               {summary['chunks']}")
    print(f"  By section_kind:      {summary['by_section_kind']}")
    print(
        f"  Distinct articles:    {summary['articles_distinct']}  "
        f"chapters: {summary['chapters_distinct']}"
    )
    print(
        f"  Outbound refs:        {summary['total_refs']}  "
        f"(with sub={summary['refs_with_sub']}, "
        f"without sub={summary['refs_without_sub']})"
    )
    print(f"  Refs by kind:         {summary['by_ref_kind']}")
    print(
        f"  Chunks with refs:     {summary['chunks_with_refs']} "
        f"({(summary['chunks_with_refs'] / summary['chunks'] * 100):.1f}%)"
    )

    if details > 0 and summary["total_refs"] > 0:
        print(f"\n  First {details} reference examples:")
        for chunk, ref in _iter_detailed_examples(chunks, limit=details):
            print(f"    [{_format_citing(chunk)}]")
            print(f"      → {_format_ref(ref)}")
            print(f"      raw: {ref.raw_text!r}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase B.2 chunker over the live corpus.")
    parser.add_argument(
        "--details",
        type=int,
        default=5,
        help="Number of detailed reference examples to print per document (default: 5)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_DIR,
        help="Path to the docs_to_ingest directory (default: <repo>/docs_to_ingest)",
    )
    args = parser.parse_args()

    if not args.corpus.is_dir():
        print(f"ERROR: corpus directory not found: {args.corpus}")
        return 2

    md_files = sorted(args.corpus.glob("*.md"))
    if not md_files:
        print(f"No .md files in {args.corpus}")
        return 2

    print(f"Validating Phase B.2 chunker over {len(md_files)} document(s) in {args.corpus}")
    print("=" * 70)

    aggregate: dict[str, int | dict[str, int]] = {
        "total_chunks": 0,
        "total_refs": 0,
        "refs_with_sub": 0,
        "refs_without_sub": 0,
        "ref_kind_totals": Counter[str](),
        "section_kind_totals": Counter[str](),
    }
    for path in md_files:
        summary = _run_one_document(path, details=args.details)
        aggregate["total_chunks"] += summary.get("chunks", 0)
        aggregate["total_refs"] += summary.get("total_refs", 0)
        aggregate["refs_with_sub"] += summary.get("refs_with_sub", 0)
        aggregate["refs_without_sub"] += summary.get("refs_without_sub", 0)
        for k, v in (summary.get("by_ref_kind") or {}).items():
            aggregate["ref_kind_totals"][k] += v
        for k, v in (summary.get("by_section_kind") or {}).items():
            aggregate["section_kind_totals"][k] += v

    print("\n" + "=" * 70)
    print("CROSS-CORPUS TOTALS")
    print("=" * 70)
    print(f"  Documents:            {len(md_files)}")
    print(f"  Total chunks:         {aggregate['total_chunks']}")
    print(f"  Section kinds:        {dict(aggregate['section_kind_totals'])}")
    print(f"  Total outbound refs:  {aggregate['total_refs']}")
    print(
        f"    ├─ with sub-ref:    {aggregate['refs_with_sub']} "
        f"({(aggregate['refs_with_sub'] / aggregate['total_refs'] * 100):.1f}%)"
        if aggregate["total_refs"]
        else "    ├─ with sub-ref:    0"
    )
    print(
        f"    └─ without sub-ref: {aggregate['refs_without_sub']} "
        f"({(aggregate['refs_without_sub'] / aggregate['total_refs'] * 100):.1f}%)"
        if aggregate["total_refs"]
        else "    └─ without sub-ref: 0"
    )
    print(f"  Refs by kind:         {dict(aggregate['ref_kind_totals'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
