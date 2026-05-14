"""Batch ingestion driver: re-embed every file in ``docs_to_ingest/``.

One ``VectorKnowledgeIngestor`` instance is reused across all documents so
the embedding client (Gemini or OpenAI) initialises once.  Each document
has its own ``DocumentRegistration``; populate the per-doc metadata below
before running.

Behaviour:

- Reads every entry in ``DOCUMENTS`` from ``docs_to_ingest/``.
- Calls ``store.replace_document_chunks`` under the hood, which deletes
  any existing rows for the same ``document_id`` and re-inserts.  Safe to
  re-run.
- Catches and reports per-document errors so one bad file does not abort
  the rest of the batch.

Prerequisites:
- ``schemas/knowledge_vector.sql`` already applied to the target DB.
- Env vars set (see ``docs/active/VECTOR_KNOWLEDGE_ROLLOUT.md`` — provider
  / model / dimension must match what the deployed app uses for queries).

Run:
    python ingest_all_documents.py
"""

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
        topics=["exchange_rules", "day_ahead_market", "intraday_market", "auction_rules"],
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
        topics=["market_structure", "general_definitions","licencins", "GNERC", "market_particionat", "markets", "system_operators"],
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
        published_date="2025-09-15",
        effective_date=None,
        version_label=None,
        topics=["market_structure", "balancing_price", "general_definitions", "transitory_market_rules", "balancing_price", "transitory_export_import_rules"],
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
        except Exception as exc:  # noqa: BLE001 — batch driver, report-and-continue
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
