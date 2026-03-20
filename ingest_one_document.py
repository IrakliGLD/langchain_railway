from pathlib import Path

from contracts.vector_knowledge import DocumentRegistration
from knowledge.vector_ingestion import VectorKnowledgeIngestor


def main() -> None:
    # Replace this with your local .md or .txt file path.
    text_path = Path(
        r"D:\Enaiapp\langchain_railway\docs_to_ingest\gnerc_electricity_market_rules_excerpt_ka.md"
    )
    text_content = text_path.read_text(encoding="utf-8")

    document = DocumentRegistration(
        # Replace these values before running.
        source_key="exchange_rules_july24",
        title="Electricity Day-Ahead and Intraday Market Rules",
        document_type="regulation",
        issuer="GNERC",
        language="ka",
        source_url=None,
        storage_path=None,
        logical_key="gnerc-electricity-market-rules-2020-46-annex-1",
        published_date="2020-08-11",
        effective_date="2024-07-01",
        effective_end_date=None,
        version_label="2024-07-01",
        is_latest=True,
        is_active=True,
        abolished=False,
        supersedes_document_id=None,
        metadata={
            "country": "georgia",
            "city": "tbilisi",
            "resolution_number": "46",
            "annex_number": "1",
            "notes": "Pilot ingestion for exchange rules",
        },
    )

    # Replace these topic tags with the best matching retrieval topics.
    topics = ["market_structure", "exchange_rules", "day_ahead_market", "intraday_market", "financial_clearing", "exchange_price_formation", "exchange_participation", "exchange_operations"]

    ingestor = VectorKnowledgeIngestor()
    result = ingestor.ingest_text_document(
        document=document,
        text_content=text_content,
        topics=topics,
    )

    print("Ingestion result:")
    print(result.model_dump())


if __name__ == "__main__":
    main()
