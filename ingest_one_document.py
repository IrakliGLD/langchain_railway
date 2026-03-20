from pathlib import Path

from contracts.vector_knowledge import DocumentRegistration
from knowledge.vector_ingestion import VectorKnowledgeIngestor


def main() -> None:
    # Replace this with your local .md or .txt file path.
    text_path = Path(
        r"D:\Enaiapp\langchain_railway\docs_to_ingest\market_concept_design.md"
    )
    text_content = text_path.read_text(encoding="utf-8")

    document = DocumentRegistration(
        # Replace these values before running.
        source_key="electricity_market_concept_jun2025",
        title="Electricity Market Model Concept",
        document_type="regulation",
        issuer="Government of Georgia",
        language="ka",
        source_url=None,
        storage_path=None,
        logical_key="market_concept_jun25",
        published_date="2020-04-16",
        effective_date="2020-04-16",
        effective_end_date=None,
        version_label="2025-06-26",
        is_latest=True,
        is_active=True,
        abolished=False,
        supersedes_document_id=None,
        metadata={
            "country": "georgia",
            "city": "tbilisi",
            "resolution_number": "246",
            "annex_number": "1",
            "notes": "Pilot ingestion for market concept design",
        },
    )

    # Replace these topic tags with the best matching retrieval topics.
    topics = ["market_design", "electricity_market_transitory_model", "electricity_market_target_model", "day_ahead_market", "intraday_market", "balancing_market", "deregulation_plan"]

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
