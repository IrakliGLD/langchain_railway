from pathlib import Path

from contracts.vector_knowledge import DocumentRegistration
from knowledge.vector_ingestion import VectorKnowledgeIngestor


def main() -> None:
    # Replace this with your local .md or .txt file path.
    text_path = Path(
        r"D:\Enaiapp\langchain_railway\docs_to_ingest\electricity_retail_market_rules.md"
    )
    text_content = text_path.read_text(encoding="utf-8")

    document = DocumentRegistration(
        source_key="Electricity_retail_market_runes_______only_net_metering_and_net_billing_sections_20250731",
        title="Exctract about net metering and net billing from the 'Electricity Retail Market Rules'",
        issuer="GNERC",
        language="ka",
        source_url=None,
        storage_path=None,
        logical_key="Exctract_about_net_metering_and_net_billing_from_the_Electricity_Retail_Market_Rules'",
        published_date="2020-08-14",
        effective_date="2020-08-14",
        effective_end_date=None,
        version_label="2025-07-31",
        is_latest=True,
        is_active=True,
        abolished=False,
        supersedes_document_id=None,
        metadata={
            "country": "georgia",
            "city": "tbilisi",
            "resolution_number": "47",
            "annex_number": "NA",
            "notes": "Excerpt focused on net metering and net billing",
        },
    )

    topics = [
        "net_metering",
        "net_billing",
        "microgeneration",
        "consumer_engagement",
        "renewable_energy_integration",
        "renewable_energy_communities",

    ]

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
