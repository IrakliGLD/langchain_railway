from pathlib import Path

from contracts.vector_knowledge import DocumentRegistration
from knowledge.vector_ingestion import VectorKnowledgeIngestor


def main() -> None:
    # Replace this with your local .md or .txt file path.
    text_path = Path(
        r"D:\Enaiapp\langchain_railway\docs_to_ingest\law_on_energy_and_water_supply.md"
    )
    text_content = text_path.read_text(encoding="utf-8")

    document = DocumentRegistration(
        source_key="Law_of_Georgia_on_Energy_and_water_supply_2025_electricity_part",
        title="Law of Georgia on Energy and Water Supply",
        document_type="regulation",
        issuer="Parliament of Georgia",
        language="ka",
        source_url=None,
        storage_path=None,
        logical_key="Law_of_Georgia_on_Energy_and_water_supply",
        published_date="2019-12-27",
        effective_date="2019-12-27",
        effective_end_date=None,
        version_label="2025-06-12",
        is_latest=True,
        is_active=True,
        abolished=False,
        supersedes_document_id=None,
        metadata={
            "country": "georgia",
            "city": "tbilisi",
            "resolution_number": "5646-რს",
            "annex_number": "NA",
            "notes": "Excerpt focused on electricity",
        },
    )

    topics = [
        "Georgian_electricity_sector_market_participants",
        "market_structure",
        "general_direction_of_the_electricity_sector",
        "Law_general_definitions",
        "responsibilities_of_market_participants",
        "functions_of_market_participants",
        "electricity_policy_making",
        "sanctioning_mechanisms",
        
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
