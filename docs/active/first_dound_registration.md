# First Document Registration

This file gives a starter `DocumentRegistration` block and topic list for each markdown source currently under `docs_to_ingest/`.

Assumptions:
- `document_type="regulation"` is used for all three because they are normative/legal-rule style documents.
- `is_latest=True` and `is_active=True` are starter defaults. Change them if you ingest an older superseded version or a document that is no longer in force.
- `source_url` and `storage_path` are left `None` because the current local-ingestion flow reads from local markdown files.

## 1. `docs_to_ingest/market_concept_design.md`

```python
document = DocumentRegistration(
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

topics = [
    "market_design",
    "electricity_market_transitory_model",
    "electricity_market_target_model",
    "market_transition",
    "day_ahead_market",
    "intraday_market",
    "balancing_market",
    "capacity_market",
    "cross_border_trade",
    "deregulation_plan",
]
```

Why these topics:
- The concept document is a high-level market architecture text.
- It covers the transition from the transitory model to the target model.
- It also touches day-ahead, intraday, balancing, capacity, and cross-border design principles.

## 2. `docs_to_ingest/transitory_market_rules.md`

```python
document = DocumentRegistration(
    source_key="transitory_capacity_market_rules_aug2006",
    title="Electricity (Capacity) Market Rules",
    document_type="regulation",
    issuer="Ministry of Energy of Georgia",
    language="ka",
    source_url=None,
    storage_path=None,
    logical_key="transitory_capacity_market_rules",
    published_date="2006-08-30",
    effective_date="2006-08-30",
    effective_end_date=None,
    version_label="2006-08-30",
    is_latest=True,
    is_active=True,
    abolished=False,
    supersedes_document_id=None,
    metadata={
        "country": "georgia",
        "city": "tbilisi",
        "order_number": "77",
        "notes": "Rules applied during the transitory electricity market model period",
    },
)

topics = [
    "electricity_balancing_transitory_model",
    "capacity_market",
    "balancing_market",
    "balancing_energy_buyers",
    "wholesale_market_participants",
    "electricity_export",
    "electricity_import",
    "cross_border_trade",
    "eligible_participants",
    "market_rules",
    "balancing_electricity",
    "balancing_electricity_buyers",
    "balancing_electricity_sellers",
]
```

Why these topics:
- This document is the strongest candidate for transitory-model questions.
- It is also the better fit for balancing-participant, import/export, and capacity-market questions than the day-ahead exchange rules.
- The added participant/export topics should give retrieval more precise hooks than generic `market_structure`.

## 3. `docs_to_ingest/gnerc_electricity_market_rules_excerpt_ka.md`

```python
document = DocumentRegistration(
    source_key="electricity_day_ahead_intraday_rules_aug2020",
    title="Electricity Day-Ahead and Intraday Market Rules",
    document_type="regulation",
    issuer="GNERC",
    language="ka",
    source_url=None,
    storage_path=None,
    logical_key="day_ahead_intraday_market_rules",
    published_date="2020-08-11",
    effective_date="2020-08-11",
    effective_end_date=None,
    version_label="2020-08-11",
    is_latest=True,
    is_active=True,
    abolished=False,
    supersedes_document_id=None,
    metadata={
        "country": "georgia",
        "city": "tbilisi",
        "resolution_number": "46",
        "annex_number": "1",
        "notes": "Excerpt focused on electricity day-ahead and intraday exchange rules",
    },
)

topics = [
    "day_ahead_market",
    "intraday_market",
    "exchange_rules",
    "exchange_participation",
    "participant_registration",
    "exchange_price_formation",
    "financial_clearing",
    "market_operator_procedures",
    "market_structure",
]
```

Why these topics:
- This document should remain the strongest source for exchange registration, trading process, and day-ahead/intraday price-formation questions.
- It should not be the only document used for export, balancing-buyer, or transitory-model questions, so its topics stay exchange-focused.

## Suggested Ingestion Order

1. `market_concept_design.md`
2. `transitory_market_rules.md`
3. `gnerc_electricity_market_rules_excerpt_ka.md`

That order gives the retriever:
- one conceptual market-design document,
- one transitory/capacity/balancing rule document,
- one exchange/day-ahead/intraday rule document.

## Important Note On Topics

If a question is likely to ask:
- who can trade during the transitory model,
- who buys balancing electricity,
- what procedures apply to import/export,

then the document topics must include those concepts explicitly. Broad tags like `market_structure` alone are not enough to reliably pull the right source when multiple market-rule documents are present.
