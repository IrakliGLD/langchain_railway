"""Heuristic query classification.

Rule-based ``(query) -> label`` helpers extracted from ``core/llm.py`` (P0-1,
architecture-audit 2026-06-30). Dependency-free (stdlib ``re`` only) so leaf
consumers such as ``visualization/`` can depend on it downward without importing
the LLM hub. ``core.llm`` re-exports both functions for backward compatibility.
"""
import re


def classify_query_type(user_query: str) -> str:
    """
    Classify query into specific types for better chart/answer decisions.

    Returns:
        - "single_value": One specific value requested
        - "list": Enumeration/listing of items
        - "comparison": Comparing two or more things
        - "trend": Time series analysis
        - "table": Detailed data display
        - "unknown": Cannot determine type
    """
    query_lower = user_query.lower()
    regulation_procedure_patterns = [
        "who is eligible", "who can participate", "who may participate",
        "who can register", "who may register", "what documents are required",
        "what documents do i need", "what are the requirements",
        "requirements for registration", "registration process",
        "how to register", "how can i register", "what is the procedure",
        "licensing procedure", "participation conditions", "deadline for registration",
    ]
    regulation_data_patterns = [
        "how many", "count", "total", "number of", "statistics", "breakdown",
    ]

    # Single value indicators (highest priority)
    if any(p in query_lower for p in [
        "what is the", "what was the", "how much is", "how much was",
        "რა არის", "რა იყო", "сколько"
    ]) and any(p in query_lower for p in [
        "in june", "in 2024", "for june", "for 2024", "latest", "last month",
        "during", "იუნის", "წელს", "в июне", "в 2024",
        "jan ", "feb ", "mar ", "apr ", "may ", "jun ",
        "jul ", "aug ", "sep ", "oct ", "nov ", "dec ",
        "january", "february", "march", "april", "june", "july",
        "august", "september", "october", "november", "december",
    ]):
        return "single_value"

    # Regulatory procedure indicators should win over the broad
    # "what are the" list fallback.
    if any(p in query_lower for p in regulation_procedure_patterns):
        if not any(p in query_lower for p in regulation_data_patterns):
            return "regulatory_procedure"

    # List indicators
    if any(p in query_lower for p in [
        "list all", "show all", "enumerate", "which entities",
        "what are the", "name all", "give me all entities",
        "ჩამოთვალე", "ყველა", "перечисли", "какие"
    ]):
        return "list"

    # Comparison indicators
    if any(p in query_lower for p in [
        "compare", " vs ", " vs. ", "versus", "difference between",
        "compared to", "შედარება", "შედარებით", "сравни", "по сравнению"
    ]):
        return "comparison"

    # Trend indicators
    if any(p in query_lower for p in [
        "trend", "over time", "dynamics", "evolution", "change over",
        "from 20", "between 20", "since 20",
        "динамика", "ტენდენცია", "დინამიკა"
    ]):
        return "trend"
    if (
        re.search(r"\b(monthly|daily|yearly|weekly|quarterly)\b", query_lower)
        and re.search(r"\b\d{4}\b", query_lower)
    ):
        return "trend"

    # Table indicators
    if any(p in query_lower for p in [
        "show me all", "give me all", "detailed", "breakdown", "show data",
        "table", "tabular"
    ]):
        return "table"

    # Date-range patterns → trend (catches "jan 2024 to dec 2024", "during 2024")
    if re.search(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\s+to\s+", query_lower):
        return "trend"
    if "during" in query_lower and re.search(r"\d{4}", query_lower):
        return "trend"

    return "unknown"


def get_query_focus(user_query: str) -> str:
    """
    Determine the main focus of the query to filter domain knowledge appropriately.

    Returns:
        - "cpi": Consumer Price Index queries
        - "tariff": Tariff-focused queries
        - "generation": Electricity generation queries
        - "regulation": Registration, eligibility, procedure queries
        - "energy_security": Energy security, import dependence queries
        - "balancing": Balancing market/price queries
        - "trade": Import/export/trade queries
        - "general": Cannot determine or multiple focuses
    """
    query_lower = user_query.lower()

    # CPI focus (check first - very specific)
    if any(k in query_lower for k in ["cpi", "inflation", "consumer price index", "ინფლაცია"]):
        return "cpi"

    # Tariff focus (check before balancing - tariff is more specific)
    if any(k in query_lower for k in ["tariff", "ტარიფი", "тариф"]) and \
       not any(k in query_lower for k in ["balancing", "საბალანსო", "баланс"]):
        return "tariff"

    # Generation focus
    if any(k in query_lower for k in ["generation", "generated", "produce", "გენერაცია", "генерация", "производство"]) and \
       not any(k in query_lower for k in ["price", "ფასი", "цена"]):
        return "generation"

    # Regulation / procedure focus (check before trade — registration queries
    # about exchange participation, eligibility, etc. should get regulation
    # guidance, not trade guidance).  Exclude data-intent queries that happen
    # to mention generic tokens like "participant" or "license".
    _data_intent = any(k in query_lower for k in [
        "how many", "count", "total", "number of", "breakdown", "statistics",
        "რამდენი", "სულ", "сколько", "количество",
    ])
    if not _data_intent and any(k in query_lower for k in [
        "register", "registration", "eligible", "eligibility",
        "procedure", "requirement", "participant", "license", "licence",
        "რეგისტრაცია", "მონაწილე", "регистрация", "участник",
    ]):
        return "regulation"

    # Energy security focus (check before trade — "import dependence" is security, not trade)
    if any(k in query_lower for k in [
        "energy security", "უსაფრთხოება", "энергобезопасность",
        "import dependence", "import reliance", "self-sufficient",
        "იმპორტზე დამოკიდებულება",
    ]):
        return "energy_security"

    # Trade focus
    if any(k in query_lower for k in ["import", "export", "trade", "იმპორტი", "ექსპორტი", "импорт", "экспорт"]) and \
       not any(k in query_lower for k in ["price", "ფასი", "цена"]):
        return "trade"

    # Balancing focus (check last - most common)
    if any(k in query_lower for k in ["balancing", "p_bal", "საბალანსო", "баланс", "balance market"]):
        return "balancing"

    return "general"
