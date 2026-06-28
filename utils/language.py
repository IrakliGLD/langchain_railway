"""
Language detection and instruction generation.

Supports Georgian, Russian, and English language detection
based on Unicode character ranges.
"""


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    Supports:
    - Georgian (ka): Unicode range \u10a0-\u10ff
    - Russian (ru): Unicode range \u0400-\u04ff
    - English (en): Default fallback

    Args:
        text: Input text to analyze

    Returns:
        Language code: 'ka' for Georgian, 'ru' for Russian, 'en' for English

    Examples:
        >>> detect_language("What was the price?")
        'en'
        >>> detect_language("რა იყო ფასი?")
        'ka'
        >>> detect_language("Какая была цена?")
        'ru'
    """
    # Georgian unicode range check
    if any('\u10a0' <= char <= '\u10ff' for char in text):
        return "ka"

    # Russian/Cyrillic unicode range check
    if any('\u0400' <= char <= '\u04ff' for char in text):
        return "ru"

    # Default to English
    return "en"


def get_language_instruction(lang_code: str) -> str:
    """
    Get instruction for LLM to respond in the detected language.

    Args:
        lang_code: Language code ('ka', 'ru', or 'en')

    Returns:
        Language-specific instruction string for LLM prompt

    Examples:
        >>> get_language_instruction('ka')
        'IMPORTANT: Respond in Georgian language (ქართული ენა)...'
        >>> get_language_instruction('en')
        'Respond in English.'
    """
    language_instructions = {
        "ka": "IMPORTANT: Respond in Georgian language (ქართული ენა). Use Georgian characters and natural Georgian phrasing.",
        "ru": "IMPORTANT: Respond in Russian language (русский язык). Use Cyrillic characters and natural Russian phrasing.",
        "en": "Respond in English."
    }
    return language_instructions.get(lang_code, language_instructions["en"])


def get_grounding_fallback_message(lang_code: str) -> str:
    """Localized conservative message shown when the grounding guardrail rejects
    a generated answer. Returned in the user's language so a non-English query
    doesn't get an English non-answer (default English for unknown codes).

    Args:
        lang_code: Language code ('ka', 'ru', or 'en').
    """
    messages = {
        "ka": (
            "მოწოდებული მონაცემების საფუძველზე ვერ მოვახერხე დეტალური "
            "პასუხის სრულად დასაბუთება. გთხოვთ, დააზუსტოთ კითხვა ან "
            "შეავიწროვოთ პერიოდი უფრო ზუსტი, დასაბუთებული პასუხისთვის."
        ),
        "ru": (
            "Не удалось полностью обосновать развёрнутый ответ на основе "
            "предоставленных данных. Уточните запрос или сузьте период для "
            "более точного обоснованного ответа."
        ),
        "en": (
            "I could not fully ground a detailed narrative from the provided "
            "data preview. Please refine the query or narrow the period for a "
            "more precise grounded answer."
        ),
    }
    return messages.get(lang_code, messages["en"])
