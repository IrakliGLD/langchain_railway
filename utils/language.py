"""
Language detection and instruction generation.

Supports Georgian, Russian, and English language detection
based on Unicode character ranges.
"""

import re

_REQUESTED_LANGUAGE_PATTERNS = {
    "ka": (
        re.compile(
            r"\b(?:answer|respond|reply|write|prepare|provide|produce|generate)"
            r"(?:\s+(?:the|this|my|your|a))?"
            r"(?:\s+(?:answer|response|report|analysis|output))?"
            r"\s+in\s+georgian\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:მიპასუხე|უპასუხე|დაწერე|მომიმზადე|მოამზადე|წარმოადგინე)"
            r"[^.!?\n]{0,80}ქართულად",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:ответь|ответьте|напиши|напишите|подготовь|подготовьте|составь|составьте)"
            r"[^.!?\n]{0,80}(?:на грузинском|по-грузински)",
            re.IGNORECASE,
        ),
    ),
    "ru": (
        re.compile(
            r"\b(?:answer|respond|reply|write|prepare|provide|produce|generate)"
            r"(?:\s+(?:the|this|my|your|a))?"
            r"(?:\s+(?:answer|response|report|analysis|output))?"
            r"\s+in\s+russian\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:მიპასუხე|უპასუხე|დაწერე|მომიმზადე|მოამზადე|წარმოადგინე)"
            r"[^.!?\n]{0,80}რუსულად",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:ответь|ответьте|напиши|напишите|подготовь|подготовьте|составь|составьте)"
            r"[^.!?\n]{0,80}(?:на русском|по-русски)",
            re.IGNORECASE,
        ),
    ),
    "en": (
        re.compile(
            r"\b(?:answer|respond|reply|write|prepare|provide|produce|generate)"
            r"(?:\s+(?:the|this|my|your|a))?"
            r"(?:\s+(?:answer|response|report|analysis|output))?"
            r"\s+in\s+english\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:მიპასუხე|უპასუხე|დაწერე|მომიმზადე|მოამზადე|წარმოადგინე)"
            r"[^.!?\n]{0,80}ინგლისურად",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:ответь|ответьте|напиши|напишите|подготовь|подготовьте|составь|составьте)"
            r"[^.!?\n]{0,80}(?:на английском|по-английски)",
            re.IGNORECASE,
        ),
    ),
}


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


def resolve_answer_language(text: str) -> str:
    """Honor an explicit answer-language instruction, else detect the query."""

    for language_code, patterns in _REQUESTED_LANGUAGE_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            return language_code
    return detect_language(text)


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
            "ხელმისაწვდომ მტკიცებულებებში პასუხის სრულად დასაბუთება ვერ "
            "მოვახერხე. გთხოვთ, დააზუსტოთ, რომელი ფაქტის ან საკითხის "
            "შემოწმება გსურთ."
        ),
        "ru": (
            "Не удалось полностью обосновать ответ доступными доказательствами. "
            "Уточните, какой факт или аспект нужно проверить."
        ),
        "en": (
            "I could not fully ground the answer in the available evidence. "
            "Please clarify which fact or aspect you want checked."
        ),
    }
    return messages.get(lang_code, messages["en"])


def get_evidence_unavailable_message(lang_code: str) -> str:
    """Localized transparent message for a data request whose evidence could
    not be retrieved (P4.4, finding H12).

    Deliberately carries NO figures: a data-primary request whose SQL failed
    validation or relevance must not be dressed up as a plausible domain
    narrative with invented numbers. It states the limitation honestly and
    invites a refinement. Default English for unknown codes.

    Args:
        lang_code: Language code ('ka', 'ru', or 'en').
    """
    messages = {
        "ka": (
            "ამ კითხვისთვის საჭირო მონაცემების მოძიება ვერ მოხერხდა, "
            "ამიტომ რიცხობრივ პასუხს ვერ დავასაბუთებ. გთხოვთ, დააზუსტოთ "
            "მაჩვენებელი, პერიოდი ან ერთეული და სცადოთ თავიდან."
        ),
        "ru": (
            "Не удалось получить данные, необходимые для этого запроса, "
            "поэтому обоснованный числовой ответ невозможен. Пожалуйста, "
            "уточните показатель, период или объект и попробуйте снова."
        ),
        "en": (
            "I could not retrieve the data required for this request, so I "
            "cannot give a grounded numeric answer. Please refine the metric, "
            "period, or entity and try again."
        ),
    }
    return messages.get(lang_code, messages["en"])


def get_transient_failure_message(lang_code: str) -> str:
    """Localized, content-free response for a temporary provider failure."""
    messages = {
        "ka": (
            "მოთხოვნის დამუშავება დროებით ვერ დასრულდა. "
            "გთხოვთ, ცოტა ხანში ხელახლა სცადოთ."
        ),
        "ru": (
            "Обработку запроса временно не удалось завершить. "
            "Повторите попытку немного позже."
        ),
        "en": (
            "The request could not be completed because the service is "
            "temporarily unavailable. Please try again after a short wait."
        ),
    }
    return messages.get(lang_code, messages["en"])
