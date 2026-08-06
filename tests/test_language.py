"""Language resolution shared by report planning."""

import pytest

from utils.language import resolve_answer_language


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Assess electricity prices.", "en"),
        ("შეაფასე ელექტროენერგიის ფასები.", "ka"),
        ("Оцените цены на электроэнергию.", "ru"),
        ("Assess electricity prices. Answer in Georgian.", "ka"),
        ("შეაფასე ფასები. მიპასუხე ინგლისურად.", "en"),
        ("Оцените цены. Ответьте на грузинском.", "ka"),
        ("Analyze Russian market prices.", "en"),
    ],
)
def test_resolve_answer_language_uses_query_unless_explicitly_overridden(
    query: str,
    expected: str,
):
    assert resolve_answer_language(query) == expected
