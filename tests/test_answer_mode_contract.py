from dataclasses import fields

from models import AnswerMode, QueryContext


def test_answer_mode_wire_values_are_stable() -> None:
    assert [mode.value for mode in AnswerMode] == [
        "brief",
        "standard",
        "report",
    ]


def test_query_context_defaults_to_standard_answer_mode() -> None:
    assert QueryContext(query="Explain the balancing market.").answer_mode == "standard"


def test_answer_mode_is_appended_without_shifting_existing_context_positions() -> None:
    assert fields(QueryContext)[-1].name == "answer_mode"
