"""Catalog-level regressions for analyzer contract wording."""

import pathlib
import re

from contracts.question_analysis import KnowledgeTopicName
from contracts.question_analysis_catalogs import (
    QUESTION_ANALYSIS_ANSWER_KIND_GUIDE,
    QUESTION_ANALYSIS_QUERY_TYPE_GUIDE,
    QUESTION_ANALYSIS_TOPIC_CATALOG,
)

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ``sql_examples`` is a KnowledgeTopicName and has a markdown file, but it is
# deliberately absent from the analyzer's topic-catalog mirror: it carries
# few-shot SQL patterns, not a domain the analyzer should ever *select*.
_TOPICS_NOT_IN_ANALYZER_MIRROR = {"sql_examples"}


def _entry(entries, name: str) -> dict:
    for entry in entries:
        if entry["name"] == name:
            return entry
    raise AssertionError(f"missing entry: {name}")


def test_answer_kind_guide_distinguishes_historical_trend_from_forecast():
    timeseries = _entry(QUESTION_ANALYSIS_ANSWER_KIND_GUIDE, "timeseries")["use_for"].lower()
    forecast = _entry(QUESTION_ANALYSIS_ANSWER_KIND_GUIDE, "forecast")["use_for"].lower()

    assert "historical trend" in timeseries
    assert "historical trend" in forecast
    assert "not forecast" in forecast


def test_query_type_guide_distinguishes_historical_trend_from_forecast():
    data_retrieval = _entry(QUESTION_ANALYSIS_QUERY_TYPE_GUIDE, "data_retrieval")["use_for"].lower()
    forecast = _entry(QUESTION_ANALYSIS_QUERY_TYPE_GUIDE, "forecast")["use_for"].lower()

    assert "trend summaries" in data_retrieval
    assert "not forecast" in forecast


class TestTopicRegistryAlignment:
    """A knowledge topic lives in four parallel registries.

    Adding one to some but not all is silent: the analyzer can emit a topic
    with no catalog guidance, or a markdown file can exist that nothing ever
    routes to.  These tests make the drift loud.
    """

    def test_enum_and_catalog_agree(self):
        enum_names = {topic.value for topic in KnowledgeTopicName}
        catalog_names = {entry["name"] for entry in QUESTION_ANALYSIS_TOPIC_CATALOG}

        assert enum_names - catalog_names == set(), "KnowledgeTopicName values with no catalog entry"
        assert catalog_names - enum_names == set(), "Catalog entries that are not KnowledgeTopicName values"

    def test_every_topic_has_a_knowledge_file(self):
        stems = {path.stem for path in (_ROOT / "knowledge").glob("*.md")}
        missing = sorted({topic.value for topic in KnowledgeTopicName} - stems)

        assert not missing, f"KnowledgeTopicName values with no knowledge/<name>.md: {missing}"

    def test_analyzer_skill_mirror_lists_every_topic(self):
        mirror = (_ROOT / "skills" / "question-analyzer" / "references" / "topic-catalog.md")
        headings = set(re.findall(r"^## (\w+)", mirror.read_text(encoding="utf-8"), re.MULTILINE))

        expected = {topic.value for topic in KnowledgeTopicName} - _TOPICS_NOT_IN_ANALYZER_MIRROR
        assert not expected - headings, f"Topics missing from the analyzer skill mirror: {sorted(expected - headings)}"


class TestTariffTopicDisambiguation:
    """``tariffs`` (generation side) and ``network_supply_tariffs`` (retail side)
    are adjacent topics.  Without explicit cross-references the analyzer routes
    every tariff question to whichever it saw first, answering half the subject.

    Same pattern as cross_border_trade <-> cross_border_capacity.
    """

    def test_each_tariff_topic_points_at_the_other(self):
        generation = _entry(QUESTION_ANALYSIS_TOPIC_CATALOG, "tariffs")["use_for"]
        retail = _entry(QUESTION_ANALYSIS_TOPIC_CATALOG, "network_supply_tariffs")["use_for"]

        assert "network_supply_tariffs" in generation
        assert "tariffs" in retail

    def test_the_two_topics_are_scoped_by_side_of_the_market(self):
        generation = _entry(QUESTION_ANALYSIS_TOPIC_CATALOG, "tariffs")["use_for"]
        retail = _entry(QUESTION_ANALYSIS_TOPIC_CATALOG, "network_supply_tariffs")["use_for"]

        assert "GEL/MWh" in generation and "GENERATION" in generation
        assert "GEL/kWh" in retail and "RETAIL" in retail


class TestKnowledgeTopicRouting:
    """TOPIC_MAP keyword routing for the retail-tariff topic."""

    def test_end_user_questions_reach_the_retail_topic(self):
        from knowledge import TOPIC_MAP

        queries = [
            "what is the end-user electricity tariff",
            "how much does a household pay per kwh",
            "breakdown of the distribution tariff",
            "what is included in the supply tariff",
            "telasi network tariff",
        ]
        for query in queries:
            matched = {
                stem
                for keyword, stems in TOPIC_MAP.items()
                if keyword in query.lower()
                for stem in stems
            }
            assert "network_supply_tariffs" in matched, f"{query!r} did not route to the retail topic"

    def test_bare_tariff_keyword_reaches_both_tariff_topics(self):
        """A bare 'tariff' question must not silently get only the generation file."""
        from knowledge import TOPIC_MAP

        matched = {
            stem
            for keyword, stems in TOPIC_MAP.items()
            if keyword in "how did the tariff change last year"
            for stem in stems
        }
        assert {"tariffs", "network_supply_tariffs"} <= matched

    def test_no_topic_keyword_is_a_substring_of_a_common_english_word(self):
        """TOPIC_MAP matches by bare substring, so short keys fire inside words.

        'eps' (a real supply-company code) would match "steps"/"keeps" and is
        deliberately excluded; this guards the next such addition.
        """
        from knowledge import TOPIC_MAP

        decoys = ["steps", "keeps", "epsilon", "solar", "telephone", "development", "response"]
        offenders = [
            keyword
            for keyword in TOPIC_MAP
            if keyword.isascii() and len(keyword) <= 4 and any(keyword in word for word in decoys)
        ]
        assert not offenders, f"TOPIC_MAP keys that fire inside unrelated words: {offenders}"
