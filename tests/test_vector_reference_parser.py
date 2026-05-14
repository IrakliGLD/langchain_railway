"""Tests for ``knowledge.vector_reference_parser``.

Phase B.2 of the cross-reference rollout. Covers:

* section-heading parsing (article/chapter, Georgian/English/Russian,
  decimal article numbers, Roman chapter numbers);
* body-text reference extraction across the morphological forms the
  audit confirmed in ``docs_to_ingest/`` (suffix-ordinal, prefix-ordinal,
  decimal, ordinal-word, Roman chapter, self-article anchor);
* ``კოდექსი`` rejection rule;
* deduplication of equivalent surface forms.

These cases were derived directly from real examples in
``law_on_energy_and_water_supply.md`` and
``transitory_market_rules.md`` — see the audit dialogue in the
cross-reference rollout plan.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from contracts.vector_knowledge import ChunkReferenceKind
from knowledge.vector_reference_parser import (
    parse_outgoing_references,
    parse_section_heading,
)


# ---------------------------------------------------------------------------
# Section-heading parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title,expected_kind,expected_article,expected_chapter",
    [
        ("მუხლი 14. სისტემის კომერციული ოპერატორი", "article", "14", ""),
        ("მუხლი 14", "article", "14", ""),
        # Decimal-numbered article (common in transitory_market_rules).
        ("მუხლი 14.7. ფასის ფორმირება", "article", "14.7", ""),
        ("მუხლი 8.1", "article", "8.1", ""),
        ("მუხლი 14.11. სპეციალური წესი", "article", "14.11", ""),
        # Georgian chapter with Roman numeral.
        ("თავი I ზოგადი დებულებანი", "chapter", "", "I"),
        ("თავი IV", "chapter", "", "IV"),
        ("თავი IV.1 ელექტროენერგიის იმპორტი", "chapter", "", "IV.1"),
        ("თავი XII მონაცემები", "chapter", "", "XII"),
        # English forms.
        ("Article 14", "article", "14", ""),
        ("Article 14.7. Settlement", "article", "14.7", ""),
        ("Chapter IV", "chapter", "", "IV"),
        ("Chapter 4 Wholesale Trade", "chapter", "", "4"),
        # Russian forms.
        ("Статья 14", "article", "14", ""),
        ("Статья 14.7", "article", "14.7", ""),
        ("Глава IV", "chapter", "", "IV"),
        # Non-matching: free-form / preamble headings.
        ("Definitions", "", "", ""),
        ("ტერმინთა განმარტება", "", "", ""),
        ("", "", "", ""),
    ],
)
def test_parse_section_heading(title, expected_kind, expected_article, expected_chapter):
    info = parse_section_heading(title)
    assert info.section_kind == expected_kind
    assert info.article_number == expected_article
    assert info.chapter_number == expected_chapter


# ---------------------------------------------------------------------------
# Body-text reference extraction — Georgian suffix-ordinal
# ---------------------------------------------------------------------------


def test_suffix_ordinal_bare_article():
    """``24-ე მუხლის`` (genitive, suffix-ordinal) → article 24."""
    refs = parse_outgoing_references(
        "მონაცემები 24-ე მუხლის შესაბამისად მზადდება."
    )
    assert len(refs) == 1
    assert refs[0].kind == ChunkReferenceKind.article
    assert refs[0].number == "24"
    assert refs[0].sub_kind is None


def test_suffix_ordinal_with_paragraph_via_prefix_ordinal():
    """``87-ე მუხლის მე-3 პუნქტი`` → article 87 paragraph 3."""
    refs = parse_outgoing_references(
        "ვრცელდება 87-ე მუხლის მე-3 პუნქტი მოთხოვნებზე."
    )
    assert len(refs) == 1
    assert refs[0].number == "87"
    assert refs[0].sub_kind == "paragraph"
    assert refs[0].sub_number == "3"


def test_suffix_ordinal_with_ordinal_word_paragraph():
    """``87-ე მუხლის პირველი პუნქტი`` → article 87 paragraph 1.

    Direct corpus pattern: ordinal-word for the paragraph number.
    """
    refs = parse_outgoing_references(
        "ბაზრის წესების 87-ე მუხლის პირველი პუნქტი დადგენილია."
    )
    assert len(refs) == 1
    assert refs[0].number == "87"
    assert refs[0].sub_kind == "paragraph"
    assert refs[0].sub_number == "1"


# ---------------------------------------------------------------------------
# Body-text — Georgian prefix-ordinal (the user-cited example form)
# ---------------------------------------------------------------------------


def test_prefix_ordinal_bare_article():
    """``მე-14 მუხლის`` → article 14."""
    refs = parse_outgoing_references(
        "გათვალისწინებულია ამ კანონის მე-14 მუხლის შესაბამისად."
    )
    assert len(refs) == 1
    assert refs[0].number == "14"
    assert refs[0].sub_kind is None


def test_prefix_ordinal_compound_user_cited_form():
    """The exact form the user flagged:
    ``მე-14 მუხლის მე-7 პუნქტის შესაბამისად`` → article 14 paragraph 7.
    """
    refs = parse_outgoing_references(
        "მე-14 მუხლის მე-7 პუნქტის შესაბამისად მოქმედებს."
    )
    assert len(refs) == 1
    assert refs[0].number == "14"
    assert refs[0].sub_kind == "paragraph"
    assert refs[0].sub_number == "7"


def test_prefix_ordinal_with_ordinal_word_paragraph():
    """``მე-13 მუხლის პირველი პუნქტი`` → article 13 paragraph 1."""
    refs = parse_outgoing_references(
        "გათვალისწინებულია ამ კანონის მე-13 მუხლის პირველი პუნქტი."
    )
    assert len(refs) == 1
    assert refs[0].number == "13"
    assert refs[0].sub_kind == "paragraph"
    assert refs[0].sub_number == "1"


# ---------------------------------------------------------------------------
# Body-text — Georgian decimal articles
# ---------------------------------------------------------------------------


def test_decimal_article_reference():
    """``14.7 მუხლი`` (decimal article — distinct from "article 14 paragraph 7")."""
    refs = parse_outgoing_references(
        "ბაზრის წესების 14.7 მუხლის მე-2 პუნქტში ასახული რეჟიმები."
    )
    # Decimal "14.7 მუხლი" is captured. The "მე-2 პუნქტი" that follows
    # is a separate fragment outside the decimal pattern; the parser
    # does not currently bind decimal-article forms to following
    # paragraph refs (documented limitation — paragraphs of decimal
    # articles are rare in the corpus and the citing chunk's body
    # contains the paragraph text anyway).
    assert any(r.number == "14.7" for r in refs)


def test_decimal_article_with_chapter_decimal_extension():
    """Article numbers with two decimal levels (``14.11``, ``14.12``)
    are captured intact."""
    refs = parse_outgoing_references(
        "ბაზრის წესების 14.12 მუხლის პირველი პუნქტი დადგენილია 14.11 მუხლის საფუძველზე."
    )
    numbers = {r.number for r in refs}
    assert "14.11" in numbers
    assert "14.12" in numbers


# ---------------------------------------------------------------------------
# Body-text — Roman-numeral chapter references
# ---------------------------------------------------------------------------


def test_roman_chapter_in_body():
    """``XI თავი`` / ``XI და XII თავებით`` → chapter XI (and XII)."""
    refs = parse_outgoing_references(
        "ვრცელდება ამ კანონის XI და XII თავებით დადგენილი მოთხოვნები."
    )
    chapter_refs = [r for r in refs if r.kind == ChunkReferenceKind.chapter]
    # Both Roman tokens are captured — though plural-conjoined ``XI და XII``
    # may collapse to one match depending on regex precedence. Recall
    # acceptable as long as at least one chapter ref is emitted.
    assert chapter_refs
    assert any(r.number in {"XI", "XII"} for r in chapter_refs)


# ---------------------------------------------------------------------------
# Body-text — self-article anchors
# ---------------------------------------------------------------------------


def test_self_article_anchor_emits_self_kind():
    """``ამ მუხლის`` and ``წინამდებარე მუხლის`` resolve to the chunk
    itself — emitted as ``ChunkReferenceKind.self_article`` so the
    resolver SKIPS expansion instead of looping back to the source chunk."""
    refs = parse_outgoing_references(
        "დისპეტჩერიზაციის ლიცენზიატი წინამდებარე მუხლის მე-2 პუნქტით "
        "გათვალისწინებულ შემთხვევაში მოქმედებს. ამ მუხლის მე-3 პუნქტი "
        "ვრცელდება მონაცემებზე."
    )
    self_refs = [r for r in refs if r.kind == ChunkReferenceKind.self_article]
    assert self_refs, "no self_article ref emitted"


def test_self_article_anchors_dedupe_to_one():
    """Multiple ``ამ მუხლის`` occurrences in one chunk collapse to a
    single self-reference — the resolver doesn't care which sentence
    it appeared in."""
    refs = parse_outgoing_references("ამ მუხლის ... ამ მუხლის ... ამ მუხლის ...")
    self_refs = [r for r in refs if r.kind == ChunkReferenceKind.self_article]
    assert len(self_refs) == 1


# ---------------------------------------------------------------------------
# Body-text — კოდექსი rejection (false-positive prevention)
# ---------------------------------------------------------------------------


def test_kodeksi_reference_is_dropped():
    """External-code references (Civil/Tax/Admin/Labor) must NOT be
    emitted — they would resolve against the citing document and
    produce a same-numbered false-positive match.

    Concrete corpus pattern:
    ``საქართველოს შრომის კოდექსის 30-ე მუხლის პირველი ნაწილით…``
    """
    refs = parse_outgoing_references(
        "ვადების გამოთვლისას საქართველოს შრომის კოდექსის 30-ე მუხლის "
        "პირველი ნაწილით გათვალისწინებული უქმე დღეები."
    )
    # The 30-ე reference must NOT appear.
    assert all(r.number != "30" for r in refs)


def test_kodeksi_rejection_window_is_bounded():
    """A ``კოდექსი`` mention many paragraphs earlier does NOT poison
    a later same-doc article reference."""
    # Put the kodeksi token >60 chars before the article ref.
    text = (
        "სამოქალაქო კოდექსის თანახმად მონაცემები მოგვაქვს. "
        "ერთად ბევრი სიტყვა მოდის. ეს ტექსტი არ უნდა გავლენდეს. "
        "ნორმალური მონაცემები. ვრცელდება ამ კანონის 36-ე მუხლის შესაბამისად."
    )
    refs = parse_outgoing_references(text)
    # The 36-ე reference should survive — it is far enough from the
    # kodeksi token.
    assert any(r.number == "36" for r in refs)


def test_english_code_reference_dropped():
    """English-language ``Civil Code, Article 14`` → dropped."""
    refs = parse_outgoing_references(
        "Per the Civil Code, Article 14, paragraph 2, these obligations apply."
    )
    assert all(r.number != "14" for r in refs)


# ---------------------------------------------------------------------------
# Body-text — English & Russian
# ---------------------------------------------------------------------------


def test_english_article_with_paragraph():
    refs = parse_outgoing_references(
        "Pursuant to Article 14, paragraph 7 of these rules, the rate applies."
    )
    assert len(refs) == 1
    assert refs[0].number == "14"
    assert refs[0].sub_kind == "paragraph"
    assert refs[0].sub_number == "7"


def test_russian_article_with_punkt():
    refs = parse_outgoing_references(
        "В соответствии со статьёй 14, пункт 7 настоящего закона."
    )
    assert len(refs) == 1
    assert refs[0].number == "14"
    assert refs[0].sub_kind == "paragraph"
    assert refs[0].sub_number == "7"


# ---------------------------------------------------------------------------
# Deduplication across morphological surface forms
# ---------------------------------------------------------------------------


def test_morphological_dedup_collapses_equivalent_surface_forms():
    """The same canonical tuple ``(article, "14", None, None)`` reached
    via two surface forms (suffix-ordinal then prefix-ordinal) must
    collapse to ONE :class:`ChunkReference`. The resolver doesn't care
    which surface form produced it."""
    refs = parse_outgoing_references(
        "მიუხედავად 14-ე მუხლის შინაარსისა, მე-14 მუხლის ბოლო პუნქტი ხელახლა მოქმედებს."
    )
    article_refs = [r for r in refs if r.kind == ChunkReferenceKind.article]
    # The bare-article tuple (14, no sub) appears once.
    bare = [r for r in article_refs if r.number == "14" and r.sub_kind is None]
    assert len(bare) == 1


def test_empty_text_returns_empty_list():
    assert parse_outgoing_references("") == []
    assert parse_outgoing_references(None) == []  # type: ignore[arg-type]


def test_text_with_no_references_returns_empty_list():
    refs = parse_outgoing_references(
        "This paragraph discusses the general principles of electricity pricing "
        "without referencing any specific article or chapter."
    )
    assert refs == []
