"""Cross-reference parser for vector-knowledge chunking.

Phase B.2 of the cross-reference rollout. Two responsibilities:

1. :func:`parse_section_heading` — extract structural metadata
   (``section_kind``, ``article_number``, ``chapter_number``) from a chunk's
   heading text. Used by the chunker to populate ``ChunkIngestRecord``'s
   canonical fields at ingest time.

2. :func:`parse_outgoing_references` — extract outbound cross-references
   from chunk body text and return them as a list of
   :class:`ChunkReference` tuples. Used by the chunker to populate
   ``ChunkIngestRecord.outgoing_refs``; the retrieval-side resolver
   (Phase B.3) consumes the same tuples for one-hop expansion.

Catalog scope and rejection policy were validated against the actual
corpus in ``docs_to_ingest/``. See
``docs/active/VECTOR_KNOWLEDGE_ROLLOUT.md`` for the rollout plan.

Patterns covered (Georgian unless noted):

- Suffix-ordinal: ``N-ე მუხლი[case-suffix]`` (e.g. ``24-ე მუხლის``)
- Prefix-ordinal: ``მე-N მუხლი[case-suffix]`` (e.g. ``მე-14 მუხლის``)
- Decimal article: ``N.M მუხლი`` / ``N.M.K მუხლი`` (e.g. ``14.7 მუხლი``)
- Compound article+paragraph: any article form followed by a paragraph
  reference (ordinal-word, ``მე-N``, or ``N-ე``)
- Roman chapter: ``თავი ROMAN`` (e.g. ``თავი XI``)
- Self-article: ``ამ მუხლის`` / ``წინამდებარე მუხლის`` → emitted as
  ``ChunkReferenceKind.self_article`` so the resolver skips expansion
- English/Russian forms for foreign-language regulations

Rejection rule:
- ``კოდექსი`` references within ~60 chars before an article ref are
  dropped — the corpus's external code references (Civil, Tax, Admin,
  Labor) are not in our ingestion set and would generate false
  positives if resolved against the citing document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from contracts.vector_knowledge import ChunkReference, ChunkReferenceKind

# ---------------------------------------------------------------------------
# Canonical-number helpers
# ---------------------------------------------------------------------------

# Georgian ordinal-word → integer string. Used in body-text parsing to
# normalise ``პირველი პუნქტი`` ("first paragraph") to ``paragraph=1``.
# Bounded list — paragraph numbers above ~12 are written numerically in
# the corpus.
_GEORGIAN_ORDINAL_WORDS: dict[str, str] = {
    "პირველი": "1",
    "მეორე": "2",
    "მესამე": "3",
    "მეოთხე": "4",
    "მეხუთე": "5",
    "მეექვსე": "6",
    "მეშვიდე": "7",
    "მერვე": "8",
    "მეცხრე": "9",
    "მეათე": "10",
    "მეთერთმეტე": "11",
    "მეთორმეტე": "12",
}

# Same idea for Russian, used by the foreign-language patterns. Bounded.
_RUSSIAN_ORDINAL_WORDS: dict[str, str] = {
    "первый": "1", "первая": "1", "первое": "1",
    "второй": "2", "вторая": "2", "второе": "2",
    "третий": "3", "третья": "3", "третье": "3",
    "четвёртый": "4", "четвертый": "4",
    "пятый": "5",
    "шестой": "6",
    "седьмой": "7",
    "восьмой": "8",
    "девятый": "9",
    "десятый": "10",
}


def _normalize_ordinal_word_ka(word: str) -> Optional[str]:
    return _GEORGIAN_ORDINAL_WORDS.get(str(word or "").strip().lower())


def _normalize_ordinal_word_ru(word: str) -> Optional[str]:
    return _RUSSIAN_ORDINAL_WORDS.get(str(word or "").strip().lower())


# Number forms we accept in body-text article references:
#   * ``14``       → ``14``        (plain integer)
#   * ``14(9)``    → ``14(9)``     (sub-article: "Article 14¹" Georgian notation)
#   * ``14.7``     → ``14.7``      (legacy decimal article — kept for forward compat)
#   * ``14.7.3``   → ``14.7.3``    (rare but valid legacy form)
#   * ``14(9).2``  → ``14(9).2``   (combined; unusual but accepted)
#
# Note: the ``M(N)`` parens notation is the canonical Georgian regulatory
# form. ``M.N`` is being migrated to ``M(N)`` across the corpus; both
# forms are accepted here for a smooth migration.
_NUMBER_DECIMAL = r"\d+(?:\(\d+\))?(?:\.\d+){0,2}"

# Paragraph-reference number forms (slice of ``_NUMBER_DECIMAL`` that
# excludes legacy decimal). Used in standalone paragraph references like
# ``1(1) პუნქტი`` where the number is the paragraph identifier itself.
_PARAGRAPH_NUM = r"\d+(?:\(\d+\))?"

# Georgian case-suffix character class on ``მუხლი``. Lists every suffix
# letter the corpus shows: ``-ის`` (gen), ``-ით`` (instr), ``-ში`` (loc),
# ``-ზე`` (postp), ``-სა`` (gen-conjoined), ``-ად`` (transformative),
# ``-დან`` (ablative), plus the bare nominative. Order in the class is
# irrelevant; presence matters.
_KA_NOUN_SUFFIX = r"[ისითზებაადან]*"


# ---------------------------------------------------------------------------
# Section-heading parser (chunker-side)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionHeadingInfo:
    section_kind: str = ""
    article_number: str = ""
    chapter_number: str = ""


# Roman-numeral set bounded at 30 (XXX). Chapters in the corpus rarely
# exceed XII; XXX leaves headroom without admitting nonsense like
# ``MMXIV``.
_ROMAN_NUMERAL = r"(?:XX[IVX]*|X[IVX]*|VIII|VII|VI|V|IV|III|II|I)"

# Terminator after a heading article/chapter number.  ``\b`` is unsuitable
# because Python ``\b`` is a transition between word and non-word chars,
# and a number ending in ``)`` (like ``14(9)``) followed by ``.`` is two
# non-word chars in a row — no word boundary, so ``\b`` fails.  We accept
# any common terminator (period, whitespace, punctuation) or end-of-string.
_HEADING_NUM_END = r"(?=[.\s,;:!?\-]|$)"

# Heading patterns. Each captures the canonical number into group(1).
_HEADING_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    # Georgian article: "მუხლი 14" / "მუხლი 14.7" / "მუხლი 14(9). Title".
    (
        "article",
        "article_number",
        re.compile(rf"^\s*მუხლი\s+({_NUMBER_DECIMAL}){_HEADING_NUM_END}"),
    ),
    # Georgian chapter: "თავი IV" / "თავი IV.1 Title". Roman numerals,
    # optionally with a decimal sub-chapter ("IV.1").
    (
        "chapter",
        "chapter_number",
        re.compile(rf"^\s*თავი\s+({_ROMAN_NUMERAL}(?:\.\d+)?){_HEADING_NUM_END}"),
    ),
    # English article: "Article 14" / "Article 14(9). Title".
    (
        "article",
        "article_number",
        re.compile(rf"^\s*Article\s+({_NUMBER_DECIMAL}){_HEADING_NUM_END}", re.IGNORECASE),
    ),
    # English chapter: "Chapter IV" / "Chapter 4".
    (
        "chapter",
        "chapter_number",
        re.compile(
            rf"^\s*Chapter\s+({_ROMAN_NUMERAL}(?:\.\d+)?|\d+(?:\.\d+)?){_HEADING_NUM_END}",
            re.IGNORECASE,
        ),
    ),
    # Russian article: "Статья 14" / "Статья 14(9)".
    (
        "article",
        "article_number",
        re.compile(rf"^\s*Статья\s+({_NUMBER_DECIMAL}){_HEADING_NUM_END}", re.IGNORECASE),
    ),
    # Russian chapter: "Глава IV" / "Глава 4".
    (
        "chapter",
        "chapter_number",
        re.compile(
            rf"^\s*Глава\s+({_ROMAN_NUMERAL}(?:\.\d+)?|\d+(?:\.\d+)?){_HEADING_NUM_END}",
            re.IGNORECASE,
        ),
    ),
]


def parse_section_heading(title: str) -> SectionHeadingInfo:
    """Extract canonical structural fields from a markdown section heading.

    Returns empty fields when the heading doesn't match a known
    article/chapter shape (intro paragraphs, free-form titles).
    """
    s = str(title or "").strip()
    if not s:
        return SectionHeadingInfo()
    for kind, field, pattern in _HEADING_PATTERNS:
        match = pattern.match(s)
        if match:
            number = match.group(1)
            if field == "article_number":
                return SectionHeadingInfo(section_kind=kind, article_number=number)
            return SectionHeadingInfo(section_kind=kind, chapter_number=number)
    return SectionHeadingInfo()


# ---------------------------------------------------------------------------
# Body-text reference parser
# ---------------------------------------------------------------------------

# Detect ``კოდექსი`` (Civil/Tax/Admin/Labor codes — external, not in our
# corpus). When the parser sees this within REJECTION_WINDOW chars BEFORE
# an article reference, the article ref is dropped — the cited article
# is not in any of our ingested documents, and resolving against the
# citing document would produce a false-positive match on a same-numbered
# article that means something completely different.
_REJECTION_NOUN_RE = re.compile(rf"\bკოდექს{_KA_NOUN_SUFFIX}\b")
# English ``Code`` (Civil/Tax/Admin/Labor) and Russian ``кодекс`` —
# same idea, foreign-language analogues. Word-boundary anchored to
# avoid matching ``Code Red`` or ``codec``.
_REJECTION_NOUN_EN_RE = re.compile(r"\bCode\b")
_REJECTION_NOUN_RU_RE = re.compile(r"\bкодекс[аеуом]?\b", re.IGNORECASE)
REJECTION_WINDOW = 60  # chars before the article ref to scan for "კოდექსი"

# Self-document anchors. Treated as "same document" when found in the
# preamble of an article reference. They don't change resolution
# semantics (all article refs resolve same-doc by default) but the
# parser exposes them for trace observability later.
# - ``ამ კანონის`` / ``ამ წესების`` / ``ბაზრის წესების`` — explicit
#   same-document signal in the citing text.
_SELF_DOC_ANCHOR_RE = re.compile(
    r"ამ\s+(კანონ|წეს)" + _KA_NOUN_SUFFIX
    + r"|ბაზრის\s+წეს" + _KA_NOUN_SUFFIX
)

# Self-article anchors. ``ამ მუხლის`` / ``წინამდებარე მუხლის`` mean
# "this article" — the citing chunk itself. Emitted as a
# :data:`ChunkReferenceKind.self_article` so the resolver skips
# expansion (would loop the chunk back to itself).
_SELF_ARTICLE_RE = re.compile(
    r"(?:ამ|წინამდებარე)\s+მუხლ" + _KA_NOUN_SUFFIX
)

# Article-reference patterns. Each must capture the article number into
# group ``num`` and (when present) sub-references into ``sub_num`` plus
# ``sub_kind`` (paragraph/part) into group ``sub_unit``.
#
# Ordering matters: longer/more-specific patterns are tried first so a
# compound ``მე-14 მუხლის მე-7 პუნქტი`` is captured whole and not
# truncated to ``მე-14 მუხლი`` by a less specific pattern.
#
# The ``num`` capture is a Python regex group name; the helper iterates
# all matches and converts each to a :class:`ChunkReference`.

# Paragraph-sub-reference alternatives, shared across all article patterns.
# Captures one of four forms after the article number:
#   * prefix-ordinal  ``მე-7 პუნქტი``       (sub_pre=7)
#   * suffix-ordinal  ``7-ე პუნქტი``        (sub_post=7)
#   * ordinal word    ``მეშვიდე პუნქტი``    (sub_word=მეშვიდე)
#   * parens form     ``5(7) პუნქტი``       (sub_paren_num=5(7))
_PARAGRAPH_SUB_GROUP = (
    r"(?:\s+(?:"
    r"(?P<sub_prefix>მე-(?P<sub_pre>\d+))|"
    r"(?P<sub_suffix>(?P<sub_post>\d+)-ე)|"
    r"(?P<sub_word>" + "|".join(_GEORGIAN_ORDINAL_WORDS) + r")|"
    rf"(?P<sub_paren_num>\d+\(\d+\))"
    rf")\s+(?P<sub_unit>პუნქტ|ნაწილ){_KA_NOUN_SUFFIX})?"
)

# Georgian suffix-ordinal: "24-ე მუხლი[suffix]" possibly followed by a
# paragraph reference.  Plain integer or decimal article number (parens
# form like ``14(9)-ე`` is not used in the corpus — parens form supersedes
# the ordinal-prefix grammar).
_KA_SUFFIX_ARTICLE = re.compile(
    rf"(?P<full_num>\d+(?:\.\d+){{0,2}})-ე\s+მუხლ{_KA_NOUN_SUFFIX}"
    + _PARAGRAPH_SUB_GROUP
)

# Georgian prefix-ordinal: "მე-14 მუხლი[suffix]" possibly followed by a
# paragraph reference.  Plain integer article number only (same reason).
_KA_PREFIX_ARTICLE = re.compile(
    rf"მე-(?P<full_num>\d+)\s+მუხლ{_KA_NOUN_SUFFIX}"
    + _PARAGRAPH_SUB_GROUP
)

# Georgian compound article: "14(9) მუხლი" (parens sub-article) or
# "14.7 მუხლი" (legacy decimal).  Negative lookbehind ``(?<![-\d])``
# prevents capturing the ``2`` in ``მე-2``.  Optionally followed by a
# paragraph reference.
_KA_COMPOUND_ARTICLE = re.compile(
    rf"(?<![-\d])(?P<full_num>\d+\(\d+\)(?:\.\d+)?|\d+\.\d+(?:\.\d+)?)\s+მუხლ{_KA_NOUN_SUFFIX}"
    + _PARAGRAPH_SUB_GROUP
)

# Georgian forward article (rare in body, common in target headers but
# we don't parse from headings here): "მუხლი 14" / "მუხლი 14(9)" /
# "მუხლი 14.7".  Word-boundary on ``მუხლი`` is the start anchor; the
# trailing terminator is implicit (greedy ``_NUMBER_DECIMAL`` stops at
# first non-matching char).
_KA_FORWARD_ARTICLE = re.compile(
    rf"\bმუხლი\s+(?P<full_num>{_NUMBER_DECIMAL})(?=[.\s,;:!?\-)]|$)"
)

# Georgian standalone paragraph reference: "1(1) პუნქტი", "3(6) პუნქტი".
# Used when body text references a paragraph of the current article
# without preceding article reference.  Negative lookbehind prevents
# capturing the digits inside a parenthesised group like ``(9)``.
# Emitted as :data:`ChunkReferenceKind.self_article` (current article) with
# ``sub_kind=paragraph`` and ``sub_number=<paren-form>``.
_KA_SELF_PARAGRAPH = re.compile(
    rf"(?<![-\d(])(?P<para_num>\d+\(\d+\))\s+(?P<sub_unit>პუნქტ|ნაწილ){_KA_NOUN_SUFFIX}"
)

# Georgian chapter: "თავი XI" / "XI თავი" / "XI და XII თავებით".
_KA_CHAPTER = re.compile(
    rf"\b(?P<roman>{_ROMAN_NUMERAL})\s+თავ{_KA_NOUN_SUFFIX}|"
    rf"\bთავი\s+(?P<roman2>{_ROMAN_NUMERAL})\b"
)

# English article: "Article 14" / "Article 14.7" / "Article 14, paragraph 7".
_EN_ARTICLE = re.compile(
    rf"\bArticle\s+(?P<full_num>{_NUMBER_DECIMAL})\b"
    rf"(?:[,\s]+(?:paragraph|para\.?|section)\s+(?P<sub_num>\d+))?",
    re.IGNORECASE,
)

# Russian article: "Статья 14" / "статьи 14" / "по статье 14" /
# "статьёй 14, пункт 7". The noun's case suffix is 1-3 letters; the
# character class admits ``й`` so instrumental ``статьёй`` matches.
_RU_ARTICLE = re.compile(
    rf"\bстать[аяёеиуюй]{{1,3}}\s+(?P<full_num>{_NUMBER_DECIMAL})\b"
    rf"(?:[,\s]+(?:пункт|часть)\s+(?P<sub_num>\d+))?",
    re.IGNORECASE,
)


def _sub_from_match_ka(match: re.Match[str]) -> tuple[Optional[str], Optional[str]]:
    """Resolve the (sub_kind, sub_number) from a Georgian compound match.

    Four named-group alternatives exist for the paragraph number:
    prefix-ordinal (``მე-N``), suffix-ordinal (``N-ე``), ordinal-word
    (``პირველი`` etc.), or parens form (``N(M)``). Exactly one is set
    per match; the rest are ``None``.
    """
    groups = match.groupdict()
    sub_pre = groups.get("sub_pre")
    sub_post = groups.get("sub_post")
    sub_word = groups.get("sub_word")
    sub_paren = groups.get("sub_paren_num")
    sub_unit_raw = groups.get("sub_unit")
    if not sub_unit_raw:
        return None, None

    sub_unit = "paragraph" if sub_unit_raw.startswith("პუნქტ") else "part"

    if sub_pre:
        return sub_unit, sub_pre
    if sub_post:
        return sub_unit, sub_post
    if sub_word:
        canonical = _normalize_ordinal_word_ka(sub_word)
        if canonical:
            return sub_unit, canonical
    if sub_paren:
        return sub_unit, sub_paren
    return None, None


def _is_inside_rejection_window(
    text: str,
    article_start: int,
    rejection_positions: list[int],
) -> bool:
    """``True`` when any ``კოდექსი`` token sits within
    :data:`REJECTION_WINDOW` chars before this article reference."""
    window_start = max(0, article_start - REJECTION_WINDOW)
    return any(window_start <= pos < article_start for pos in rejection_positions)


def _collect_rejection_positions(text: str) -> list[int]:
    positions: list[int] = []
    for rgx in (_REJECTION_NOUN_RE, _REJECTION_NOUN_EN_RE, _REJECTION_NOUN_RU_RE):
        for match in rgx.finditer(text):
            positions.append(match.start())
    return positions


def _make_article_ref(
    *,
    number: str,
    raw_text: str,
    sub_kind: Optional[str] = None,
    sub_number: Optional[str] = None,
) -> Optional[ChunkReference]:
    try:
        return ChunkReference(
            kind=ChunkReferenceKind.article,
            number=number,
            sub_kind=sub_kind,
            sub_number=sub_number,
            raw_text=raw_text,
        )
    except (TypeError, ValueError):
        return None


def parse_outgoing_references(text: str) -> List[ChunkReference]:
    """Parse outbound cross-references from a chunk's body text.

    Returns a deduplicated list of :class:`ChunkReference` tuples. Each
    morphological surface form collapses to the canonical
    ``(kind, number, sub_kind, sub_number)`` tuple — the resolver
    (Phase B.3) treats them identically.

    Same-document is implicit. The corpus's ``კოდექსი`` references
    (Civil/Tax/Admin/Labor codes) are rejected here because resolving
    them against the citing document would generate false positives.
    """
    body = str(text or "")
    if not body:
        return []

    refs: List[ChunkReference] = []
    seen_keys: set[tuple] = set()
    consumed_ranges: list[tuple[int, int]] = []  # (start, end) of already-matched spans

    def _add_ref(ref: Optional[ChunkReference], start: int, end: int) -> None:
        if ref is None:
            return
        # Dedup by canonical tuple regardless of surface form.
        key = (ref.kind, ref.number, ref.sub_kind, ref.sub_number)
        if key in seen_keys:
            return
        seen_keys.add(key)
        consumed_ranges.append((start, end))
        refs.append(ref)

    def _overlaps_consumed(start: int, end: int) -> bool:
        for (cstart, cend) in consumed_ranges:
            # Half-open intervals: [start, end) overlaps [cstart, cend)
            # when start < cend and cstart < end.
            if start < cend and cstart < end:
                return True
        return False

    rejection_positions = _collect_rejection_positions(body)

    # Self-article anchors first — they need to be tagged before any
    # other pattern consumes the same span.
    for match in _SELF_ARTICLE_RE.finditer(body):
        try:
            ref = ChunkReference(
                kind=ChunkReferenceKind.self_article,
                number="self",
                raw_text=match.group(0),
            )
        except (TypeError, ValueError):
            continue
        _add_ref(ref, match.start(), match.end())

    # Compound + bare article forms. Order matters: more-specific
    # (compound) patterns first so we don't truncate a compound to its
    # bare article fragment.  ``_KA_FORWARD_ARTICLE`` ("მუხლი N") has
    # no paragraph-sub group — pass empty sub.  The other three patterns
    # all carry the unified paragraph-sub group.
    for pattern in (
        _KA_SUFFIX_ARTICLE,
        _KA_PREFIX_ARTICLE,
        _KA_COMPOUND_ARTICLE,
        _KA_FORWARD_ARTICLE,
    ):
        for match in pattern.finditer(body):
            if _overlaps_consumed(match.start(), match.end()):
                continue
            if _is_inside_rejection_window(body, match.start(), rejection_positions):
                continue
            number = match.group("full_num")
            if pattern is _KA_FORWARD_ARTICLE:
                sub_kind, sub_number = None, None
            else:
                sub_kind, sub_number = _sub_from_match_ka(match)
            _add_ref(
                _make_article_ref(
                    number=number,
                    raw_text=match.group(0),
                    sub_kind=sub_kind,
                    sub_number=sub_number,
                ),
                match.start(),
                match.end(),
            )

    # Georgian standalone paragraph references: "1(1) პუნქტი" /
    # "3(6) პუნქტი" — paragraphs of the current article without an
    # article prefix.  Emitted as self_article so the resolver does NOT
    # follow them (the paragraph is in the same article and reachable
    # via adjacency).  Captured for trace observability and future
    # same-article paragraph lookup.
    for match in _KA_SELF_PARAGRAPH.finditer(body):
        if _overlaps_consumed(match.start(), match.end()):
            continue
        if _is_inside_rejection_window(body, match.start(), rejection_positions):
            continue
        sub_unit_raw = match.group("sub_unit")
        sub_kind = "paragraph" if sub_unit_raw.startswith("პუნქტ") else "part"
        try:
            ref = ChunkReference(
                kind=ChunkReferenceKind.self_article,
                number="self",
                sub_kind=sub_kind,
                sub_number=match.group("para_num"),
                raw_text=match.group(0),
            )
        except (TypeError, ValueError):
            continue
        _add_ref(ref, match.start(), match.end())

    # Georgian Roman-numeral chapter references.
    for match in _KA_CHAPTER.finditer(body):
        if _overlaps_consumed(match.start(), match.end()):
            continue
        if _is_inside_rejection_window(body, match.start(), rejection_positions):
            continue
        roman = match.group("roman") or match.group("roman2")
        if not roman:
            continue
        try:
            ref = ChunkReference(
                kind=ChunkReferenceKind.chapter,
                number=roman,
                raw_text=match.group(0),
            )
        except (TypeError, ValueError):
            continue
        _add_ref(ref, match.start(), match.end())

    # English / Russian article references.
    for pattern in (_EN_ARTICLE, _RU_ARTICLE):
        for match in pattern.finditer(body):
            if _overlaps_consumed(match.start(), match.end()):
                continue
            if _is_inside_rejection_window(body, match.start(), rejection_positions):
                continue
            number = match.group("full_num")
            sub_num = match.groupdict().get("sub_num")
            sub_kind = "paragraph" if sub_num else None
            _add_ref(
                _make_article_ref(
                    number=number,
                    raw_text=match.group(0),
                    sub_kind=sub_kind,
                    sub_number=sub_num,
                ),
                match.start(),
                match.end(),
            )

    return refs


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "REJECTION_WINDOW",
    "SectionHeadingInfo",
    "parse_outgoing_references",
    "parse_section_heading",
]
