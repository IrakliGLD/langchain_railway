"""Whether a question about regulated retail prices should be answered from data.

Lives in its own module because both the pipeline (response-mode derivation)
and the evidence planner (tool selection) need the same answer, and the
pipeline already imports the planner -- so the predicate cannot live in either
without a cycle. Two callers disagreeing about this is what let a question be
routed to the data path with no data step planned.
"""
import logging

from agent.router import looks_like_retail_tariff_question
from agent.tools.end_user_price_tools import (
    asks_for_wholesale_comparison,
    scope_haystack,
)
from agent.tools.end_user_price_tools import (
    resolve_scope as resolve_end_user_scope,
)
from contracts.question_analysis import AnswerKind, KnowledgeTopicName, QueryType
from models import QueryContext

log = logging.getLogger("Enai")


#: Wording that names a plant-fleet dataset: installed capacity bands,
#: commissioning vintage, capacity factors, ownership concentration. These are
#: served from materialized views (by_capacity, by_commissioning,
#: capacity_factor, ownership_concentration) via SQL -- there is no typed tool.
#:
#: "capacity" alone is deliberately absent: it collides with the guaranteed
#: capacity FEE and with cross-border interconnection capacity, which are
#: different subjects served by different data.
_PLANT_FLEET_MARKERS = frozenset({
    "installed capacity",
    "capacity band",
    "capacity category",
    "capacity factor",
    "load factor",
    "commissioning",
    "commissioned",
    "vintage",
    "plant age",
    "fleet",
    "ownership concentration",
    "market concentration",
    "herfindahl",
    "hhi",
    "owner share",
    "number of plants",
    "plant count",
    "facility count",
    "how many plants",
    "how many power plants",
})


def looks_like_plant_fleet_question(text: str) -> bool:
    """Whether the question is about fleet structure rather than output."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _PLANT_FLEET_MARKERS)


def is_plant_fleet_data_question(ctx: QueryContext) -> bool:
    """Whether a plant-fleet question should be answered from data.

    Same failure the retail tool had, from the other direction: on 2026-08-16
    two fleet questions came back ``query_type=ambiguous``,
    ``preferred_path=knowledge``, ``candidate_tools=[]`` (confidence 0.88 and
    0.32) and were answered as prose from generation_mix knowledge -- with no
    data at all, while by_capacity, by_commissioning, capacity_factor and
    ownership_concentration sat one query away.

    There is no typed tool for these views, so routing to data means the SQL
    path. That is adequate here: the views are narrow and flat, and
    PLANT_FLEET_EXAMPLES already shows the shapes.

    Ambiguity about which technology or band is meant is not a reason to
    withhold the data -- it is a reason to show it and offer to narrow, the
    same rule the domain owner set for retail.
    """
    if not ctx.has_authoritative_question_analysis:
        return False

    analysis = ctx.question_analysis
    question = ctx.resolved_query or analysis.canonical_query_en or ctx.query
    if not looks_like_plant_fleet_question(question):
        return False

    return (
        analysis.classification.query_type == QueryType.AMBIGUOUS
        or analysis.answer_kind == AnswerKind.CLARIFY
    )


def is_retail_data_question(ctx: QueryContext) -> bool:
    """Whether a retail question should be answered from data.

    2026-08-15, from the domain owner, after seeing both behaviours: "i prefer
    general answer and then ask to clarify to provide targeted information and
    assessment", and make all three questions behave the same way.

    So retail questions are never blocked on a clarifying question. They are
    answered from the data -- every category, never averaged -- and the answer
    closes by offering to narrow. The narrowing offer lives in the summarizer
    guidance, not here.

    This routes to data even when the analyzer says ``preferred_path=knowledge``
    with ``query_type=ambiguous``, which is how the same three questions took
    three different paths on 2026-08-15: one clarified, one fetched data, and
    one wrote an essay -- the last because ``network_supply_tariffs`` ranked
    third among candidate topics rather than first.

    A genuine definition question ("what is a supply tariff") keeps the
    knowledge path: it is classified conceptual_definition, not ambiguous.
    """
    if not ctx.has_authoritative_question_analysis:
        return False

    analysis = ctx.question_analysis
    topics = {
        candidate.name.value
        for candidate in (analysis.knowledge.candidate_topics or [])
    }
    question = ctx.resolved_query or analysis.canonical_query_en or ctx.query
    haystack = scope_haystack(analysis.entity_scope, question)

    # Topic nomination is not reliable enough to be the only key. On
    # 2026-08-15 the retail comparison came back with candidate_topics
    # [tariffs, market_structure, balancing_price] -- network_supply_tariffs
    # absent entirely -- so the question was answered as prose while the data
    # sat one call away. The WORDING is the second key.
    is_retail = (
        KnowledgeTopicName.NETWORK_SUPPLY_TARIFFS.value in topics
        or looks_like_retail_tariff_question(haystack)
    )
    if not is_retail:
        return False

    # Ambiguity about WHICH company or category is not a reason to withhold
    # the data; it is a reason to show all of it and offer to narrow.
    if analysis.classification.query_type == QueryType.AMBIGUOUS:
        return True
    if analysis.answer_kind == AnswerKind.CLARIFY:
        return True

    # A comparison against the wholesale side is a data question by nature.
    if asks_for_wholesale_comparison(haystack):
        return True

    # Already scoped: certainly a data question.
    supplier, category = resolve_end_user_scope(haystack)
    return bool(supplier or category)

