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

