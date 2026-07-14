from dataclasses import asdict

import pandas as pd

from agent.answer_provenance import build_answer_provenance
from agent.pipeline import _build_and_attach_evidence_frame
from agent.provenance import source_rows_hash, stamp_provenance, tool_invocation_hash
from agent.summarizer import _build_scenario_frame
from agent.tools.types import ToolInvocation
from models import QueryContext


def test_tool_frame_receives_query_and_exact_source_hashes():
    ctx = QueryContext(query="price")
    ctx.df = pd.DataFrame({"date": ["2024-01-01"], "p_bal_gel": [150.0]})
    ctx.cols = list(ctx.df.columns)
    ctx.rows = [tuple(row) for row in ctx.df.itertuples(index=False, name=None)]
    invocation = ToolInvocation(name="get_prices", params={"currency": "gel"})
    query_hash = tool_invocation_hash(invocation.name, invocation.params)
    stamp_provenance(ctx, ctx.cols, ctx.rows, source="tool", query_hash=query_hash)

    _build_and_attach_evidence_frame(ctx, invocation)

    assert ctx.evidence_frame is not None
    refs = ctx.evidence_frame.provenance_refs
    assert f"query:tool:{query_hash}" in refs
    assert f"source:rows:{source_rows_hash(ctx.cols, ctx.rows)}" in refs


def test_source_hash_changes_when_source_values_change():
    cols = ["date", "value"]
    assert source_rows_hash(cols, [("2024-01", 1)]) != source_rows_hash(
        cols, [("2024-01", 2)]
    )


def test_derived_scenario_frame_preserves_all_parent_refs_and_serializes_them():
    parent_refs = ["query:tool:a", "source:rows:b", "query:tool:c"]
    ctx = QueryContext(query="scenario", provenance_refs=parent_refs)
    ctx.analysis_evidence = [{
        "record_type": "scenario",
        "derived_metric_name": "scenario_scale",
        "metric": "p_bal_gel",
        "scenario_factor": 1.2,
        "scenario_volume": None,
        "aggregate_result": 120.0,
        "source_row_count": 2,
        "period_range": "2023 to 2024",
        "min_period_value": 50.0,
        "max_period_value": 60.0,
        "mean_period_value": 55.0,
        "formula": "price * 1.2",
    }]
    frame = _build_scenario_frame(ctx)
    assert frame is not None
    assert frame.provenance_refs == parent_refs
    assert asdict(frame)["provenance_refs"] == parent_refs
    assert build_answer_provenance(ctx)["provenance_refs"] == parent_refs
