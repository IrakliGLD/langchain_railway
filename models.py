"""
Pydantic models for API requests and responses.

Extracted from monolithic main.py for better organization.
"""
from dataclasses import dataclass, field as dc_field
from typing import Optional, List, Dict, Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator
from contracts.question_analysis import QuestionAnalysis


# ---------------------------------------------------------------------------
# QueryContext: mutable state object that flows through the pipeline stages
# ---------------------------------------------------------------------------

@dataclass
class QueryContext:
    """Carries all mutable state between pipeline stages.

    Created once per /ask request. Each stage reads what it needs and
    writes its outputs back onto the same object.
    """
    # --- inputs (set once at creation) ---
    query: str
    lang_code: str = ""
    lang_instruction: str = ""
    mode: str = "light"                          # "light" | "analyst"
    conversation_history: Optional[List[Dict[str, str]]] = None
    trace_id: str = ""
    session_id: str = ""
    question_analysis: Optional[QuestionAnalysis] = None
    question_analysis_error: str = ""
    question_analysis_source: str = ""

    # --- planner outputs ---
    plan: Dict[str, Any] = dc_field(default_factory=dict)
    raw_sql: Optional[str] = None
    safe_sql: Optional[str] = None
    is_conceptual: bool = False
    skip_sql: bool = False
    skip_sql_reason: str = ""
    aggregation_intent: Dict[str, bool] = dc_field(default_factory=dict)
    used_tool: bool = False
    tool_name: Optional[str] = None
    tool_params: Dict[str, Any] = dc_field(default_factory=dict)
    tool_match_reason: str = ""
    tool_confidence: float = 0.0
    tool_fallback_reason: str = ""
    agent_outcome: str = ""  # "data_exit" | "conceptual_exit" | "fallback_exit"
    agent_rounds: int = 0
    agent_tool_calls: int = 0
    agent_fallback_reason: str = ""
    agent_trace: List[Dict[str, Any]] = dc_field(default_factory=list)

    # --- sql_executor outputs ---
    df: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    rows: list = dc_field(default_factory=list)
    cols: list = dc_field(default_factory=list)
    provenance_rows: list = dc_field(default_factory=list)
    provenance_cols: list = dc_field(default_factory=list)
    provenance_query_hash: str = ""
    provenance_source: str = ""  # "sql" | "tool" | ""
    sql_is_relevant: bool = True
    skip_chart_due_to_relevance: bool = False

    # --- analyzer outputs ---
    preview: str = ""
    stats_hint: str = ""
    share_summary_override: Optional[str] = None
    why_summary_override: Optional[str] = None
    why_summary_claims: List[str] = dc_field(default_factory=list)
    analysis_evidence: List[Dict[str, Any]] = dc_field(default_factory=list)
    correlation_results: Dict[str, Any] = dc_field(default_factory=dict)
    add_trendlines: bool = False
    trendline_extend_to: Optional[str] = None

    # --- summarizer outputs ---
    summary: str = ""
    summary_source: str = ""
    summary_claims: List[str] = dc_field(default_factory=list)
    summary_citations: List[str] = dc_field(default_factory=list)
    summary_confidence: float = 0.0
    summary_claim_provenance: List[Dict[str, Any]] = dc_field(default_factory=list)
    summary_provenance_coverage: float = 0.0
    summary_provenance_gate_passed: bool = True
    summary_provenance_gate_reason: str = ""

    # --- chart_pipeline outputs ---
    chart_data: Optional[List[Dict[str, Any]]] = None
    chart_type: Optional[str] = None
    chart_meta: Optional[Dict[str, Any]] = None

    # --- timing ---
    exec_time: float = 0.0
    stage_timings_ms: Dict[str, float] = dc_field(default_factory=dict)


class Question(BaseModel):
    """
    User question model for /ask endpoint.

    Attributes:
        query: Natural language query in English, Georgian, or Russian
        user_id: Optional user identifier for tracking
        conversation_history: Optional list of previous Q&A pairs for context
    """
    query: str = Field(..., max_length=2000, description="Natural language query")
    user_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Last 1-3 Q&A pairs: [{'question': '...', 'answer': '...'}]"
    )

    @field_validator("query")
    @classmethod
    def _not_empty(cls, v):
        """Validate query is not empty."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class APIResponse(BaseModel):
    """
    API response model for /ask endpoint.

    Attributes:
        answer: Natural language answer to the query
        chart_data: Optional chart data as list of dictionaries
        chart_type: Optional chart type (line, bar, pie, stackedbar, etc.)
        chart_metadata: Optional metadata about the chart
        execution_time: Total execution time in seconds
    """
    answer: str
    chart_data: Optional[List[Dict[str, Any]]] = None
    chart_type: Optional[str] = None
    chart_metadata: Optional[Dict[str, Any]] = None
    execution_time: float


class MetricsResponse(BaseModel):
    """
    Metrics endpoint response model.

    Attributes:
        requests: Total number of requests processed
        llm_calls: Total number of LLM API calls
        sql_queries: Total number of SQL queries executed
        errors: Total number of errors
        avg_request_time: Average request processing time
        avg_llm_time: Average LLM call time
        avg_sql_time: Average SQL execution time
    """
    requests: int
    llm_calls: int
    sql_queries: int
    errors: int
    avg_request_time: float
    avg_llm_time: float
    avg_sql_time: float
