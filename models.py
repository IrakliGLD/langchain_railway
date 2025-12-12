"""
Pydantic models for API requests and responses.

Extracted from monolithic main.py for better organization.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


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
