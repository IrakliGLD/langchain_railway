"""A failure that never reaches the driver must still say what went wrong.

2026-08-15: ``get_end_user_prices`` failed with a SQLAlchemy StatementError.
Because no DBAPI call happened there was no SQLSTATE, so the log line read
"operation=typed_tool_query type=StatementError sqlstate=unknown" and named
nothing. The real message -- "A value is required for bind parameter 'name'"
-- identifies the defect immediately.
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from sqlalchemy.exc import ProgrammingError, StatementError  # noqa: E402

from core.db_gateway import _safe_error_detail  # noqa: E402


def test_a_pre_driver_error_reports_its_reason():
    error = StatementError(
        "A value is required for bind parameter 'name'", "SELECT 1", {}, None
    )
    detail = _safe_error_detail(error)
    assert "bind parameter" in detail
    assert "name" in detail


def test_a_driver_error_reports_nothing_extra():
    """Once the driver was reached the message can echo parameter VALUES.

    Those may carry user data, so that path stays on the SQLSTATE code alone.
    """
    original = Exception("duplicate key value violates unique constraint")
    error = ProgrammingError("INSERT ...", {"secret": "value"}, original)

    assert _safe_error_detail(error) == ""


def test_an_unrelated_error_reports_nothing_extra():
    assert _safe_error_detail(ValueError("boom")) == ""


def test_the_detail_is_bounded():
    error = StatementError("x" * 5000, "SELECT 1", {}, None)
    assert len(_safe_error_detail(error)) <= 200


def test_only_the_first_line_is_kept():
    """SQLAlchemy appends the statement and parameters after a blank line."""
    error = StatementError(
        "A value is required for bind parameter 'name'\n"
        "[SQL: SELECT :name]\n"
        "[parameters: {'secret': 'value'}]",
        "SELECT 1",
        {},
        None,
    )
    detail = _safe_error_detail(error)
    assert "secret" not in detail
    assert "parameters" not in detail
