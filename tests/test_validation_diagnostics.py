"""The one place a ValidationError becomes safe telemetry."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError


class _Inner(BaseModel):
    text: str


class _Outer(BaseModel):
    items: list[_Inner]


def test_locations_name_the_rejected_fields_and_nothing_else():
    from utils.validation_diagnostics import validation_error_locations

    with pytest.raises(ValidationError) as caught:
        _Outer.model_validate({"items": [{"text": 1}, {}]})

    locations = validation_error_locations(caught.value)

    assert locations == ["items.0.text", "items.1.text"]


def test_a_non_pydantic_exception_yields_no_locations():
    """Callers log this on any failure path; it must never raise itself."""

    from utils.validation_diagnostics import validation_error_locations

    assert validation_error_locations(RuntimeError("boom")) == []
