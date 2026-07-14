import pytest

from utils.forecasting import extract_forecast_horizon_years


@pytest.mark.parametrize(
    "text",
    [
        "forecast for 10 years",
        "forecast for 10-Year horizon",
        "forecast for ten years",
        "FORECAST FOR TEN-YEAR HORIZON",
        "forecast for the next decade",
    ],
)
def test_ten_year_forms_resolve_identically(text):
    assert extract_forecast_horizon_years(text) == 10


def test_structured_horizon_is_authoritative_over_conflicting_text():
    assert extract_forecast_horizon_years(
        "forecast for 3 years",
        structured_horizon_years=12,
    ) == 12
