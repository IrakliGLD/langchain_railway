from analysis.stats import quick_stats


def test_quick_stats_is_order_invariant():
    rows = [(2024, 120.0), (2021, 100.0), (2023, 115.0), (2022, 110.0)]
    assert quick_stats(rows, ["year", "price"]) == quick_stats(
        list(reversed(rows)), ["year", "price"]
    )


def test_equal_endpoint_values_report_stable_not_decreasing():
    result = quick_stats([(2023, 100.0), (2024, 100.0)], ["year", "price"])
    assert "stable (0.0%)" in result
    assert "decreasing" not in result


def test_quick_stats_never_averages_incompatible_numeric_columns_together():
    rows = [(2023, 100.0, 10.0), (2024, 120.0, 5.0)]
    result = quick_stats(rows, ["year", "price", "generation"])
    assert "Trend (Yearly Avg, price, 2023→2024): increasing (20.0%)" in result
    assert "Trend (Yearly Total, generation, 2023→2024): decreasing (-50.0%)" in result


def test_duplicate_entity_rows_do_not_fake_monthly_completeness():
    rows = []
    for month in range(1, 7):
        for entity in ("hydro", "thermal"):
            rows.append((f"2024-{month:02d}-01", entity, 10.0))
    result = quick_stats(rows, ["date", "entity", "generation"])
    assert "Insufficient data for yearly comparison" in result
