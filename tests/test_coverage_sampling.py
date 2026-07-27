"""Coverage-preserving bounded sampling tests."""

from itertools import islice

from utils.coverage_sampling import coverage_priority_indices


def test_coverage_priority_starts_with_boundaries_and_spans_large_ranges():
    sample = list(
        islice(
            coverage_priority_indices(10_000_000),
            5,
        )
    )

    assert sample[:2] == [0, 9_999_999]
    assert all(0 <= index < 10_000_000 for index in sample)
    assert len(sample) == len(set(sample))


def test_coverage_priority_places_relevant_rows_before_generic_interior_rows():
    sample = list(
        islice(
            coverage_priority_indices(
                100,
                preferred_indices=[25, 73],
            ),
            5,
        )
    )

    assert sample[:4] == [0, 99, 73, 25]
    assert sample[4] not in {0, 99, 73, 25}
