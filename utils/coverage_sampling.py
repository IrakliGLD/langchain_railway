"""Deterministic index ordering for bounded, coverage-preserving samples."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from heapq import heappop, heappush


def coverage_priority_indices(
    total: int,
    *,
    preferred_indices: Iterable[int] = (),
) -> Iterator[int]:
    """Yield boundaries, diverse preferred rows, then diverse remaining rows."""

    if total < 0:
        raise ValueError("total must be non-negative.")
    if total == 0:
        return

    selected: set[int] = set()

    def emit(index: int) -> int | None:
        if 0 <= index < total and index not in selected:
            selected.add(index)
            return index
        return None

    for boundary in (0, total - 1):
        if (index := emit(boundary)) is not None:
            yield index

    def farthest(candidates: set[int]) -> int:
        return max(
            candidates,
            key=lambda index: (
                min(abs(index - chosen) for chosen in selected),
                -index,
            ),
        )

    preferred = {
        index
        for index in preferred_indices
        if 0 <= index < total and index not in selected
    }
    while preferred:
        index = farthest(preferred)
        selected.add(index)
        preferred.remove(index)
        yield index

    intervals: list[tuple[int, int, int]] = []

    def add_interval(left: int, right: int) -> None:
        if right - left > 1:
            heappush(intervals, (-(right - left), left, right))

    ordered = sorted(selected)
    for left, right in zip(ordered, ordered[1:], strict=False):
        add_interval(left, right)

    while intervals:
        _, left, right = heappop(intervals)
        index = (left + right) // 2
        if index in selected:
            continue
        selected.add(index)
        yield index
        add_interval(left, index)
        add_interval(index, right)
