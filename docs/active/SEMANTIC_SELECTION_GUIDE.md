# Semantic Selection Guide

## Purpose

Describe how the system selects relevant domain knowledge and SQL examples before generation.

## Inputs

- User query text
- Language/context hints
- Routing/tool metadata (when available)

## Knowledge Sources

- Topic markdown files in `knowledge/*.md`
- Topic loader/selector logic in `knowledge/__init__.py`
- SQL example retrieval in `knowledge/sql_example_selector.py`

## Selection Behavior

1. Detect high-signal intent terms from the query.
2. Map intent to one or more domain topics.
3. Pull only relevant topic text and SQL examples.
4. Keep prompt context bounded; avoid dumping full knowledge corpus.

## Design Rules

- Prefer precision over broad context stuffing.
- Keep selection deterministic where possible.
- Keep data retrieval concerns separate from analysis concerns.
- Do not embed runtime-irrelevant historical guidance into active prompts.

## Extending Topic Coverage

When adding a new topic:

1. Add a new markdown file under `knowledge/`.
2. Register topic metadata and trigger terms in `knowledge/__init__.py`.
3. Add representative SQL examples if needed in `knowledge/sql_examples.md`.
4. Update selector heuristics/tests.

## Regression Guard

Any selector change should include tests for:

- Correct topic hit for positive cases
- Noisy-query false positives
- Fallback behavior when no topic confidently matches
