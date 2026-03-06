# Testing Guide

## Scope

This guide covers practical testing for API behavior, routing, fallback, and output consistency.

## Recommended Command Order

1. Quick smoke test:

```bash
pytest -q tests/test_main.py
```

2. Routing/tool behavior:

```bash
pytest -q tests/test_router.py tests/test_tool_adapter.py
```

3. Pipeline behavior:

```bash
pytest -q tests/test_pipeline_agent_mode.py tests/test_orchestrator.py
```

4. Full suite:

```bash
pytest -q
```

## What to Verify

- Correct route choice for known intents
- Graceful fallback to SQL path when tools do not match
- Uniform tabular handoff (`df`, `cols`, `rows`)
- Stable summary and chart payload generation
- No regressions in `test_main.py`

## Manual Endpoint Validation

Run local server:

```bash
uvicorn main:app --reload --port 8000
```

Run sample request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-App-Key: <APP_KEY>" \
  -d '{"query":"Compare tariffs for 2024","mode":"light"}'
```

## Debugging Order

1. Check failing assertion and test fixture setup
2. Check router decision and tool args
3. Check SQL fallback branch and validation
4. Check analyzer/summarizer/chart stages
