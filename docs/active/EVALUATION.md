# Evaluation Guide

## Goal

Validate correctness, stability, and safety of query handling across tool-first and SQL fallback paths.

## Test Layers

### 1. Unit/Module Tests

Run all automated tests:

```bash
pytest -q
```

Run critical path tests when iterating quickly:

```bash
pytest -q tests/test_main.py tests/test_router.py tests/test_tool_adapter.py
```

### 2. Pipeline/Integration Tests

Focus on orchestration and fallback behavior:

```bash
pytest -q tests/test_pipeline_agent_mode.py tests/test_orchestrator.py
```

### 3. Manual API Checks

Start server locally:

```bash
uvicorn main:app --reload --port 8000
```

Test endpoint:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-App-Key: <GATEWAY_SHARED_SECRET>" \
  -d '{"query":"What is balancing price trend in 2024?","mode":"light"}'
```

`/evaluate` is now separate: call it with `X-App-Key: <EVALUATE_ADMIN_SECRET>`.

## Validation Checklist

- Fast-route intents bypass LLM where expected
- Typed tool outputs satisfy the shared tabular contract
- SQL fallback still works when tools do not match
- Summaries and chart payloads are consistent with retrieved data
- No unsafe SQL interpolation paths are introduced

## Failure Triage

1. Reproduce with a single focused test
2. Confirm router decision and fallback branch
3. Confirm data contract (`df`, `cols`, `rows`) at handoff boundaries
4. Confirm summarizer/chart stage expectations
