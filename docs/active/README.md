# langchain_railway Documentation

This folder contains the active project documentation.

## Document Index

- `README.md`: documentation index and quick start
- `DEVELOPER_GUIDE.md`: architecture and contributor workflow
- `EVALUATION.md`: testing and quality validation
- `TESTING_GUIDE.md`: practical test execution checklist
- `SEMANTIC_SELECTION_GUIDE.md`: semantic topic/example selection behavior
- `architectural_assessment.md`: phased architecture plan and decisions
- `CHANGELOG.md`: maintained change log for architecture and docs
- `COMPREHENSIVE_AUDIT.md`: preserved historical audit reference

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run tests:

```bash
pytest -q
```

3. Start API server:

```bash
uvicorn main:app --reload --port 8000
```

4. Check API docs:

- Open `/docs` after the server starts.

## Notes

- Domain knowledge files used by runtime live in `knowledge/` and are not part of this folder.
- Historical one-off migration and review markdown files were intentionally removed during cleanup.
