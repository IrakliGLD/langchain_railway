# `langchain_railway` Documentation

This folder contains the active project documentation.

## Document Index

| File | Purpose |
|---|---|
| [`query_pipeline_architecture.md`](query_pipeline_architecture.md) | **Main reference.** Stage-by-stage architecture of the query pipeline, the §2.3 Ideal Decision Tree it implements, module responsibilities, and what's still open. **Source of truth: the code; when this doc disagrees, update the doc.** |
| [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) | Deployment environments, auth modes, env baselines, local workflow, manual endpoint validation. |
| [`VECTOR_KNOWLEDGE_ROLLOUT.md`](VECTOR_KNOWLEDGE_ROLLOUT.md) | Runbook for ingesting a new regulatory document into the vector knowledge store. |

## Related References (outside this folder)

| Location | What it covers |
|---|---|
| [`skills/developer-phased-audit/`](../../skills/developer-phased-audit/SKILL.md) | Phased-implementation workflow + [targeted-suite definition](../../skills/developer-phased-audit/references/targeted-suite.md) (the test set required green before any phase audit). |
| [`skills/pipeline-failure-diagnostics/`](../../skills/pipeline-failure-diagnostics/SKILL.md) | Playbook for triaging Q&A failures — failure taxonomy, fix-layer selection. |
| `knowledge/*.md` | Runtime curated domain knowledge consumed by the LLM at request time. |
| `docs_to_ingest/*.md` | Ingestion-ready source documents for the vector knowledge store. |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run the targeted test suite (per the phased-audit workflow)
pytest tests/ --ignore=tests/security -q

# Start the API locally
uvicorn main:app --reload --port 8000
```

See [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) for env-var setup and manual endpoint validation.

## Documentation Policy

- Keep active docs in this folder. Historical audit / migration / handoff documents are not preserved here — git history is the durable record.
- Runtime knowledge for the LLM lives in `knowledge/`, not in this folder.
- Avoid adding new top-level audit / assessment / handoff markdown files. If a one-off audit is needed, write it into the relevant skill's reference set or into a single phase commit body instead.
