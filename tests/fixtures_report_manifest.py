"""Shared report evidence and plan fixtures.

Extracted from ``tests/test_report_planner.py``, which nine test modules were
importing for these builders. A fixture hub living inside a test module for one
subsystem means deleting that subsystem takes the hub with it; keeping them here
lets the planner tests be retired without touching the modules that only ever
wanted a manifest.
"""

from __future__ import annotations

from contracts.report import REPORT_PLAN_CONTRACT_VERSION
from contracts.report_evidence import ReportEvidenceManifest

TABLE_REF = "evidence:table:" + "1" * 16
STATS_REF = "evidence:statistics:" + "2" * 16
LIMIT_REF = "evidence:limitation:" + "3" * 16


def _manifest() -> ReportEvidenceManifest:
    return ReportEvidenceManifest.model_validate(
        {
            "contract_version": "report-evidence-manifest-v1",
            "manifest_id": "manifest:" + "4" * 32,
            "query_digest": "5" * 64,
            "items": [
                {
                    "evidence_ref": TABLE_REF,
                    "kind": "table",
                    "title": "Prices",
                    "source": "tool",
                    "provenance_refs": ["query:tool:prices"],
                    "columns": ["period", "price"],
                    "rows": [
                        {"period": "2026-01", "price": 120.0},
                        {"period": "2026-02", "price": 130.0},
                    ],
                    "content": "",
                    "unit_by_column": {"price": "GEL/MWh"},
                    "total_row_count": 2,
                    "truncated": False,
                },
                {
                    "evidence_ref": STATS_REF,
                    "kind": "statistics",
                    "title": "Statistics",
                    "source": "derived",
                    "provenance_refs": [TABLE_REF],
                    "columns": [],
                    "rows": [],
                    "content": "Average price was 125 GEL/MWh.",
                    "unit_by_column": {},
                    "total_row_count": 0,
                    "truncated": False,
                },
                {
                    "evidence_ref": LIMIT_REF,
                    "kind": "limitation",
                    "title": "Evidence boundary",
                    "source": "system",
                    "provenance_refs": [],
                    "columns": [],
                    "rows": [],
                    "content": "Only the supplied periods and sources may be used.",
                    "unit_by_column": {},
                    "total_row_count": 0,
                    "truncated": False,
                },
            ],
        }
    )


def _plan_payload() -> dict:
    return {
        "contract_version": REPORT_PLAN_CONTRACT_VERSION,
        "title": "Electricity price trend report",
        "objective": "Explain the observed price trend from supplied evidence.",
        "language_code": "en",
        "target_words": 900,
        "evidence_manifest_id": _manifest().manifest_id,
        "sections": [
            {
                "section_id": "executive_summary",
                "kind": "executive_summary",
                "title": "Executive summary",
                "objective": "Summarize the principal evidence-backed finding.",
                "target_words": 120,
                "required_evidence_refs": [STATS_REF],
                "chart_refs": [],
            },
            {
                "section_id": "scope_and_evidence",
                "kind": "scope_and_evidence",
                "title": "Scope and evidence",
                "objective": "Describe the period, source, and measurement.",
                "target_words": 120,
                "required_evidence_refs": [TABLE_REF],
                "chart_refs": [],
            },
            {
                "section_id": "key_findings",
                "kind": "key_findings",
                "title": "Key findings",
                "objective": "Explain the observed values and direction.",
                "target_words": 420,
                "required_evidence_refs": [TABLE_REF, STATS_REF],
                "chart_refs": ["price_trend"],
            },
            {
                "section_id": "limitations",
                "kind": "limitations",
                "title": "Limitations",
                "objective": "State the evidence boundary.",
                "target_words": 100,
                "required_evidence_refs": [LIMIT_REF],
                "chart_refs": [],
            },
            {
                "section_id": "conclusion",
                "kind": "conclusion",
                "title": "Conclusion",
                "objective": "Answer the question within the evidence boundary.",
                "target_words": 140,
                "required_evidence_refs": [STATS_REF],
                "chart_refs": [],
            },
        ],
        "charts": [
            {
                "chart_id": "price_trend",
                "section_id": "key_findings",
                "purpose": "trend",
                "title": "Observed electricity price",
                "evidence_refs": [TABLE_REF],
                "required": True,
            }
        ],
    }
