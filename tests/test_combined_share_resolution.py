"""Focused tests for deterministic combined-share resolution."""

from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.analyzer import _build_share_summary_artifact


def test_combined_share_artifact_sums_full_requested_bucket():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-06-01", "2020-07-01"]),
            "share_renewable_ppa": [0.20, 0.10],
            "share_thermal_ppa": [0.15, 0.10],
            "share_cfd_scheme": [0.10, 0.05],
            "share_regulated_hpp": [0.15, 0.15],
            "share_regulated_new_tpp": [0.12, 0.10],
            "share_regulated_old_tpp": [0.14, 0.10],
            "share_deregulated_hydro": [0.135, 0.10],
            "share_import": [0.005, 0.30],
        }
    )

    summary, grounding = _build_share_summary_artifact(
        df,
        plan={},
        user_query=(
            "In which months was the aggregated share of renewable ppa, thermal ppa, CfD scheme, "
            "regulated hydro, all regulated thermal and deregulated hydro in balancing electricity more than 99%?"
        ),
    )

    assert summary is not None
    assert "June 2020" in summary
    assert "July 2020" not in summary
    assert "99.5%" in summary
    assert "Deregulated Hydro Generation exceeded" not in summary
    assert "share_regulated_new_tpp" in grounding
    assert "share_cfd_scheme" in grounding


def test_combined_share_artifact_fails_closed_when_bucket_is_only_partially_resolved():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-06-01"]),
            "share_renewable_ppa": [0.4],
            "share_regulated_hpp": [0.6],
            "share_import": [0.6],
        }
    )

    summary, grounding = _build_share_summary_artifact(
        df,
        plan={},
        user_query=(
            "In which months was the aggregated share of renewable ppa, regulated hydro, and mystery source in balancing electricity more than 99%?"
        ),
    )

    assert summary is None
    assert grounding == ""


def test_combined_share_artifact_renders_exact_one_as_100_percent():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-06-01"]),
            "share_renewable_ppa": [0.4],
            "share_regulated_hpp": [0.6],
        }
    )

    summary, grounding = _build_share_summary_artifact(
        df,
        plan={},
        user_query=(
            "In which months was the aggregated share of renewable ppa and regulated hydro in balancing electricity more than 99%?"
        ),
    )

    assert summary is not None
    assert "100.0%" in summary
    assert "1.0%" not in summary
    assert "100.0%" in grounding
