"""Tests for the published chat-gateway contract (P6.A, finding M5).

The committed JSON Schema artifact is the versioned wire contract that
edge/frontend consumers generate DTOs from. These tests fail whenever a request
or response field is renamed, added, removed, or retyped without regenerating
the artifact — the "consumer-driven contract test" P6.1 requires.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from contracts.chat_gateway_contract import (
    CONTRACT_ARTIFACT_PATH,
    build_contract_document,
    load_committed_contract,
    serialize_contract_document,
)
from models import CHAT_GATEWAY_CONTRACT_VERSION


class TestContractArtifact:
    def test_committed_artifact_exists(self):
        assert CONTRACT_ARTIFACT_PATH.exists(), (
            "Contract artifact missing; run "
            "`python -m scripts.export_chat_gateway_contract`"
        )

    def test_no_drift_between_models_and_committed_artifact(self):
        # Byte-for-byte: the committed file must equal a fresh render of the
        # current models. Regenerate with the export script when this fails.
        fresh = serialize_contract_document()
        committed = CONTRACT_ARTIFACT_PATH.read_text(encoding="utf-8")
        assert fresh == committed, (
            "Chat-gateway contract drift: models.py changed without "
            "regenerating contracts/chat_gateway_v1.schema.json. Run "
            "`python -m scripts.export_chat_gateway_contract`."
        )

    def test_version_matches_backend_constant(self):
        doc = load_committed_contract()
        assert doc["contract_version"] == CHAT_GATEWAY_CONTRACT_VERSION


class TestContractSurface:
    """Explicit presence checks so a silent field drop is caught by name."""

    def setup_method(self):
        self.doc = build_contract_document()
        self.schemas = self.doc["schemas"]

    def test_request_fields_present(self):
        props = self.schemas["Question"]["properties"]
        assert set(props) == {"query", "user_id", "conversation_history"}
        assert self.schemas["Question"]["required"] == ["query"]
        # Unknown request fields are rejected (extra=forbid).
        assert self.schemas["Question"]["additionalProperties"] is False

    def test_conversation_turn_shape(self):
        props = self.schemas["ConversationTurn"]["properties"]
        assert set(props) == {"question", "answer"}

    def test_response_fields_present(self):
        props = self.schemas["APIResponse"]["properties"]
        assert set(props) == {
            "answer", "charts", "chart_data", "chart_type",
            "chart_metadata", "execution_time",
        }
        # Multiple charts survive as a list (M5: charts[] is part of the contract).
        assert props["charts"]["anyOf"][0]["type"] == "array"

    def test_error_envelope_shape(self):
        # Nested via $defs; the top-level error wrapper points at the detail.
        err = self.schemas["APIErrorResponse"]
        detail = err["$defs"]["APIErrorDetail"]["properties"]
        assert set(detail) == {"code", "message", "retryable", "request_id"}

    def test_document_is_canonical_json(self):
        # Deterministic, sorted, newline-terminated — so diffs are meaningful.
        serialized = serialize_contract_document()
        assert serialized.endswith("\n")
        assert '"contract_version": "chat-gateway-v1"' in serialized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
