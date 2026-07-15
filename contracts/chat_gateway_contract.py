"""Versioned chat-gateway API contract (P6.A, finding M5).

Publishes the request/response/error wire contract of the ``/ask`` endpoint as
a single versioned JSON Schema document, generated deterministically from the
Pydantic models in ``models.py``. The models are the source of truth; this
module renders them into the artifact that edge/frontend consumers (P6.B)
generate DTOs from and pin contract tests against.

Building the document from the Pydantic models directly (not the FastAPI app)
keeps generation fast and side-effect-free: no app startup, DB, or secrets.

Regenerate the committed artifact with::

    python -m scripts.export_chat_gateway_contract

and the drift test in ``tests/test_chat_gateway_contract.py`` fails whenever a
field is renamed, added, removed, or retyped without regenerating it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from models import (
    CHAT_GATEWAY_CONTRACT_VERSION,
    APIErrorResponse,
    APIResponse,
    ConversationTurn,
    Question,
)

# The committed artifact. Lives beside this module so both the exporter and the
# drift test resolve it the same way.
CONTRACT_ARTIFACT_PATH = Path(__file__).resolve().parent / "chat_gateway_v1.schema.json"

# The models that form the public wire contract, keyed by their contract role.
# Order is stable so the generated document is byte-deterministic.
_CONTRACT_MODELS = {
    "Question": Question,               # request body
    "ConversationTurn": ConversationTurn,  # request sub-object
    "APIResponse": APIResponse,         # success response body
    "APIErrorResponse": APIErrorResponse,  # error envelope
}


def build_contract_document() -> Dict[str, Any]:
    """Render the versioned contract document from the current Pydantic models.

    The returned dict is JSON-serializable and deterministic for a given set of
    models, so a byte-for-byte comparison against the committed artifact is a
    reliable drift signal.
    """
    schemas = {
        name: model.model_json_schema() for name, model in _CONTRACT_MODELS.items()
    }
    return {
        "contract_version": CHAT_GATEWAY_CONTRACT_VERSION,
        "endpoint": "/ask",
        "description": (
            "Versioned request/response/error contract for the Enai chat gateway. "
            "Generated from models.py by scripts/export_chat_gateway_contract.py; "
            "do not edit by hand."
        ),
        "schemas": schemas,
    }


def serialize_contract_document(document: Dict[str, Any] | None = None) -> str:
    """Serialize the contract document to canonical JSON (sorted keys, 2-space)."""
    document = document if document is not None else build_contract_document()
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def load_committed_contract() -> Dict[str, Any]:
    """Load the committed contract artifact from disk."""
    return json.loads(CONTRACT_ARTIFACT_PATH.read_text(encoding="utf-8"))
