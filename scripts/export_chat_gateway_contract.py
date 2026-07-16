"""Regenerate the committed chat-gateway contract artifact (P6.A).

Writes ``contracts/chat_gateway_v1.schema.json`` from the Pydantic models that
define the ``/ask`` wire contract. Run after any intended change to Question,
ConversationTurn, APIResponse, or APIErrorResponse:

    python -m scripts.export_chat_gateway_contract

The drift test (tests/test_chat_gateway_contract.py) fails until this is run,
so an unintended contract change cannot merge silently.
"""

from __future__ import annotations

import os

# Config validation runs at import; supply placeholders when unset so the
# exporter works from a clean shell without real secrets.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "export-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "export-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "export-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "export-openai-key")

from contracts.chat_gateway_contract import (  # noqa: E402
    CONTRACT_ARTIFACT_PATH,
    V2_CONTRACT_ARTIFACT_PATH,
    serialize_contract_document,
)
from models import (  # noqa: E402
    CHAT_GATEWAY_CONTRACT_VERSION,
    CHAT_GATEWAY_V2_CONTRACT_VERSION,
)


def main() -> None:
    artifacts = (
        (CHAT_GATEWAY_CONTRACT_VERSION, CONTRACT_ARTIFACT_PATH),
        (CHAT_GATEWAY_V2_CONTRACT_VERSION, V2_CONTRACT_ARTIFACT_PATH),
    )
    for contract_version, artifact_path in artifacts:
        payload = serialize_contract_document(contract_version=contract_version)
        artifact_path.write_text(payload, encoding="utf-8")
        print(f"Wrote {artifact_path} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
