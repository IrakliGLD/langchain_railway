"""Immutable backend release-identity contract tests."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Importing ``config`` performs the production startup validation.  Supply the
# same minimal baseline used by the dedicated configuration test module.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from config import validate_runtime_settings
from core.release_identity import (
    RELEASE_IDENTITY_SCHEMA_VERSION,
    load_release_sha,
    normalize_release_sha,
    write_release_identity,
)

VALID_SHA = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
NORMALIZED_SHA = VALID_SHA.lower()
ROOT = Path(__file__).resolve().parents[1]


def test_normalize_release_sha_accepts_only_a_full_commit_sha():
    assert normalize_release_sha(f"  {VALID_SHA}  ") == NORMALIZED_SHA
    assert normalize_release_sha(None) is None
    assert normalize_release_sha("") is None

    for invalid in ("abc123", "g" * 40, "a" * 39, "a" * 41):
        with pytest.raises(RuntimeError, match="40-character hexadecimal"):
            normalize_release_sha(invalid)


def test_embedded_release_identity_is_deterministic_and_authoritative(tmp_path: Path):
    identity_path = tmp_path / "release-identity.json"

    write_release_identity(identity_path, VALID_SHA)

    assert json.loads(identity_path.read_text(encoding="utf-8")) == {
        "git_sha": NORMALIZED_SHA,
        "schema_version": RELEASE_IDENTITY_SCHEMA_VERSION,
    }
    assert load_release_sha(identity_path=identity_path) == NORMALIZED_SHA
    assert load_release_sha(
        identity_path=identity_path,
        environment_value=VALID_SHA,
    ) == NORMALIZED_SHA


def test_runtime_release_identity_cannot_override_the_embedded_image(tmp_path: Path):
    identity_path = tmp_path / "release-identity.json"
    write_release_identity(identity_path, VALID_SHA)

    with pytest.raises(RuntimeError, match="does not match embedded image identity"):
        load_release_sha(
            identity_path=identity_path,
            environment_value="1" * 40,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": "unexpected", "git_sha": NORMALIZED_SHA},
        {"schema_version": RELEASE_IDENTITY_SCHEMA_VERSION, "git_sha": "short"},
        {"schema_version": RELEASE_IDENTITY_SCHEMA_VERSION, "git_sha": NORMALIZED_SHA, "extra": True},
    ],
)
def test_embedded_release_identity_rejects_malformed_or_extended_payloads(
    tmp_path: Path,
    payload: dict,
):
    identity_path = tmp_path / "release-identity.json"
    identity_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="release identity"):
        load_release_sha(identity_path=identity_path)


def test_development_can_use_environment_identity_when_no_image_file_exists(tmp_path: Path):
    assert load_release_sha(
        identity_path=tmp_path / "missing.json",
        environment_value=VALID_SHA,
    ) == NORMALIZED_SHA
    assert load_release_sha(identity_path=tmp_path / "missing.json") is None


def _validate_runtime_release(*, deployment_env: str, release_sha: str | None) -> None:
    validate_runtime_settings(
        supabase_db_url="postgresql://user:pass@localhost/db",
        gateway_shared_secret="gateway",
        session_signing_secret="session",
        evaluate_admin_secret="evaluate",
        auth_mode="gateway_only",
        deployment_env=deployment_env,
        release_sha=release_sha,
        supabase_jwt_secret=None,
        enable_evaluate_endpoint=False,
        allow_evaluate_endpoint=False,
        model_type="openai",
        openai_api_key="test-openai-key",
        google_api_key=None,
    )


@pytest.mark.parametrize("deployment_env", ["staging", "production"])
def test_deployed_runtime_requires_an_exact_release_identity(deployment_env: str):
    with pytest.raises(RuntimeError, match="ENAI_RELEASE_SHA is required"):
        _validate_runtime_release(deployment_env=deployment_env, release_sha=None)

    _validate_runtime_release(deployment_env=deployment_env, release_sha=NORMALIZED_SHA)


@pytest.mark.parametrize("deployment_env", ["development", "test"])
def test_non_deployed_runtime_allows_missing_release_identity(deployment_env: str):
    _validate_runtime_release(deployment_env=deployment_env, release_sha=None)


def test_release_manifest_records_and_verifies_image_revision(tmp_path: Path):
    manifest_path = tmp_path / "backend-release-manifest.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_backend_release_manifest.py"),
        "--git-sha",
        NORMALIZED_SHA,
        "--image-id",
        "sha256:image-id",
        "--image-revision",
        NORMALIZED_SHA,
        "--output",
        str(manifest_path),
    ]

    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "git_sha": NORMALIZED_SHA,
        "image_id": "sha256:image-id",
        "image_revision": NORMALIZED_SHA,
        "inputs": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("Dockerfile", "requirements.txt", "requirements-lock.txt", "railway.json")
        },
        "schema_version": "backend-release-manifest-v2",
    }


def test_release_manifest_rejects_image_revision_mismatch(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_backend_release_manifest.py"),
            "--git-sha",
            NORMALIZED_SHA,
            "--image-id",
            "sha256:image-id",
            "--image-revision",
            "1" * 40,
            "--output",
            str(tmp_path / "manifest.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "must match" in completed.stderr
