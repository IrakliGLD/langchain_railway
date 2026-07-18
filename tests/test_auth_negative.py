"""Negative JWT and gateway-only auth tests (F10-SEC-01 B2.A.3).

Pins the PyJWT 2.13 rejection matrix for utils.auth: unsupported algorithms,
malformed tokens, critical-header handling, claim/expiry/audience policy, and
gateway-only behavior. Public bearer auth stays disabled in production
(ENAI_AUTH_MODE=gateway_only); these tests exist so the verification policy is
locked even while that surface is dormant.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import json
import time

import jwt
import pytest
from fastapi import HTTPException

from utils import auth as auth_module
from utils.auth import authenticate_request

# RFC 7518 recommends >= 32 bytes for HS256; PyJWT 2.13 warns below that.
TEST_SECRET = "unit-test-supabase-jwt-secret-0123456789abcdef"
SUBJECT = "00000000-0000-0000-0000-00000000abcd"


def _claims(**overrides) -> dict:
    base = {"sub": SUBJECT, "aud": "authenticated", "exp": int(time.time()) + 3600}
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def _token(claims: dict | None = None, *, secret: str = TEST_SECRET,
           algorithm: str = "HS256", headers: dict | None = None) -> str:
    return jwt.encode(claims if claims is not None else _claims(),
                      secret, algorithm=algorithm, headers=headers)


def _b64seg(data: dict) -> str:
    return jwt.utils.base64url_encode(json.dumps(data).encode()).decode()


@pytest.fixture()
def bearer_enabled(monkeypatch):
    monkeypatch.setattr(auth_module, "SUPABASE_JWT_SECRET", TEST_SECRET)
    monkeypatch.setattr(auth_module, "ENABLE_PUBLIC_BEARER_AUTH", True)


def _authenticate(token: str):
    return authenticate_request(authorization=f"Bearer {token}", request_id="req-1")


# ---------------------------------------------------------------------------
# Unsupported algorithms
# ---------------------------------------------------------------------------


class TestUnsupportedAlgorithms:
    def test_alg_none_rejected(self, bearer_enabled):
        unsigned = f"{_b64seg({'alg': 'none', 'typ': 'JWT'})}.{_b64seg(_claims())}."
        with pytest.raises(HTTPException) as exc:
            _authenticate(unsigned)
        assert exc.value.status_code == 401

    def test_hs512_rejected_even_with_correct_secret(self, bearer_enabled):
        with pytest.raises(HTTPException) as exc:
            _authenticate(_token(algorithm="HS512"))
        assert exc.value.status_code == 401

    def test_wrong_secret_rejected(self, bearer_enabled):
        with pytest.raises(HTTPException) as exc:
            _authenticate(_token(secret="attacker-controlled-secret-0123456789abcdef"))
        assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# Malformed tokens and headers
# ---------------------------------------------------------------------------


class TestMalformedTokens:
    @pytest.mark.parametrize("token", [
        "",  # empty
        "AAAA",  # one segment
        f"{_b64seg({'alg': 'HS256'})}.e30",  # two segments
        f"AAAA.{_b64seg({'sub': SUBJECT})}.AAAA",  # header segment is not JSON
        "not.base64url.!!!",  # undecodable segments
    ])
    def test_malformed_rejected(self, bearer_enabled, token):
        with pytest.raises(HTTPException) as exc:
            _authenticate(token)
        assert exc.value.status_code == 401


class TestCriticalHeaders:
    def test_unknown_critical_extension_rejected(self, bearer_enabled):
        # PyJWT 2.13 enforces RFC 7515 4.1.11: unsupported crit params fail.
        token = _token(headers={"crit": ["x-custom"], "x-custom": 1})
        with pytest.raises(HTTPException) as exc:
            _authenticate(token)
        assert exc.value.status_code == 401

    def test_malformed_crit_header_rejected(self, bearer_enabled):
        token = _token(headers={"crit": "exp"})
        with pytest.raises(HTTPException) as exc:
            _authenticate(token)
        assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# Expiry, required claims, audience/issuer policy
# ---------------------------------------------------------------------------


class TestClaimPolicy:
    @pytest.mark.parametrize("claims", [
        _claims(exp=int(time.time()) - 10),  # expired
        _claims(exp=None),  # missing required exp
        _claims(sub=None),  # missing required sub
        _claims(aud=None),  # missing audience
        _claims(aud="other-project"),  # wrong audience
        _claims(sub="   "),  # whitespace-only subject normalizes to empty
    ])
    def test_rejected(self, bearer_enabled, claims):
        with pytest.raises(HTTPException) as exc:
            _authenticate(_token(claims))
        assert exc.value.status_code == 401

    def test_issuer_is_not_part_of_current_policy(self, bearer_enabled):
        # Documented policy: audience is enforced, issuer is not. Production
        # remains gateway_only, so bearer tokens are not accepted there at all;
        # revisit issuer pinning if public bearer auth is ever enabled.
        ctx = _authenticate(_token(_claims(iss="https://unrelated.example")))
        assert ctx.auth_mode == "public_bearer"
        assert ctx.subject_id == f"user:{SUBJECT}"


# ---------------------------------------------------------------------------
# Gateway-only behavior (production posture)
# ---------------------------------------------------------------------------


class TestGatewayOnlyBehavior:
    def test_valid_bearer_rejected_when_bearer_disabled(self, monkeypatch):
        monkeypatch.setattr(auth_module, "SUPABASE_JWT_SECRET", TEST_SECRET)
        monkeypatch.setattr(auth_module, "ENABLE_PUBLIC_BEARER_AUTH", False)
        with pytest.raises(HTTPException) as exc:
            _authenticate(_token())
        assert exc.value.status_code == 401

    def test_gateway_secret_still_authenticates_when_bearer_disabled(self, monkeypatch):
        monkeypatch.setattr(auth_module, "ENABLE_PUBLIC_BEARER_AUTH", False)
        monkeypatch.setattr(auth_module, "GATEWAY_ACTOR_ASSERTION_MODE", "optional")
        ctx = authenticate_request(x_app_key="test-gateway-key", request_id="req-1")
        assert ctx.auth_mode == "gateway"
        assert ctx.subject_id == "gateway:internal"

    def test_wrong_gateway_secret_with_bearer_disabled_rejected(self, monkeypatch):
        monkeypatch.setattr(auth_module, "ENABLE_PUBLIC_BEARER_AUTH", False)
        with pytest.raises(HTTPException) as exc:
            authenticate_request(x_app_key="wrong-key", request_id="req-1")
        assert exc.value.status_code == 401

    def test_no_credentials_rejected(self):
        with pytest.raises(HTTPException) as exc:
            authenticate_request(request_id="req-1")
        assert exc.value.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
