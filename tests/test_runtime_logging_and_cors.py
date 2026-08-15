from __future__ import annotations

import io
import logging

from utils.cors_config import parse_allowed_origins
from utils.logging_config import build_application_log_handlers


def test_cors_parser_drops_path_bearing_origin_and_warns_by_count(caplog):
    raw = "https://dashboard.galdava.com,https://dashboard.galdava.com/chat"

    with caplog.at_level(logging.WARNING, logger="Enai"):
        origins = parse_allowed_origins(raw)

    assert origins == ["https://dashboard.galdava.com"]
    assert "Ignoring invalid CORS origins: count=1" in caplog.text
    assert "/chat" not in caplog.text


def test_cors_parser_normalizes_root_slash_and_deduplicates():
    origins = parse_allowed_origins(
        "https://dashboard.galdava.com/, https://dashboard.galdava.com"
    )

    assert origins == ["https://dashboard.galdava.com"]


def test_cors_parser_lowercases_host_so_browser_origins_match():
    """Browsers send ``Origin`` with a lower-cased host.

    Without case-folding, a mixed-case configured origin never matches a real
    request and the failure looks like the configuration being ignored.
    """
    assert parse_allowed_origins("https://Dashboard.Galdava.COM") == [
        "https://dashboard.galdava.com"
    ]
    assert parse_allowed_origins("HTTPS://Dashboard.Galdava.com:8443") == [
        "https://dashboard.galdava.com:8443"
    ]


def test_cors_parser_deduplicates_across_case():
    origins = parse_allowed_origins(
        "https://Dashboard.Galdava.com,https://dashboard.galdava.com"
    )

    assert origins == ["https://dashboard.galdava.com"]


def test_cors_parser_fails_closed_for_non_origins():
    origins = parse_allowed_origins(
        "*,ftp://dashboard.galdava.com,https://user:pass@dashboard.galdava.com,"
        "https://dashboard.galdava.com?source=test"
    )

    assert origins == []


def test_application_log_handlers_split_info_from_warning_and_error():
    stdout = io.StringIO()
    stderr = io.StringIO()
    logger = logging.Logger("stream-split-test", level=logging.INFO)
    logger.propagate = False
    for handler in build_application_log_handlers(stdout=stdout, stderr=stderr):
        logger.addHandler(handler)

    logger.info("normal lifecycle")
    logger.warning("operator warning")
    logger.error("terminal error")

    assert "normal lifecycle" in stdout.getvalue()
    assert "operator warning" not in stdout.getvalue()
    assert "terminal error" not in stdout.getvalue()
    assert "normal lifecycle" not in stderr.getvalue()
    assert "operator warning" in stderr.getvalue()
    assert "terminal error" in stderr.getvalue()
