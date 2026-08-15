"""Application logging configuration with severity-based stream routing."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


class _BelowWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


def build_application_log_handlers(
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    format_string: str = DEFAULT_LOG_FORMAT,
) -> tuple[logging.Handler, logging.Handler]:
    """Build non-overlapping handlers: INFO to stdout, WARNING+ to stderr."""
    formatter = logging.Formatter(format_string)

    info_handler = logging.StreamHandler(stdout if stdout is not None else sys.stdout)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(_BelowWarningFilter())
    info_handler.setFormatter(formatter)

    warning_handler = logging.StreamHandler(stderr if stderr is not None else sys.stderr)
    warning_handler.setLevel(logging.WARNING)
    warning_handler.setFormatter(formatter)

    return info_handler, warning_handler


def configure_application_logging(*, level: int = logging.INFO) -> None:
    """Configure root logging while respecting an existing host configuration."""
    logging.basicConfig(
        level=level,
        handlers=list(build_application_log_handlers()),
    )
