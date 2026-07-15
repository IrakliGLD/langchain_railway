# P7.A authoritative Railway runtime image.
# Python patch release and multi-platform manifest are both pinned.  Update the
# tag and digest together after rebuilding, scanning, and smoke-testing.
FROM python:3.11.15-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --gid 10001 enai \
    && useradd --uid 10001 --gid enai --create-home --shell /usr/sbin/nologin enai

WORKDIR /app

# Runtime dependencies only.  Development/test/scanning tools never enter the
# production image.
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --requirement requirements.txt

# Explicit runtime allow-list: no repository-wide COPY and no tests, reports,
# exports, VCS data, operator scripts, or local environment files.
COPY --chown=enai:enai main.py config.py context.py models.py ./
COPY --chown=enai:enai agent ./agent
COPY --chown=enai:enai analysis ./analysis
COPY --chown=enai:enai config_metrics ./config_metrics
COPY --chown=enai:enai contracts ./contracts
COPY --chown=enai:enai core ./core
COPY --chown=enai:enai guardrails ./guardrails
COPY --chown=enai:enai knowledge ./knowledge
COPY --chown=enai:enai schemas ./schemas
COPY --chown=enai:enai skills ./skills
COPY --chown=enai:enai utils ./utils
COPY --chown=enai:enai visualization ./visualization

USER enai
EXPOSE 8000
CMD ["python", "main.py"]
