# P7.A authoritative Railway runtime image.
# Python patch release and multi-platform manifest are both pinned.  Update the
# tag and digest together after rebuilding, scanning, and smoke-testing.
FROM python:3.11.15-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba

# Railway supplies RAILWAY_GIT_COMMIT_SHA for Git-triggered builds.  The
# explicit override lets the protected evidence workflow build any exact SHA.
ARG RAILWAY_GIT_COMMIT_SHA
ARG ENAI_RELEASE_SHA=${RAILWAY_GIT_COMMIT_SHA}
LABEL org.opencontainers.image.revision="${ENAI_RELEASE_SHA}"

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

# Fail the image build unless it carries a full immutable source identity.
# Runtime configuration may confirm this value, but cannot override it.
RUN ENAI_RELEASE_SHA="${ENAI_RELEASE_SHA}" python -c 'import os; from pathlib import Path; from core.release_identity import write_release_identity; write_release_identity(Path("/app/release-identity.json"), os.environ.get("ENAI_RELEASE_SHA", ""))'

USER enai
EXPOSE 3000
CMD ["python", "main.py"]
