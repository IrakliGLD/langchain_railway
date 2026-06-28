# Dockerfile v1.6
# Python 3.11 for compatibility with langsmith/pydantic v1
# Python 3.12 has breaking changes in typing.ForwardRef._evaluate() that cause pydantic v1 to fail
FROM python:3.11-slim

# No system packages required: psycopg2-binary and psycopg[binary] bundle libpq,
# and every other pinned dependency ships a manylinux wheel (no compiler needed).
# This removes the previous `apt-get install libpq-dev gcc` layer, which both
# bloated the image and was the source of an apt-fetch build failure (exit 100).

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . .

# Run the application via Python entrypoint (handles PORT from environment)
CMD ["python", "main.py"]
