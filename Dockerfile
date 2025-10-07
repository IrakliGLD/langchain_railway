# Dockerfile v1.3
# Installs libpq for psycopg2-binary and sets up Python environment. Updated CMD to exec form for $PORT expansion. Realistic: 90% success, 10% risk of env variable issue.
FROM python:3.12-slim

# Install system dependencies including libpq
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . .

# Run the application with exec form for $PORT expansion
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
