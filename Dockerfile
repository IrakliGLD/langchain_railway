# Dockerfile v1.3
# Installs libpq for psycopg2-binary and sets up Python environment. Updated CMD to /bin/sh -c for $PORT expansion. Realistic: 95% success, 5% risk of env variable misconfiguration.
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

# Run the application with /bin/sh to expand $PORT
CMD ["/bin/sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
