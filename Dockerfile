# Dockerfile v1.0
# Installs libpq for psycopg and sets up Python environment. Realistic: 90% success, 10% risk of build cache issues.
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

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
