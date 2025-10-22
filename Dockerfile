# Dockerfile v1.5
# Python 3.11 for compatibility with langsmith/pydantic v1
# Python 3.12 has breaking changes in typing.ForwardRef._evaluate() that cause pydantic v1 to fail
FROM python:3.11-slim

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

# Run the application via Python entrypoint (handles PORT from environment)
CMD ["python", "main.py"]
