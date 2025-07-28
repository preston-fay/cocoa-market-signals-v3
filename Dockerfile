# Dockerfile for Cocoa Market Signals Dashboard
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_minimal.txt .
RUN pip install --no-cache-dir -r requirements_minimal.txt

# Copy application files
COPY src/ ./src/
COPY templates/ ./templates/
COPY data/ ./data/
COPY *.py ./

# Expose port (Railway will override this)
EXPOSE 8002

# Set environment variable for port
ENV PORT=8002

# Run the comprehensive dashboard
CMD ["python", "src/dashboard/app_comprehensive.py"]