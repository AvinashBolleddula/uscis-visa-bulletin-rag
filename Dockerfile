# Use official Python image (already has Python installed)
FROM python:3.12-slim

# Prevent Python buffering issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cloud Run expects port 8080
ENV PORT=8080

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (better Docker caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies INTO container
RUN uv sync --no-dev

# Copy app code
COPY . .

# Expose Cloud Run port
EXPOSE 8080

# Start Streamlit correctly for Cloud Run
CMD ["streamlit", "run", "ui/app.py", "--server.port=8080", "--server.address=0.0.0.0"]