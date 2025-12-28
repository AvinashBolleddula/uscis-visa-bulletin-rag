# Cloud SDK image includes gsutil (needed by scripts/gcs_sync.py)
FROM gcr.io/google.com/cloudsdktool/cloud-sdk:slim

# Install Python 3.12 + build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-distutils \
    ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

# Make python3 point to 3.12
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3

WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock ./

# Create venv and install deps
RUN uv sync --frozen

# Copy the rest of the repo
COPY . .

# Cloud Run expects the app to listen on $PORT
ENV PORT=8080

# Streamlit config (optional but helps on Cloud Run)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=${PORT}

EXPOSE 8080

# Start Streamlit
CMD ["uv", "run", "streamlit", "run", "ui/app.py", "--server.port=8080", "--server.address=0.0.0.0"]