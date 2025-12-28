# Dockerfile
FROM python:3.12-slim

# System deps (for pdfplumber/pypdf + curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg \
  && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK (gcloud + gsutil)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list \
  && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
  && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
  && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install -U uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install deps into system site-packages
RUN uv sync --frozen

# Copy the rest of the repo
COPY . .

# Cloud Run listens on $PORT
ENV PORT=8080

# Streamlit must bind to 0.0.0.0 and $PORT
CMD ["bash", "-lc", "uv run streamlit run ui/app.py --server.address 0.0.0.0 --server.port $PORT"]