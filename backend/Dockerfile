FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF and other native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (layer cached separately from source code)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[all]" 2>/dev/null || pip install --no-cache-dir -e .

# Copy source
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/qdrant_storage

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
