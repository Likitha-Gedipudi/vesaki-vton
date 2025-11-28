# Vesaki-VTON Docker Image
# Multi-stage build for optimized production image

# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements
COPY requirements_advanced.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements_advanced.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY models/ ./models/
COPY losses/ ./losses/
COPY data/ ./data/
COPY utils/ ./utils/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY *.py ./
COPY haarcascade_frontalface_default.xml ./

# Create directories for data and checkpoints
RUN mkdir -p /app/checkpoints /app/logs /app/dataset /app/results

# Download model weights (if not mounting volume)
RUN python3 scripts/download_models.py || echo "Model download skipped"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run API server
CMD ["python3", "api_server.py"]


# Alternative: GPU-enabled image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# GPU version continues from here
FROM gpu-base as gpu-dependencies

COPY requirements_advanced.txt .

RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install -r requirements_advanced.txt

FROM gpu-dependencies as gpu-application

COPY models/ ./models/
COPY losses/ ./losses/
COPY data/ ./data/
COPY utils/ ./utils/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY *.py ./
COPY haarcascade_frontalface_default.xml ./

RUN mkdir -p /app/checkpoints /app/logs /app/dataset /app/results
RUN python3 scripts/download_models.py || echo "Model download skipped"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["python3", "api_server.py"]

