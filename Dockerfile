# Multi-stage Dockerfile for ZANE Drug Discovery Platform
# Optimized for GPU workloads with PyTorch, PennyLane, and OpenMM

# Stage 1: Build dependencies
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    cmake \
    swig \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install key scientific libraries
RUN pip3 install --no-cache-dir \
    pennylane \
    openmm \
    pdbfixer

# Install remaining dependencies from requirements.txt
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage 2: Final runtime image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

LABEL maintainer="cosmic-hydra"
LABEL description="ZANE Drug Discovery Platform - GPU Optimized"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libxml2 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Install the package in editable mode
RUN pip3 install --no-cache-dir -e .

# Environment variables for distributed computing and MLOps
ENV RAY_DASHBOARD_HOST=0.0.0.0

# Create non-root user for security
RUN groupadd -r zane && useradd -r -g zane -d /app zane \
    && chown -R zane:zane /app
USER zane

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Set default command
CMD ["uvicorn", "infrastructure.api_gateway:app", "--host", "0.0.0.0", "--port", "8000"]
