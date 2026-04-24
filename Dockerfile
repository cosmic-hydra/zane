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
    git \
    wget \
    cmake \
    swig \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

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

LABEL maintainer="ZANE AI <engineering@zane.ai>"
LABEL description="ZANE Drug Discovery Platform - GPU Optimized"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libxml2 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Environment variables for distributed computing and MLOps
ENV RAY_DASHBOARD_HOST=0.0.0.0
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV DVC_REMOTE_URL=s3://zane-data-lake/dvc-store

# Set default command
CMD ["python3", "-m", "drug_discovery.pipeline"]
