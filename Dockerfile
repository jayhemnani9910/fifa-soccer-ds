# Multi-stage build for FIFA Soccer DS Analytics
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python --version

WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY pyproject.toml .
COPY Makefile .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY configs/ ./configs/
COPY data/raw/ ./data/raw/

# Install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Development stage
FROM base AS development

RUN pip install \
    pytest \
    pytest-cov \
    pre-commit \
    black \
    ruff \
    isort \
    mypy

# Copy additional dev files
COPY .pre-commit-config.yaml .
COPY AGENTS.md .
COPY README.md .

# Production stage
FROM base AS production

# Create non-root user with proper groups
RUN useradd -m -u 1000 -G users soccer && \
    mkdir -p /home/soccer/app/{data,outputs,mlruns} && \
    chown -R soccer:soccer /home/soccer

# Set working directory
WORKDIR /home/soccer/app

# Copy application files with proper ownership
COPY --from=base --chown=soccer:soccer /app . 

# Switch to non-root user
USER soccer

# Add environment variables for production
ENV PYTHONPATH=/home/soccer/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import src.pipeline_full; print('OK')" || exit 1
