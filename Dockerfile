# syntax=docker/dockerfile:1.7

# CUDA 12.6 is the newest GPU profile exercised by this repository. The digest
# pins the multi-platform image inspected on 2026-07-18.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@sha256:8aef630a54bc5c5146ae5ce68e6af5caa3df0fb690bb91544175c91f307e4356
FROM ${CUDA_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
        python3 \
        python3-pip \
        python3-venv \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY configs/ ./configs/

# Install the officially paired PyTorch 2.13 / torchvision 0.28 CUDA 12.6
# wheels first. The project install then reuses this exact local-version build.
RUN python -m pip install --upgrade \
        "pip==26.1.2" "setuptools==83.0.0" "wheel==0.47.0" \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cu126 \
        "torch==2.13.0" "torchvision==0.28.0" \
    && python -m pip install . \
    && python -m pip check

RUN groupadd --gid 10001 soccer \
    && useradd --uid 10001 --gid soccer --create-home --shell /usr/sbin/nologin soccer \
    && mkdir -p /app/artifacts /app/data /app/outputs /app/state \
    && chown -R soccer:soccer /app /home/soccer

USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3).read()"]


FROM runtime AS development
USER root
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY Makefile requirements.txt .pre-commit-config.yaml ./
RUN python -m pip install ".[dev]" && python -m pip check
USER 10001:10001
CMD ["bash"]


FROM runtime AS mlops
USER root
RUN python -m pip install ".[mlops]" && python -m pip check
USER 10001:10001


FROM runtime AS production
