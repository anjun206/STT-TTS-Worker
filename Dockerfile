FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# 필수 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential git git-lfs curl ca-certificates \
    ffmpeg sox libsndfile1 libsndfile1-dev \
    libgl1 libglib2.0-0 \
    ninja-build cmake pkg-config \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Python 3.11 (deadsnakes)
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

# 전용 가상환경 생성 및 활성 경로 설정
RUN python3.11 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pip 최신화
RUN python -m ensurepip && python -m pip install --upgrade pip setuptools wheel

# ⚠️ 주의: TORCH_CUDA_ARCH_LIST는 SM 버전으로!
# L4는 8.9이므로 8.9로 설정(잘못된 "12.0+PTX" 제거)
ENV TORCH_CUDA_ARCH_LIST="8.9"

# cu12.8 PyTorch 계열 설치
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# 나머지 파이썬 패키지
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY entrypoint.sh /app/entrypoint.sh

ENV HF_HOME=/app/cache/hf \
    TRANSFORMERS_CACHE=/app/cache/hf \
    HUGGINGFACE_HUB_CACHE=/app/cache/hf \
    TTS_HOME=/app/cache/tts \
    MT_KO_EN_MODEL="Helsinki-NLP/opus-mt-tc-big-ko-en"

RUN chmod +x /app/entrypoint.sh && \
    useradd -m appuser && \
    mkdir -p /app/cache/hf /app/cache/tts && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
ENV APP_MODE=worker
CMD ["/app/entrypoint.sh"]
