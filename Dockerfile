# Dockerfile cho Face Recognition project
# Base: PyTorch với CUDA support để có thể train trên GPU
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Metadata
LABEL maintainer="Group 12 - Biometric Project"
LABEL description="Face recognition with partial facial visibility"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (tận dụng Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ project
COPY . .

# Set Python path
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Default command - có thể override khi chạy container
CMD ["/bin/bash"]

# === Build & Run ===
# Build:
#   docker build -t biometric-project .
# Run (CPU):
#   docker run -it --rm -v $(pwd):/workspace biometric-project
# Run (GPU):
#   docker run -it --rm --gpus all -v $(pwd):/workspace biometric-project
# Train:
#   docker run --rm --gpus all -v $(pwd):/workspace biometric-project \
#       python scripts/train.py --config configs/train/exp_001_baseline.yaml
