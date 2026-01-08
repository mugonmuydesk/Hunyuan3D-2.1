# Dockerfile for Hunyuan3D-2.1 on RunPod Serverless
# Image-to-3D generation with PBR materials
# Requires ~29GB VRAM (RTX 4090, A100, etc.)

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libgomp1 \
    build-essential \
    ninja-build \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-dev python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default (required for bpy==4.0)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python -m ensurepip --upgrade

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch 2.5.1 with CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Set working directory
WORKDIR /app

# Clone Hunyuan3D-2.1 repository
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git . && \
    git lfs pull

# Install Python dependencies (fix bpy version - 4.0 doesn't exist, use 4.2.0)
# Use --ignore-installed to handle distutils-installed packages in base image
RUN sed -i 's/bpy==4.0/bpy>=4.2.0/' requirements.txt && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt

# Install RunPod SDK
RUN pip install --no-cache-dir runpod huggingface_hub[cli]

# Build custom rasterizer
WORKDIR /app/hy3dpaint/custom_rasterizer
RUN pip install --no-cache-dir .

# Build mesh painter
WORKDIR /app/hy3dpaint
RUN bash compile_mesh_painter.sh || echo "Mesh painter compilation optional"

# Download RealESRGAN weights
WORKDIR /app
RUN mkdir -p weights && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -O weights/RealESRGAN_x4plus.pth || echo "RealESRGAN weights optional"

# Create model cache directory
RUN mkdir -p /models

# Pre-download model weights (makes image large but faster cold starts)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/Hunyuan3D-2.1', local_dir='/models/Hunyuan3D-2.1')" || echo "Model download will happen at runtime"

# Copy handler
COPY handler.py /app/handler.py

# Set environment variables
ENV HF_HOME=/models
ENV MAX_NUM_VIEW=6
ENV TEXTURE_RESOLUTION=512

# Expose port (optional)
EXPOSE 8000

# Start handler
WORKDIR /app
CMD ["python", "-u", "handler.py"]
