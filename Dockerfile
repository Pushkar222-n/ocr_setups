# ── Base image ────────────────────────────────────────────────────────────────
# CUDA 12.1 runtime on Ubuntu 22.04. Adjust the tag to match your RunPod
# GPU type (e.g. cu118 for older hardware, cu124 for H100/Hopper).
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ── System packages ────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ── Install uv ─────────────────────────────────────────────────────────────────
RUN curl -Ls https://astral.sh/uv/install.sh | bash
ENV PATH="/root/.local/bin:$PATH"

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /workspace/ocr-pipeline

# ── Copy project files ─────────────────────────────────────────────────────────
COPY pyproject.toml ./
COPY ocr/ ./ocr/
COPY scripts/ ./scripts/
COPY README.md ./

# ── Install Python dependencies via uv ─────────────────────────────────────────
# This installs into a .venv managed by uv, honouring pyproject.toml exactly.
RUN uv sync --no-dev

# ── HuggingFace cache inside container (override with -v mount on RunPod) ──────
ENV HF_HOME=/workspace/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/workspace/hf_cache

# ── Volume mount points (RunPod: attach network volumes here) ──────────────────
# /workspace/input_pdfs   → source PDFs
# /workspace/output_md    → generated Markdown
# /workspace/.checkpoints → checkpoint JSONs
# /workspace/hf_cache     → model weights cache (recommended for RunPod)

# ── Default command ─────────────────────────────────────────────────────────────
# Override ARGS at pod launch, e.g.:
#   --model unlimited_ocr --input_dir /workspace/input_pdfs
CMD ["uv", "run", "scripts/process.py", \
     "--model", "dolphin", \
     "--input_dir", "/workspace/input_pdfs", \
     "--output_dir", "/workspace/output_md", \
     "--checkpoint_dir", "/workspace/.checkpoints", \
     "--device", "cuda", \
     "--dpi", "150"]
