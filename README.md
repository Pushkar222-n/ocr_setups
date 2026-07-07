# OCR Pipeline

A modular, production-grade PDF OCR pipeline supporting:

| Model | Source | Mode |
|---|---|---|
| **Dolphin** (ByteDance) | [github.com/bytedance/Dolphin](https://github.com/bytedance/Dolphin) · [HuggingFace](https://huggingface.co/ByteDance/Dolphin-v2) | VisionEncoderDecoder, page-level |
| **Unlimited-OCR** (Baidu) | [github.com/baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) · [HuggingFace](https://huggingface.co/baidu/Unlimited-OCR) | VLM, gundam/base mode |

**Key features:**
- ✅ Recursive PDF discovery (nested subfolders supported)
- ✅ Folder-mirrored `.md` output structure
- ✅ **Page-level checkpointing** — interrupt and resume from the exact last page
- ✅ `uv` dependency management (no pip install)
- ✅ RunPod-compatible (bare-metal pods, HuggingFace Transformers inference)
- ✅ Easily extensible — add new models by subclassing `OCRModel`

---

## Project Structure

```
ocr-pipeline/
├── pyproject.toml          # uv project config & dependencies
├── Dockerfile              # RunPod / container deployment
├── ocr/
│   ├── base.py             # Abstract OCRModel base class
│   ├── checkpoint.py       # Page-level crash-recovery checkpoints
│   ├── pdf_utils.py        # PDF → page images (PyMuPDF)
│   ├── pipeline.py         # Main orchestration loop
│   └── models/
│       ├── dolphin.py      # ByteDance Dolphin adapter
│       └── unlimited_ocr.py  # Baidu Unlimited-OCR adapter
└── scripts/
    └── process.py          # CLI entrypoint (uv run scripts/process.py)
```

---

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- CUDA-capable GPU (recommended; CPU works but is very slow)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/ocr-pipeline.git
cd ocr-pipeline

# Install all dependencies
uv sync
```

### Adding new dependencies

```bash
uv add <package-name>
```

---

## Usage

### Basic — run Dolphin on all PDFs

```bash
uv run scripts/process.py \
  --model dolphin \
  --input_dir ./input_pdfs \
  --output_dir ./output_md
```

### Baidu Unlimited-OCR

```bash
uv run scripts/process.py \
  --model unlimited_ocr \
  --input_dir ./input_pdfs \
  --output_dir ./output_md
```

### Full options

```bash
uv run scripts/process.py \
  --model        dolphin           \  # dolphin | unlimited_ocr
  --input_dir    ./input_pdfs      \  # source folder (recursive)
  --output_dir   ./output_md       \  # output folder (mirrored structure)
  --checkpoint_dir ./.checkpoints  \  # crash recovery state
  --model_path   ByteDance/Dolphin-v2 \  # HF hub ID or local path
  --device       cuda              \  # cuda | cpu
  --dpi          150               \  # PDF render resolution
  --force                          \  # reprocess completed PDFs
  --log_level    INFO                 # DEBUG | INFO | WARNING
```

### Interrupt and resume

The pipeline saves a checkpoint after every page. If it's interrupted (pod
stops, OOM, etc.), just re-run the exact same command — it will skip
already-completed pages and continue from where it stopped.

---

## Input / Output Structure

Input:
```
input_pdfs/
├── annual_report.pdf
└── research/
    ├── paper_a.pdf
    └── 2024/
        └── paper_b.pdf
```

Output (mirrors input exactly):
```
output_md/
├── annual_report.md
└── research/
    ├── paper_a.md
    └── 2024/
        └── paper_b.md
```

Each `.md` file contains page markers:
```markdown
<!-- page: 1 -->
[OCR text for page 1]

<!-- page: 2 -->
[OCR text for page 2]
```

---

## Adding a New Model

1. Create `ocr/models/my_model.py`:

```python
from ocr.base import OCRModel
from pathlib import Path

class MyModel(OCRModel):
    @property
    def name(self) -> str:
        return "my_model"

    def load_model(self) -> None:
        # load weights here
        self._model = ...

    def process_page(self, image_path: str | Path) -> str:
        # run inference, return markdown string
        return "..."
```

2. Register it in `ocr/models/__init__.py`:

```python
from ocr.models.my_model import MyModel

MODEL_REGISTRY = {
    "dolphin": DolphinModel,
    "unlimited_ocr": UnlimitedOCRModel,
    "my_model": MyModel,   # ← add this
}
```

3. Use it:

```bash
uv run scripts/process.py --model my_model --input_dir ./input_pdfs
```

---

## RunPod Deployment

### Option A: Docker (recommended)

```bash
# Build image
docker build -t ocr-pipeline .

# Push to registry
docker push YOUR_REGISTRY/ocr-pipeline:latest
```

In RunPod:
- Use your image as the custom Docker image
- Mount a **Network Volume** at `/workspace` for persistent storage
- Set the override command with your chosen model and paths

### Option B: Bare RunPod pod (no custom Docker)

SSH into your RunPod pod and run:

```bash
# Install uv
curl -Ls https://astral.sh/uv/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"

# Clone your repo
git clone https://github.com/YOUR_USERNAME/ocr-pipeline.git
cd ocr-pipeline

# Install deps
uv sync

# Run
uv run scripts/process.py \
  --model dolphin \
  --input_dir /workspace/input_pdfs \
  --output_dir /workspace/output_md
```

### Environment variables

```bash
# Override HuggingFace cache location (recommended: point to network volume)
export HF_HOME=/workspace/hf_cache

# Select GPU
export CUDA_VISIBLE_DEVICES=0
```

---

## Checkpoint Format

`.checkpoints/<sha256_of_relative_path>.json`:

```json
{
  "pdf_path": "research/paper_a.pdf",
  "total_pages": 42,
  "completed_pages": [0, 1, 2, 3, 4],
  "page_texts": {
    "0": "## Introduction\n...",
    "1": "## Related Work\n..."
  },
  "complete": false
}
```

---

## License

MIT
