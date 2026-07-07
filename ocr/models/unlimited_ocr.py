"""Baidu Unlimited-OCR adapter.

Official repository: https://github.com/baidu/Unlimited-OCR
HuggingFace model:  https://huggingface.co/baidu/Unlimited-OCR
Paper:              https://arxiv.org/abs/2606.23050

Architecture
------------
Unlimited-OCR is a vision-language model that supports two inference configs:

  gundam mode (image_size=640, crop_mode=True):
    • Optimised for single images / single pages
    • Best quality-to-speed ratio for per-page OCR
    • Uses base_size=1024

  base mode (image_size=1024, crop_mode=False):
    • Used for multi-page / full-document parsing in one shot
    • This pipeline uses gundam mode for per-page processing

The model exposes:
  model.infer()       — single image → output file + returned text
  model.infer_multi() — multiple images (multi-page) → output file

This adapter calls ``model.infer()`` in gundam mode for each page, since the
pipeline already handles page splitting and checkpointing.

Usage of trust_remote_code
--------------------------
The model ships custom modelling code on HuggingFace.  Both AutoModel and
AutoTokenizer require ``trust_remote_code=True``.

Inference requirements (tested upstream)
-----------------------------------------
  torch==2.10.0
  torchvision==0.25.0
  transformers==4.57.1
  einops==0.8.2
  addict==2.4.0
  easydict==1.13
  pymupdf==1.27.2.2
  psutil==7.2.2
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

print("Loading PyTorch...")
import torch
print("Loading PIL...")
from PIL import Image

from ocr.base import OCRModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "baidu/Unlimited-OCR"


class UnlimitedOCRModel(OCRModel):
    """Adapter for Baidu Unlimited-OCR model (gundam mode per-page inference)."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cuda",
        # Gundam mode settings (single-page, best quality)
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        max_length: int = 32768,
        no_repeat_ngram_size: int = 35,
        ngram_window: int = 128,
    ) -> None:
        """
        Args:
            model_path:           HuggingFace repo ID or local directory.
            device:               "cuda" or "cpu".
            base_size:            Base size for image processing (gundam=1024).
            image_size:           Target image size (gundam=640).
            crop_mode:            Enable crop mode (gundam=True).
            max_length:           Max generation length in tokens.
            no_repeat_ngram_size: NGram repeat penalty size.
            ngram_window:         NGram window for custom logit processor.
        """
        super().__init__(model_path=model_path, device=device)
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.max_length = max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.ngram_window = ngram_window

        # Set after load_model()
        self._model = None
        self._tokenizer = None

    # ── OCRModel interface ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "unlimited_ocr"

    def load_model(self) -> None:
        """Load Unlimited-OCR weights from HuggingFace or local path."""
        print("Loading HuggingFace transformers (this can take a moment)...")
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        
        # Monkey-patch is_torch_fx_available to prevent ImportError in older cached deepseekv2 models
        import transformers.utils.import_utils
        if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
            transformers.utils.import_utils.is_torch_fx_available = lambda: False

        logger.info("Loading Unlimited-OCR from %s …", self.model_path)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        
        # Load config and patch missing attributes for newer transformers
        # versions. Instead of intercepting AttributeError lazily on the
        # class (which previously invented an incomplete rope_parameters
        # dict and caused a KeyError), we compute correct values once and
        # set them directly on this config instance.
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)

        _rope_theta = getattr(config, "rope_theta", 10000.0)
        _rope_scaling = getattr(config, "rope_scaling", None) or {}

        _defaults = {
            "pad_token_id": getattr(config, "eos_token_id", 0),
            "attention_dropout": 0.0,
            "attention_bias": False,
            "mlp_bias": False,
            "rope_scaling": None,
            "rope_parameters": {
                "rope_type": _rope_scaling.get("rope_type", _rope_scaling.get("type", "default")),
                "rope_theta": _rope_theta,
                **{k: v for k, v in _rope_scaling.items() if k not in ("rope_type", "type")},
            },
        }
        for attr, value in _defaults.items():
            if not hasattr(config, attr):
                setattr(config, attr, value)

        self._model = AutoModel.from_pretrained(
            self.model_path,
            config=config,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=dtype,
        )
        self._model.eval()

        if self.device != "cpu":
            self._model = self._model.cuda()
        else:
            self._model = self._model.cpu()

        logger.info(
            "Unlimited-OCR loaded — dtype=%s, device=%s", dtype, self.device
        )

    def process_page(self, image_path: str | Path) -> str:
        """Run Unlimited-OCR on a single page image (gundam mode).

        The official ``model.infer()`` API writes results to an output directory.
        We capture the text output and return it as a string.

        Args:
            image_path: Path to a rendered page image (PNG/JPEG).

        Returns:
            Markdown string.  Returns an empty string on error.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call ensure_loaded() first.")

        image_path = Path(image_path)
        if not image_path.exists():
            logger.error("Image not found: %s", image_path)
            return ""

        try:
            logger.info("  [%s] Starting inference for %s ...", self.name, image_path.name)
            return self._run_inference(image_path)
        except Exception as exc:
            logger.error("Unlimited-OCR inference failed on %s: %s", image_path.name, exc)
            return ""

    # ── Internal helpers ───────────────────────────────────────────────────

    def _run_inference(self, image_path: Path) -> str:
        """Call model.infer() and retrieve generated text.

        The upstream model.infer() can save results to disk and also returns
        the generated text directly.  We use a temp output_path to avoid
        polluting the working directory, then read the text return value.
        """
        with tempfile.TemporaryDirectory(prefix="uocr_out_") as tmp_out:
            # The official API:
            # model.infer(
            #     tokenizer, prompt, image_file, output_path,
            #     base_size, image_size, crop_mode,
            #     max_length, no_repeat_ngram_size, ngram_window,
            #     save_results
            # )
            result = self._model.infer(
                tokenizer=self._tokenizer,
                prompt="<image>document parsing.",
                image_file=str(image_path),
                output_path=tmp_out,
                base_size=self.base_size,
                image_size=self.image_size,
                crop_mode=self.crop_mode,
                max_length=self.max_length,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                ngram_window=self.ngram_window,
                save_results=True,
            )

        # model.infer() returns the generated text directly.
        # If it returns None (some versions only write to disk), we fall back
        # to an empty string; the caller will log an OCR error.
        if isinstance(result, str):
            return result.strip()

        # Some model versions return a dict with 'text' or 'markdown' key
        if isinstance(result, dict):
            for key in ("text", "markdown", "output", "content"):
                if key in result and isinstance(result[key], str):
                    return result[key].strip()

        logger.warning(
            "Unlimited-OCR returned unexpected type %s for %s — returning empty.",
            type(result).__name__,
            image_path.name,
        )
        return ""
