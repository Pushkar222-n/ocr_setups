"""Dolphin OCR adapter — ByteDance/Dolphin (VisionEncoderDecoderModel).

Official repository: https://github.com/bytedance/Dolphin
HuggingFace model:  https://huggingface.co/ByteDance/Dolphin
                    https://huggingface.co/ByteDance/Dolphin-v2  (recommended)

Architecture
------------
Dolphin is a VisionEncoderDecoderModel (Swin Transformer encoder + MBart decoder)
that performs page-level document parsing, outputting structured Markdown.

The "analyze-then-parse" paradigm:
  Stage 1: Layout analysis — identifies anchors (text blocks, tables, formulae …)
  Stage 2: Parallel parsing of each anchor element → Markdown

This adapter uses the standard HuggingFace Transformers API, which works on
any GPU environment (including RunPod bare-metal pods) without a server process.

Usage
-----
The adapter is intentionally thin:  it loads the model once, then calls
``process_page()`` for each page image.  The pipeline handles temp dirs,
checkpointing, and output assembly.

Note on model IDs
-----------------
- ``ByteDance/Dolphin``    → original model (ACL 2025 paper version)
- ``ByteDance/Dolphin-v2`` → improved model with 21-category element support
  The default is Dolphin-v2.  Override with ``--model_path ByteDance/Dolphin``
  if you need the v1 behaviour.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

print("Loading PyTorch...")
import torch
print("Loading PIL...")
from PIL import Image

from ocr.base import OCRModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "ByteDance/Dolphin-v2"


class DolphinModel(OCRModel):
    """Adapter for ByteDance Dolphin document parsing model."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cuda",
        max_new_tokens: int = 4096,
    ) -> None:
        """
        Args:
            model_path:     HuggingFace repo ID or local directory.
            device:         "cuda" or "cpu".
            max_new_tokens: Maximum tokens to generate per page.
        """
        super().__init__(model_path=model_path, device=device)
        self.max_new_tokens = max_new_tokens

        # Set after load_model()
        self._model = None
        self._processor = None
        self._tokenizer = None

    # ── OCRModel interface ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "dolphin"

    def load_model(self) -> None:
        """Load Dolphin weights from HuggingFace or local path."""
        print("Loading HuggingFace transformers (this can take a moment)...")
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
            Qwen2_5_VLForConditionalGeneration,
        )

        logger.info("Loading Dolphin from %s …", self.model_path)

        # Dolphin uses a custom processor bundled in the model repo.
        # trust_remote_code is required to load it correctly.
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
        except Exception:
            # Some Dolphin checkpoints use a plain image processor
            from transformers import AutoImageProcessor
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self._model.eval()
        self._model.to(self.device)
        logger.info("Dolphin loaded — dtype=%s, device=%s", dtype, self.device)

    def process_page(self, image_path: str | Path) -> str:
        """Run Dolphin page-level parsing on a single page image.

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
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            logger.error("Cannot open image %s: %s", image_path, exc)
            return ""

        try:
            logger.info("  [%s] Starting generation for %s ...", self.name, image_path.name)
            return self._run_inference(image)
        except Exception as exc:
            logger.error("Dolphin inference failed on %s: %s", image_path.name, exc)
            return ""

    # ── Internal helpers ───────────────────────────────────────────────────

    def _run_inference(self, image: "Image.Image") -> str:
        """Run Qwen2.5-VL text generation using chat templates."""
        # Qwen2.5-VL expects inputs via the chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract all text, tables, and formatting from this document page into Markdown."},
                ],
            }
        ]

        # Apply chat template
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self._processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        if hasattr(self._model, "dtype"):
            inputs = inputs.to(self._model.dtype)

        # Generate
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        # Trim the prompt tokens from the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode output
        output_text = self._tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()
