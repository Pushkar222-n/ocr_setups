"""Abstract base class for all OCR model adapters.

To add a new model:
  1. Create a new file in ocr/models/<your_model>.py
  2. Subclass OCRModel and implement load_model() and process_page()
  3. Register it in ocr/models/__init__.py MODEL_REGISTRY
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class OCRModel(abc.ABC):
    """Abstract base for OCR model adapters.

    Each adapter wraps a specific model and exposes a uniform interface
    so the pipeline doesn't need to know about model internals.
    """

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        """
        Args:
            model_path: HuggingFace model ID or absolute local path.
            device: Torch device string, e.g. "cuda" or "cpu".
        """
        self.model_path = model_path
        self.device = device
        self._loaded = False

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier for this model (used in logs and checkpoints)."""
        ...

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load weights into memory.  Called once before any process_page()."""
        ...

    @abc.abstractmethod
    def process_page(self, image_path: str | Path) -> str:
        """Run OCR on a single page image and return Markdown text.

        Args:
            image_path: Path to a PNG/JPEG page image.

        Returns:
            Markdown string for that page.  May be empty string on failure.
        """
        ...

    def ensure_loaded(self) -> None:
        """Idempotent model loader — safe to call multiple times."""
        if not self._loaded:
            logger.info("[%s] Loading model from %s …", self.name, self.model_path)
            self.load_model()
            self._loaded = True
            logger.info("[%s] Model ready on %s.", self.name, self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_path={self.model_path!r}, device={self.device!r})"
