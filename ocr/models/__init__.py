"""Model registry — maps CLI names to adapter classes.

To register a new model:
  1. Create ocr/models/<name>.py with a class subclassing OCRModel.
  2. Import it here and add it to MODEL_REGISTRY.
"""

from __future__ import annotations

from ocr.models.dolphin import DolphinModel
from ocr.models.unlimited_ocr import UnlimitedOCRModel

MODEL_REGISTRY: dict[str, type] = {
    "dolphin": DolphinModel,
    "unlimited_ocr": UnlimitedOCRModel,
}

__all__ = ["MODEL_REGISTRY", "DolphinModel", "UnlimitedOCRModel"]
