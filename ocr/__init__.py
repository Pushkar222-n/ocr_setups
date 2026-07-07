"""OCR Pipeline — top-level package."""
from ocr.base import OCRModel
from ocr.pipeline import run_pipeline

__all__ = ["OCRModel", "run_pipeline"]
