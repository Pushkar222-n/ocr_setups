"""PDF utility functions — converts PDF pages to images via PyMuPDF.

PyMuPDF (fitz) is a fast, dependency-light PDF renderer that works
well in headless Linux/RunPod environments (no display required).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)



def get_page_count(pdf_path: str | Path) -> int:
    """Return the number of pages in a PDF without rendering anything."""
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def pdf_to_page_images(
    pdf_path: str | Path,
    dpi: int = 150,
    output_dir: str | Path | None = None,
) -> List[Path]:
    """Render every page of a PDF to a PNG image.

    Args:
        pdf_path:   Path to the source PDF.
        dpi:        Render resolution (150 dpi balances quality vs. speed).
                    Use 200-300 for high-quality scanned docs.
        output_dir: Where to save page images.  If None, a temporary directory
                    is created automatically (caller owns cleanup).

    Returns:
        Sorted list of Path objects, one per page, in page order.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)

    if output_dir is None:
        tmp = tempfile.mkdtemp(prefix="pdf_ocr_")
        out_dir = Path(tmp)
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    image_paths: List[Path] = []

    logger.debug("Rendering %d pages from %s at %d dpi …", n_pages, pdf_path.name, dpi)

    for i, page in enumerate(doc):
        out_path = out_dir / f"page_{i:04d}.png"
        if not out_path.exists():
            page.get_pixmap(matrix=mat, alpha=False).save(str(out_path))
        image_paths.append(out_path)

    doc.close()
    logger.debug("Rendered %d pages to %s", n_pages, out_dir)
    return sorted(image_paths)


def render_single_page(
    pdf_path: str | Path,
    page_idx: int,
    dpi: int = 150,
    output_dir: str | Path | None = None,
) -> Path:
    """Render a single page from a PDF.

    Args:
        pdf_path:  Path to the PDF.
        page_idx:  0-indexed page number.
        dpi:       Render resolution.
        output_dir: Save location (temp dir if None).

    Returns:
        Path to the rendered PNG.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    if page_idx >= len(doc):
        raise IndexError(f"Page index {page_idx} out of range (PDF has {len(doc)} pages).")

    if output_dir is None:
        tmp = tempfile.mkdtemp(prefix="pdf_ocr_")
        out_dir = Path(tmp)
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"page_{page_idx:04d}.png"
    if not out_path.exists():
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        page = doc[page_idx]
        page.get_pixmap(matrix=mat, alpha=False).save(str(out_path))

    doc.close()
    return out_path
