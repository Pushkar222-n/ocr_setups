"""Main pipeline orchestration.

This module wires together:
  - Recursive PDF discovery
  - Per-PDF output path mirroring
  - Page-wise checkpointing
  - Model inference calls
  - Markdown file assembly
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterator, List

from ocr.base import OCRModel
from ocr.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from ocr.pdf_utils import get_page_count, render_single_page

logger = logging.getLogger(__name__)

# Markdown page separator template
PAGE_MARKER = "<!-- page: {page_num} -->"


# ---------------------------------------------------------------------------
# PDF discovery
# ---------------------------------------------------------------------------

def discover_pdfs(input_dir: Path) -> List[Path]:
    """Recursively find all PDF files under input_dir.

    Matches both .pdf and .PDF extensions, sorted for deterministic order.

    Returns:
        Absolute paths to all discovered PDFs.
    """
    pdfs: List[Path] = []
    for pattern in ("**/*.pdf", "**/*.PDF"):
        pdfs.extend(input_dir.rglob(pattern[3:]))  # rglob drops the **/ prefix

    # deduplicate (case-insensitive FS may double-count)
    seen: set[Path] = set()
    unique: List[Path] = []
    for p in pdfs:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)

    return sorted(unique)


# ---------------------------------------------------------------------------
# Output path mirroring
# ---------------------------------------------------------------------------

def get_output_md_path(input_dir: Path, output_dir: Path, pdf_path: Path) -> Path:
    """Mirror the PDF's relative position under output_dir with .md extension.

    Example:
        input_dir  = /data/input_pdfs
        pdf_path   = /data/input_pdfs/research/2024/paper.pdf
        output_dir = /data/output_md
        → returns  /data/output_md/research/2024/paper.md
    """
    relative = pdf_path.relative_to(input_dir)
    return output_dir / relative.with_suffix(".md")


def get_relative_path(input_dir: Path, pdf_path: Path) -> str:
    """Return the PDF path relative to input_dir as a POSIX string (used as checkpoint key)."""
    return pdf_path.relative_to(input_dir).as_posix()


# ---------------------------------------------------------------------------
# Single-PDF processing
# ---------------------------------------------------------------------------

def _assemble_markdown(ckpt: Checkpoint) -> str:
    """Stitch per-page texts into a single Markdown document."""
    parts: List[str] = []
    for page_idx in range(ckpt.total_pages):
        page_num = page_idx + 1  # 1-indexed for readability
        parts.append(PAGE_MARKER.format(page_num=page_num))
        parts.append("")  # blank line after marker
        text = ckpt.page_texts.get(page_idx, "")
        parts.append(text.strip())
        parts.append("")  # blank line between pages
    return "\n".join(parts).rstrip() + "\n"


def process_pdf(
    model: OCRModel,
    pdf_path: Path,
    input_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    dpi: int = 150,
    force: bool = False,
) -> bool:
    """Process a single PDF, resuming from checkpoint if interrupted.

    Args:
        model:          Loaded OCR model.
        pdf_path:       Absolute path to the PDF.
        input_dir:      Root input directory (for relative path calculations).
        output_dir:     Root output directory.
        checkpoint_dir: Where checkpoint JSONs are stored.
        dpi:            PDF render resolution.
        force:          If True, reprocess even if output already exists.

    Returns:
        True on success, False if processing failed.
    """
    rel_path = get_relative_path(input_dir, pdf_path)
    out_md_path = get_output_md_path(input_dir, output_dir, pdf_path)

    # ── Skip if already complete ──────────────────────────────────────────
    if not force and out_md_path.exists():
        logger.info("✓ Already done: %s — skipping.", rel_path)
        return True

    logger.info("→ Processing: %s", rel_path)

    # ── Determine page count ───────────────────────────────────────────────
    try:
        total_pages = get_page_count(pdf_path)
    except Exception as exc:
        logger.error("Cannot open PDF %s: %s", rel_path, exc)
        return False

    if total_pages == 0:
        logger.warning("PDF %s has 0 pages — skipping.", rel_path)
        return False

    # ── Load or restore checkpoint ─────────────────────────────────────────
    ckpt = load_checkpoint(checkpoint_dir, rel_path, total_pages)

    remaining = ckpt.remaining_pages
    if not remaining:
        logger.info("Checkpoint says all pages done for %s — writing output.", rel_path)
        _write_output(ckpt, out_md_path)
        return True

    logger.info(
        "  Pages remaining: %d / %d", len(remaining), total_pages
    )

    # ── Create a per-PDF temp dir for page images ──────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="pdf_ocr_"))

    try:
        for page_idx in remaining:
            page_num = page_idx + 1
            t0 = time.perf_counter()

            # Render this specific page to an image
            try:
                img_path = render_single_page(pdf_path, page_idx, dpi=dpi, output_dir=tmp_dir)
            except Exception as exc:
                logger.error("  Page %d render error: %s", page_num, exc)
                ckpt.mark_page_done(page_idx, f"<!-- render error: {exc} -->")
                save_checkpoint(checkpoint_dir, ckpt)
                continue

            # Run OCR
            try:
                text = model.process_page(img_path)
            except Exception as exc:
                logger.error("  Page %d OCR error: %s", page_num, exc)
                text = f"<!-- ocr error on page {page_num}: {exc} -->"

            elapsed = time.perf_counter() - t0
            logger.info(
                "  [%s] Page %d/%d — %.1fs — %d chars",
                model.name, page_num, total_pages, elapsed, len(text),
            )

            ckpt.mark_page_done(page_idx, text)
            save_checkpoint(checkpoint_dir, ckpt)

            # Clean up this page's image to save disk space
            try:
                img_path.unlink(missing_ok=True)
            except OSError:
                pass

    finally:
        # Clean up temp dir regardless of success/failure
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Write final Markdown output ────────────────────────────────────────
    _write_output(ckpt, out_md_path)
    logger.info("✓ Complete: %s → %s", rel_path, out_md_path)
    return True


def _write_output(ckpt: Checkpoint, out_path: Path) -> None:
    """Write assembled Markdown to disk, creating parent dirs as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = _assemble_markdown(ckpt)
    out_path.write_text(markdown, encoding="utf-8")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    model: OCRModel,
    input_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    dpi: int = 150,
    force: bool = False,
) -> None:
    """Discover all PDFs and process them with checkpointing.

    Args:
        model:          An OCRModel instance (already loaded or will be loaded lazily).
        input_dir:      Root folder containing PDFs (searched recursively).
        output_dir:     Root folder for Markdown outputs (mirrors input structure).
        checkpoint_dir: Folder for per-PDF checkpoint JSON files.
        dpi:            PDF render resolution passed to PyMuPDF.
        force:          Reprocess PDFs that already have output files.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    checkpoint_dir = checkpoint_dir.resolve()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover PDFs
    pdfs = discover_pdfs(input_dir)
    if not pdfs:
        logger.warning("No PDF files found under %s.", input_dir)
        return

    logger.info("Found %d PDF(s) under %s", len(pdfs), input_dir)

    # Ensure model is loaded before starting the loop
    model.ensure_loaded()

    # Process each PDF
    success_count = 0
    fail_count = 0

    for i, pdf_path in enumerate(pdfs, start=1):
        logger.info("── [%d/%d] %s", i, len(pdfs), pdf_path.name)
        ok = process_pdf(
            model=model,
            pdf_path=pdf_path,
            input_dir=input_dir,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            dpi=dpi,
            force=force,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    logger.info(
        "Pipeline complete — %d succeeded, %d failed.",
        success_count, fail_count,
    )
