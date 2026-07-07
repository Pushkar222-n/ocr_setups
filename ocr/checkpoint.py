"""Page-level checkpoint persistence.

One JSON file is written per PDF, keyed by the SHA-256 hash of its
path relative to the input root.  This keeps .checkpoints/ flat
regardless of how deeply nested the source PDFs are.

Checkpoint schema
-----------------
{
    "pdf_path":        "subdir/report.pdf",          # relative to input_dir
    "total_pages":     42,
    "completed_pages": [0, 1, 2, 3],                 # 0-indexed page numbers
    "page_texts": {
        "0": "## Page 1\\n...",
        "1": "## Page 2\\n..."
    },
    "complete":        false
}
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Checkpoint:
    """In-memory representation of a single-PDF checkpoint."""

    def __init__(
        self,
        pdf_path: str,
        total_pages: int,
        completed_pages: Optional[List[int]] = None,
        page_texts: Optional[Dict[int, str]] = None,
        complete: bool = False,
    ) -> None:
        self.pdf_path = pdf_path          # relative path (str)
        self.total_pages = total_pages
        self.completed_pages: List[int] = sorted(completed_pages or [])
        self.page_texts: Dict[int, str] = {
            int(k): v for k, v in (page_texts or {}).items()
        }
        self.complete = complete

    @property
    def next_page(self) -> int:
        """Return the next 0-indexed page to process."""
        if not self.completed_pages:
            return 0
        return max(self.completed_pages) + 1

    @property
    def remaining_pages(self) -> List[int]:
        """Return list of 0-indexed page numbers not yet processed."""
        done = set(self.completed_pages)
        return [i for i in range(self.total_pages) if i not in done]

    def mark_page_done(self, page_idx: int, text: str) -> None:
        """Record a successfully processed page."""
        self.page_texts[page_idx] = text
        if page_idx not in self.completed_pages:
            self.completed_pages.append(page_idx)
            self.completed_pages.sort()
        if len(self.completed_pages) >= self.total_pages:
            self.complete = True

    def to_dict(self) -> dict:
        return {
            "pdf_path": self.pdf_path,
            "total_pages": self.total_pages,
            "completed_pages": self.completed_pages,
            "page_texts": {str(k): v for k, v in self.page_texts.items()},
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        return cls(
            pdf_path=data["pdf_path"],
            total_pages=data["total_pages"],
            completed_pages=data.get("completed_pages", []),
            page_texts=data.get("page_texts", {}),
            complete=data.get("complete", False),
        )


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _checkpoint_key(relative_pdf_path: str) -> str:
    """Stable filename: SHA-256 of the relative PDF path."""
    return hashlib.sha256(relative_pdf_path.encode()).hexdigest()


def checkpoint_file(checkpoint_dir: Path, relative_pdf_path: str) -> Path:
    """Return the .json path for a given PDF."""
    key = _checkpoint_key(relative_pdf_path)
    return checkpoint_dir / f"{key}.json"


def load_checkpoint(
    checkpoint_dir: Path,
    relative_pdf_path: str,
    total_pages: int,
) -> Checkpoint:
    """Load an existing checkpoint or create a fresh one.

    Args:
        checkpoint_dir: Directory where checkpoint files live.
        relative_pdf_path: PDF path relative to input_dir (used as key).
        total_pages: Expected total page count for this PDF.

    Returns:
        A Checkpoint object — either restored or newly initialised.
    """
    path = checkpoint_file(checkpoint_dir, relative_pdf_path)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ckpt = Checkpoint.from_dict(data)
            logger.debug(
                "Checkpoint restored for %s: %d/%d pages done.",
                relative_pdf_path,
                len(ckpt.completed_pages),
                ckpt.total_pages,
            )
            return ckpt
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "Corrupt checkpoint for %s (%s) — starting fresh.", relative_pdf_path, exc
            )

    return Checkpoint(
        pdf_path=relative_pdf_path,
        total_pages=total_pages,
    )


def save_checkpoint(checkpoint_dir: Path, ckpt: Checkpoint) -> None:
    """Persist a checkpoint to disk atomically (write-then-replace)."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_file(checkpoint_dir, ckpt.pdf_path)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ckpt.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.debug("Checkpoint saved for %s.", ckpt.pdf_path)
