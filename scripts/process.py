"""CLI entrypoint for the OCR pipeline.

Usage
-----
    uv run scripts/process.py --model dolphin --input_dir ./input_pdfs

Full options:
    uv run scripts/process.py \\
        --model        dolphin            # or: unlimited_ocr
        --input_dir    ./input_pdfs       # folder with PDFs (searched recursively)
        --output_dir   ./output_md        # where .md files are written
        --checkpoint_dir ./.checkpoints   # per-page crash recovery state
        --model_path   ByteDance/Dolphin-v2  # HF model ID or local path
        --device       cuda               # cuda / cpu
        --dpi          150                # PDF render resolution
        --force                           # reprocess already-done PDFs
        --log_level    INFO               # DEBUG / INFO / WARNING

Environment variables
---------------------
    HF_HOME / HUGGINGFACE_HUB_CACHE  — override HuggingFace cache location
    CUDA_VISIBLE_DEVICES             — select GPU (e.g. "0" or "0,1")
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on the path when running as a script
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ocr.models import MODEL_REGISTRY
from ocr.pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "dolphin"
DEFAULT_INPUT_DIR = "./input_pdfs"
DEFAULT_OUTPUT_DIR = "./output_md"
DEFAULT_CHECKPOINT_DIR = "./.checkpoints"
DEFAULT_DPI = 150
DEFAULT_DEVICE = "cuda"
DEFAULT_LOG_LEVEL = "INFO"

MODEL_DEFAULT_PATHS: dict[str, str] = {
    "dolphin": "ByteDance/Dolphin-v2",
    "unlimited_ocr": "baidu/Unlimited-OCR",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr-process",
        description=(
            "Modular PDF OCR pipeline.\n"
            "Processes PDFs recursively and generates per-PDF Markdown files.\n"
            "Supports page-level checkpointing — safe to interrupt and resume."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default=DEFAULT_MODEL,
        help=f"OCR model to use. Available: {', '.join(MODEL_REGISTRY)}. Default: %(default)s",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help="Folder containing PDFs (searched recursively). Default: %(default)s",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Folder for output Markdown files. Default: %(default)s",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path(DEFAULT_CHECKPOINT_DIR),
        help="Folder for checkpoint JSON files. Default: %(default)s",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "HuggingFace model ID or local path. "
            "Defaults to the recommended hub ID for each model."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Torch device: 'cuda' or 'cpu'. Default: %(default)s",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="PDF render resolution in DPI. Higher = better quality, slower. Default: %(default)s",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess PDFs that already have output Markdown files.",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=DEFAULT_LOG_LEVEL,
        help="Logging verbosity. Default: %(default)s",
    )

    # Model-specific tuning (advanced)
    adv = parser.add_argument_group("Advanced / model-specific options")
    adv.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="[Dolphin] Maximum tokens to generate per page. Default: 4096",
    )
    adv.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="[Unlimited-OCR] Max generation length in tokens. Default: 32768",
    )
    adv.add_argument(
        "--image_mode",
        choices=["gundam", "base"],
        default="gundam",
        help="[Unlimited-OCR] Inference mode. 'gundam' for per-page, 'base' for multi-page. Default: gundam",
    )

    return parser


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace):
    """Construct the correct model adapter from parsed args."""
    model_name = args.model
    model_cls = MODEL_REGISTRY[model_name]
    model_path = args.model_path or MODEL_DEFAULT_PATHS[model_name]

    # Build kwargs common to all models
    kwargs: dict = {
        "model_path": model_path,
        "device": args.device,
    }

    # Model-specific kwargs
    if model_name == "dolphin":
        if args.max_new_tokens is not None:
            kwargs["max_new_tokens"] = args.max_new_tokens

    elif model_name == "unlimited_ocr":
        if args.max_length is not None:
            kwargs["max_length"] = args.max_length
        if args.image_mode == "base":
            # base mode: larger image, no crop
            kwargs["image_size"] = 1024
            kwargs["crop_mode"] = False
            kwargs["ngram_window"] = 1024

    return model_cls(**kwargs)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    logger = logging.getLogger("ocr.main")

    # Validate input directory
    if not args.input_dir.exists():
        logger.error("Input directory does not exist: %s", args.input_dir)
        sys.exit(1)

    # Log configuration
    logger.info("═══════════════════════════════════════")
    logger.info("  OCR Pipeline")
    logger.info("  model        : %s", args.model)
    logger.info("  input_dir    : %s", args.input_dir.resolve())
    logger.info("  output_dir   : %s", args.output_dir.resolve())
    logger.info("  checkpoint   : %s", args.checkpoint_dir.resolve())
    logger.info("  device       : %s", args.device)
    logger.info("  dpi          : %d", args.dpi)
    logger.info("  force        : %s", args.force)
    logger.info("═══════════════════════════════════════")

    # Build model
    model = build_model(args)
    logger.info("Model adapter: %r", model)

    # Run pipeline
    try:
        run_pipeline(
            model=model,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            dpi=args.dpi,
            force=args.force,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.  Progress saved in checkpoints.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
