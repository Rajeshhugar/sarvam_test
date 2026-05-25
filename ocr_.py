#!/usr/bin/env python3
"""
Enhanced Sarvam Vision OCR Pipeline
- Concurrent processing (ThreadPoolExecutor)
- Retry logic with exponential backoff
- File + console logging
- Supports JSON, TXT, and MD output formats
"""

import json
import os
import zipfile
import tempfile
import argparse
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sarvamai import SarvamAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}

# Layout tags we don't want in the extracted text output
SKIP_TAGS = {
    "header", "footnote", "figure", "page-number", "footer", "sidebar",
    "number", "footer_image", "header_image", "vision_footnote", "image",
    "seal", "aside_text", "chart", "advertisement", "photograph",
    "diagram"
}


# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

def setup_logging(log_file: str = "sarvam_ocr.log") -> logging.Logger:
    """
    Sets up a logger that writes to both the console and a log file.

    Why: Helps track what was processed, skipped, or errored across runs,
    especially useful when processing hundreds of files.
    """
    logger = logging.getLogger("sarvam_vision")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler — shows INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler — captures everything including DEBUG
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────

def _is_image_dir(p: Path) -> bool:
    """
    Checks if a directory is a 'leaf image directory' — i.e., it contains
    images but no further nested subdirectories.

    Why: The API treats a folder of images as a single multi-page document
    (like a scanned book split into per-page PNGs). We detect this pattern
    so we can send the whole folder as one job, not individual images.
    """
    if not p.is_dir():
        return False
    has_img = any(f.suffix.lower() in IMAGE_EXTENSIONS for f in p.iterdir())
    has_sub = any(f.is_dir() and any(f.iterdir()) for f in p.iterdir())
    return has_img and not has_sub


def _is_processable_file(p: Path) -> bool:
    """True for PDFs and supported image file types."""
    return p.is_file() and (p.suffix.lower() == '.pdf' or p.suffix.lower() in IMAGE_EXTENSIONS)


def discover_input_files(path: Path, base_in: Path, base_out: Path, root=True):
    """
    Recursively walks the input path and builds a list of (input, output, name) tuples.

    Returns:
        List of (input_path, output_md_path, display_name)

    Why: Handles arbitrary folder structures — flat dirs, nested dirs, mixed
    PDF + image folders — and maps each to a corresponding output .md path.
    """
    items = []
    if path.is_file():
        if _is_processable_file(path):
            rel = path.relative_to(base_in) if path != base_in else Path(path.name)
            items.append((path, base_out / rel.with_suffix('.md'), str(rel)))
    elif path.is_dir():
        if not root and _is_image_dir(path):
            rel = path.relative_to(base_in) if path != base_in else Path(path.name)
            items.append((path, base_out / (str(rel) + '.md'), str(rel)))
        else:
            for item in sorted(path.iterdir()):
                if item.name.startswith('.'):
                    continue
                if _is_processable_file(item):
                    rel = item.relative_to(base_in)
                    items.append((item, base_out / rel.with_suffix('.md'), str(rel)))
                elif item.is_dir():
                    if _is_image_dir(item):
                        rel = item.relative_to(base_in)
                        items.append((item, base_out / (str(rel) + '.md'), str(rel)))
                    else:
                        items.extend(discover_input_files(item, base_in, base_out, root=False))
    return items


# ─────────────────────────────────────────────
# ZIP EXTRACTION
# ─────────────────────────────────────────────

def extract_text_from_zip(zpath: Path) -> str:
    """
    Extracts and assembles text from the API's ZIP output.

    The ZIP contains JSON files (one per page), each with a list of 'blocks'.
    Blocks are text regions detected on the page, tagged by layout type
    (e.g. paragraph, header, footnote). We sort by reading_order and skip
    layout types in SKIP_TAGS to produce clean, readable text.

    Returns:
        A single string with all pages joined by double newlines.
    """
    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(zpath, 'r') as zf:
            zf.extractall(d)
        root = Path(d)
        json_dir = root / 'metadata' if (root / 'metadata').exists() else root
        pages = []
        for jf in sorted(json_dir.glob('*.json')):
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            blocks = sorted(data.get('blocks', []), key=lambda x: x.get('reading_order', 0))
            parts = [b['text'] for b in blocks
                     if b.get('layout_tag') not in SKIP_TAGS and b.get('text')]
            if parts:
                pages.append('\n\n'.join(parts))
        return '\n\n'.join(pages)


# ─────────────────────────────────────────────
# RETRY LOGIC
# ─────────────────────────────────────────────

def run_with_retry(fn, retries: int = 3, backoff: float = 2.0, logger=None):
    """
    Calls fn() and retries up to `retries` times on failure,
    with exponential backoff between attempts.

    Why: The Sarvam API (like all cloud APIs) can have transient failures —
    rate limits, timeouts, network blips. Retrying automatically makes the
    pipeline robust without manual intervention.

    Args:
        fn: A zero-argument callable to attempt.
        retries: Max number of total attempts.
        backoff: Base wait time in seconds (doubles each retry).
    """
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            wait = backoff ** attempt
            if logger:
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {wait:.1f}s...")
            if attempt == retries:
                raise
            time.sleep(wait)


# ─────────────────────────────────────────────
# CORE OCR FUNCTION
# ─────────────────────────────────────────────

def ocr_and_save_document(
    client: SarvamAI,
    inp: Path,
    out_md: Path,
    output_format: str = "md",
    retries: int = 3,
    logger=None,
) -> bool:
    """
    Runs OCR on a single document (PDF or image directory) and saves the result.

    Steps:
      1. Collect files (single PDF, or all images in a dir).
      2. For each file: create API job → upload → start → wait → download ZIP.
      3. Extract text from all ZIPs and join into one string.
      4. Save as .md, .txt, or .json depending on output_format.

    Args:
        client: Authenticated SarvamAI client.
        inp: Input path (file or directory).
        out_md: Output file path (extension adjusted per format).
        output_format: One of "md", "txt", "json".
        retries: Number of API retry attempts per file.
        logger: Optional logger instance.

    Returns:
        True if successful, False if no files found.
    """
    files = [inp] if inp.is_file() else sorted(
        f for f in inp.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        return False

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zips = []

        for i, f in enumerate(files):
            def _run_job(f=f, i=i):
                job = client.document_intelligence.create_job(language="en-IN", output_format="md")
                job.upload_file(str(f))
                job.start()
                job.wait_until_complete()
                zpath = tmp / f"out_{i}.zip"
                job.download_output(str(zpath))
                return zpath

            zpath = run_with_retry(_run_job, retries=retries, logger=logger)
            zips.append(zpath)

        content = '\n\n'.join(extract_text_from_zip(z) for z in zips)

        # Adjust output path extension and save
        out_md.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "txt":
            out_path = out_md.with_suffix('.txt')
            out_path.write_text(content, encoding='utf-8')

        elif output_format == "json":
            out_path = out_md.with_suffix('.json')
            payload = {"source": str(inp), "content": content}
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

        else:  # default: markdown
            out_md.write_text(content, encoding='utf-8')

    return True


# ─────────────────────────────────────────────
# WORKER (used by thread pool)
# ─────────────────────────────────────────────

def _worker(args_tuple):
    """
    Thin wrapper around ocr_and_save_document for use with ThreadPoolExecutor.

    Why a wrapper: ThreadPoolExecutor.submit() needs a single callable.
    We pack all arguments into a tuple and unpack here.
    """
    client, path, out_md, kwargs, logger = args_tuple
    name = kwargs.pop("name", str(path))

    if out_md.exists():
        if logger:
            logger.info(f"Skip (exists): {name}")
        return name, "skipped"

    try:
        ocr_and_save_document(client, path, out_md, logger=logger, **kwargs)
        if logger:
            logger.info(f"Done: {name}")
        return name, "ok"
    except Exception as e:
        if logger:
            logger.error(f"Error [{name}]: {e}")
        return name, f"error: {e}"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sarvam Vision OCR — enhanced pipeline.")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("output", help="Output directory for result files")
    parser.add_argument("-s", "--start", type=int, default=None, help="Start index (slice)")
    parser.add_argument("-e", "--end", type=int, default=None, help="End index (slice)")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="Number of concurrent workers (default: 4)")
    parser.add_argument("-r", "--retries", type=int, default=3,
                        help="API retry attempts per file (default: 3)")
    parser.add_argument("--output-format", choices=["md", "txt", "json"], default="md",
                        help="Output format: md (default), txt, or json")
    parser.add_argument("--log-file", default="sarvam_ocr.log",
                        help="Path to log file (default: sarvam_ocr.log)")
    parser.add_argument(
        "--join-line-break-hyphens", dest="join_line_break_hyphens",
        action=argparse.BooleanOptionalAction, default=False,
        help="Post-process: join words split by hyphens at line breaks",
    )
    parser.add_argument(
        "--unwrap-simple-math-latex", dest="unwrap_simple_math_latex",
        action=argparse.BooleanOptionalAction, default=False,
        help="Post-process: unwrap simple math/LaTeX expressions",
    )
    args = parser.parse_args()

    logger = setup_logging(args.log_file)

    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        logger.error("SARVAM_API_KEY environment variable is not set.")
        return

    client = SarvamAI(api_subscription_key=key)

    inp = Path(args.input).resolve()
    out = Path(args.output).resolve()
    items = discover_input_files(inp, inp, out)

    if not items:
        logger.error("No PDFs or images found in input path.")
        return

    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else len(items)
    items = items[start:end]

    logger.info(f"Processing {len(items)} file(s) with {args.workers} worker(s).")

    # Build task list for the thread pool
    tasks = [
        (
            client, path, out_md,
            {
                "name": name,
                "output_format": args.output_format,
                "retries": args.retries,
            },
            logger,
        )
        for path, out_md, name in items
    ]

    results = {"ok": 0, "skipped": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Processing", unit="file") as bar:
            for future in as_completed(futures):
                name, status = future.result()
                if status == "ok":
                    results["ok"] += 1
                elif status == "skipped":
                    results["skipped"] += 1
                else:
                    results["error"] += 1
                bar.update(1)

    logger.info(
        f"Finished. ✓ {results['ok']} processed | "
        f"⏭ {results['skipped']} skipped | "
        f"✗ {results['error']} errors. "
        f"See {args.log_file} for details."
    )


if __name__ == "__main__":
    main()