#!/usr/bin/env python3
"""
Batch OCR runner — processes every invoice image in data/{language}/images/
and saves results to output/{language}/{image_name}.md

Output structure:
    output/
        hindi/
            invoice_xxx_0001.md
            invoice_xxx_0002.md
            ...
        bengali/
            ...
        kannada/ ...
        marathi/ ...
        tamil/  ...
        telugu/ ...
"""

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from sarvamai import SarvamAI
from tqdm import tqdm

from ocr_ import ocr_and_save_document, setup_logging

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
OUTPUT_DIR  = Path("output")
WORKERS     = 4       # concurrent API calls
RETRIES     = 3       # retries per image on transient failures
OUT_FORMAT  = "md"    # md | txt | json
LOG_FILE    = "sarvam_ocr.log"
# ─────────────────────────────────────────────────────────────────────────────


def collect_tasks(data_dir: Path, output_dir: Path):
    """Walk data/{lang}/images/*.png and build (img_path, out_path) pairs."""
    tasks = []
    for lang_dir in sorted(data_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        images_dir = lang_dir / "images"
        if not images_dir.exists():
            continue
        lang = lang_dir.name
        out_lang = output_dir / lang
        for img in sorted(images_dir.glob("*.png")):
            out_path = out_lang / img.with_suffix(".md").name
            tasks.append((lang, img, out_path))
    return tasks


def _worker(args):
    client, lang, img_path, out_path, logger = args

    if out_path.exists():
        logger.info(f"Skip (exists): {lang}/{img_path.name}")
        return lang, img_path.name, "skipped"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ocr_and_save_document(
            client, img_path, out_path,
            output_format=OUT_FORMAT,
            retries=RETRIES,
            logger=logger,
        )
        logger.info(f"Done: {lang}/{img_path.name}")
        return lang, img_path.name, "ok"
    except Exception as e:
        logger.error(f"Error [{lang}/{img_path.name}]: {e}")
        return lang, img_path.name, f"error: {e}"


def main():
    load_dotenv()
    logger = setup_logging(LOG_FILE)

    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        logger.error("SARVAM_API_KEY is not set in .env or environment.")
        return

    client = SarvamAI(api_subscription_key=key)

    tasks = collect_tasks(DATA_DIR, OUTPUT_DIR)
    if not tasks:
        logger.error(f"No images found under {DATA_DIR}/")
        return

    logger.info(
        f"Found {len(tasks)} image(s) across "
        f"{len({t[0] for t in tasks})} language(s). "
        f"Workers={WORKERS}, Retries={RETRIES}, Format={OUT_FORMAT}"
    )

    work = [(client, lang, img, out, logger) for lang, img, out in tasks]
    results = {"ok": 0, "skipped": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_worker, w): w for w in work}
        with tqdm(total=len(work), desc="OCR progress", unit="img") as bar:
            for future in as_completed(futures):
                _, _, status = future.result()
                if status == "ok":
                    results["ok"] += 1
                elif status == "skipped":
                    results["skipped"] += 1
                else:
                    results["error"] += 1
                bar.update(1)

    logger.info(
        f"Finished. "
        f"✓ {results['ok']} processed | "
        f"⏭  {results['skipped']} skipped | "
        f"✗ {results['error']} errors. "
        f"See {LOG_FILE} for details."
    )
    logger.info(f"Outputs saved under: {OUTPUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
