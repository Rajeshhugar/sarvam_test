#!/usr/bin/env python3
"""
Batch translation runner — walks output/{language}/*.md and writes English
translations to output_en/{language}/*.md, mirroring the source structure.

Run after run_all.py has produced OCR markdown. Existing translations are
skipped so the script is idempotent and resumable.
"""

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from sarvamai import SarvamAI
from tqdm import tqdm

from ocr_ import setup_logging
from translate_ import LANG_CODE, translate_markdown_text

# ── Config ────────────────────────────────────────────────────────────────────
SRC_DIR     = Path("output")       # OCR markdown from run_all.py
DST_DIR     = Path("output_en")    # translated markdown lands here
TARGET_LANG = "en-IN"
MODEL       = "sarvam-translate:v1"
WORKERS     = 4                    # concurrent translate calls
RETRIES     = 3
LOG_FILE    = "sarvam_translate.log"
# ─────────────────────────────────────────────────────────────────────────────


def collect_tasks(src_dir: Path, dst_dir: Path):
    """Walk src/{lang}/*.md and build (lang, src_path, dst_path) tuples."""
    tasks = []
    for lang_dir in sorted(src_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        out_lang = dst_dir / lang
        for md in sorted(lang_dir.glob("*.md")):
            tasks.append((lang, md, out_lang / md.name))
    return tasks


def _worker(args):
    client, lang, src_path, dst_path, logger = args

    if dst_path.exists():
        logger.info(f"Skip (exists): {lang}/{src_path.name}")
        return lang, src_path.name, "skipped"

    # Map folder name (e.g. "hindi") to a Sarvam source code; fall back to auto.
    src_code = LANG_CODE.get(lang.lower(), "auto")

    try:
        text = src_path.read_text(encoding="utf-8")
        translated = translate_markdown_text(
            client, text,
            source_language_code=src_code,
            target_language_code=TARGET_LANG,
            model=MODEL,
            retries=RETRIES,
            logger=logger,
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text(translated, encoding="utf-8")
        logger.info(f"Done: {lang}/{src_path.name}")
        return lang, src_path.name, "ok"
    except Exception as e:
        logger.error(f"Error [{lang}/{src_path.name}]: {e}")
        return lang, src_path.name, f"error: {e}"


def main():
    load_dotenv()
    logger = setup_logging(LOG_FILE)

    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        logger.error("SARVAM_API_KEY is not set in .env or environment.")
        return

    client = SarvamAI(api_subscription_key=key)

    tasks = collect_tasks(SRC_DIR, DST_DIR)
    if not tasks:
        logger.error(f"No markdown files found under {SRC_DIR}/")
        return

    logger.info(
        f"Found {len(tasks)} file(s) across "
        f"{len({t[0] for t in tasks})} language(s). "
        f"Workers={WORKERS}, Target={TARGET_LANG}, Model={MODEL}"
    )

    work = [(client, lang, src, dst, logger) for lang, src, dst in tasks]
    results = {"ok": 0, "skipped": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_worker, w): w for w in work}
        with tqdm(total=len(work), desc="Translate", unit="file") as bar:
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
        f"✓ {results['ok']} translated | "
        f"⏭  {results['skipped']} skipped | "
        f"✗ {results['error']} errors. "
        f"See {LOG_FILE} for details."
    )
    logger.info(f"Translations saved under: {DST_DIR.resolve()}/")


if __name__ == "__main__":
    main()
