"""
Translate multilingual Markdown/HTML invoice files
while preserving original formatting and table structure.

Features:
- Processes ALL language folders under output/
- Skips files already present in translated_output/ (resumable)
- Translates only Indic-script text nodes
- Preserves HTML tables/layout
- Uses Sarvam Translate API
- Caches translations to avoid duplicate API calls
- Fails fast on quota exhaustion (402) instead of wasting retries
"""

import os
import re
import sys
import time
from pathlib import Path
from bs4 import BeautifulSoup
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError
from dotenv import load_dotenv

load_dotenv()

# Force UTF-8 output so Indic characters print correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_ROOT  = "output"              # walk all language sub-folders here
OUTPUT_ROOT = "translated_output"   # mirror structure written here

TARGET_LANGUAGE = "en-IN"
MODEL           = "sarvam-translate:v1"
MAX_CHUNK_CHARS = 900   # safely below the API's ~1000-char limit

# folder name → Sarvam source language code
LANGUAGE_MAP = {
    "bengali":   "bn-IN",
    "hindi":     "hi-IN",
    "kannada":   "kn-IN",
    "malayalam": "ml-IN",
    "marathi":   "mr-IN",
    "tamil":     "ta-IN",
    "telugu":    "te-IN",
    "gujarati":  "gu-IN",
}
# ─────────────────────────────────────────────────────────────────────────────

key = os.environ.get("SARVAM_API_KEY", "")
if not key:
    print("❌  SARVAM_API_KEY not set. Add it to your .env file and retry.")
    sys.exit(1)

client = SarvamAI(api_subscription_key=key)

translation_cache: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_numeric_or_code(text: str) -> bool:
    """Return True for text that needs no translation (numbers, codes, IDs)."""
    text = text.strip()
    if not text:
        return True
    patterns = [
        r'^[\d\s.,:%()/\-₹]+$',   # numbers / punctuation / ₹
        r'^[A-Z0-9]{3,}[A-Z0-9\-]*$',  # all-caps codes (GSTIN, PAN, IFSC …)
        r'^\d+$',
    ]
    return any(re.fullmatch(p, text) for p in patterns)


def contains_indic(text: str) -> bool:
    """Detect any Indic-script character (Devanagari → Malayalam range)."""
    return bool(re.search(r'[ऀ-ൿ]', text))


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    """
    Split on newline boundaries so each chunk stays within max_chars.
    Lines longer than max_chars are hard-cut.
    """
    chunks, current = [], ""
    for line in text.splitlines(keepends=True):
        if len(line) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(line), max_chars):
                chunks.append(line[i: i + max_chars])
        elif len(current) + len(line) > max_chars:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks or [text]


def _call_translate_api(text: str, source_lang: str, retries: int = 3) -> str:
    """Single API call with retry/backoff. Fails immediately on quota errors."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            response = client.text.translate(
                input=text,
                source_language_code=source_lang,
                target_language_code=TARGET_LANGUAGE,
                model=MODEL,
            )
            return response.translated_text
        except ApiError as e:
            if e.status_code == 402:
                print(
                    "\n❌  SARVAM API QUOTA EXHAUSTED (402).\n"
                    "    Recharge credits at https://dashboard.sarvam.ai\n"
                    "    Re-run afterwards — already-saved files will be skipped.\n"
                )
                raise  # no point retrying; credits won't appear between attempts
            last_err = e
            wait = 2 ** attempt
            print(f"  Attempt {attempt} failed (HTTP {e.status_code}). Retrying in {wait}s…")
            if attempt < retries:
                time.sleep(wait)
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  Attempt {attempt} failed: {e}. Retrying in {wait}s…")
            if attempt < retries:
                time.sleep(wait)
    raise last_err


def translate_text(text: str, source_lang: str) -> str:
    """Translate with caching, chunking, and retry logic."""
    text = text.strip()
    if not text:
        return text
    if text in translation_cache:
        return translation_cache[text]

    chunks = _chunk_text(text)
    translated_parts = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or not contains_indic(chunk):
            translated_parts.append(chunk)
            continue
        try:
            translated = _call_translate_api(chunk, source_lang)
            print(f"  ✓ {repr(chunk)[:60]} → {repr(translated)[:60]}")
            translated_parts.append(translated)
            time.sleep(0.3)   # polite delay
        except ApiError as e:
            if e.status_code == 402:
                raise   # propagate quota error to main() so it can stop cleanly
            print(f"  ERROR translating [{chunk[:60]}]: {e}")
            translated_parts.append(chunk)   # keep original on non-quota API errors
        except Exception as e:
            print(f"  ERROR translating [{chunk[:60]}]: {e}")
            translated_parts.append(chunk)   # keep original on failure

    result = " ".join(p for p in translated_parts if p)
    translation_cache[text] = result
    return result


def process_md_file(input_file: Path, output_file: Path, source_lang: str) -> None:
    """Read an MD/HTML invoice file, translate Indic text nodes, and write output."""
    content = input_file.read_text(encoding="utf-8")
    soup = BeautifulSoup(content, "html.parser")

    for text_node in soup.find_all(string=True):
        original = text_node.strip()
        if not original:
            continue
        if is_numeric_or_code(original):
            continue
        if not contains_indic(original):
            continue

        translated = translate_text(original, source_lang)
        text_node.replace_with(text_node.replace(original, translated))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(str(soup), encoding="utf-8")
    print(f"  Saved → {output_file}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    input_root  = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)

    if not input_root.exists():
        print(f"❌  Input folder '{INPUT_ROOT}' not found.")
        sys.exit(1)

    total = skipped = done = errors = 0

    for lang_folder in sorted(input_root.iterdir()):
        if not lang_folder.is_dir():
            continue

        lang_name = lang_folder.name.lower()
        source_lang = LANGUAGE_MAP.get(lang_name)

        if source_lang is None:
            print(f"\nSkipping unsupported language folder: {lang_name}")
            continue

        md_files = sorted(lang_folder.glob("*.md"))
        if not md_files:
            continue

        # Count pending vs already-done before printing the header
        pending = [
            f for f in md_files
            if not (output_root / lang_folder.name / f.name).exists()
        ]

        if not pending:
            print(f"\n⏭  {lang_name}: all {len(md_files)} file(s) already translated — skipping.")
            skipped += len(md_files)
            total   += len(md_files)
            continue

        print("\n" + "=" * 60)
        print(
            f"LANGUAGE: {lang_name}  "
            f"({len(pending)} to translate, "
            f"{len(md_files) - len(pending)} already done)"
        )
        print("=" * 60)

        for md_file in md_files:
            total += 1
            output_path = output_root / lang_folder.name / md_file.name

            # ── Skip if already translated ──────────────────────────────────
            if output_path.exists():
                skipped += 1
                continue   # silent skip — summary shown in the header above

            try:
                process_md_file(md_file, output_path, source_lang)
                done += 1
            except ApiError as e:
                if e.status_code == 402:
                    print("  Stopping: API quota exhausted.")
                    sys.exit(1)   # exit cleanly; re-run after recharge
                errors += 1
                print(f"  API error on {md_file.name}: {e}")
            except Exception as e:
                errors += 1
                print(f"  Error on {md_file.name}: {e}")

    print(
        f"\n{'='*60}\n"
        f"DONE — {total} file(s) found | "
        f"✓ {done} translated | "
        f"⏭  {skipped} skipped | "
        f"✗ {errors} errors\n"
        f"Translations saved under: {output_root.resolve()}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
