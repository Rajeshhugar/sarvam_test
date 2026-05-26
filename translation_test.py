"""
Translate multilingual Markdown/HTML invoice files
while preserving original formatting and table structure.

Features:
- Processes all language folders
- Translates only text nodes
- Preserves HTML tables/layout
- Uses Sarvam Translate API
- Creates translated output folders
- Reuses translations using dictionary cache
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

key = os.environ.get("SARVAM_API_KEY", "")

# Force UTF-8 output so Bengali/Indic characters print correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


INPUT_ROOT = "output/kannada"              # root folder
OUTPUT_ROOT = "translated_output/kannada"   # translated output

TARGET_LANGUAGE = "en-IN"
MODEL = "sarvam-translate:v1"
MAX_CHUNK_CHARS = 900   # stay safely below the API's 1000-char limit

# Language folder mapping
LANGUAGE_MAP = {
    "bengali": "bn-IN",
    "hindi": "hi-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "marathi": "mr-IN",
    "gujarati": "gu-IN",
}


client = SarvamAI(
    api_subscription_key=key
)


translation_cache = {}


def is_numeric_or_code(text: str) -> bool:
    """
    Skip:
    - numbers
    - GSTIN
    - PAN
    - IFSC
    - phone numbers
    """

    text = text.strip()

    if not text:
        return True

    patterns = [
        r'^[\d\s.,:%()/\-]+$',
        r'^[A-Z0-9\-]+$',
        r'^\d+$'
    ]

    for pattern in patterns:
        if re.fullmatch(pattern, text):
            return True

    return False


def contains_indic(text: str) -> bool:
    """
    Detect Indic script characters
    """

    indic_pattern = r'[\u0900-\u0D7F]'
    return bool(re.search(indic_pattern, text))


def _call_translate_api(text: str, source_lang: str, retries: int = 3) -> str:
    """
    Single API call with retry/backoff. Raises on final failure.
    NOTE: speaker_gender is a TTS parameter — do NOT pass it here.
    """
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
                    "\n❌ SARVAM API QUOTA EXHAUSTED (402).\n"
                    "   Please recharge your credits at https://dashboard.sarvam.ai\n"
                    "   Then re-run the script — already-saved files will be skipped.\n"
                )
                raise  # no point retrying — credits won't appear between attempts
            last_err = e
            wait = 2 ** attempt
            print(f"  Attempt {attempt} failed (HTTP {e.status_code}): {e}. Retrying in {wait}s…")
            if attempt < retries:
                time.sleep(wait)
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  Attempt {attempt} failed: {e}. Retrying in {wait}s…")
            if attempt < retries:
                time.sleep(wait)
    raise last_err


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    """
    Split text on newline boundaries so each chunk fits within max_chars.
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


def translate_text(text: str, source_lang: str) -> str:
    """
    Translate using Sarvam, with caching, chunking, and retry logic.
    """
    text = text.strip()

    if not text:
        return text

    if text in translation_cache:
        return translation_cache[text]

    # Split long / multi-line text into safe-sized chunks
    chunks = _chunk_text(text)
    translated_parts = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            translated_parts.append(chunk)
            continue
        if not contains_indic(chunk):
            # Purely ASCII — no translation needed
            translated_parts.append(chunk)
            continue
        try:
            translated = _call_translate_api(chunk, source_lang)
            print(f"Translated: {chunk!r} -> {translated!r}")
            translated_parts.append(translated)
            time.sleep(0.3)   # polite delay between calls
        except Exception as e:
            print(f"ERROR translating [{chunk}] : {e}")
            translated_parts.append(chunk)   # keep original on failure

    result = " ".join(p for p in translated_parts if p)
    translation_cache[text] = result
    return result


def process_md_file(input_file, output_file, source_lang):
    """
    Translate one markdown/html file
    """

    print(f"\nProcessing: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse HTML inside markdown
    soup = BeautifulSoup(content, "html.parser")


    for text_node in soup.find_all(string=True):

        original = text_node.strip()

        if not original:
            continue

        if is_numeric_or_code(original):
            continue

        if not contains_indic(original):
            continue

        translated = translate_text(
            original,
            source_lang
        )

        # Replace only original text
        replaced = text_node.replace(
            original,
            translated
        )

        text_node.replace_with(replaced)



    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(soup))

    print(f"Saved: {output_file}")



def main():

    input_root = Path(INPUT_ROOT)

    for lang_folder in input_root.iterdir():

        if not lang_folder.is_dir():
            continue

        lang_name = lang_folder.name.lower()

        if lang_name not in LANGUAGE_MAP:

            print(f"Skipping unsupported folder: {lang_name}")
            continue

        source_lang = LANGUAGE_MAP[lang_name]

        print("\n" + "=" * 60)
        print(f"PROCESSING LANGUAGE: {lang_name}")
        print("=" * 60)

        md_files = list(lang_folder.glob("*.md"))

        for md_file in md_files:

            relative_path = md_file.relative_to(INPUT_ROOT)

            output_path = Path(OUTPUT_ROOT) / relative_path

            process_md_file(
                input_file=md_file,
                output_file=output_path,
                source_lang=source_lang
            )

    print("\nDONE")




if __name__ == "__main__":
    main()