#!/usr/bin/env python3
"""
Sarvam translation helper for OCR markdown outputs.

Translates a markdown string to a target language while preserving paragraph
structure. The Sarvam translate API has a per-request size limit, so input is
chunked on paragraph boundaries (\\n\\n) and reassembled after translation.
"""

from __future__ import annotations

import time
from typing import Optional

from sarvamai import SarvamAI


# Sarvam translate accepts up to ~2000 chars per request for sarvam-translate:v1.
# Stay comfortably below to leave headroom for expansion in the target language.
DEFAULT_CHUNK_CHARS = 1500

# Folder name -> Sarvam language code. Used by batch runners.
LANG_CODE = {
    "bengali":  "bn-IN",
    "english":  "en-IN",
    "hindi":    "hi-IN",
    "kannada":  "kn-IN",
    "malayalam":"ml-IN",
    "marathi":  "mr-IN",
    "tamil":    "ta-IN",
    "telugu":   "te-IN",
}


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    """
    Split text on paragraph boundaries (\\n\\n), packing paragraphs into chunks
    no larger than max_chars. Paragraphs that exceed max_chars on their own are
    further split on single newlines.
    """
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0

    for para in text.split("\n\n"):
        if len(para) > max_chars:
            # Oversized paragraph — split by single newline, then by hard cut.
            flush()
            sub_buf: list[str] = []
            sub_len = 0
            for line in para.split("\n"):
                if len(line) > max_chars:
                    if sub_buf:
                        chunks.append("\n".join(sub_buf))
                        sub_buf, sub_len = [], 0
                    for i in range(0, len(line), max_chars):
                        chunks.append(line[i:i + max_chars])
                    continue
                add = len(line) + (1 if sub_buf else 0)
                if sub_len + add > max_chars:
                    chunks.append("\n".join(sub_buf))
                    sub_buf, sub_len = [line], len(line)
                else:
                    sub_buf.append(line)
                    sub_len += add
            if sub_buf:
                chunks.append("\n".join(sub_buf))
            continue

        add = len(para) + (2 if buf else 0)
        if buf_len + add > max_chars:
            flush()
            buf, buf_len = [para], len(para)
        else:
            buf.append(para)
            buf_len += add

    flush()
    return chunks


def _translate_chunk(
    client: SarvamAI,
    chunk: str,
    source_language_code: str,
    target_language_code: str,
    model: str,
    retries: int,
    backoff: float,
    logger=None,
) -> str:
    """Translate a single chunk with retry/backoff. Returns translated text."""
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.text.translate(
                input=chunk,
                source_language_code=source_language_code,
                target_language_code=target_language_code,
                model=model,
            )
            return resp.translated_text
        except Exception as e:
            last_err = e
            wait = backoff ** attempt
            if logger:
                logger.warning(
                    f"Translate attempt {attempt} failed: {e}. "
                    f"Retrying in {wait:.1f}s..."
                )
            if attempt == retries:
                break
            time.sleep(wait)
    raise last_err  # type: ignore[misc]


def translate_markdown_text(
    client: SarvamAI,
    text: str,
    source_language_code: str = "auto",
    target_language_code: str = "en-IN",
    model: str = "sarvam-translate:v1",
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    retries: int = 3,
    backoff: float = 2.0,
    logger=None,
) -> str:
    """
    Translate markdown content, preserving paragraph structure.

    Empty input returns "". Whitespace-only chunks are passed through untouched.
    """
    if not text or not text.strip():
        return text

    chunks = _split_into_chunks(text, chunk_chars)
    translated: list[str] = []
    for idx, chunk in enumerate(chunks, 1):
        if not chunk.strip():
            translated.append(chunk)
            continue
        if logger:
            logger.debug(f"Translating chunk {idx}/{len(chunks)} ({len(chunk)} chars)")
        translated.append(
            _translate_chunk(
                client, chunk,
                source_language_code=source_language_code,
                target_language_code=target_language_code,
                model=model,
                retries=retries,
                backoff=backoff,
                logger=logger,
            )
        )
    return "\n\n".join(translated)
