#!/usr/bin/env python3
"""
Sarvam translation helper for OCR markdown outputs.

Translates a markdown string to a target language while preserving paragraph
structure and HTML table structure.

Strategy
--------
1. The input is split into segments: plain-text paragraphs and raw HTML blocks
   (anything between <table>…</table> tags).
2. Plain-text segments are chunked on paragraph/line boundaries (existing logic)
   and translated as plain text.
3. HTML table segments are parsed with BeautifulSoup.  Every <td> and <th>
   cell's *text content* is extracted, translated, and put back — the HTML
   skeleton (tags, attributes, rowspan/colspan) is never sent to the API, so
   the structure is always preserved.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag
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

# Regex to detect non-ASCII (Indian script) characters — only these cells need
# to be translated; purely numeric / English cells are left untouched.
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")

# Regex to split raw markdown into alternating [text, table, text, table, ...]
# segments.  Tables are captured as a whole block.
_TABLE_RE = re.compile(r"(<table[\s\S]*?</table>)", re.IGNORECASE)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    """
    Split plain text on paragraph boundaries (\\n\\n), packing paragraphs into
    chunks no larger than max_chars.  Paragraphs that exceed max_chars on their
    own are further split on single newlines, then by hard cut.
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
            flush()
            sub_buf: list[str] = []
            sub_len = 0
            for line in para.split("\n"):
                if len(line) > max_chars:
                    if sub_buf:
                        chunks.append("\n".join(sub_buf))
                        sub_buf, sub_len = [], 0
                    for i in range(0, len(line), max_chars):
                        chunks.append(line[i : i + max_chars])
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
    """Translate a single plain-text chunk with retry/backoff."""
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
                    f"Retrying in {wait:.1f}s…"
                )
            if attempt == retries:
                break
            time.sleep(wait)
    raise last_err  # type: ignore[misc]


# ── HTML-table-aware translation ──────────────────────────────────────────────

def _cell_plain_text(cell: Tag) -> str:
    """
    Extract raw text from a <td>/<th> cell, collapsing <br/> tags to spaces
    so the translator sees a single readable string.
    """
    parts: list[str] = []
    for child in cell.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag) and child.name in ("br", "br/"):
            parts.append(" ")
        else:
            parts.append(child.get_text(" ", strip=False))
    return " ".join(parts).strip()


def _translate_html_table(
    client: SarvamAI,
    table_html: str,
    source_language_code: str,
    target_language_code: str,
    model: str,
    retries: int,
    backoff: float,
    logger=None,
) -> str:
    """
    Parse an HTML <table> block, translate the text content of every <td>/<th>
    cell that contains non-ASCII (Indian-script) text, and return the modified
    HTML string with the original structure intact.
    """
    soup = BeautifulSoup(table_html, "html.parser")
    cells = soup.find_all(["td", "th"])

    for cell in cells:
        plain = _cell_plain_text(cell)
        if not plain or not _NON_ASCII_RE.search(plain):
            # All-ASCII or empty — no translation needed.
            continue

        if len(plain) > DEFAULT_CHUNK_CHARS:
            # Very long cell — split and translate in pieces.
            pieces = _split_into_chunks(plain, DEFAULT_CHUNK_CHARS)
            translated_pieces = []
            for piece in pieces:
                if not piece.strip() or not _NON_ASCII_RE.search(piece):
                    translated_pieces.append(piece)
                else:
                    translated_pieces.append(
                        _translate_chunk(
                            client, piece,
                            source_language_code=source_language_code,
                            target_language_code=target_language_code,
                            model=model,
                            retries=retries,
                            backoff=backoff,
                            logger=logger,
                        )
                    )
            translated_text = " ".join(translated_pieces)
        else:
            translated_text = _translate_chunk(
                client, plain,
                source_language_code=source_language_code,
                target_language_code=target_language_code,
                model=model,
                retries=retries,
                backoff=backoff,
                logger=logger,
            )

        # Replace the cell's contents with the translated plain text, keeping
        # all attributes (rowspan, colspan, etc.) on the tag itself.
        cell.clear()
        cell.append(NavigableString(translated_text))

    return str(soup)


# ── Public entry-point ────────────────────────────────────────────────────────

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
    Translate markdown content that may contain embedded HTML tables.

    • Plain-text paragraphs are chunked and translated as before.
    • HTML <table> blocks are translated cell-by-cell so structure is preserved.

    Empty input returns "".
    """
    if not text or not text.strip():
        return text

    # Split into alternating [text_seg, table_seg, text_seg, …] segments.
    segments = _TABLE_RE.split(text)
    result_parts: list[str] = []

    for seg in segments:
        if not seg:
            continue

        if _TABLE_RE.match(seg):
            # ── HTML table segment ──────────────────────────────────────────
            if logger:
                logger.debug(f"Translating HTML table ({len(seg)} chars)")
            result_parts.append(
                _translate_html_table(
                    client, seg,
                    source_language_code=source_language_code,
                    target_language_code=target_language_code,
                    model=model,
                    retries=retries,
                    backoff=backoff,
                    logger=logger,
                )
            )
        else:
            # ── Plain-text segment ──────────────────────────────────────────
            if not seg.strip():
                result_parts.append(seg)
                continue
            chunks = _split_into_chunks(seg, chunk_chars)
            translated_chunks: list[str] = []
            for idx, chunk in enumerate(chunks, 1):
                if not chunk.strip():
                    translated_chunks.append(chunk)
                    continue
                if not _NON_ASCII_RE.search(chunk):
                    # Already English — pass through untouched.
                    translated_chunks.append(chunk)
                    continue
                if logger:
                    logger.debug(
                        f"Translating text chunk {idx}/{len(chunks)} "
                        f"({len(chunk)} chars)"
                    )
                translated_chunks.append(
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
            result_parts.append("\n\n".join(translated_chunks))

    return "".join(result_parts)
