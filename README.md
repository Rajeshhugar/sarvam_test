# Sarvam Vision OCR

Concurrent OCR pipeline for PDFs and images using the Sarvam AI API.

## Install

```bash
pip install sarvamai tqdm
```

Requires a local `postproccessing.py` with `join_line_break_hyphens` and `unwrap_simple_math_latex`.

## Setup

```bash
export SARVAM_API_KEY="your_key"          # Linux/Mac
$env:SARVAM_API_KEY="your_key"            # PowerShell
```

## Usage

```bash
python ocr_.py <input> <output> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `input` | — | File or folder (PDF / image / folder of images) |
| `output` | — | Output directory |
| `-s`, `-e` | all | Slice the file list |
| `-w` | 4 | Worker threads |
| `-r` | 3 | Retry attempts |
| `--output-format` | md | `md` / `txt` / `json` |
| `--log-file` | sarvam_ocr.log | Log path |
| `--join-line-break-hyphens` | off | Fix `word-\nbreak` |
| `--unwrap-simple-math-latex` | off | Simplify LaTeX |

## Examples

```bash
# Single file
python ocr_.py input.pdf ./out

# Folder, 8 workers, JSON output
python ocr_.py ./docs ./out -w 8 --output-format json

# With post-processing
python ocr_.py ./docs ./out --join-line-break-hyphens --unwrap-simple-math-latex

# Slice (resume / partition)
python ocr_.py ./docs ./out -s 100 -e 200
```

## Notes

- A folder of images (leaf dir) is treated as one multi-page document.
- Existing outputs are skipped — delete to re-OCR.
- Layout tags like `header`, `footer`, `page-number`, `figure` are stripped.
- Retry backoff: `2^attempt` seconds.

## Programmatic

```python
from pathlib import Path
from sarvamai import SarvamAI
from ocr_ import ocr_and_save_document, setup_logging

client = SarvamAI(api_subscription_key="...")
ocr_and_save_document(client, Path("in.pdf"), Path("out/in.md"),
                      output_format="md", logger=setup_logging())
```
