"""Smoke test: send a page image of a sample PDF to Gemma 4 via Ollama.

This verifies the vision path works before we wire it into the app.
Renders the first page of the bond-clause employment agreement at 144 DPI,
base64-encodes the PNG, and hits Ollama's /api/chat with an ``images`` field.

Run:
    .venv/bin/python scripts/smoke_test_vision.py
"""
from __future__ import annotations

import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
import requests


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = ROOT / "sample_docs" / "employment_agreement_bond_clause.pdf"
MODEL = "gemma4:e4b"
OLLAMA_URL = "http://localhost:11434/api/chat"


def render_first_page_to_png_bytes(pdf_path: Path, dpi: int = 144) -> bytes:
    with fitz.open(pdf_path) as doc:
        page = doc[0]
        zoom = dpi / 72  # PyMuPDF default is 72 DPI
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        buf = BytesIO()
        buf.write(pix.tobytes("png"))
        return buf.getvalue()


def main() -> int:
    if not SAMPLE_PDF.exists():
        print(f"Missing sample: {SAMPLE_PDF}")
        return 2

    print(f"Rendering page 1 of {SAMPLE_PDF.name} to PNG ...")
    png = render_first_page_to_png_bytes(SAMPLE_PDF)
    print(f"  PNG size: {len(png) / 1024:.1f} KB")
    img_b64 = base64.b64encode(png).decode("ascii")

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Gemma 4. You will be shown an image of an employment "
                    "document. Respond with strict JSON only using this schema: "
                    '{"document_type": str, "role": str, "salary": str, '
                    '"working_hours": str, "bond_clause": str, "notice_period": str}. '
                    "Use \"Not found\" for any field that is not visible."
                ),
            },
            {
                "role": "user",
                "content": "Extract the fields from this employment document image.",
                "images": [img_b64],
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "num_predict": 1024,
        },
    }

    print("Calling Ollama /api/chat with vision payload ...")
    started = time.monotonic()
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    elapsed = time.monotonic() - started
    print(f"  HTTP {response.status_code} in {elapsed:.1f}s")

    if response.status_code != 200:
        print(response.text[:500])
        return 1

    data = response.json()
    content = data.get("message", {}).get("content", "")
    print("\nModel response:\n")
    try:
        parsed = json.loads(content)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(content)
        print("\n(!) response was not valid JSON")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
