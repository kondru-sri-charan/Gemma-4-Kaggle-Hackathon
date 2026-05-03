"""Vision-only smoke test: simulate a scanned PDF by passing empty text.

Exercises the pure-vision branch of the analyzer: no extracted text, only
page images. This is the path a worker's phone-photo of a printed letter
takes through the system.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402
from services.analyzer import analyze_document_with_status  # noqa: E402
from services.gemma_client import describe_model_mode, get_gemma_config  # noqa: E402

SAMPLE = ROOT / "sample_docs" / "employment_agreement_bond_clause.pdf"


def main() -> int:
    config = get_gemma_config()
    print(f"Model mode: {describe_model_mode(config)}")
    print(f"Sample: {SAMPLE.name} (simulating no selectable text)")

    # Only keep the rendered page images, discard the extracted text so the
    # analyzer has to rely purely on vision.
    full = app.extract_pdf_text(SAMPLE.read_bytes())
    print(f"Rendered {len(full.images)} page image(s)")

    started = time.monotonic()
    result = analyze_document_with_status(
        extracted_text="",
        target_language="Telugu",
        images=full.images if config.model_mode == "real" else None,
    )
    elapsed = time.monotonic() - started
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Analysis source: {result.source}")
    print()
    print(json.dumps(result.analysis, indent=2, ensure_ascii=False))

    return 0 if result.source == "gemma_real" else 1


if __name__ == "__main__":
    sys.exit(main())
