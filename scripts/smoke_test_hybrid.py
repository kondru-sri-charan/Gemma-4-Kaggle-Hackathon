"""End-to-end hybrid smoke test.

Loads a sample PDF, runs the new hybrid extractor (text + page images), and
calls analyze_document_with_status in real mode with both streams. Prints
the analysis source and the resulting JSON for quality review.

Run:
    MODEL_MODE=real .venv/bin/python scripts/smoke_test_hybrid.py [sample.pdf]
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


DEFAULT_SAMPLE = (
    ROOT / "sample_docs" / "employment_agreement_bond_clause.pdf"
)


def main() -> int:
    sample_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAMPLE
    if not sample_path.exists():
        print(f"Sample not found: {sample_path}")
        return 2

    config = get_gemma_config()
    print(f"Model mode: {describe_model_mode(config)}")
    print(f"Sample: {sample_path.name}")
    print()

    file_bytes = sample_path.read_bytes()
    extraction = app.extract_pdf_text(file_bytes)
    print(f"Evidence: {extraction.evidence}")
    print(f"Extracted text length: {len(extraction.text)} chars")
    print(f"Rendered page images: {len(extraction.images)}")
    if extraction.error:
        print(f"Extractor note: {extraction.error}")
    print()

    print("Running analyzer (hybrid text + vision) ...")
    started = time.monotonic()
    result = analyze_document_with_status(
        extracted_text=extraction.text,
        target_language="Hindi",
        images=extraction.images if config.model_mode == "real" else None,
    )
    elapsed = time.monotonic() - started

    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Analysis source: {result.source}")
    print()
    print(json.dumps(result.analysis, indent=2, ensure_ascii=False))

    if result.source == "gemma_real":
        print("\n✓ Real Gemma 4 hybrid response passed strict validation.")
        return 0
    if result.source == "fallback_after_model_failure":
        print("\n⚠ Fell back to mock. See risk_flags[0] for the reason.")
        return 1
    print(f"\n? Unexpected source: {result.source}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
