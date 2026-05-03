"""Off-scope classification test with the vision path.

Renders an entirely fictional health cover card as a one-page PDF,
extracts it through the hybrid extractor (text + page images), and runs
the analyzer. This exercises the full vision path for an off-scope
document.

All names, IDs, URLs, and organisation names in this file are invented
for testing and do not refer to any real person, insurer, or policy.
"""
from __future__ import annotations

import json
import sys
import time
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402
from services.analyzer import analyze_document_with_status  # noqa: E402
from services.gemma_client import describe_model_mode, get_gemma_config  # noqa: E402


# Fully fictional card content. Any resemblance to a real insurer, member,
# or policy is coincidental.
CARD_TEXT = [
    "HEALTH COVER CARD -- SilverPine MutualCare",
    "",
    "Member Name: Anaya Example",
    "Member ID: SPMC-0000-0001",
    "Policy Number: COVER-000-TEST",
    "Valid Until: 31 March 2099",
    "",
    "TERMS AND CONDITIONS",
    "",
    "This card is generated from information that your employer or HR",
    "team shares with us. If you spot any errors, please raise the",
    "correction request through your employer.",
    "",
    "We do not issue a plastic card. A printed copy of this document, in",
    "colour or black and white, is accepted at all listed hospitals.",
    "",
    "Any listed hospital will accept the printed card and will contact",
    "SilverPine MutualCare for pre-authorisation in the event of an",
    "in-patient admission.",
    "",
    "Your cover stays active as long as both conditions are true: your",
    "policy has not expired, and you continue to be employed with the",
    "same employer that enrolled you. If either ends, the cover ends.",
    "",
    "Use of this card after the policy expiry date will not be honoured.",
]


def make_card_pdf_bytes() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in CARD_TEXT:
        page.insert_text((72, y), line, fontsize=11)
        y += 16
    return doc.write()


def main() -> int:
    config = get_gemma_config()
    print(f"Model mode: {describe_model_mode(config)}")
    print("Sample: synthetic off-scope health cover card PDF (fictional)")
    print()

    pdf_bytes = make_card_pdf_bytes()
    extraction = app.extract_pdf_text(pdf_bytes)
    print(
        f"Evidence: {extraction.evidence}  |  "
        f"text={len(extraction.text)}c  images={len(extraction.images)}"
    )

    started = time.monotonic()
    result = analyze_document_with_status(
        extracted_text=extraction.text,
        target_language="Hindi",
        images=extraction.images if config.model_mode == "real" else None,
    )
    elapsed = time.monotonic() - started

    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Analysis source: {result.source}")
    print(f"Tool calls: {len(result.tool_log)}")
    print()
    print(f"document_type: {result.analysis['document_type']!r}")
    print(f"risk_flags count: {len(result.analysis['risk_flags'])}")
    print()

    expected = "Unsupported / Non-employment document"
    if result.analysis["document_type"] == expected and not result.analysis["risk_flags"]:
        print("✓ Correctly classified as off-scope through the vision path.")
        return 0

    print(f"✗ REGRESSION: expected {expected!r}")
    print()
    print(json.dumps(result.analysis, indent=2, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    sys.exit(main())
