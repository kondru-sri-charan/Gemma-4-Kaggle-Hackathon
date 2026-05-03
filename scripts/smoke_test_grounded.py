"""End-to-end smoke test for the labor-law grounded analysis.

Runs the hybrid (text + vision) analysis on the bond-clause sample and
prints the final JSON along with the tool_log, so we can see which
statutes Gemma consulted while producing the risk flags.
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
    sample_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE
    if not sample_path.is_absolute():
        sample_path = ROOT / sample_path

    config = get_gemma_config()
    print(f"Model mode: {describe_model_mode(config)}")
    print(f"Sample: {sample_path.name}")

    extraction = app.extract_pdf_text(sample_path.read_bytes())
    print(f"Evidence: {extraction.evidence}  |  "
          f"text={len(extraction.text)}c  images={len(extraction.images)}")
    print()

    print("Running grounded analysis (tool calling + vision)...")
    started = time.monotonic()
    result = analyze_document_with_status(
        extracted_text=extraction.text,
        target_language="Hindi",
        images=extraction.images if config.model_mode == "real" else None,
    )
    elapsed = time.monotonic() - started

    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Analysis source: {result.source}")
    print(f"Tool calls recorded: {len(result.tool_log)}")
    print()

    if result.tool_log:
        print("=== Tool calls ===")
        for call in result.tool_log:
            print(f"\n{call['name']}({call['arguments']}):")
            rows = call["result"] if isinstance(call["result"], list) else [call["result"]]
            for row in rows:
                if isinstance(row, dict) and "title" in row:
                    print(f"  - {row.get('title')}")
                    print(f"    {row.get('statute_reference')}")
        print()

    print("=== Analysis JSON ===")
    print(json.dumps(result.analysis, indent=2, ensure_ascii=False))

    if result.source == "gemma_real":
        return 0 if result.tool_log else 3  # 3 = tool calling did not fire
    return 1


if __name__ == "__main__":
    sys.exit(main())
