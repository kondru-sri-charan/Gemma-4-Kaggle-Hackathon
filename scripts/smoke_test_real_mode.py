"""End-to-end smoke test: run the analyzer against real Ollama Gemma 4.

Run with:
    MODEL_MODE=real .venv/bin/python scripts/smoke_test_real_mode.py

Expects Ollama running locally with the gemma4:e4b model pulled.
Prints the analysis source (gemma_real / fallback_after_model_failure / mock)
and a pretty-printed JSON analysis so we can eyeball quality.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path regardless of cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.analyzer import analyze_document_with_status  # noqa: E402
from services.gemma_client import describe_model_mode, get_gemma_config  # noqa: E402


SAMPLE_OFFER_LETTER = """\
OFFER LETTER

Employee Name: Ravi Kumar
Designation: Delivery Associate
Date of Joining: 15 June 2026

Compensation:
- Monthly gross salary: Rs 18,000
- Statutory deductions: PF Rs 900, Professional Tax Rs 200

Working hours:
- 12 hours per day, 6 days per week.
- Overtime may be required during peak demand.

Leave:
- 8 paid leaves per year.
- Weekly off on Sunday.

Notice period:
- 60 days notice required from the employee side.
- Employer may terminate with 15 days notice during probation.

Service bond:
- Employee agrees to a 24-month service bond.
- If employee leaves before 24 months, Rs 50,000 bond recovery will be deducted
  from final salary or dues payable.

By signing below, the employee accepts these terms.
"""


def main() -> int:
    config = get_gemma_config()
    print(f"Model mode: {describe_model_mode(config)}")
    print(f"Provider: {config.provider}")
    print(f"Model: {config.model}")
    print(f"Ollama URL: {config.ollama_url}")
    print(f"Timeout: {config.timeout_seconds}s")
    print()

    print("Running analyzer against sample offer letter...")
    started = time.monotonic()
    result = analyze_document_with_status(SAMPLE_OFFER_LETTER, "Hindi")
    elapsed = time.monotonic() - started

    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Analysis source: {result.source}")
    print()
    print(json.dumps(result.analysis, indent=2, ensure_ascii=False))

    if result.source == "gemma_real":
        print("\n✓ Real Gemma 4 response passed strict validation.")
        return 0
    if result.source == "fallback_after_model_failure":
        print(
            "\n⚠ Gemma response failed validation, fell back to mock. "
            "See risk_flags[0].risk_explanation for the reason."
        )
        return 1
    print(f"\n? Unexpected source: {result.source}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
