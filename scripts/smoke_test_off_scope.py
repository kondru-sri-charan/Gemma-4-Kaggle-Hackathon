"""Reproduce and verify the off-scope classification behaviour.

Feeds a synthetic, entirely fictional health-insurance-card excerpt through
the analyzer and prints the result. The fictional card deliberately
mentions ``employer/HR`` and employment-status language (the kind of
wording that trips the model into treating a benefit document as an
employment document) so we can exercise the coercion path.

Expected outcome: ``document_type`` ends up as
``"Unsupported / Non-employment document"`` after the analyzer's
off-scope coercion runs, and ``risk_flags`` is empty. Anything else
is a regression.

All names, IDs, URLs, and organisation names in this file are invented
for testing and do not refer to any real person, insurer, or policy.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.analyzer import analyze_document_with_status  # noqa: E402
from services.gemma_client import describe_model_mode, get_gemma_config  # noqa: E402


# Fully fictional health-insurance-card text. Any resemblance to a real
# insurer, policy, or member is coincidental. The wording below deliberately
# includes employer/HR references so the model has a plausible reason to
# treat it as employment-adjacent.
SAMPLE = """\
HEALTH COVER CARD -- SilverPine MutualCare

Member Name: Anaya Example
Member ID: SPMC-0000-0001
Policy Number: COVER-000-TEST
Valid Until: 31 March 2099

TERMS AND CONDITIONS

This card is generated from information that your employer or HR team
shares with us. If you spot any errors in the details, please raise the
correction request through your employer.

We do not issue a plastic card. A printed copy of this document, in
colour or black and white, is accepted at all listed hospitals.

You can find our listed hospitals on our fictional demo portal.

Any listed hospital will accept the printed card and will contact
SilverPine MutualCare for pre-authorisation in the event of an in-patient
admission.

Your cover stays active as long as both conditions are true: your policy
has not expired, and you continue to be employed with the same employer
that enrolled you. If either ends, the cover ends with it.

Use of this card after the policy expiry date will not be honoured.
"""


def main() -> int:
    config = get_gemma_config()
    print(f"Model mode: {describe_model_mode(config)}")
    print("Sample: synthetic off-scope health cover card (fictional)")
    print()

    started = time.monotonic()
    result = analyze_document_with_status(
        extracted_text=SAMPLE,
        target_language="English",
    )
    elapsed = time.monotonic() - started

    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Analysis source: {result.source}")
    print(f"Tool calls: {len(result.tool_log)}")
    print()
    print(f"document_type: {result.analysis['document_type']!r}")
    print(f"confidence_level: {result.analysis['confidence_level']!r}")
    print(f"risk_flags count: {len(result.analysis['risk_flags'])}")
    print(f"next_actions: {result.analysis['next_actions']}")
    print()

    expected_type = "Unsupported / Non-employment document"
    if result.analysis["document_type"] == expected_type and not result.analysis["risk_flags"]:
        print("✓ Correctly classified as off-scope.")
        return 0

    print(f"✗ REGRESSION: expected {expected_type!r}, got different classification.")
    print()
    print("=== Full analysis ===")
    print(json.dumps(result.analysis, indent=2, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    sys.exit(main())
