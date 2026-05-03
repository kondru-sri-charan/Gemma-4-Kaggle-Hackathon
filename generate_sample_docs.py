#!/usr/bin/env python3
"""Generate sample PDF files for Sahaayak AI testing.

Produces six realistic Indian employment documents:

1. employment_agreement_bond_clause.pdf  - contract with restrictive bond
2. salary_slip_deductions.pdf            - payslip with PF/ESI/TDS
3. non_employment_aws_sample.pdf         - off-scope canary
4. hindi_english_offer_letter.pdf        - mixed Hindi + English offer letter
5. gig_worker_agreement.pdf              - delivery platform gig contract
6. karnataka_factory_offer_letter.pdf    - Karnataka-specific to trigger
                                           state-scoped law lookup

Run:
    .venv/bin/python generate_sample_docs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError:
    print("Error: reportlab is not installed.", file=sys.stderr)
    print("Install it with: pip install reportlab", file=sys.stderr)
    sys.exit(1)


SAMPLE_DIR = Path(__file__).parent / "sample_docs"
SAMPLE_DIR.mkdir(exist_ok=True)

_STYLES = getSampleStyleSheet()
TITLE_STYLE = ParagraphStyle(
    "CustomTitle",
    parent=_STYLES["Heading1"],
    fontSize=14,
    textColor=colors.HexColor("#000000"),
    spaceAfter=12,
)


# ---------------------------------------------------------------------------
# Devanagari font registration for the Hindi-English mixed offer letter.
# ReportLab's built-in fonts do not carry Devanagari glyphs, so we register
# the macOS system Devanagari font if available. On other platforms the
# Hindi paragraphs fall back to a bracketed English transliteration so the
# generator still succeeds.
# ---------------------------------------------------------------------------
_DEVANAGARI_FONT_NAME: str | None = None


def _try_register_devanagari_font() -> str | None:
    candidate_paths = [
        "/System/Library/Fonts/Supplemental/DevanagariMT.ttc",  # macOS
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",  # Debian/Ubuntu
        "/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf",  # Arch
        "C:/Windows/Fonts/Nirmala.ttf",  # Windows 10+
    ]
    for path in candidate_paths:
        if not Path(path).exists():
            continue
        try:
            if path.endswith(".ttc"):
                pdfmetrics.registerFont(TTFont("Devanagari", path, subfontIndex=0))
            else:
                pdfmetrics.registerFont(TTFont("Devanagari", path))
            return "Devanagari"
        except Exception:
            continue
    return None


_DEVANAGARI_FONT_NAME = _try_register_devanagari_font()


def _make_hindi_style() -> ParagraphStyle:
    if _DEVANAGARI_FONT_NAME:
        return ParagraphStyle(
            "Hindi",
            parent=_STYLES["Normal"],
            fontName=_DEVANAGARI_FONT_NAME,
            fontSize=11,
        )
    # No Devanagari font available; fall back to the normal style so the
    # transliterated English shows up in place of the Hindi.
    return _STYLES["Normal"]


HINDI_STYLE = _make_hindi_style()


# ---------------------------------------------------------------------------
# 1. Employment Agreement with restrictive Bond Clause
# ---------------------------------------------------------------------------
def build_employment_agreement_bond_clause() -> None:
    doc = SimpleDocTemplate(
        str(SAMPLE_DIR / "employment_agreement_bond_clause.pdf"), pagesize=letter
    )
    story = []
    story.append(Paragraph("EMPLOYMENT AGREEMENT", TITLE_STYLE))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Employee Name: John Doe", _STYLES["Normal"]))
    story.append(Paragraph("Employee ID: EMP-2024-001", _STYLES["Normal"]))
    story.append(Paragraph("Department: Software Engineering", _STYLES["Normal"]))
    story.append(Paragraph("Date of Joining: 01 January 2024", _STYLES["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("<b>1. POSITION AND RESPONSIBILITIES</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "The Employee is appointed as Senior Software Engineer in the Software "
            "Development Department.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>2. COMPENSATION</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "Monthly Salary: Rs 1,20,000 (Gross CTC: Rs 15,60,000)",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>3. WORKING HOURS</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "Standard working hours: 9:00 AM to 6:00 PM, Monday to Friday. Total 40 "
            "hours per week.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>4. LEAVE ENTITLEMENT</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "Annual Leave: 18 days per annum, Sick Leave: 10 days per annum, "
            "Casual Leave: 5 days per annum",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(
        Paragraph("<b>5. SERVICE BOND CLAUSE (RESTRICTIVE)</b>", _STYLES["Normal"])
    )
    story.append(
        Paragraph(
            "Employee acknowledges receipt of comprehensive training valued at "
            "Rs 5,00,000. If Employee voluntarily resigns within 2 years of "
            "completion of training, Employee shall pay bond recovery amount of "
            "Rs 2,50,000 to the Company.",
            _STYLES["Normal"],
        )
    )
    story.append(
        Paragraph(
            "In case of breach, the Company reserves the right to recover the amount "
            "from any pending salary, bonus, or gratuity.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>6. NOTICE PERIOD</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "During probation (first 6 months): 15 days notice required. After "
            "probation: 30 days notice required.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>7. PROBATION</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "The first 6 months of employment shall be probationary period.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.3 * inch))
    story.append(
        Paragraph(
            "This is a sample anonymized document for testing purposes.",
            _STYLES["Italic"],
        )
    )

    doc.build(story)
    print("✓ Created: employment_agreement_bond_clause.pdf")


# ---------------------------------------------------------------------------
# 2. Salary Slip with statutory deductions
# ---------------------------------------------------------------------------
def build_salary_slip_deductions() -> None:
    doc = SimpleDocTemplate(
        str(SAMPLE_DIR / "salary_slip_deductions.pdf"), pagesize=letter
    )
    story = []
    story.append(Paragraph("SALARY SLIP - PAYROLL STATEMENT", TITLE_STYLE))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Employee ID: EMP-2024-001", _STYLES["Normal"]))
    story.append(Paragraph("Employee Name: John Doe", _STYLES["Normal"]))
    story.append(Paragraph("Salary Month: March 2024", _STYLES["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("<b>EARNINGS</b>", _STYLES["Heading3"]))
    earnings = [
        ["Description", "Amount (Rs)"],
        ["Basic Salary", "60,000"],
        ["Dearness Allowance (DA)", "15,000"],
        ["House Rent Allowance (HRA)", "25,000"],
        ["Other Allowances", "10,000"],
        ["<b>Total Earnings</b>", "<b>1,10,000</b>"],
    ]
    story.append(_money_table(earnings))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("<b>DEDUCTIONS</b>", _STYLES["Heading3"]))
    deductions = [
        ["Description", "Amount (Rs)"],
        ["Provident Fund (PF)", "8,000"],
        ["Professional Tax", "200"],
        ["TDS (Tax Deducted at Source)", "3,500"],
        ["ESI (Employee State Insurance)", "1,200"],
        ["<b>Total Deductions</b>", "<b>12,900</b>"],
    ]
    story.append(_money_table(deductions))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("<b>Net Salary:</b> Rs 97,100", _STYLES["Heading3"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(
        Paragraph(
            "This is a sample anonymized document for testing purposes.",
            _STYLES["Italic"],
        )
    )

    doc.build(story)
    print("✓ Created: salary_slip_deductions.pdf")


def _money_table(data: list[list[str]]) -> Table:
    table = Table(data, colWidths=[3 * inch, 2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, -1), (-1, -1), colors.lightgrey),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    return table


# ---------------------------------------------------------------------------
# 3. Non-employment canary: AWS documentation
# ---------------------------------------------------------------------------
def build_non_employment_aws_sample() -> None:
    doc = SimpleDocTemplate(
        str(SAMPLE_DIR / "non_employment_aws_sample.pdf"), pagesize=letter
    )
    story = []
    story.append(Paragraph("AWS EC2 TECHNICAL DOCUMENTATION", TITLE_STYLE))
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph("<b>Amazon Elastic Compute Cloud (EC2)</b>", _STYLES["Heading2"])
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<b>1. Overview</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "Amazon EC2 is a web service that provides resizable compute capacity "
            "in the cloud. It is designed to make web-scale cloud computing easier "
            "for developers.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<b>2. Features</b>", _STYLES["Normal"]))
    for feature in (
        "Virtual computing environments (instances)",
        "Pre-configured templates for instances (AMIs)",
        "Various configurations of CPU, memory, storage, and networking",
        "Secure login mechanism (key pairs)",
    ):
        story.append(Paragraph("&bull; " + feature, _STYLES["Normal"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<b>3. Pricing Models</b>", _STYLES["Normal"]))
    for pricing in (
        "On-Demand: Pay for compute capacity by the hour",
        "Reserved Instances: Reserve capacity for 1 or 3 years",
        "Spot Instances: Bid for unused capacity",
    ):
        story.append(Paragraph("&bull; " + pricing, _STYLES["Normal"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(
        Paragraph(
            "This is a sample non-employment document for testing unsupported "
            "document classification.",
            _STYLES["Italic"],
        )
    )

    doc.build(story)
    print("✓ Created: non_employment_aws_sample.pdf")


# ---------------------------------------------------------------------------
# 4. Hindi-English mixed offer letter (blue-collar hiring)
# ---------------------------------------------------------------------------
# Models how many Indian hiring letters for entry-level roles are written:
# the formal clauses stay in English, but a worker-facing summary at the
# bottom restates them in Hindi. Also a deliberate red flag: the salary
# mentioned in the body (Rs 14,000) is below the Karnataka zone-1 unskilled
# minimum wage, giving Gemma something concrete to flag once it consults
# the labor-law tool.
def build_hindi_english_offer_letter() -> None:
    doc = SimpleDocTemplate(
        str(SAMPLE_DIR / "hindi_english_offer_letter.pdf"), pagesize=letter
    )
    story = []
    story.append(Paragraph("OFFER OF EMPLOYMENT", TITLE_STYLE))
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "Bluemango Facility Solutions Pvt Ltd (fictional demo company), Bengaluru",
            _STYLES["Normal"],
        )
    )
    story.append(Paragraph("Date: 10 May 2026", _STYLES["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Dear Ramesh Kumar,", _STYLES["Normal"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "We are pleased to offer you the position of <b>Housekeeping Associate</b> "
            "with effective date of joining 01 June 2026, on the following terms and "
            "conditions.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>1. Compensation</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "Your monthly gross wages shall be <b>Rs 14,000</b>. Wages will be paid "
            "by bank transfer on or before the 10th of each month.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>2. Working hours</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "Your standard working hours are 12 hours per day, 6 days per week. "
            "Overtime may be required as per operational needs.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>3. Leave</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "You will be entitled to 8 paid leaves per year. Weekly off on Sunday.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>4. Notice period</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "You are required to give 30 days notice if you wish to resign. The "
            "Company may terminate the engagement with 7 days notice.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>5. Uniform deposit</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "A uniform deposit of Rs 2,000 will be recovered from your first "
            "month's wages and refunded on exit, subject to return of uniform in "
            "good condition.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.3 * inch))

    # Worker-facing Hindi summary.
    story.append(
        Paragraph("<b>Summary for the worker (in Hindi):</b>", _STYLES["Normal"])
    )
    story.append(Spacer(1, 0.1 * inch))

    hindi_lines = [
        "प्रिय रमेश कुमार,",
        "ब्लूमैंगो फैसिलिटी सोल्यूशंस (काल्पनिक डेमो कंपनी) आपको हाउसकीपिंग एसोसिएट के रूप में रखना "
        "चाहती है। आपकी मासिक तनख्वाह 14,000 रुपये होगी।",
        "काम के घंटे रोज़ 12 घंटे, सप्ताह में 6 दिन होंगे। साप्ताहिक छुट्टी रविवार को रहेगी।",
        "एक साल में 8 पेड लीव मिलेगी।",
        "अगर आप नौकरी छोड़ना चाहें तो 30 दिन पहले बताना होगा। कंपनी 7 दिन में नौकरी खत्म कर सकती है।",
        "पहली तनख्वाह से 2,000 रुपये वर्दी की जमा राशि (deposit) काटी जाएगी।",
    ]
    if _DEVANAGARI_FONT_NAME:
        for line in hindi_lines:
            story.append(Paragraph(line, HINDI_STYLE))
            story.append(Spacer(1, 0.07 * inch))
    else:
        story.append(
            Paragraph(
                "(Devanagari font unavailable on this system; Hindi summary "
                "omitted. Install a Devanagari-capable font like Noto Sans "
                "Devanagari to regenerate with full Hindi content.)",
                _STYLES["Italic"],
            )
        )

    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            "Please sign and return a copy to accept this offer.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            "This is a sample anonymized document for testing purposes.",
            _STYLES["Italic"],
        )
    )

    doc.build(story)
    print("✓ Created: hindi_english_offer_letter.pdf")


# ---------------------------------------------------------------------------
# 5. Gig Worker Engagement Contract (delivery platform)
# ---------------------------------------------------------------------------
# Models a platform-style "independent contractor" engagement that disclaims
# the employer-employee relationship and loads the rider up with penalties
# -- a common pattern in Indian gig work that Gemma should flag even though
# there is no traditional salary.
def build_gig_worker_agreement() -> None:
    doc = SimpleDocTemplate(
        str(SAMPLE_DIR / "gig_worker_agreement.pdf"), pagesize=letter
    )
    story = []
    story.append(Paragraph("DELIVERY PARTNER ENGAGEMENT AGREEMENT", TITLE_STYLE))
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "Between Parrotbox Rides Pvt Ltd (a fictional demo platform, \"the "
            "Platform\") and the Delivery Partner (\"the Partner\").",
            _STYLES["Normal"],
        )
    )
    story.append(Paragraph("Effective Date: 15 May 2026", _STYLES["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("<b>1. Nature of engagement</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "The Partner is engaged as an independent contractor. Nothing in this "
            "agreement shall be construed to create an employer-employee "
            "relationship, partnership, or agency. The Partner is not entitled to "
            "salary, provident fund, ESI, gratuity, paid leave, or any other "
            "statutory benefit available to employees.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>2. Earnings</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "The Partner will earn a per-delivery fee as determined by the "
            "Platform from time to time in its sole discretion. Earnings will be "
            "credited weekly after deduction of the Platform's service fee "
            "(currently 20% of gross earnings) and any outstanding penalties.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>3. Working hours</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "The Partner is free to log in at any time. However, the Partner must "
            "maintain a minimum of 10 hours of active login per day and 60 hours "
            "per week to retain priority status.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>4. Penalties</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "The following penalties will be deducted from weekly earnings: "
            "cancelled delivery: Rs 50 per cancellation; rejected delivery after "
            "acceptance: Rs 100; missed delivery: Rs 200; uniform damage: Rs 500.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>5. Device security deposit</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "The Partner shall furnish a refundable security deposit of Rs 5,000 "
            "for the Platform-issued delivery device. The deposit will be "
            "recovered from the Partner's earnings in four equal weekly "
            "installments and refunded on return of the device in working "
            "condition.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>6. Non-compete</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "During the term of this agreement and for 90 days after termination, "
            "the Partner shall not provide similar services to any competing "
            "platform operating in the same city. Breach of this clause shall "
            "attract liquidated damages of Rs 25,000.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>7. Termination</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "The Platform may deactivate the Partner's account at any time, with "
            "or without cause, without any notice. The Partner may terminate this "
            "agreement by giving 15 days written notice.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.3 * inch))
    story.append(
        Paragraph(
            "This is a sample anonymized document for testing purposes.",
            _STYLES["Italic"],
        )
    )

    doc.build(story)
    print("✓ Created: gig_worker_agreement.pdf")


# ---------------------------------------------------------------------------
# 6. Karnataka factory offer letter
# ---------------------------------------------------------------------------
# Triggers the state-specific labor-law lookup. The document mentions
# Karnataka explicitly and sits on the edge of the Karnataka Shops &
# Commercial Establishments Act coverage, with a notice period that looks
# inconsistent with Section 39 of that Act. Intended to make Gemma consult
# the Karnataka-specific entry in our labor-law DB.
def build_karnataka_factory_offer_letter() -> None:
    doc = SimpleDocTemplate(
        str(SAMPLE_DIR / "karnataka_factory_offer_letter.pdf"), pagesize=letter
    )
    story = []
    story.append(Paragraph("APPOINTMENT LETTER", TITLE_STYLE))
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "Brassgear Components Pvt Ltd (Unit II) — a fictional demo employer, "
            "Peenya Industrial Area, Bengaluru, Karnataka - 560058",
            _STYLES["Normal"],
        )
    )
    story.append(Paragraph("Ref: BGC/APT/2026/0421", _STYLES["Normal"]))
    story.append(Paragraph("Date: 03 May 2026", _STYLES["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    story.append(
        Paragraph(
            "Shri. Suresh Kumar<br/>S/O Late Shri. Ram Kumar<br/>Peenya 2nd Stage, "
            "Bengaluru",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph("Sub: Appointment as Machine Operator (Lathe)", _STYLES["Normal"])
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(
        Paragraph(
            "Dear Sir, we are pleased to appoint you as a Machine Operator "
            "(Lathe) at our Peenya unit with effective date of joining 15 May "
            "2026, subject to the following terms and conditions.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>1. Place of work</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "Your place of work shall be our Unit II at Peenya Industrial Area, "
            "Bengaluru, Karnataka. The Company may transfer you to any other unit "
            "in Karnataka or Andhra Pradesh at its discretion.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>2. Compensation</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "Gross monthly wages: Rs 18,500, comprising Basic Rs 11,000, "
            "Dearness Allowance Rs 4,500, HRA Rs 3,000. Wages will be paid on "
            "the 7th of each month by bank transfer. Statutory deductions for "
            "Provident Fund (12% of basic), Employee State Insurance (0.75% of "
            "gross), and Professional Tax (Rs 200) will apply.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>3. Working hours and overtime</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "Your working hours shall be 9 hours per day including a 30-minute "
            "rest interval, with one weekly off on Sunday. Overtime, if worked at "
            "the Company's request, will be compensated at the ordinary rate of "
            "wages (i.e. single rate).",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>4. Leave</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "You will be entitled to earned leave as per the Factories Act, 1948 "
            "and the Karnataka Factories Rules. Sick leave and casual leave as "
            "per Company policy.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>5. Probation and confirmation</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "You shall be on probation for an initial period of 6 months, which "
            "may be extended at the Company's discretion.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>6. Notice period</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "Either side may terminate this appointment by giving written notice: "
            "7 days during probation from your side and 3 days from the Company's "
            "side; after confirmation, 30 days from your side and 15 days from "
            "the Company's side.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>7. Conduct</b>", _STYLES["Normal"]))
    story.append(
        Paragraph(
            "You shall abide by the Company's Standing Orders, safety rules, and "
            "discipline policy. Unauthorised absence for more than 3 consecutive "
            "days shall be treated as voluntary abandonment of service.",
            _STYLES["Normal"],
        )
    )
    story.append(Spacer(1, 0.3 * inch))
    story.append(
        Paragraph("For Brassgear Components Pvt Ltd", _STYLES["Normal"])
    )
    story.append(Paragraph("Authorised Signatory", _STYLES["Normal"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            "This is a sample anonymized document for testing purposes.",
            _STYLES["Italic"],
        )
    )

    doc.build(story)
    print("✓ Created: karnataka_factory_offer_letter.pdf")


def main() -> int:
    if _DEVANAGARI_FONT_NAME:
        print(f"Devanagari font registered: {_DEVANAGARI_FONT_NAME}")
    else:
        print("Devanagari font NOT found; Hindi summary will be omitted.")

    build_employment_agreement_bond_clause()
    build_salary_slip_deductions()
    build_non_employment_aws_sample()
    build_hindi_english_offer_letter()
    build_gig_worker_agreement()
    build_karnataka_factory_offer_letter()

    print(f"\n✅ All sample PDF files created in {SAMPLE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
