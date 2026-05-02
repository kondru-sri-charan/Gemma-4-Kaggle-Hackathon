#!/usr/bin/env python3
"""Generate sample PDF files for Sahaayak AI testing."""

from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
except ImportError:
    print("Error: reportlab is not installed.")
    print("Install it with: pip install reportlab")
    exit(1)

sample_docs_dir = Path(__file__).parent / "sample_docs"
sample_docs_dir.mkdir(exist_ok=True)

styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=14,
    textColor=colors.HexColor('#000000'),
    spaceAfter=12,
)

# 1. Employment Agreement with Bond Clause
doc1 = SimpleDocTemplate(str(sample_docs_dir / "employment_agreement_bond_clause.pdf"), pagesize=letter)
story1 = []

story1.append(Paragraph("EMPLOYMENT AGREEMENT", title_style))
story1.append(Spacer(1, 0.2 * inch))
story1.append(Paragraph("Employee Name: John Doe", styles['Normal']))
story1.append(Paragraph("Employee ID: EMP-2024-001", styles['Normal']))
story1.append(Paragraph("Department: Software Engineering", styles['Normal']))
story1.append(Paragraph("Date of Joining: 01 January 2024", styles['Normal']))
story1.append(Spacer(1, 0.3 * inch))

story1.append(Paragraph("<b>1. POSITION AND RESPONSIBILITIES</b>", styles['Normal']))
story1.append(Paragraph("The Employee is appointed as Senior Software Engineer in the Software Development Department.", styles['Normal']))
story1.append(Spacer(1, 0.2 * inch))

story1.append(Paragraph("<b>2. COMPENSATION</b>", styles['Normal']))
story1.append(Paragraph("Monthly Salary: ₹1,20,000 (Gross CTC: ₹15,60,000)", styles['Normal']))
story1.append(Spacer(1, 0.2 * inch))

story1.append(Paragraph("<b>3. WORKING HOURS</b>", styles['Normal']))
story1.append(Paragraph("Standard working hours: 9:00 AM to 6:00 PM, Monday to Friday. Total 40 hours per week.", styles['Normal']))
story1.append(Spacer(1, 0.2 * inch))

story1.append(Paragraph("<b>4. LEAVE ENTITLEMENT</b>", styles['Normal']))
story1.append(Paragraph("Annual Leave: 18 days per annum, Sick Leave: 10 days per annum, Casual Leave: 5 days per annum", styles['Normal']))
story1.append(Spacer(1, 0.2 * inch))

story1.append(Paragraph("<b>5. SERVICE BOND CLAUSE (RESTRICTIVE)</b>", styles['Normal']))
story1.append(Paragraph("Employee acknowledges receipt of comprehensive training valued at ₹5,00,000. If Employee voluntarily resigns within 2 years of completion of training, Employee shall pay bond recovery amount of ₹2,50,000 to the Company.", styles['Normal']))
story1.append(Paragraph("In case of breach, the Company reserves the right to recover the amount from any pending salary, bonus, or gratuity.", styles['Normal']))
story1.append(Spacer(1, 0.2 * inch))

story1.append(Paragraph("<b>6. NOTICE PERIOD</b>", styles['Normal']))
story1.append(Paragraph("During probation (first 6 months): 15 days notice required. After probation: 30 days notice required.", styles['Normal']))
story1.append(Spacer(1, 0.2 * inch))

story1.append(Paragraph("<b>7. PROBATION</b>", styles['Normal']))
story1.append(Paragraph("The first 6 months of employment shall be probationary period.", styles['Normal']))
story1.append(Spacer(1, 0.3 * inch))

story1.append(Paragraph("This is a sample anonymized document for testing purposes.", styles['Italic']))

doc1.build(story1)
print("✓ Created: employment_agreement_bond_clause.pdf")

# 2. Salary Slip with Deductions
doc2 = SimpleDocTemplate(str(sample_docs_dir / "salary_slip_deductions.pdf"), pagesize=letter)
story2 = []

story2.append(Paragraph("SALARY SLIP - PAYROLL STATEMENT", title_style))
story2.append(Spacer(1, 0.2 * inch))
story2.append(Paragraph("Employee ID: EMP-2024-001", styles['Normal']))
story2.append(Paragraph("Employee Name: John Doe", styles['Normal']))
story2.append(Paragraph("Salary Month: March 2024", styles['Normal']))
story2.append(Spacer(1, 0.3 * inch))

# Earnings Table
story2.append(Paragraph("<b>EARNINGS</b>", styles['Heading3']))
earnings_data = [
    ['Description', 'Amount (₹)'],
    ['Basic Salary', '60,000'],
    ['Dearness Allowance (DA)', '15,000'],
    ['House Rent Allowance (HRA)', '25,000'],
    ['Other Allowances', '10,000'],
    ['<b>Total Earnings</b>', '<b>1,10,000</b>'],
]
earnings_table = Table(earnings_data, colWidths=[3 * inch, 2 * inch])
earnings_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
    ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))
story2.append(earnings_table)
story2.append(Spacer(1, 0.3 * inch))

# Deductions Table
story2.append(Paragraph("<b>DEDUCTIONS</b>", styles['Heading3']))
deductions_data = [
    ['Description', 'Amount (₹)'],
    ['Provident Fund (PF)', '8,000'],
    ['Professional Tax', '200'],
    ['TDS (Tax Deducted at Source)', '3,500'],
    ['ESI (Employee State Insurance)', '1,200'],
    ['<b>Total Deductions</b>', '<b>12,900</b>'],
]
deductions_table = Table(deductions_data, colWidths=[3 * inch, 2 * inch])
deductions_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
    ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))
story2.append(deductions_table)
story2.append(Spacer(1, 0.3 * inch))

story2.append(Paragraph("<b>Net Salary:</b> ₹97,100", styles['Heading3']))
story2.append(Spacer(1, 0.3 * inch))
story2.append(Paragraph("This is a sample anonymized document for testing purposes.", styles['Italic']))

doc2.build(story2)
print("✓ Created: salary_slip_deductions.pdf")

# 3. Non-Employment Document (AWS Sample)
doc3 = SimpleDocTemplate(str(sample_docs_dir / "non_employment_aws_sample.pdf"), pagesize=letter)
story3 = []

story3.append(Paragraph("AWS EC2 TECHNICAL DOCUMENTATION", title_style))
story3.append(Spacer(1, 0.2 * inch))
story3.append(Paragraph("<b>Amazon Elastic Compute Cloud (EC2)</b>", styles['Heading2']))
story3.append(Spacer(1, 0.2 * inch))

story3.append(Paragraph("<b>1. Overview</b>", styles['Normal']))
story3.append(Paragraph("Amazon EC2 is a web service that provides resizable compute capacity in the cloud. It is designed to make web-scale cloud computing easier for developers.", styles['Normal']))
story3.append(Spacer(1, 0.2 * inch))

story3.append(Paragraph("<b>2. Features</b>", styles['Normal']))
story3.append(Paragraph("• Virtual computing environments (instances)", styles['Normal']))
story3.append(Paragraph("• Pre-configured templates for instances (AMIs)", styles['Normal']))
story3.append(Paragraph("• Various configurations of CPU, memory, storage, and networking", styles['Normal']))
story3.append(Paragraph("• Secure login mechanism (key pairs)", styles['Normal']))
story3.append(Spacer(1, 0.2 * inch))

story3.append(Paragraph("<b>3. Instance Types</b>", styles['Normal']))
story3.append(Paragraph("• General Purpose (t2, t3, m5, m6i)", styles['Normal']))
story3.append(Paragraph("• Compute Optimized (c5, c6i)", styles['Normal']))
story3.append(Paragraph("• Memory Optimized (r5, r6i)", styles['Normal']))
story3.append(Paragraph("• Storage Optimized (i3, d2)", styles['Normal']))
story3.append(Spacer(1, 0.2 * inch))

story3.append(Paragraph("<b>4. Pricing Models</b>", styles['Normal']))
story3.append(Paragraph("• On-Demand: Pay for compute capacity by the hour", styles['Normal']))
story3.append(Paragraph("• Reserved Instances: Reserve capacity for 1 or 3 years", styles['Normal']))
story3.append(Paragraph("• Spot Instances: Bid for unused capacity", styles['Normal']))
story3.append(Spacer(1, 0.3 * inch))

story3.append(Paragraph("This is a sample non-employment document for testing unsupported document classification.", styles['Italic']))

doc3.build(story3)
print("✓ Created: non_employment_aws_sample.pdf")

print("\n✅ All sample PDF files created successfully in sample_docs/")
