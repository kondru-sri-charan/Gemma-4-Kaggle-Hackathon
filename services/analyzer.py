from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from services.gemma_client import (
    MODEL_MODE_REAL,
    GemmaClientError,
    generate_gemma_response,
    get_gemma_config,
)
from services.prompt_loader import PromptLoaderError, render_document_analysis_prompt


CONFIDENCE_LEVELS = {"High", "Medium", "Low"}
UNREADABLE_DOCUMENT_TYPE = "Unreadable / No selectable text"
UNREADABLE_SIMPLE_EXPLANATION = (
    "No readable text could be extracted from this document. The file may be scanned "
    "or image-based, so Sahaayak AI cannot analyze its employment terms yet."
)
UNSUPPORTED_DOCUMENT_TYPE = "Unsupported / Non-employment document"
UNSUPPORTED_SIMPLE_EXPLANATION = (
    "This document does not appear to be an employment-related document. "
    "Sahaayak AI is currently designed for offer letters, contracts, payslips, "
    "termination letters, and similar worker-facing documents."
)
EMPLOYMENT_SIGNALS = (
    "salary",
    "wage",
    "wages",
    "employee",
    "employer",
    "employment",
    "appointment",
    "appointed",
    "offer letter",
    "contract",
    "notice period",
    "probation",
    "working hours",
    "work hours",
    "leave",
    "deduction",
    "deduct",
    "payslip",
    "pay slip",
    "salary slip",
    "termination",
    "overtime",
    "bond",
    "designation",
    "job title",
    "ctc",
    "compensation",
    "remuneration",
    "stipend",
)
UNREADABLE_TEXT_MARKERS = (
    "no selectable text was found",
    "no readable text",
    "could not extract text",
    "no text was detected",
)
LEAVE_ENTITLEMENT_TERMS = (
    "annual leave",
    "paid leave",
    "paid leaves",
    "sick leave",
    "casual leave",
    "earned leave",
    "holiday",
    "holidays",
    "weekly off",
    "leave entitlement",
)
STATUTORY_DEDUCTION_TERMS = (
    "pf",
    "provident fund",
    "professional tax",
    "esi",
    "tds",
    "income tax",
)
GENERIC_DEDUCTION_TERMS = (
    "deduction",
    "deductions",
    "deduct",
    "deducted",
    "withhold",
    "withholding",
)
HIGH_RISK_DEDUCTION_TERMS = (
    "penalty",
    "fine",
    "recovery",
    "recover",
    "damages",
    "bond deduction",
    "bond recovery",
    "salary withholding",
    "unauthorized deduction",
    "unauthorised deduction",
    "uniform deduction",
    "safety gear deduction",
    "tool deduction",
    "training recovery",
)
REQUIRED_FIELDS = [
    "document_type",
    "simple_explanation",
    "key_points",
    "risk_flags",
    "next_actions",
    "questions_to_ask",
    "source_references",
    "local_language_summary",
    "confidence_level",
]
RISK_FIELDS = [
    "risk_title",
    "risk_explanation",
    "severity",
    "source_text",
    "suggested_action",
]


DocumentAnalysis = dict[str, Any]
RiskFlag = dict[str, str]
ANALYSIS_SOURCE_MOCK = "mock_analyzer"
ANALYSIS_SOURCE_GEMMA_REAL = "gemma_real"
ANALYSIS_SOURCE_MODEL_FALLBACK = "fallback_after_model_failure"


@dataclass(frozen=True)
class AnalysisResult:
    analysis: DocumentAnalysis
    source: str


class ModelResponseError(RuntimeError):
    """Raised when a model response is not strict valid analysis JSON."""


def analyze_document(extracted_text: str, target_language: str) -> DocumentAnalysis:
    """Build a validated analysis result using mock mode or a real Gemma provider."""
    return analyze_document_with_status(extracted_text, target_language).analysis


def analyze_document_with_status(extracted_text: str, target_language: str) -> AnalysisResult:
    """Build a validated analysis result with a UI-safe source label."""
    config = get_gemma_config()

    try:
        prompt = render_document_analysis_prompt(extracted_text, target_language)
    except (PromptLoaderError, Exception) as exc:
        source = (
            ANALYSIS_SOURCE_MODEL_FALLBACK
            if config.model_mode == MODEL_MODE_REAL
            else ANALYSIS_SOURCE_MOCK
        )
        return AnalysisResult(
            analysis=build_fallback_analysis(
                extracted_text=extracted_text,
                target_language=target_language,
                reason=str(exc),
            ),
            source=source,
        )

    if config.model_mode != MODEL_MODE_REAL:
        return AnalysisResult(
            analysis=build_safe_analyzer_response(extracted_text, target_language),
            source=ANALYSIS_SOURCE_MOCK,
        )

    try:
        model_response = generate_gemma_response(prompt, config)
        analysis = parse_model_json_response(model_response)
        return AnalysisResult(
            analysis=validate_strict_analysis_or_raise(analysis),
            source=ANALYSIS_SOURCE_GEMMA_REAL,
        )
    except (GemmaClientError, ModelResponseError, Exception):
        return AnalysisResult(
            analysis=build_safe_analyzer_response(extracted_text, target_language),
            source=ANALYSIS_SOURCE_MODEL_FALLBACK,
        )


def build_safe_analyzer_response(extracted_text: str, target_language: str) -> DocumentAnalysis:
    try:
        analysis = build_mock_model_response(extracted_text, target_language)
        return validate_or_fallback(analysis, extracted_text, target_language)
    except Exception as exc:
        return build_fallback_analysis(
            extracted_text=extracted_text,
            target_language=target_language,
            reason=str(exc),
        )


def parse_model_json_response(raw_response: str) -> DocumentAnalysis:
    stripped = (raw_response or "").strip()
    if not stripped:
        raise ModelResponseError("Model returned an empty response.")

    if not stripped.startswith("{") or not stripped.endswith("}"):
        raise ModelResponseError("Model response contained text outside the JSON object.")

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ModelResponseError(f"Model response was not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ModelResponseError("Model response JSON must be an object.")

    return parsed


def validate_strict_analysis_or_raise(analysis: DocumentAnalysis) -> DocumentAnalysis:
    errors = validate_analysis(analysis)
    if errors:
        raise ModelResponseError("; ".join(errors))

    return {field: analysis[field] for field in REQUIRED_FIELDS}


def build_mock_model_response(extracted_text: str, target_language: str) -> DocumentAnalysis:
    text = normalize_text(extracted_text)

    if is_unreadable_text(text):
        return build_unreadable_document_analysis(extracted_text, target_language)

    if not has_employment_signal(text):
        return build_unsupported_document_analysis(extracted_text, target_language)

    document_type = infer_document_type(text)
    sources = collect_source_clues(text)
    missing_topics = find_missing_topics(sources)
    risk_flags = build_risk_flags(text, sources, missing_topics)
    key_points = build_key_points(document_type, sources, missing_topics)

    return {
        "document_type": document_type,
        "simple_explanation": build_simple_explanation(document_type, sources, missing_topics),
        "key_points": key_points,
        "risk_flags": risk_flags,
        "next_actions": build_next_actions(missing_topics, risk_flags),
        "questions_to_ask": build_questions_to_ask(missing_topics, risk_flags),
        "source_references": build_source_references(sources),
        "local_language_summary": build_local_language_summary(
            target_language,
            document_type,
            sources,
        ),
        "confidence_level": infer_confidence_level(text, sources),
    }


def validate_or_fallback(
    analysis: DocumentAnalysis,
    extracted_text: str,
    target_language: str,
) -> DocumentAnalysis:
    errors = validate_analysis(analysis)
    if not errors:
        return analysis

    return build_fallback_analysis(
        extracted_text=extracted_text,
        target_language=target_language,
        reason="Analysis validation failed: " + "; ".join(errors),
    )


def validate_analysis(analysis: DocumentAnalysis) -> list[str]:
    errors: list[str] = []

    if not isinstance(analysis, dict):
        return ["Analysis must be a JSON object"]

    for field in REQUIRED_FIELDS:
        if field not in analysis:
            errors.append(f"Missing field: {field}")

    unexpected_fields = [field for field in analysis if field not in REQUIRED_FIELDS]
    for field in unexpected_fields:
        errors.append(f"Unexpected field: {field}")

    if errors:
        return errors

    for field in [
        "document_type",
        "simple_explanation",
        "local_language_summary",
        "confidence_level",
    ]:
        if not isinstance(analysis[field], str) or not analysis[field].strip():
            errors.append(f"{field} must be a non-empty string")

    for field in [
        "key_points",
        "risk_flags",
        "next_actions",
        "questions_to_ask",
        "source_references",
    ]:
        if not isinstance(analysis[field], list):
            errors.append(f"{field} must be a list")

    if analysis["confidence_level"] not in CONFIDENCE_LEVELS:
        errors.append("confidence_level must be High, Medium, or Low")

    risk_flags = analysis["risk_flags"]
    if not isinstance(risk_flags, list):
        return errors

    for index, risk in enumerate(risk_flags):
        if not isinstance(risk, dict):
            errors.append(f"risk_flags[{index}] must be an object")
            continue

        for field in RISK_FIELDS:
            value = risk.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"risk_flags[{index}].{field} must be a non-empty string")

        if risk.get("severity") not in CONFIDENCE_LEVELS:
            errors.append(f"risk_flags[{index}].severity must be High, Medium, or Low")

    return errors


def build_fallback_analysis(
    extracted_text: str,
    target_language: str,
    reason: str,
) -> DocumentAnalysis:
    source_text = first_useful_line(extracted_text) or "Not found in extracted text"

    return {
        "document_type": "Unknown employment document",
        "simple_explanation": (
            "The app could not produce a confident structured analysis from the extracted "
            "text. Review the extracted text manually and try a clearer document if needed."
        ),
        "key_points": [
            "The document appears to need manual review.",
            "Important employment details may be missing, unclear, or unreadable.",
        ],
        "risk_flags": [
            {
                "risk_title": "Low confidence analysis",
                "risk_explanation": reason,
                "severity": "Low",
                "source_text": source_text,
                "suggested_action": "Check the extracted text and upload a clearer file if key clauses are missing.",
            }
        ],
        "next_actions": [
            "Verify salary, job role, working hours, leave, deductions, and notice period manually.",
            "Ask the employer or advisor to explain any clause that is hard to read.",
        ],
        "questions_to_ask": [
            "What are the exact salary, working hours, leave, and notice terms?",
            "Are there any deductions, penalties, bonds, or unpaid overtime requirements?",
        ],
        "source_references": [source_text],
        "local_language_summary": (
            f"Mock {target_language} summary: the document needs careful manual review "
            "because the extracted text was not enough for a confident analysis."
        ),
        "confidence_level": "Low",
    }


def build_unsupported_document_analysis(
    extracted_text: str,
    target_language: str,
) -> DocumentAnalysis:
    source_text = first_useful_line(extracted_text) or "No readable document text was found."

    return {
        "document_type": UNSUPPORTED_DOCUMENT_TYPE,
        "simple_explanation": UNSUPPORTED_SIMPLE_EXPLANATION,
        "key_points": [
            "No clear employment-related signals were found in the extracted text.",
            "Sahaayak AI should be used with worker-facing employment documents.",
        ],
        "risk_flags": [],
        "next_actions": [
            "Upload an employment-related document for a useful analysis.",
        ],
        "questions_to_ask": [],
        "source_references": [
            f"Checked extracted text sample: {source_text}",
        ],
        "local_language_summary": (
            f"Mock {target_language} summary: this does not appear to be an employment-related "
            "document. Upload an offer letter, contract, payslip, termination letter, or similar "
            "worker-facing document."
        ),
        "confidence_level": "Low",
    }


def build_unreadable_document_analysis(
    extracted_text: str,
    target_language: str,
) -> DocumentAnalysis:
    source_text = first_useful_line(extracted_text) or "No readable document text was found."

    return {
        "document_type": UNREADABLE_DOCUMENT_TYPE,
        "simple_explanation": UNREADABLE_SIMPLE_EXPLANATION,
        "key_points": [
            "No readable text was available for analysis.",
            "The uploaded file may be a scanned PDF, image-only PDF, or unreadable upload.",
        ],
        "risk_flags": [],
        "next_actions": [
            "Upload a text-based PDF or enable OCR for scanned/image-based files.",
        ],
        "questions_to_ask": [],
        "source_references": [
            f"Extraction status: {source_text}",
        ],
        "local_language_summary": (
            f"Mock {target_language} summary: no readable text could be extracted. "
            "Upload a text-based PDF or enable OCR for scanned/image-based files."
        ),
        "confidence_level": "Low",
    }


def is_unreadable_text(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned:
        return True

    lowered = cleaned.lower()
    return any(marker in lowered for marker in UNREADABLE_TEXT_MARKERS) and not has_employment_signal(cleaned)


def has_employment_signal(text: str) -> bool:
    return line_has_any_term(text or "", EMPLOYMENT_SIGNALS)


def normalize_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text or "").strip()


def split_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def first_useful_line(text: str, max_length: int = 220) -> str:
    lines = split_lines(text)
    if not lines:
        return ""
    return trim_source(lines[0], max_length=max_length)


def find_source(text: str, keywords: tuple[str, ...]) -> str:
    for line in split_lines(text):
        if line_has_any_term(line, keywords):
            return trim_source(line)
    return ""


def find_leave_source(text: str) -> str:
    return find_source(text, LEAVE_ENTITLEMENT_TERMS)


def find_statutory_deduction_source(text: str) -> str:
    for line in split_lines(text):
        if line_has_any_term(line, STATUTORY_DEDUCTION_TERMS):
            return trim_source(line)
    return ""


def find_high_risk_deduction_source(text: str) -> str:
    return find_source(text, HIGH_RISK_DEDUCTION_TERMS)


def find_generic_deduction_source(text: str) -> str:
    for line in split_lines(text):
        has_generic_deduction = line_has_any_term(line, GENERIC_DEDUCTION_TERMS)
        has_statutory_deduction = line_has_any_term(line, STATUTORY_DEDUCTION_TERMS)
        has_high_risk_deduction = line_has_any_term(line, HIGH_RISK_DEDUCTION_TERMS)

        if not has_generic_deduction or has_high_risk_deduction:
            continue
        if has_statutory_deduction:
            continue

        return trim_source(line)

    return ""


def line_has_any_term(line: str, terms: tuple[str, ...]) -> bool:
    lowered = line.lower()
    for term in terms:
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, lowered):
            return True
    return False


def collect_source_clues(text: str) -> dict[str, str]:
    return {
        "role": find_source(
            text,
            ("designation", "position", "job title", "role", "appointed as", "employed as"),
        ),
        "salary": find_source(
            text,
            ("salary", "wage", "pay", "ctc", "compensation", "remuneration", "stipend"),
        ),
        "hours": find_source(
            text,
            ("working hours", "work hours", "shift", "overtime", "hours per", "weekly hours"),
        ),
        "leave": find_leave_source(text),
        "notice": find_source(text, ("notice period", "termination", "resignation", "terminate")),
        "deductions": find_generic_deduction_source(text),
        "high_risk_deductions": find_high_risk_deduction_source(text),
        "statutory_deductions": find_statutory_deduction_source(text),
        "probation": find_source(text, ("probation", "trial period", "probationary")),
        "start_date": find_source(text, ("start date", "joining date", "effective date", "commencement")),
    }


def find_missing_topics(sources: dict[str, str]) -> list[str]:
    required_topics = {
        "salary": "salary or wage amount",
        "hours": "working hours or overtime rules",
        "notice": "notice or termination terms",
        "leave": "leave or holiday entitlement",
    }
    return [label for key, label in required_topics.items() if not sources.get(key)]


def infer_document_type(text: str) -> str:
    lowered = text.lower()

    if any(term in lowered for term in ("offer letter", "appointment letter", "joining letter")):
        return "Employment Offer / Appointment Letter"
    if any(term in lowered for term in ("employment agreement", "employment contract", "contract of employment")):
        return "Employment Contract"
    if any(term in lowered for term in ("salary slip", "payslip", "pay slip", "wage statement")):
        return "Payslip / Wage Statement"
    if any(term in lowered for term in ("termination letter", "relieving letter", "resignation accepted")):
        return "Termination / Separation Document"
    if any(term in lowered for term in ("employee", "employer", "salary", "wage", "notice period")):
        return "Employment Document"

    return "Employment Document"


def build_simple_explanation(
    document_type: str,
    sources: dict[str, str],
    missing_topics: list[str],
) -> str:
    known_parts = []
    if sources.get("role"):
        known_parts.append("the role or designation")
    if sources.get("salary"):
        known_parts.append("pay details")
    if sources.get("hours"):
        known_parts.append("working hours")
    if sources.get("notice"):
        known_parts.append("notice or termination terms")

    if known_parts:
        details = ", ".join(known_parts)
        explanation = f"This looks like a {document_type}. The extracted text includes {details}."
    else:
        explanation = f"This looks like a {document_type}, but the extracted text is limited."

    if missing_topics:
        explanation += " Some important details are not clearly visible: " + ", ".join(missing_topics) + "."

    return explanation


def build_key_points(
    document_type: str,
    sources: dict[str, str],
    missing_topics: list[str],
) -> list[str]:
    points = [f"Document appears to be: {document_type}."]

    labels = {
        "role": "Role/designation",
        "salary": "Salary or pay",
        "hours": "Working hours",
        "leave": "Leave or holiday terms",
        "notice": "Notice or termination terms",
        "statutory_deductions": "Statutory deductions",
        "probation": "Probation terms",
        "start_date": "Start or joining date",
    }

    for key, label in labels.items():
        if sources.get(key):
            points.append(f"{label} found: {sources[key]}")

    if missing_topics:
        points.append("Missing or unclear topics: " + ", ".join(missing_topics) + ".")

    return points[:7]


def build_risk_flags(
    text: str,
    sources: dict[str, str],
    missing_topics: list[str],
) -> list[RiskFlag]:
    risks: list[RiskFlag] = []
    lowered = text.lower()

    if "salary or wage amount" in missing_topics:
        risks.append(
            make_risk(
                "Salary details missing or unclear",
                "The extracted text does not clearly show the salary, wage, or pay amount.",
                "High",
                "Not found in extracted text",
                "Ask for the exact gross pay, take-home pay, payment date, and deduction details in writing.",
            )
        )

    if "working hours or overtime rules" in missing_topics:
        risks.append(
            make_risk(
                "Working hours not clearly stated",
                "The document does not clearly explain normal working hours or overtime rules.",
                "Medium",
                "No working-hours or overtime clause found in extracted text.",
                "Ask for daily hours, weekly hours, rest days, and overtime pay rules before agreeing.",
            )
        )
    elif has_long_hours_risk(text):
        risks.append(
            make_risk(
                "Long working hours may apply",
                "The extracted text suggests long work hours that may need closer review.",
                "High",
                sources.get("hours") or first_useful_line(text),
                "Confirm the expected schedule, weekly limit, breaks, rest days, and overtime rate.",
            )
        )

    if sources.get("high_risk_deductions"):
        risks.append(
            make_risk(
                "Deductions or penalties mentioned",
                "The document appears to mention penalties, recovery, withholding, damages, or other high-risk deductions.",
                "High",
                sources["high_risk_deductions"],
                "Ask which deductions are allowed, when they apply, and whether they are legal and documented.",
            )
        )
    elif sources.get("deductions"):
        risks.append(
            make_risk(
                "Deductions or penalties mentioned",
                "The document appears to mention wage deductions, but the reason is not clearly explained.",
                "Medium",
                sources["deductions"],
                "Ask which deductions are allowed, when they apply, and whether they are legal and documented.",
            )
        )

    if "notice or termination terms" in missing_topics:
        risks.append(
            make_risk(
                "Exit terms not clearly visible",
                "The extracted text does not clearly show notice period or termination conditions.",
                "Medium",
                "Not found in extracted text",
                "Ask what notice is required and what happens if either side ends the job early.",
            )
        )

    if "overtime" in lowered and not any(term in lowered for term in ("overtime pay", "overtime rate", "paid overtime")):
        risks.append(
            make_risk(
                "Overtime payment may be unclear",
                "Overtime is mentioned, but the extracted text does not clearly show the payment rate.",
                "Medium",
                sources.get("hours") or find_source(text, ("overtime",)),
                "Ask for the overtime rate and when overtime becomes payable.",
            )
        )

    if any(term in lowered for term in ("non-compete", "non compete", "bond", "liquidated damages")):
        risks.append(
            make_risk(
                "Restrictive clause may need review",
                "The document may include a bond, non-compete, or damages clause that could limit future choices.",
                "High",
                find_source(text, ("non-compete", "non compete", "bond", "liquidated damages")),
                "Review this clause with a trusted advisor before signing.",
            )
        )

    if not risks:
        risks.append(
            make_risk(
                "No major risk detected in mock review",
                "The extracted text did not trigger the current mock risk checks.",
                "Low",
                first_useful_line(text) or "Not found in extracted text",
                "Still verify all important terms before signing or accepting the document.",
            )
        )

    return risks


def make_risk(
    title: str,
    explanation: str,
    severity: str,
    source_text: str,
    suggested_action: str,
) -> RiskFlag:
    return {
        "risk_title": title,
        "risk_explanation": explanation,
        "severity": severity,
        "source_text": source_text or "Not found in extracted text",
        "suggested_action": suggested_action,
    }


def build_next_actions(missing_topics: list[str], risk_flags: list[RiskFlag]) -> list[str]:
    actions = [
        "Compare the extracted terms with what was promised verbally or in messages.",
        "Keep a copy of the document and any employer clarifications.",
    ]

    if missing_topics:
        actions.insert(0, "Ask for written clarification on: " + ", ".join(missing_topics) + ".")

    if any(risk["severity"] == "High" for risk in risk_flags):
        actions.append("Speak with a trusted advisor or legal aid service before signing.")
    else:
        actions.append("Review the document one more time before signing or accepting.")

    return actions


def build_questions_to_ask(missing_topics: list[str], risk_flags: list[RiskFlag]) -> list[str]:
    questions = [
        "What is the exact gross pay and take-home pay?",
        "What are the normal work hours, rest days, and overtime rules?",
        "What deductions can be made from wages?",
        "What notice period applies if either side ends the job?",
    ]

    if "leave or holiday entitlement" in missing_topics:
        questions.append("How many paid leave, sick leave, and holidays are provided?")

    if any("Restrictive" in risk["risk_title"] for risk in risk_flags):
        questions.append("Does this clause restrict future jobs or require payment if the worker leaves?")

    return questions


def build_source_references(sources: dict[str, str]) -> list[str]:
    references = []
    labels = {
        "role": "Role/designation",
        "salary": "Salary/pay",
        "hours": "Working hours",
        "leave": "Leave",
        "notice": "Notice/termination",
        "deductions": "Deductions",
        "high_risk_deductions": "High-risk deductions/penalties",
        "statutory_deductions": "Statutory deductions",
        "probation": "Probation",
        "start_date": "Start date",
    }

    for key, label in labels.items():
        if sources.get(key):
            references.append(f"{label}: {sources[key]}")

    return references or ["No strong source references were found in the extracted text."]


def build_local_language_summary(
    target_language: str,
    document_type: str,
    sources: dict[str, str],
) -> str:
    found_terms = []
    if sources.get("salary"):
        found_terms.append("pay")
    if sources.get("hours"):
        found_terms.append("working hours")
    if sources.get("notice"):
        found_terms.append("notice terms")

    details = ", ".join(found_terms) if found_terms else "limited readable details"
    return (
        f"Mock {target_language} summary: this appears to be a {document_type}. "
        f"The readable text includes {details}. Please verify unclear terms before signing."
    )


def infer_confidence_level(text: str, sources: dict[str, str]) -> str:
    found_source_count = sum(1 for source in sources.values() if source)

    if len(text) > 500 and found_source_count >= 4:
        return "High"
    if len(text) > 120 and found_source_count >= 2:
        return "Medium"
    return "Low"


def has_long_hours_risk(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in ("12 hours", "12-hour", "12 hour", "60 hours", "72 hours")):
        return True

    hour_values = [int(value) for value in re.findall(r"\b(\d{2})\s*(?:hours|hrs)\b", lowered)]
    return any(value >= 12 for value in hour_values)


def trim_source(source: str, max_length: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", source).strip()
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 3].rstrip() + "..."
