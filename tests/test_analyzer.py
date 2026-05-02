from __future__ import annotations

import json
from pathlib import Path

import pytest

from services import analyzer
from services.gemma_client import GemmaClientError
from services.analyzer import (
    ANALYSIS_SOURCE_GEMMA_REAL,
    ANALYSIS_SOURCE_MOCK,
    ANALYSIS_SOURCE_MODEL_FALLBACK,
    REQUIRED_FIELDS,
    RISK_FIELDS,
    UNREADABLE_DOCUMENT_TYPE,
    UNREADABLE_SIMPLE_EXPLANATION,
    UNSUPPORTED_DOCUMENT_TYPE,
    UNSUPPORTED_SIMPLE_EXPLANATION,
    analyze_document,
    analyze_document_with_status,
    build_fallback_analysis,
    has_employment_signal,
    parse_model_json_response,
    validate_analysis,
    validate_or_fallback,
)
from services.prompt_loader import PromptLoaderError


CASES_PATH = Path(__file__).parent / "fixtures" / "document_cases.json"
DOCUMENT_CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))


def build_valid_model_analysis() -> dict:
    return {
        "document_type": "Employment Offer / Appointment Letter",
        "simple_explanation": "This is a model-generated explanation of the offer letter.",
        "key_points": ["Salary and working hours are stated."],
        "risk_flags": [
            {
                "risk_title": "Model risk",
                "risk_explanation": "The model found a possible issue.",
                "severity": "Medium",
                "source_text": "Working hours: 10 hours per day",
                "suggested_action": "Ask the employer to clarify working hours.",
            }
        ],
        "next_actions": ["Ask for written clarification before signing."],
        "questions_to_ask": ["What is the overtime rate?"],
        "source_references": ["Working hours clause"],
        "local_language_summary": "Mock local language summary from model.",
        "confidence_level": "High",
    }


def assert_schema_is_valid(analysis: dict) -> None:
    assert list(analysis.keys()) == REQUIRED_FIELDS
    assert validate_analysis(analysis) == []

    for field in ["key_points", "risk_flags", "next_actions", "questions_to_ask", "source_references"]:
        assert isinstance(analysis[field], list)

    for risk in analysis["risk_flags"]:
        assert list(risk.keys()) == RISK_FIELDS
        assert risk["severity"] in {"High", "Medium", "Low"}


@pytest.mark.parametrize("case", DOCUMENT_CASES, ids=[case["id"] for case in DOCUMENT_CASES])
def test_document_corpus(case: dict) -> None:
    analysis = analyze_document(case["text"], case["target_language"])
    assert_schema_is_valid(analysis)

    expected_document_type = case.get("expected_document_type")

    if expected_document_type == UNREADABLE_DOCUMENT_TYPE:
        assert analysis["document_type"] == UNREADABLE_DOCUMENT_TYPE
        assert analysis["simple_explanation"] == UNREADABLE_SIMPLE_EXPLANATION
        assert analysis["confidence_level"] == "Low"
        assert analysis["risk_flags"] == []
        assert analysis["next_actions"] == [
            "Upload a text-based PDF or enable OCR for scanned/image-based files."
        ]
    elif case["expected_supported"]:
        assert analysis["document_type"] != UNSUPPORTED_DOCUMENT_TYPE
        assert analysis["risk_flags"]
        assert analysis["next_actions"] != ["Upload an employment-related document for a useful analysis."]
    else:
        assert analysis["document_type"] == UNSUPPORTED_DOCUMENT_TYPE
        assert analysis["simple_explanation"] == UNSUPPORTED_SIMPLE_EXPLANATION
        assert analysis["confidence_level"] == "Low"
        assert analysis["risk_flags"] == []
        assert analysis["next_actions"] == ["Upload an employment-related document for a useful analysis."]
        assert analysis["questions_to_ask"] == []

    if case["category"] == "canary":
        assert set(analysis.keys()) == set(REQUIRED_FIELDS)
        assert "admin_password" not in analysis


def test_non_employment_document_gets_exact_unsupported_response() -> None:
    analysis = analyze_document(
        "Invoice No 99\nItem: notebook\nTotal due: Rs 90",
        "English",
    )

    assert_schema_is_valid(analysis)
    assert analysis["document_type"] == UNSUPPORTED_DOCUMENT_TYPE
    assert analysis["simple_explanation"] == UNSUPPORTED_SIMPLE_EXPLANATION
    assert analysis["confidence_level"] == "Low"
    assert analysis["risk_flags"] == []
    assert analysis["next_actions"] == ["Upload an employment-related document for a useful analysis."]


@pytest.mark.parametrize("text", ["", "   \n\t  ", "No selectable text was found in this PDF. It may be scanned."])
def test_unreadable_document_gets_exact_unreadable_response(text: str) -> None:
    analysis = analyze_document(text, "English")

    assert_schema_is_valid(analysis)
    assert analysis["document_type"] == UNREADABLE_DOCUMENT_TYPE
    assert analysis["simple_explanation"] == UNREADABLE_SIMPLE_EXPLANATION
    assert analysis["confidence_level"] == "Low"
    assert analysis["risk_flags"] == []
    assert analysis["next_actions"] == [
        "Upload a text-based PDF or enable OCR for scanned/image-based files."
    ]


def test_valid_employment_document_keeps_existing_analysis_path() -> None:
    analysis = analyze_document(
        "Offer Letter\nEmployee: Ravi\nSalary: Rs 18000 per month\n"
        "Working hours: 12 hours per day\nLeave: 12 days\nNotice period: 30 days",
        "Marathi",
    )

    assert_schema_is_valid(analysis)
    assert analysis["document_type"] == "Employment Offer / Appointment Letter"
    assert analysis["risk_flags"]
    assert any(risk["risk_title"] == "Long working hours may apply" for risk in analysis["risk_flags"])


def test_deduction_and_bond_risks_are_detected() -> None:
    analysis = analyze_document(
        "Employment Contract\nSalary: Rs 25000\nWorking hours: 8 hours\nLeave: 10 days\n"
        "Notice period: 30 days\nDeduction may apply for damage.\nEmployee bond for 12 months.",
        "Hindi",
    )

    titles = {risk["risk_title"] for risk in analysis["risk_flags"]}
    assert "Deductions or penalties mentioned" in titles
    assert "Restrictive clause may need review" in titles


def test_statutory_deductions_only_are_key_point_not_high_risk() -> None:
    analysis = analyze_document(
        "Payslip\nEmployee: Meena\nEmployer: City Retail\nSalary: Rs 25000\n"
        "Working hours: 8 hours per day\nAnnual leave: 12 days\nNotice period: 30 days\n"
        "Deduction: PF Rs 1200, ESI Rs 200, TDS Rs 500, Professional Tax Rs 200",
        "English",
    )

    assert_schema_is_valid(analysis)
    assert any("Statutory deductions found" in point for point in analysis["key_points"])
    assert not any(
        risk["risk_title"] == "Deductions or penalties mentioned" and risk["severity"] == "High"
        for risk in analysis["risk_flags"]
    )


def test_bond_deduction_and_recovery_trigger_high_risk() -> None:
    analysis = analyze_document(
        "Employment Contract\nEmployee: Arjun\nSalary: Rs 22000\nWorking hours: 8 hours\n"
        "Annual leave: 12 days\nNotice period: 30 days\n"
        "Bond deduction and training recovery may be taken from salary if employee leaves early.",
        "Hindi",
    )

    assert_schema_is_valid(analysis)
    assert any(
        risk["risk_title"] == "Deductions or penalties mentioned" and risk["severity"] == "High"
        for risk in analysis["risk_flags"]
    )


def test_employee_leaves_before_12_months_is_not_leave_entitlement() -> None:
    analysis = analyze_document(
        "Employment Contract\nEmployee: Ravi\nSalary: Rs 18000\nWorking hours: 8 hours\n"
        "Notice period: 30 days\nIf the employee leaves before 12 months, unserved months may be recovered.",
        "English",
    )

    assert_schema_is_valid(analysis)
    assert not any("Leave or holiday terms found" in point for point in analysis["key_points"])
    assert any("leave or holiday entitlement" in point for point in analysis["key_points"])


def test_annual_and_sick_leave_are_detected_as_leave_entitlement() -> None:
    analysis = analyze_document(
        "Appointment Letter\nEmployee: Sana\nSalary: Rs 20000\nWorking hours: 8 hours\n"
        "Annual leave: 12 days. Sick leave: 6 days.\nNotice period: 30 days.",
        "English",
    )

    assert_schema_is_valid(analysis)
    assert any("Leave or holiday terms found" in point for point in analysis["key_points"])


def test_missing_working_hours_risk_has_specific_source_text() -> None:
    analysis = analyze_document(
        "Offer Letter\nEmployee: Neha\nSalary: Rs 21000\nAnnual leave: 12 days\nNotice period: 30 days",
        "English",
    )

    matching_risks = [
        risk for risk in analysis["risk_flags"] if risk["risk_title"] == "Working hours not clearly stated"
    ]
    assert matching_risks
    assert matching_risks[0]["source_text"] == "No working-hours or overtime clause found in extracted text."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("salary and wages are listed", True),
        ("appointment letter with probation", True),
        ("working hours and overtime", True),
        ("grocery receipt with fruit and soap", False),
        ("", False),
    ],
)
def test_employment_signal_detection(text: str, expected: bool) -> None:
    assert has_employment_signal(text) is expected


def test_validation_rejects_bad_schema() -> None:
    errors = validate_analysis(
        {
            "document_type": "Employment Contract",
            "simple_explanation": "",
            "key_points": "not a list",
            "risk_flags": ["bad risk"],
            "next_actions": [],
            "questions_to_ask": [],
            "source_references": [],
            "local_language_summary": "summary",
            "confidence_level": "Certain",
        }
    )

    assert "simple_explanation must be a non-empty string" in errors
    assert "key_points must be a list" in errors
    assert "confidence_level must be High, Medium, or Low" in errors
    assert "risk_flags[0] must be an object" in errors


def test_validate_or_fallback_returns_safe_low_confidence_result() -> None:
    result = validate_or_fallback(
        {"document_type": "Broken"},
        "Salary: Rs 10000",
        "English",
    )

    assert_schema_is_valid(result)
    assert result["confidence_level"] == "Low"
    assert result["risk_flags"][0]["risk_title"] == "Low confidence analysis"


def test_prompt_loader_failure_returns_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_prompt_error(extracted_text: str, target_language: str) -> str:
        raise PromptLoaderError("prompt missing")

    monkeypatch.setattr(analyzer, "render_document_analysis_prompt", raise_prompt_error)

    result = analyze_document("Salary: Rs 10000", "English")

    assert_schema_is_valid(result)
    assert result["confidence_level"] == "Low"
    assert result["risk_flags"][0]["risk_title"] == "Low confidence analysis"
    assert "prompt missing" in result["risk_flags"][0]["risk_explanation"]


def test_fallback_analysis_is_valid() -> None:
    result = build_fallback_analysis("Unreadable but salary appears once", "Tamil", "test reason")
    assert_schema_is_valid(result)
    assert result["confidence_level"] == "Low"


def test_mock_mode_uses_existing_analyzer_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_MODE", "mock")

    def fail_if_called(prompt: str, config: object) -> str:
        raise AssertionError("Gemma client should not be called in mock mode")

    monkeypatch.setattr(analyzer, "generate_gemma_response", fail_if_called)

    result_with_status = analyze_document_with_status(
        "Offer Letter\nEmployee: Ravi\nSalary: Rs 18000\nWorking hours: 12 hours per day\n"
        "Annual leave: 12 days\nNotice period: 30 days",
        "English",
    )
    result = result_with_status.analysis

    assert result_with_status.source == ANALYSIS_SOURCE_MOCK
    assert_schema_is_valid(result)
    assert result["document_type"] == "Employment Offer / Appointment Letter"
    assert any(risk["risk_title"] == "Long working hours may apply" for risk in result["risk_flags"])


def test_real_mode_accepts_valid_model_json(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = build_valid_model_analysis()
    monkeypatch.setenv("MODEL_MODE", "real")
    monkeypatch.setenv("GEMMA_PROVIDER", "api")
    monkeypatch.setenv("GEMMA_MODEL", "gemma4-test")

    def return_valid_json(prompt: str, config: object) -> str:
        assert "Salary: Rs 18000" in prompt
        assert config.model_mode == "real"
        assert config.provider == "api"
        assert config.model == "gemma4-test"
        return json.dumps(expected)

    monkeypatch.setattr(analyzer, "generate_gemma_response", return_valid_json)

    result_with_status = analyze_document_with_status("Offer Letter\nSalary: Rs 18000", "English")
    result = result_with_status.analysis

    assert result_with_status.source == ANALYSIS_SOURCE_GEMMA_REAL
    assert result == expected
    assert_schema_is_valid(result)


def test_real_mode_falls_back_on_model_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_MODE", "real")

    def raise_client_error(prompt: str, config: object) -> str:
        raise GemmaClientError("provider offline")

    monkeypatch.setattr(analyzer, "generate_gemma_response", raise_client_error)

    result_with_status = analyze_document_with_status(
        "Offer Letter\nEmployee: Ravi\nSalary: Rs 18000\nWorking hours: 12 hours per day\n"
        "Annual leave: 12 days\nNotice period: 30 days",
        "English",
    )
    result = result_with_status.analysis

    assert result_with_status.source == ANALYSIS_SOURCE_MODEL_FALLBACK
    assert_schema_is_valid(result)
    assert result["document_type"] == "Employment Offer / Appointment Letter"
    assert any(risk["risk_title"] == "Long working hours may apply" for risk in result["risk_flags"])


@pytest.mark.parametrize(
    "model_response",
    [
        "{not-json",
        "Here is the JSON:\n" + json.dumps(build_valid_model_analysis()),
    ],
)
def test_real_mode_falls_back_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
    model_response: str,
) -> None:
    monkeypatch.setenv("MODEL_MODE", "real")
    monkeypatch.setattr(
        analyzer,
        "generate_gemma_response",
        lambda prompt, config: model_response,
    )

    result_with_status = analyze_document_with_status(
        "Employment Contract\nEmployee: Meena\nSalary: Rs 20000\nWorking hours: 8 hours\n"
        "Annual leave: 12 days\nNotice period: 30 days",
        "English",
    )
    result = result_with_status.analysis

    assert result_with_status.source == ANALYSIS_SOURCE_MODEL_FALLBACK
    assert_schema_is_valid(result)
    assert result["document_type"] == "Employment Contract"
    assert result != build_valid_model_analysis()


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"document_type": "Employment Contract"},
        {**build_valid_model_analysis(), "admin_password": "secret"},
        {**build_valid_model_analysis(), "risk_flags": ["bad risk"]},
    ],
)
def test_real_mode_falls_back_on_schema_violation(
    monkeypatch: pytest.MonkeyPatch,
    bad_payload: dict,
) -> None:
    monkeypatch.setenv("MODEL_MODE", "real")
    monkeypatch.setattr(
        analyzer,
        "generate_gemma_response",
        lambda prompt, config: json.dumps(bad_payload),
    )

    result_with_status = analyze_document_with_status(
        "Appointment Letter\nEmployee: Sana\nSalary: Rs 20000\nWorking hours: 8 hours\n"
        "Annual leave: 12 days\nNotice period: 30 days",
        "English",
    )
    result = result_with_status.analysis

    assert result_with_status.source == ANALYSIS_SOURCE_MODEL_FALLBACK
    assert_schema_is_valid(result)
    assert result["document_type"] == "Employment Offer / Appointment Letter"
    assert "admin_password" not in result


def test_parse_model_json_response_rejects_extra_text() -> None:
    with pytest.raises(analyzer.ModelResponseError, match="outside the JSON object"):
        parse_model_json_response("Sure, here it is:\n{}")
