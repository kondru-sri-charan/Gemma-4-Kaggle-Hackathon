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

    def fail_if_called(prompt: str, config: object, **kwargs) -> str:
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

    def return_valid_json(prompt: str, config: object, **kwargs) -> str:
        assert "Salary: Rs 18000" in prompt
        assert config.model_mode == "real"
        assert config.provider == "api"
        assert config.model == "gemma4-test"
        # In text-only mode no images should be forwarded.
        assert kwargs.get("images") is None
        return json.dumps(expected)

    monkeypatch.setattr(analyzer, "generate_gemma_response", return_valid_json)

    result_with_status = analyze_document_with_status("Offer Letter\nSalary: Rs 18000", "English")
    result = result_with_status.analysis

    assert result_with_status.source == ANALYSIS_SOURCE_GEMMA_REAL
    assert result == expected
    assert_schema_is_valid(result)


def test_real_mode_falls_back_on_model_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_MODE", "real")
    monkeypatch.setenv("GEMMA_PROVIDER", "api")

    def raise_client_error(prompt: str, config: object, **kwargs) -> str:
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
    monkeypatch.setenv("GEMMA_PROVIDER", "api")
    monkeypatch.setattr(
        analyzer,
        "generate_gemma_response",
        lambda prompt, config, **kwargs: model_response,
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
    monkeypatch.setenv("GEMMA_PROVIDER", "api")
    monkeypatch.setattr(
        analyzer,
        "generate_gemma_response",
        lambda prompt, config, **kwargs: json.dumps(bad_payload),
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


def test_real_mode_forwards_images_and_wraps_prompt_for_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vision path: images must reach generate_gemma_response and the prompt
    must tell Gemma to read the image, even when no extracted text is given."""
    captured: dict = {}
    monkeypatch.setenv("MODEL_MODE", "real")
    monkeypatch.setenv("GEMMA_PROVIDER", "api")

    def capture_and_return(prompt: str, config: object, **kwargs) -> str:
        captured["prompt"] = prompt
        captured["images"] = kwargs.get("images")
        return json.dumps(build_valid_model_analysis())

    monkeypatch.setattr(analyzer, "generate_gemma_response", capture_and_return)

    fake_image_b64 = "iVBORw0KGgoAAAANSUhEUgAA"  # not a real PNG, just a token
    result_with_status = analyze_document_with_status(
        extracted_text="",
        target_language="Hindi",
        images=[fake_image_b64],
    )

    assert result_with_status.source == ANALYSIS_SOURCE_GEMMA_REAL
    assert captured["images"] == [fake_image_b64]
    # With no extracted text, the prompt should carry only the vision notice.
    assert "read the image" in captured["prompt"].lower()


def test_real_mode_vision_prompt_includes_extracted_text_as_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid path: when both text and images are present, the prompt should
    include both so Gemma can cross-reference."""
    captured: dict = {}
    monkeypatch.setenv("MODEL_MODE", "real")
    monkeypatch.setenv("GEMMA_PROVIDER", "api")

    def capture_and_return(prompt: str, config: object, **kwargs) -> str:
        captured["prompt"] = prompt
        captured["images"] = kwargs.get("images")
        return json.dumps(build_valid_model_analysis())

    monkeypatch.setattr(analyzer, "generate_gemma_response", capture_and_return)

    result_with_status = analyze_document_with_status(
        extracted_text="Offer Letter\nSalary: Rs 18000",
        target_language="English",
        images=["fakeb64=="],
    )

    assert result_with_status.source == ANALYSIS_SOURCE_GEMMA_REAL
    assert captured["images"] == ["fakeb64=="]
    # Extracted text still appears in the prompt so the model can use it as
    # a secondary hint alongside the image.
    assert "Salary: Rs 18000" in captured["prompt"]
    assert "read the image" in captured["prompt"].lower()


def test_mock_mode_ignores_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock mode cannot see images; passing them must not change its behavior."""
    monkeypatch.setenv("MODEL_MODE", "mock")

    def fail_if_called(prompt: str, config: object, **kwargs) -> str:
        raise AssertionError("Gemma client should not be called in mock mode")

    monkeypatch.setattr(analyzer, "generate_gemma_response", fail_if_called)

    result = analyze_document(
        extracted_text="Offer Letter\nEmployee: Ravi\nSalary: Rs 18000\n"
        "Working hours: 12 hours per day\nAnnual leave: 12 days\nNotice period: 30 days",
        target_language="English",
        images=["fakeb64=="],
    )

    assert_schema_is_valid(result)
    assert result["document_type"] == "Employment Offer / Appointment Letter"


def test_real_mode_with_ollama_provider_routes_through_tool_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When running real mode against the local Ollama provider (the default),
    analyze_document_with_status should invoke the tool-calling loop and
    carry the tool_log back through AnalysisResult."""
    monkeypatch.setenv("MODEL_MODE", "real")
    monkeypatch.setenv("GEMMA_PROVIDER", "ollama")

    captured: dict = {}
    fake_tool_log = [
        {
            "name": "lookup_labor_law",
            "arguments": {"topic": "bond_clause"},
            "result": [{"statute_reference": "ICA 27"}],
        }
    ]

    def fake_generate_with_tools(
        prompt, tools, execute_tool, config, images=None, max_rounds=3
    ):
        captured["prompt"] = prompt
        captured["tools"] = tools
        captured["images"] = images
        # Sanity-check: the tool list must contain the labor-law tool.
        assert tools and tools[0]["function"]["name"] == "lookup_labor_law"
        return json.dumps(build_valid_model_analysis()), fake_tool_log

    def fail_text_only(*args, **kwargs):
        raise AssertionError(
            "generate_gemma_response should not be called when tools are active"
        )

    monkeypatch.setattr(
        analyzer, "generate_gemma_response_with_tools", fake_generate_with_tools
    )
    monkeypatch.setattr(analyzer, "generate_gemma_response", fail_text_only)

    result_with_status = analyze_document_with_status(
        extracted_text="Offer Letter\nSalary: Rs 18000\nBond: Rs 50000",
        target_language="English",
    )

    assert result_with_status.source == ANALYSIS_SOURCE_GEMMA_REAL
    assert result_with_status.tool_log == fake_tool_log
    assert captured["images"] is None


def test_real_mode_use_tools_false_bypasses_tool_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicitly disabling tools must route through the single-shot path
    even with the Ollama provider, for cases where we want to record
    text-only latency."""
    monkeypatch.setenv("MODEL_MODE", "real")
    monkeypatch.setenv("GEMMA_PROVIDER", "ollama")

    def fail_tools(*args, **kwargs):
        raise AssertionError("tool loop should not be called when use_tools=False")

    def text_only(prompt, config, **kwargs):
        return json.dumps(build_valid_model_analysis())

    monkeypatch.setattr(analyzer, "generate_gemma_response_with_tools", fail_tools)
    monkeypatch.setattr(analyzer, "generate_gemma_response", text_only)

    result_with_status = analyze_document_with_status(
        extracted_text="Offer Letter\nSalary: Rs 18000",
        target_language="English",
        use_tools=False,
    )

    assert result_with_status.source == ANALYSIS_SOURCE_GEMMA_REAL
    assert result_with_status.tool_log == []


def test_real_mode_tool_loop_failure_falls_back_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the tool loop raises (network drop, runaway, etc.), we must fall
    back to the mock analyzer instead of bubbling the exception to the UI."""
    monkeypatch.setenv("MODEL_MODE", "real")
    monkeypatch.setenv("GEMMA_PROVIDER", "ollama")

    def raise_client_error(*args, **kwargs):
        raise GemmaClientError("ollama offline")

    monkeypatch.setattr(
        analyzer, "generate_gemma_response_with_tools", raise_client_error
    )

    result_with_status = analyze_document_with_status(
        extracted_text="Offer Letter\nEmployee: Ravi\nSalary: Rs 18000\n"
        "Working hours: 12 hours per day\nAnnual leave: 12 days\nNotice period: 30 days",
        target_language="English",
    )

    assert result_with_status.source == ANALYSIS_SOURCE_MODEL_FALLBACK
    assert_schema_is_valid(result_with_status.analysis)
    # Mock analyzer should still flag the long working hours.
    assert any(
        risk["risk_title"] == "Long working hours may apply"
        for risk in result_with_status.analysis["risk_flags"]
    )
    assert result_with_status.tool_log == []


def test_parse_model_json_response_strips_code_fence_wrapper() -> None:
    """Gemma sometimes wraps the final answer in ```json ...``` even though
    the prompt forbids it. The parser must tolerate this single wrapper."""
    wrapped = '```json\n{"hello": "world"}\n```'
    assert parse_model_json_response(wrapped) == {"hello": "world"}

    wrapped_no_lang = "```\n{\"hello\": \"world\"}\n```"
    assert parse_model_json_response(wrapped_no_lang) == {"hello": "world"}


def test_parse_model_json_response_still_rejects_prose_around_fence() -> None:
    """Stripping fences must not become a blanket 'ignore prose' behaviour.
    Prose outside a fence should still fail strict parsing."""
    with pytest.raises(analyzer.ModelResponseError):
        parse_model_json_response('Here is your answer:\n```json\n{"a":1}\n```\nThanks')


def test_coerce_offscope_canonicalises_non_employment_classification() -> None:
    """When the model returns an off-scope document_type with no risk flags,
    we canonicalise the header but keep the descriptive explanation."""
    raw = {
        "document_type": "Health Insurance Benefit Card",
        "simple_explanation": (
            "This is a health insurance card from your employer, not an "
            "employment contract."
        ),
        "key_points": ["It is a health card, not a job contract."],
        "risk_flags": [],
        "next_actions": ["Keep the card safe."],
        "questions_to_ask": ["What is the coverage limit?"],
        "source_references": ["Member Name: Ravi Kumar"],
        "local_language_summary": "यह स्वास्थ्य बीमा कार्ड है।",
        "confidence_level": "High",
    }

    coerced = analyzer._coerce_offscope_classification(raw)

    assert coerced["document_type"] == "Unsupported / Non-employment document"
    assert coerced["confidence_level"] == "Low"
    assert coerced["next_actions"] == [
        "Upload an employment-related document for a useful analysis."
    ]
    # Descriptive explanation and local summary are preserved so the user
    # still learns what they uploaded.
    assert "health insurance card" in coerced["simple_explanation"].lower()
    assert coerced["local_language_summary"] == "यह स्वास्थ्य बीमा कार्ड है।"


def test_coerce_offscope_does_not_clobber_employment_analyses() -> None:
    """An employment-shaped analysis with real risk flags must pass through
    untouched, even if the document_type label is unusual."""
    raw = {
        "document_type": "Gig Worker Engagement Contract",
        "simple_explanation": "This contract sets out your gig terms.",
        "key_points": ["Gig contract key points."],
        "risk_flags": [
            {
                "risk_title": "Low notice period",
                "risk_explanation": "Only 7 days' notice required.",
                "severity": "Medium",
                "source_text": "Notice: 7 days",
                "suggested_action": "Ask for 30 days.",
            }
        ],
        "next_actions": ["Review with a lawyer."],
        "questions_to_ask": ["Is this notice period standard?"],
        "source_references": ["Notice: 7 days"],
        "local_language_summary": "गिग कॉन्ट्रैक्ट का सारांश।",
        "confidence_level": "Medium",
    }

    coerced = analyzer._coerce_offscope_classification(raw)

    assert coerced == raw  # untouched


def test_coerce_offscope_preserves_canonical_employment_document_types() -> None:
    """Our own canonical employment_type strings (with no risks) must not be
    coerced. Example: a fully-compliant offer letter for an unskilled worker
    might have no red flags at all, and we shouldn't misrepresent it."""
    raw = {
        "document_type": "Employment Offer / Appointment Letter",
        "simple_explanation": "A clean offer letter.",
        "key_points": ["All terms fine."],
        "risk_flags": [],  # no risks
        "next_actions": ["Sign and keep a copy."],
        "questions_to_ask": [],
        "source_references": ["Offer letter body"],
        "local_language_summary": "यह एक साफ ऑफर लेटर है।",
        "confidence_level": "High",
    }

    coerced = analyzer._coerce_offscope_classification(raw)

    assert coerced["document_type"] == "Employment Offer / Appointment Letter"
