from __future__ import annotations

import pytest

from services.prompt_loader import (
    PromptLoaderError,
    load_prompt_template,
    render_document_analysis_prompt,
    render_prompt_template,
)


def test_load_document_analysis_prompt_template() -> None:
    template = load_prompt_template("document_analysis_prompt.txt")

    assert "Gemma 4" in template
    assert "{{target_language}}" in template
    assert "{{extracted_document_text}}" in template
    assert "strict valid JSON" in template


def test_render_document_analysis_prompt_replaces_all_placeholders() -> None:
    prompt = render_document_analysis_prompt(
        extracted_text="Salary: Rs 18000 per month",
        target_language="Hindi",
    )

    assert "Salary: Rs 18000 per month" in prompt
    # The target language appears in context now, so assert the language name
    # shows up somewhere in the rendered prompt rather than pinning an exact
    # label string.
    assert "Hindi" in prompt
    assert "{{" not in prompt
    assert "}}" not in prompt


def test_missing_prompt_template_raises_clear_error() -> None:
    with pytest.raises(PromptLoaderError, match="Prompt template not found"):
        load_prompt_template("missing_prompt.txt")


def test_missing_prompt_value_raises_clear_error() -> None:
    with pytest.raises(PromptLoaderError, match="Missing prompt value"):
        render_prompt_template(
            "document_analysis_prompt.txt",
            target_language="English",
        )

