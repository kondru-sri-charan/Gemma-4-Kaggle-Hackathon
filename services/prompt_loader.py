from __future__ import annotations

import re
from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
PLACEHOLDER_PATTERN = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


class PromptLoaderError(RuntimeError):
    """Raised when a prompt template cannot be loaded or rendered."""


def load_prompt_template(template_name: str) -> str:
    template_path = PROMPTS_DIR / template_name

    if not template_path.is_file():
        raise PromptLoaderError(f"Prompt template not found: {template_path}")

    return template_path.read_text(encoding="utf-8")


def render_prompt_template(template_name: str, **values: str) -> str:
    template = load_prompt_template(template_name)

    def replace_placeholder(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise PromptLoaderError(f"Missing prompt value for: {key}")
        return values[key]

    rendered = PLACEHOLDER_PATTERN.sub(replace_placeholder, template)
    unresolved = PLACEHOLDER_PATTERN.findall(rendered)
    if unresolved:
        missing_values = ", ".join(sorted(set(unresolved)))
        raise PromptLoaderError(f"Unresolved prompt placeholders: {missing_values}")

    return rendered


def render_document_analysis_prompt(extracted_text: str, target_language: str) -> str:
    return render_prompt_template(
        "document_analysis_prompt.txt",
        extracted_document_text=extracted_text,
        target_language=target_language,
    )

