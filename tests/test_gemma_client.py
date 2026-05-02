from __future__ import annotations

from types import SimpleNamespace

import pytest

from services import gemma_client
from services.gemma_client import (
    DEFAULT_GEMMA_MODEL,
    DEFAULT_HF_ROUTER_BASE_URL,
    GemmaClientError,
    call_api,
    extract_text_from_openai_completion,
    get_gemma_config,
)


def test_api_config_defaults_to_hugging_face_router() -> None:
    config = get_gemma_config(
        {
            "MODEL_MODE": "real",
            "HF_TOKEN": "hf_test_token",
        }
    )

    assert config.model_mode == "real"
    assert config.provider == "api"
    assert config.model == DEFAULT_GEMMA_MODEL
    assert config.api_base_url == DEFAULT_HF_ROUTER_BASE_URL
    assert config.api_key == "hf_test_token"


def test_api_config_allows_explicit_openai_compatible_base_url() -> None:
    config = get_gemma_config(
        {
            "MODEL_MODE": "real",
            "GEMMA_PROVIDER": "api",
            "GEMMA_MODEL": "google/gemma-4-31B-it:novita",
            "GEMMA_API_BASE_URL": "https://router.huggingface.co/v1",
            "GEMMA_API_KEY": "custom_key",
        }
    )

    assert config.provider == "api"
    assert config.model == "google/gemma-4-31B-it:novita"
    assert config.api_base_url == "https://router.huggingface.co/v1"
    assert config.api_key == "custom_key"


def test_call_api_uses_openai_compatible_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"document_type":"Employment Contract"}')
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, base_url: str, api_key: str):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(gemma_client, "OpenAI", FakeOpenAI)
    config = get_gemma_config(
        {
            "MODEL_MODE": "real",
            "GEMMA_MODEL": "google/gemma-4-31B-it:novita",
            "HF_TOKEN": "hf_test_token",
        }
    )

    response_text = call_api("Return JSON only.", config)

    assert response_text == '{"document_type":"Employment Contract"}'
    assert captured["base_url"] == DEFAULT_HF_ROUTER_BASE_URL
    assert captured["api_key"] == "hf_test_token"
    assert captured["model"] == "google/gemma-4-31B-it:novita"
    assert captured["messages"][0]["role"] == "user"
    assert captured["messages"][0]["content"][0]["type"] == "text"
    assert captured["messages"][0]["content"][0]["text"] == "Return JSON only."


def test_call_api_requires_hf_token_or_api_key() -> None:
    config = get_gemma_config(
        {
            "MODEL_MODE": "real",
            "GEMMA_PROVIDER": "api",
        }
    )

    with pytest.raises(GemmaClientError, match="HF_TOKEN or GEMMA_API_KEY"):
        call_api("prompt", config)


def test_extract_text_from_openai_completion_supports_string_content() -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=" strict json ")
            )
        ]
    )

    assert extract_text_from_openai_completion(completion) == "strict json"


def test_extract_text_from_openai_completion_supports_list_content() -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"type": "text", "text": "part one"},
                        {"type": "text", "text": "part two"},
                    ]
                )
            )
        ]
    )

    assert extract_text_from_openai_completion(completion) == "part one\npart two"

