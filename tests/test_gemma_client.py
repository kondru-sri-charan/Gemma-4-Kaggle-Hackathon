from __future__ import annotations

from types import SimpleNamespace

import pytest

from services import gemma_client
from services.gemma_client import (
    DEFAULT_GEMMA_MODEL,
    DEFAULT_HF_ROUTER_BASE_URL,
    DEFAULT_HF_ROUTER_MODEL,
    DEFAULT_OLLAMA_CHAT_URL,
    GemmaClientError,
    call_api,
    call_ollama,
    extract_text_from_openai_completion,
    get_gemma_config,
)


def test_real_mode_defaults_to_local_ollama_e4b() -> None:
    """Community booth pitch: local Ollama is the default path, not cloud."""
    config = get_gemma_config({"MODEL_MODE": "real"})

    assert config.model_mode == "real"
    assert config.provider == "ollama"
    assert config.model == DEFAULT_GEMMA_MODEL == "gemma4:e4b"
    assert config.ollama_url == DEFAULT_OLLAMA_CHAT_URL


def test_api_config_points_at_hugging_face_router_when_opted_in() -> None:
    config = get_gemma_config(
        {
            "MODEL_MODE": "real",
            "GEMMA_PROVIDER": "api",
            "HF_TOKEN": "hf_test_token",
        }
    )

    assert config.provider == "api"
    assert config.model == DEFAULT_HF_ROUTER_MODEL == "google/gemma-4-e4b-it"
    assert config.api_base_url == DEFAULT_HF_ROUTER_BASE_URL
    assert config.api_key == "hf_test_token"


def test_api_config_allows_explicit_openai_compatible_base_url() -> None:
    config = get_gemma_config(
        {
            "MODEL_MODE": "real",
            "GEMMA_PROVIDER": "api",
            "GEMMA_MODEL": "google/gemma-4-e4b-it",
            "GEMMA_API_BASE_URL": "https://router.huggingface.co/v1",
            "GEMMA_API_KEY": "custom_key",
        }
    )

    assert config.provider == "api"
    assert config.model == "google/gemma-4-e4b-it"
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
            "GEMMA_PROVIDER": "api",
            "GEMMA_MODEL": "google/gemma-4-e4b-it",
            "HF_TOKEN": "hf_test_token",
        }
    )

    response_text = call_api("Return JSON only.", config)

    assert response_text == '{"document_type":"Employment Contract"}'
    assert captured["base_url"] == DEFAULT_HF_ROUTER_BASE_URL
    assert captured["api_key"] == "hf_test_token"
    assert captured["model"] == "google/gemma-4-e4b-it"
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


def test_call_ollama_splits_prompt_into_system_and_user_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemma 4 natively supports a `system` role. We should use it to carry
    persona and instructions, and keep the untrusted document text in the
    user turn, splitting on the BEGIN/END markers the prompt template uses."""
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "message": {
                    "role": "assistant",
                    "content": '{"document_type":"Employment Contract"}',
                }
            }

    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(gemma_client.requests, "post", fake_post)

    prompt = (
        "You are an analyzer. Output JSON only.\n"
        "=== DOCUMENT TEXT BEGINS ===\n"
        "Employee: Ravi\nSalary: Rs 18000\n"
        "=== DOCUMENT TEXT ENDS ===\n"
    )
    config = get_gemma_config({"MODEL_MODE": "real"})

    text = call_ollama(prompt, config)

    assert text == '{"document_type":"Employment Contract"}'
    assert captured["url"] == DEFAULT_OLLAMA_CHAT_URL
    payload = captured["payload"]
    assert payload["model"] == "gemma4:e4b"
    assert payload["stream"] is False
    assert payload["format"] == "json"
    assert payload["options"]["temperature"] == 1.0
    assert payload["options"]["top_p"] == 0.95
    assert payload["options"]["top_k"] == 64

    messages = payload["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are an analyzer.")
    assert "DOCUMENT TEXT BEGINS" not in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Employee: Ravi\nSalary: Rs 18000"


def test_call_ollama_falls_back_to_single_user_turn_without_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"role": "assistant", "content": "{}"}}

    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        fake_post.seen = json  # type: ignore[attr-defined]
        return FakeResponse()

    monkeypatch.setattr(gemma_client.requests, "post", fake_post)
    config = get_gemma_config({"MODEL_MODE": "real"})

    call_ollama("plain prompt with no marker", config)

    payload = fake_post.seen  # type: ignore[attr-defined]
    messages = payload["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "plain prompt with no marker"


def test_call_ollama_raises_when_response_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"role": "assistant", "content": ""}}

    monkeypatch.setattr(
        gemma_client.requests,
        "post",
        lambda url, json, timeout: FakeResponse(),
    )
    config = get_gemma_config({"MODEL_MODE": "real"})

    with pytest.raises(GemmaClientError, match="did not contain model text"):
        call_ollama("anything", config)


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



def test_call_ollama_attaches_images_to_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vision path: images must be forwarded as the top-level `images` array
    on the user message, which is the format Ollama's /api/chat expects."""
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "message": {
                    "role": "assistant",
                    "content": '{"document_type":"Employment Contract"}',
                }
            }

    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(gemma_client.requests, "post", fake_post)

    prompt = (
        "You are an analyzer. Output JSON only.\n"
        "=== DOCUMENT TEXT BEGINS ===\n"
        "Salary: Rs 18000\n"
        "=== DOCUMENT TEXT ENDS ===\n"
    )
    config = get_gemma_config({"MODEL_MODE": "real"})

    call_ollama(prompt, config, images=["image1b64==", "image2b64=="])

    messages = captured["payload"]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["images"] == ["image1b64==", "image2b64=="]


def test_call_ollama_without_images_omits_images_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No images -> no `images` field. Keeps the payload minimal and avoids
    sending empty arrays that a future Ollama version might reject."""

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"role": "assistant", "content": "{}"}}

    captured: dict = {}

    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(gemma_client.requests, "post", fake_post)
    config = get_gemma_config({"MODEL_MODE": "real"})

    call_ollama(
        "You are an analyzer.\n"
        "=== DOCUMENT TEXT BEGINS ===\ntext\n=== DOCUMENT TEXT ENDS ===\n",
        config,
    )

    messages = captured["payload"]["messages"]
    assert "images" not in messages[1]


def test_call_api_builds_openai_multimodal_parts_when_images_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For the HF Router / any OpenAI-compatible endpoint, images go in as
    `image_url` parts (data URL format) before the text part per Gemma 4's
    recommended modality order."""
    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="{}"))
                ]
            )

    class FakeOpenAI:
        def __init__(self, base_url: str, api_key: str):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(gemma_client, "OpenAI", FakeOpenAI)
    config = get_gemma_config(
        {
            "MODEL_MODE": "real",
            "GEMMA_PROVIDER": "api",
            "HF_TOKEN": "hf_test",
        }
    )

    call_api("prompt body", config, images=["abc==", "def=="])

    parts = captured["messages"][0]["content"]
    # Two image parts come first, text part comes last.
    assert len(parts) == 3
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"] == "data:image/png;base64,abc=="
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,def=="
    assert parts[2]["type"] == "text"
    assert parts[2]["text"] == "prompt body"


# ---------------------------------------------------------------------------
# Tool-calling loop
# ---------------------------------------------------------------------------


def _make_assistant_message(
    content: str = "",
    tool_calls: list[dict] | None = None,
) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def test_generate_with_tools_returns_final_text_and_empty_log_when_no_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the model answers directly without calling any tools, we should
    return the text and an empty tool_log."""
    recorded: list[dict] = []

    def fake_post(messages, config, tools=None, force_json=False):
        recorded.append({"messages": messages, "tools": tools, "force_json": force_json})
        return _make_assistant_message(content='{"ok": true}')

    monkeypatch.setattr(gemma_client, "_post_ollama_chat", fake_post)
    config = get_gemma_config({"MODEL_MODE": "real"})

    text, tool_log = gemma_client.generate_gemma_response_with_tools(
        prompt="system\n=== DOCUMENT TEXT BEGINS ===\nbody\n=== DOCUMENT TEXT ENDS ===\n",
        tools=[{"type": "function", "function": {"name": "noop"}}],
        execute_tool=lambda name, args: {"never": "called"},
        config=config,
    )

    assert text == '{"ok": true}'
    assert tool_log == []
    assert len(recorded) == 1
    assert recorded[0]["tools"] == [{"type": "function", "function": {"name": "noop"}}]


def test_generate_with_tools_runs_execute_tool_and_feeds_result_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool-calling happy path: model emits a tool call, we execute it, and
    the next turn returns the final answer. The tool_log must record the
    call, arguments, and result."""
    turn_counter = {"n": 0}
    captured_messages: list[list[dict]] = []

    def fake_post(messages, config, tools=None, force_json=False):
        captured_messages.append([dict(m) for m in messages])
        turn_counter["n"] += 1
        if turn_counter["n"] == 1:
            return _make_assistant_message(
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "index": 0,
                            "name": "lookup_labor_law",
                            "arguments": {"topic": "bond_clause"},
                        },
                    }
                ]
            )
        return _make_assistant_message(content='{"document_type": "Contract"}')

    monkeypatch.setattr(gemma_client, "_post_ollama_chat", fake_post)
    config = get_gemma_config({"MODEL_MODE": "real"})

    def execute(name: str, arguments: dict) -> list[dict]:
        assert name == "lookup_labor_law"
        assert arguments == {"topic": "bond_clause"}
        return [
            {
                "topic": "bond_clause",
                "title": "Bond enforceability",
                "summary": "Bonds enforceable only for actual loss.",
                "statute_reference": "ICA 27",
            }
        ]

    text, tool_log = gemma_client.generate_gemma_response_with_tools(
        prompt="system\n=== DOCUMENT TEXT BEGINS ===\nbody\n=== DOCUMENT TEXT ENDS ===\n",
        tools=[{"type": "function", "function": {"name": "lookup_labor_law"}}],
        execute_tool=execute,
        config=config,
    )

    assert text == '{"document_type": "Contract"}'
    assert len(tool_log) == 1
    assert tool_log[0]["name"] == "lookup_labor_law"
    assert tool_log[0]["arguments"] == {"topic": "bond_clause"}
    assert tool_log[0]["result"][0]["statute_reference"] == "ICA 27"

    # On the second turn, the message history must include the assistant
    # tool-call message and a role=tool message with the serialised result.
    second_turn_messages = captured_messages[1]
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second_turn_messages)
    tool_msg = next(m for m in second_turn_messages if m.get("role") == "tool")
    assert tool_msg["tool_name"] == "lookup_labor_law"
    assert "ICA 27" in tool_msg["content"]


def test_generate_with_tools_parses_string_arguments_as_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some runtimes deliver tool-call arguments as a JSON string. The loop
    should decode them before invoking execute_tool."""
    recorded: list[dict] = []
    turn = {"n": 0}

    def fake_post(messages, config, tools=None, force_json=False):
        turn["n"] += 1
        if turn["n"] == 1:
            return _make_assistant_message(
                tool_calls=[
                    {
                        "function": {
                            "name": "lookup",
                            "arguments": '{"topic": "notice_period"}',
                        }
                    }
                ]
            )
        return _make_assistant_message(content='{"done": true}')

    monkeypatch.setattr(gemma_client, "_post_ollama_chat", fake_post)
    config = get_gemma_config({"MODEL_MODE": "real"})

    def execute(name: str, arguments: dict) -> dict:
        recorded.append(arguments)
        return {"ok": True}

    gemma_client.generate_gemma_response_with_tools(
        prompt="system\n=== DOCUMENT TEXT BEGINS ===\nbody\n=== DOCUMENT TEXT ENDS ===\n",
        tools=[{"type": "function"}],
        execute_tool=execute,
        config=config,
    )

    assert recorded == [{"topic": "notice_period"}]


def test_generate_with_tools_captures_execute_tool_errors_as_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If execute_tool raises, the loop must convert the exception into an
    error dict and send it back to the model rather than propagating."""
    turn = {"n": 0}

    def fake_post(messages, config, tools=None, force_json=False):
        turn["n"] += 1
        if turn["n"] == 1:
            return _make_assistant_message(
                tool_calls=[
                    {"function": {"name": "boom", "arguments": {}}}
                ]
            )
        return _make_assistant_message(content='{"ok": true}')

    monkeypatch.setattr(gemma_client, "_post_ollama_chat", fake_post)
    config = get_gemma_config({"MODEL_MODE": "real"})

    def execute(name: str, arguments: dict):
        raise RuntimeError("simulated tool failure")

    _, tool_log = gemma_client.generate_gemma_response_with_tools(
        prompt="system\n=== DOCUMENT TEXT BEGINS ===\nbody\n=== DOCUMENT TEXT ENDS ===\n",
        tools=[{"type": "function"}],
        execute_tool=execute,
        config=config,
    )

    assert len(tool_log) == 1
    assert "simulated tool failure" in tool_log[0]["result"]["error"]


def test_generate_with_tools_refuses_non_ollama_provider() -> None:
    """The HF Router path isn't wired for tools yet; caller should get a
    clear error rather than a silent fallback."""
    config = get_gemma_config(
        {
            "MODEL_MODE": "real",
            "GEMMA_PROVIDER": "api",
            "HF_TOKEN": "hf_test",
        }
    )

    with pytest.raises(gemma_client.GemmaClientError, match="GEMMA_PROVIDER=ollama"):
        gemma_client.generate_gemma_response_with_tools(
            prompt="anything",
            tools=[],
            execute_tool=lambda n, a: None,
            config=config,
        )
