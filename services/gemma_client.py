from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from openai import OpenAI, OpenAIError
import requests


MODEL_MODE_MOCK = "mock"
MODEL_MODE_REAL = "real"
PROVIDER_OLLAMA = "ollama"
PROVIDER_API = "api"
DEFAULT_HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
# Local Ollama is the primary path for the Sahaayak "community booth" offline
# deployment story. gemma4:e4b is the 4B effective edge model with text+image+
# audio support and 128K context. Run it with `ollama pull gemma4:e4b`.
DEFAULT_GEMMA_MODEL = "gemma4:e4b"
DEFAULT_OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_HF_ROUTER_MODEL = "google/gemma-4-e4b-it"


class GemmaClientError(RuntimeError):
    """Raised when a Gemma provider call cannot return usable text."""


@dataclass(frozen=True)
class GemmaConfig:
    model_mode: str
    provider: str
    model: str
    ollama_url: str
    api_base_url: str
    api_key: str
    timeout_seconds: float


def get_gemma_config(env: Mapping[str, str] | None = None) -> GemmaConfig:
    values = env or os.environ

    model_mode = values.get("MODEL_MODE", MODEL_MODE_MOCK).strip().lower()
    if model_mode not in {MODEL_MODE_MOCK, MODEL_MODE_REAL}:
        model_mode = MODEL_MODE_MOCK

    # Default to local Ollama. The "community booth" story (NGO laptop running
    # Gemma 4 offline) is the pitch, so cloud is opt-in, not the default.
    provider = values.get("GEMMA_PROVIDER", PROVIDER_OLLAMA).strip().lower()
    if provider not in {PROVIDER_OLLAMA, PROVIDER_API}:
        provider = PROVIDER_OLLAMA

    timeout_seconds = parse_timeout(values.get("GEMMA_TIMEOUT_SECONDS", "120"))

    # Pick a provider-appropriate default model ID when the user didn't set
    # GEMMA_MODEL explicitly. Ollama uses tag syntax (`gemma4:e4b`) while the
    # HF Router expects a repo id (`google/gemma-4-e4b-it`).
    explicit_model = (values.get("GEMMA_MODEL") or "").strip()
    if explicit_model:
        model = explicit_model
    elif provider == PROVIDER_API:
        model = DEFAULT_HF_ROUTER_MODEL
    else:
        model = DEFAULT_GEMMA_MODEL

    return GemmaConfig(
        model_mode=model_mode,
        provider=provider,
        model=model,
        ollama_url=values.get("GEMMA_OLLAMA_URL", DEFAULT_OLLAMA_CHAT_URL).strip(),
        api_base_url=(
            values.get("GEMMA_API_BASE_URL")
            or values.get("GEMMA_API_URL")
            or DEFAULT_HF_ROUTER_BASE_URL
        ).strip(),
        api_key=(values.get("GEMMA_API_KEY") or values.get("HF_TOKEN") or "").strip(),
        timeout_seconds=timeout_seconds,
    )


def parse_timeout(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        return 120.0

    return parsed if parsed > 0 else 120.0


def describe_model_mode(config: GemmaConfig | None = None) -> str:
    active_config = config or get_gemma_config()
    if active_config.model_mode == MODEL_MODE_REAL:
        return f"real ({active_config.provider}: {active_config.model})"
    return "mock"


def generate_gemma_response(
    prompt: str,
    config: GemmaConfig | None = None,
    images: list[str] | None = None,
) -> str:
    """Generate a Gemma 4 response.

    Args:
        prompt: Rendered prompt text.
        config: Gemma config; pulled from environment if omitted.
        images: Optional list of base64-encoded PNG/JPG images (no data URL
            prefix). When provided, they are attached to the user message so
            Gemma 4 can read the document visually instead of (or alongside)
            its extracted text.
    """
    active_config = config or get_gemma_config()

    if active_config.model_mode != MODEL_MODE_REAL:
        raise GemmaClientError("Gemma client can only be called when MODEL_MODE=real.")

    if active_config.provider == PROVIDER_OLLAMA:
        return call_ollama(prompt, active_config, images=images)

    if active_config.provider == PROVIDER_API:
        return call_api(prompt, active_config, images=images)

    raise GemmaClientError(f"Unsupported Gemma provider: {active_config.provider}")


def call_ollama(
    prompt: str,
    config: GemmaConfig,
    images: list[str] | None = None,
) -> str:
    """Call a local Ollama server running Gemma 4.

    Uses the /api/chat endpoint so we can exploit Gemma 4's native `system`
    role (recommended by the model card) and attach images directly to the
    user turn via Ollama's ``images`` field. Sampling params follow the
    official Gemma 4 recommendation: temperature=1.0, top_p=0.95, top_k=64.
    """
    messages = _build_messages(prompt, images=images)

    payload = {
        "model": config.model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            # Give the model enough room for the full structured analysis,
            # including long source_text quotes and multi-paragraph risk
            # explanations for complex contracts. 4096 tokens is ~12KB of
            # text, comfortably more than any offer letter needs.
            "num_predict": 4096,
        },
    }

    try:
        response = requests.post(
            config.ollama_url,
            json=payload,
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise GemmaClientError(f"Ollama request failed: {exc}") from exc
    except ValueError as exc:
        raise GemmaClientError("Ollama returned a non-JSON HTTP response.") from exc

    text = _extract_ollama_text(data)
    if not text:
        raise GemmaClientError("Ollama response did not contain model text.")

    return text


_DOC_BEGIN_MARKER = "=== DOCUMENT TEXT BEGINS ==="
_DOC_END_MARKER = "=== DOCUMENT TEXT ENDS ==="


def _build_messages(
    prompt: str,
    images: list[str] | None = None,
) -> list[dict[str, object]]:
    """Split the rendered prompt into system + user messages and attach images.

    The prompt template frames persona, output contract, classification and
    risk rules, and prompt-injection safety as *instructions*, and wraps the
    actual document text in explicit BEGIN/END markers. We put the
    instruction block in `system` and the document body in `user` so Gemma 4
    treats the document as data to analyze, not as further instructions.

    When ``images`` is provided, Ollama expects them as a top-level
    ``images`` array on the user message (base64-encoded, no data URL
    prefix). Gemma 4's model card recommends placing image content before
    text in the prompt; Ollama handles that ordering internally, so we just
    include the field alongside ``content``.

    If the markers are missing for any reason (custom prompt, legacy tests)
    we fall back to a single user turn so the call still succeeds.
    """
    begin_idx = prompt.find(_DOC_BEGIN_MARKER)
    end_idx = (
        prompt.find(_DOC_END_MARKER, begin_idx + len(_DOC_BEGIN_MARKER))
        if begin_idx != -1
        else -1
    )

    if begin_idx != -1 and end_idx != -1:
        system_content = prompt[:begin_idx].strip()
        document_content = prompt[begin_idx + len(_DOC_BEGIN_MARKER):end_idx].strip()
        if system_content and document_content:
            user_message: dict[str, object] = {
                "role": "user",
                "content": document_content,
            }
            if images:
                user_message["images"] = list(images)
            return [
                {"role": "system", "content": system_content},
                user_message,
            ]

    fallback_message: dict[str, object] = {"role": "user", "content": prompt}
    if images:
        fallback_message["images"] = list(images)
    return [fallback_message]


def _extract_ollama_text(data: object) -> str:
    """Extract the assistant text from an Ollama /api/chat or /api/generate response."""
    if not isinstance(data, dict):
        return ""

    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    generate_text = data.get("response")
    if isinstance(generate_text, str) and generate_text.strip():
        return generate_text.strip()

    return ""


def call_api(
    prompt: str,
    config: GemmaConfig,
    images: list[str] | None = None,
) -> str:
    if not config.api_base_url:
        raise GemmaClientError("GEMMA_API_BASE_URL is required when GEMMA_PROVIDER=api.")
    if not config.api_key:
        raise GemmaClientError("HF_TOKEN or GEMMA_API_KEY is required when GEMMA_PROVIDER=api.")

    try:
        client = OpenAI(
            base_url=config.api_base_url,
            api_key=config.api_key,
        )

        completion = client.chat.completions.create(
            model=config.model,
            messages=[
                {
                    "role": "user",
                    "content": _build_openai_content_parts(prompt, images),
                }
            ],
            timeout=config.timeout_seconds,
        )
    except OpenAIError as exc:
        raise GemmaClientError(f"Gemma API request failed: {exc}") from exc
    except Exception as exc:
        raise GemmaClientError(f"Gemma API request failed: {exc}") from exc

    text = extract_text_from_openai_completion(completion)
    if not text:
        raise GemmaClientError("Gemma API response did not contain model text.")

    return text


def _build_openai_content_parts(
    prompt: str,
    images: list[str] | None,
) -> list[dict[str, object]]:
    """Build OpenAI-compatible multimodal content parts.

    Gemma 4's model card recommends placing image content before the text in
    the prompt, so images come first. Each base64 image is wrapped as a data
    URL since the HF Router expects OpenAI-style ``image_url`` parts.
    """
    parts: list[dict[str, object]] = []
    for image_b64 in images or []:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )
    parts.append({"type": "text", "text": prompt})
    return parts


def extract_text_from_openai_completion(completion: object) -> str:
    choices = getattr(completion, "choices", None)
    if not choices:
        return ""

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text")
            else:
                value = getattr(item, "text", None)

            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())

        return "\n".join(text_parts).strip()

    return ""


def extract_text_from_api_payload(data: object) -> str:
    if isinstance(data, str):
        return data.strip()

    if not isinstance(data, dict):
        return ""

    for key in ("response", "output", "text", "content"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()

            text = first_choice.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()

    return ""


# ---------------------------------------------------------------------------
# Tool-calling support
# ---------------------------------------------------------------------------
# Ollama and Gemma 4 together support function calling (also known as tool
# use). The loop looks like:
#
#   1. Send prompt + tools to /api/chat.
#   2. If the model returns ``tool_calls``, execute each tool locally and
#      append a ``role: tool`` message with the JSON-encoded result.
#   3. Send the updated history back to /api/chat. Repeat until the model
#      returns plain text (no more tool calls) or we hit the safety cap.
#   4. Return the final text, which we still require to be strict JSON.
#
# Ollama does not allow combining ``format: "json"`` with tool calls on
# intermediate turns (the model needs to be free to emit structured tool
# calls), so we only set ``format: "json"`` on the final follow-up turn
# when we know no more tools will be called.


MAX_TOOL_ROUNDS = 3


def generate_gemma_response_with_tools(
    prompt: str,
    tools: list[dict],
    execute_tool: "Callable[[str, dict], object]",
    config: GemmaConfig | None = None,
    images: list[str] | None = None,
    max_rounds: int = MAX_TOOL_ROUNDS,
) -> tuple[str, list[dict]]:
    """Run a multi-turn tool-calling loop against Gemma 4.

    Args:
        prompt: The full prompt (system + document body) rendered from the
            template. Same format as the text-only path.
        tools: JSON schema list in Ollama's tool format.
        execute_tool: Callable ``fn(name: str, arguments: dict) -> object``
            that runs a tool and returns a JSON-serializable result. The
            caller is responsible for wrapping errors into a dict like
            ``{"error": "..."}`` so the model can recover.
        config: Gemma config.
        images: Optional base64 images to attach to the first user turn.
        max_rounds: Safety cap on the number of tool-call rounds.

    Returns:
        A tuple ``(final_text, tool_log)`` where ``tool_log`` is the ordered
        list of calls and results in shape ``{"name": str, "arguments":
        dict, "result": object}``. The final text is the model's reply
        after all tools have been exhausted.

    Raises:
        GemmaClientError: If the provider is not Ollama (API/HF path not
            yet wired for tools), if the server errors, or if the model
            exhausts ``max_rounds`` without producing a final answer.
    """
    active_config = config or get_gemma_config()

    if active_config.model_mode != MODEL_MODE_REAL:
        raise GemmaClientError(
            "Gemma client can only be called with tools when MODEL_MODE=real."
        )

    if active_config.provider != PROVIDER_OLLAMA:
        # For now we wire function calling only for the local Ollama path
        # because that is the offline "community booth" story we are
        # pitching. The HF Router supports tools too, but that is a future
        # enhancement and not required for Day 5-6 of the plan.
        raise GemmaClientError(
            "Tool calling is currently supported only with GEMMA_PROVIDER=ollama."
        )

    messages = _build_messages(prompt, images=images)
    tool_log: list[dict] = []

    for round_idx in range(max_rounds):
        is_final_round = round_idx == max_rounds - 1
        response_message = _post_ollama_chat(
            messages,
            active_config,
            tools=tools,
            # We cannot constrain to JSON while tool calls may still be
            # emitted. Only on the final round (or when no tools are
            # called) do we ask for strict JSON.
            force_json=False,
        )

        tool_calls = response_message.get("tool_calls") or []

        if not tool_calls:
            text = _extract_text_from_chat_message(response_message)
            if text:
                return text, tool_log
            # Model gave us neither text nor tool calls -- that's a
            # provider bug; force one more round with strict JSON to
            # nudge it into producing the expected shape.
            if is_final_round:
                raise GemmaClientError(
                    "Model returned neither tool calls nor text."
                )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Please now produce the final strict-JSON analysis "
                        "based on everything above. Do not call any more "
                        "tools."
                    ),
                }
            )
            continue

        # Record the assistant turn with its tool calls so the model can
        # see its own previous reasoning on the next pass.
        messages.append(
            {
                "role": "assistant",
                "content": response_message.get("content", "") or "",
                "tool_calls": tool_calls,
            }
        )

        for call in tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            arguments = fn.get("arguments") or {}
            if isinstance(arguments, str):
                # Some runtimes serialize arguments as a JSON string;
                # normalise to dict.
                import json as _json

                try:
                    arguments = _json.loads(arguments)
                except _json.JSONDecodeError:
                    arguments = {}

            try:
                result = execute_tool(name, arguments)
            except Exception as exc:  # surface errors as tool output
                result = {"error": f"Tool {name!r} raised: {exc}"}

            tool_log.append(
                {"name": name, "arguments": arguments, "result": result}
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": _serialize_tool_result(result),
                }
            )

    # Exhausted the loop without getting a final text answer. One last
    # shot: ask explicitly for the strict-JSON final analysis with no
    # tools available.
    final_message = _post_ollama_chat(
        messages
        + [
            {
                "role": "user",
                "content": (
                    "You have consulted the available tools. Now produce "
                    "the final strict-JSON analysis exactly in the schema "
                    "described in the system prompt. Do not call any more "
                    "tools."
                ),
            }
        ],
        active_config,
        tools=None,
        force_json=True,
    )
    text = _extract_text_from_chat_message(final_message)
    if not text:
        raise GemmaClientError(
            "Tool-calling loop exhausted without a final JSON answer."
        )
    return text, tool_log


def _post_ollama_chat(
    messages: list[dict],
    config: GemmaConfig,
    tools: list[dict] | None = None,
    force_json: bool = False,
) -> dict:
    payload: dict[str, object] = {
        "model": config.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            # See call_ollama for the reasoning behind 4096.
            "num_predict": 4096,
        },
    }
    if tools:
        payload["tools"] = tools
    if force_json:
        payload["format"] = "json"

    try:
        response = requests.post(
            config.ollama_url,
            json=payload,
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise GemmaClientError(f"Ollama request failed: {exc}") from exc
    except ValueError as exc:
        raise GemmaClientError("Ollama returned a non-JSON HTTP response.") from exc

    message = data.get("message")
    if not isinstance(message, dict):
        raise GemmaClientError("Ollama response did not contain a message object.")
    return message


def _extract_text_from_chat_message(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return ""


def _serialize_tool_result(result: object) -> str:
    import json as _json

    try:
        return _json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return _json.dumps({"error": "result was not JSON serialisable"})


# Declared at module bottom so the type hint on ``execute_tool`` resolves
# without forcing a top-of-file import.
from typing import Callable  # noqa: E402
