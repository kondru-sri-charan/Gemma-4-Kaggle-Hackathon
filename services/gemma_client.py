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
DEFAULT_GEMMA_MODEL = "google/gemma-4-31B-it:novita"


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

    provider = values.get("GEMMA_PROVIDER", PROVIDER_API).strip().lower()
    if provider not in {PROVIDER_OLLAMA, PROVIDER_API}:
        provider = PROVIDER_API

    timeout_seconds = parse_timeout(values.get("GEMMA_TIMEOUT_SECONDS", "60"))

    return GemmaConfig(
        model_mode=model_mode,
        provider=provider,
        model=values.get("GEMMA_MODEL", DEFAULT_GEMMA_MODEL).strip() or DEFAULT_GEMMA_MODEL,
        ollama_url=values.get("GEMMA_OLLAMA_URL", "http://localhost:11434/api/generate").strip(),
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
        return 60.0

    return parsed if parsed > 0 else 60.0


def describe_model_mode(config: GemmaConfig | None = None) -> str:
    active_config = config or get_gemma_config()
    if active_config.model_mode == MODEL_MODE_REAL:
        return f"real ({active_config.provider}: {active_config.model})"
    return "mock"


def generate_gemma_response(prompt: str, config: GemmaConfig | None = None) -> str:
    active_config = config or get_gemma_config()

    if active_config.model_mode != MODEL_MODE_REAL:
        raise GemmaClientError("Gemma client can only be called when MODEL_MODE=real.")

    if active_config.provider == PROVIDER_OLLAMA:
        return call_ollama(prompt, active_config)

    if active_config.provider == PROVIDER_API:
        return call_api(prompt, active_config)

    raise GemmaClientError(f"Unsupported Gemma provider: {active_config.provider}")


def call_ollama(prompt: str, config: GemmaConfig) -> str:
    payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
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

    text = data.get("response") or data.get("message", {}).get("content")
    if not isinstance(text, str) or not text.strip():
        raise GemmaClientError("Ollama response did not contain model text.")

    return text


def call_api(prompt: str, config: GemmaConfig) -> str:
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
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
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
