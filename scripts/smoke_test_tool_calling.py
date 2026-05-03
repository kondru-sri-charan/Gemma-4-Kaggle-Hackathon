"""Smoke test: does Gemma 4 E4B on Ollama actually emit tool_calls?

Before we invest in a multi-turn labor-law grounding loop, we need to
confirm the local runtime supports Gemma 4 function calling.

We give the model a single, obvious reason to call a tool ("look up the
notice period rules") and see whether the response contains a
``tool_calls`` array or whether Gemma just answers from memory.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "lookup_labor_law",
        "description": (
            "Look up Indian labor-law rules on a specific topic. Call this any "
            "time you need authoritative guidance on a worker-rights topic "
            "(notice period, bond clauses, minimum wage, PF/ESI rules, "
            "overtime compensation) instead of answering from memory."
        ),
        "parameters": {
            "type": "object",
            "required": ["topic"],
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "The topic to look up. Examples: 'notice_period', "
                        "'bond_clause', 'minimum_wage', 'overtime', "
                        "'pf_deductions', 'esi_deductions'."
                    ),
                },
                "state": {
                    "type": "string",
                    "description": (
                        "Optional Indian state name for state-specific rules, "
                        "e.g. 'Karnataka', 'Maharashtra'. Omit for national "
                        "rules."
                    ),
                },
            },
        },
    },
}


PROMPT = (
    "A worker in Karnataka has been asked to sign an offer letter that requires "
    "60 days notice if they resign, but only 15 days notice from the employer. "
    "What does Indian labor law actually say about notice period requirements? "
    "Use the lookup_labor_law tool to get authoritative information before answering."
)


def main() -> int:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": PROMPT},
        ],
        "stream": False,
        "tools": [TOOL_SPEC],
        "options": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        },
    }

    print(f"Testing tool calling with {MODEL} ...")
    started = time.monotonic()
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    elapsed = time.monotonic() - started
    print(f"HTTP {response.status_code} in {elapsed:.1f}s\n")

    if response.status_code != 200:
        print(response.text[:500])
        return 1

    data = response.json()
    message = data.get("message", {})
    content = message.get("content", "")
    tool_calls = message.get("tool_calls", [])

    print("=== Response content ===")
    print(content if content else "(empty)")
    print()
    print("=== Tool calls ===")
    if tool_calls:
        print(json.dumps(tool_calls, indent=2))
        print(f"\n✓ Model emitted {len(tool_calls)} tool call(s). Tool loop is viable.")
        return 0

    print("(no tool calls)")
    print("\n⚠ Model did not call the tool. Gemma 4 E4B on Ollama may not support")
    print("  function calling via the /api/chat `tools` field, or the prompt")
    print("  wasn't persuasive enough. Next step: check raw stop tokens or")
    print("  simplify the tool description.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
