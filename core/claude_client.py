"""
Claude API client with prompt caching, vision support, and structured output via tool_use.
All LLM calls in the system route through this module.

Two model tiers:
  Sonnet — deep analysis (risk manager, research, positional)
  Haiku  — high-frequency lightweight tasks (conviction scoring, quick filters)
"""

import anthropic
from typing import Any
from core.config import settings

MAX_TOKENS = 4096


def get_client() -> anthropic.Anthropic | None:
    if not settings.ANTHROPIC_API_KEY:
        print("[Claude] ANTHROPIC_API_KEY not set — LLM calls will be skipped.")
        return None
    return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def call_structured(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_text: str,
    tool_name: str,
    tool_description: str,
    tool_schema: dict,
    image_base64: str | None = None,
    cache_system: bool = True,
    use_haiku: bool = False,
) -> dict | None:
    """
    Calls Claude with tool_use to force structured JSON output.
    Uses prompt caching on the system prompt (saves cost on repeated calls).
    Returns the tool input dict or None on failure.
    """
    model = settings.CLAUDE_HAIKU_MODEL if use_haiku else settings.CLAUDE_SONNET_MODEL

    system_block: list[dict] = [{"type": "text", "text": system_prompt}]
    if cache_system:
        system_block[0]["cache_control"] = {"type": "ephemeral"}

    content: list[dict] = []
    if image_base64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": image_base64},
        })
    content.append({"type": "text", "text": user_text})

    tools = [{
        "name": tool_name,
        "description": tool_description,
        "input_schema": tool_schema,
    }]

    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_block,
            tools=tools,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": content}],
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input
    except Exception as e:
        print(f"[Claude] Structured call failed ({model}): {e}")
    return None


def call_text(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_text: str,
    image_base64: str | None = None,
    cache_system: bool = True,
    use_haiku: bool = False,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """
    Calls Claude for a plain text response (narratives, summaries, research).
    """
    model = settings.CLAUDE_HAIKU_MODEL if use_haiku else settings.CLAUDE_SONNET_MODEL

    system_block: list[dict] = [{"type": "text", "text": system_prompt}]
    if cache_system:
        system_block[0]["cache_control"] = {"type": "ephemeral"}

    content: list[dict] = []
    if image_base64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": image_base64},
        })
    content.append({"type": "text", "text": user_text})

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_block,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text
    except Exception as e:
        print(f"[Claude] Text call failed ({model}): {e}")
    return ""
