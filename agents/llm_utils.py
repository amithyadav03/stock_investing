"""
Shared LLM utilities used across agents.
Prevents duplication between nodes.py and exit_monitor.py.
"""

import os
from tools.fundamental_news import MacroContext
from core.claude_client import get_client, call_structured


def classify_macro(macro_raw: dict) -> MacroContext:
    """
    Classifies the current macro regime using Claude.
    Returns MacroContext with BULLISH/NEUTRAL/BEARISH + risk_multiplier.
    Falls back to NEUTRAL on any failure.
    """
    client = get_client()
    if not client:
        return MacroContext(sentiment_enum="NEUTRAL", risk_multiplier=1.0, summary="No LLM client.")

    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "macro_analyst.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()
        parts = content.split("---", 1)
        sys_prompt = parts[0].strip()
        user_template = parts[1].strip() if len(parts) > 1 else parts[0].strip()

        user_text = user_template.format(
            index_performance=macro_raw.get("index_performance", {}),
            headlines="\n".join(macro_raw.get("headlines", [])),
        )

        result = call_structured(
            client=client,
            system_prompt=sys_prompt,
            user_text=user_text,
            tool_name="submit_macro_context",
            tool_description="Submit the macro market regime classification",
            tool_schema={
                "type": "object",
                "properties": {
                    "sentiment_enum": {"type": "string", "enum": ["BULLISH", "NEUTRAL", "BEARISH"]},
                    "risk_multiplier": {"type": "number"},
                    "summary": {"type": "string"},
                },
                "required": ["sentiment_enum", "risk_multiplier", "summary"],
            },
        )
        if result:
            macro = MacroContext(**result)
            print(f"[Macro] Regime: {macro.sentiment_enum} (×{macro.risk_multiplier})")
            return macro
    except Exception as e:
        print(f"[Macro] Classification failed: {e}")

    return MacroContext(
        sentiment_enum="NEUTRAL",
        risk_multiplier=1.0,
        summary=f"Macro classification failed — defaulting to NEUTRAL.",
    )


def load_prompt(filename: str) -> tuple[str, str]:
    """Loads a prompt file from the prompts/ directory; splits at '---'."""
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = content.split("---", 1)
    return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ("", parts[0].strip())
