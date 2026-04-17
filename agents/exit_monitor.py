"""
Exit monitoring workflow.
Evaluates active positions to decide: HOLD, TRAIL_SL, or EXIT_NOW.
Called periodically by scripts/monitor_positions.py during market hours.
"""

import os
from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

from tools.market_data import market_data_tool
from tools.fundamental_news import fundamental_news_tool
from core.claude_client import get_client, call_structured
from core.config import settings


class ExitDecision(BaseModel):
    action: str = Field(description="'HOLD', 'TRAIL_SL', or 'EXIT_NOW'.")
    urgency: str = Field(description="'NORMAL' or 'URGENT' — urgent triggers immediate Telegram alert.")
    new_stop_loss: Optional[float] = Field(default=None, description="New trailing stop loss price (only for TRAIL_SL action).")
    rationale: str = Field(description="Clear markdown explanation of why this exit/hold decision was made.")
    exit_reason: Optional[str] = Field(default=None, description="For EXIT_NOW: 'STOP_LOSS', 'TAKE_PROFIT', 'THESIS_INVALIDATED', 'MACRO_REVERSAL', 'FORCED_EXIT'.")


def evaluate_exit(
    proposal_id: int,
    symbol: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    quantity: int,
    entry_rationale: str,
    entry_time: datetime,
) -> ExitDecision:
    """
    Full exit evaluation for a single active position.
    Returns an ExitDecision object.
    """
    # Fetch current market data
    technicals = market_data_tool.fetch_advanced_technicals(symbol)
    if "error" in technicals:
        return ExitDecision(
            action="HOLD",
            urgency="NORMAL",
            rationale=f"Could not fetch data for {symbol}: {technicals['error']}",
        )

    current_price = technicals.get("latest_price", 0.0)
    if current_price == 0.0:
        current_price = market_data_tool.get_current_price(symbol)

    days_held = (datetime.now() - entry_time).days if entry_time else 0
    pnl_pct = round(((current_price - entry_price) / entry_price) * 100, 2) if direction == "BUY" else round(((entry_price - current_price) / entry_price) * 100, 2)

    # Hard-code exits: price already hit SL or TP
    if direction == "BUY":
        if current_price <= stop_loss:
            return ExitDecision(action="EXIT_NOW", urgency="URGENT", rationale=f"Stop loss hit. Price ₹{current_price} ≤ SL ₹{stop_loss}.", exit_reason="STOP_LOSS")
        if current_price >= take_profit:
            return ExitDecision(action="EXIT_NOW", urgency="URGENT", rationale=f"Take profit hit. Price ₹{current_price} ≥ TP ₹{take_profit}.", exit_reason="TAKE_PROFIT")
    elif direction == "SELL":
        if current_price >= stop_loss:
            return ExitDecision(action="EXIT_NOW", urgency="URGENT", rationale=f"Stop loss hit. Price ₹{current_price} ≥ SL ₹{stop_loss}.", exit_reason="STOP_LOSS")
        if current_price <= take_profit:
            return ExitDecision(action="EXIT_NOW", urgency="URGENT", rationale=f"Take profit hit. Price ₹{current_price} ≤ TP ₹{take_profit}.", exit_reason="TAKE_PROFIT")

    # Macro context for exit decision
    macro_raw = fundamental_news_tool.get_macro_context()
    macro_regime = "NEUTRAL"
    macro_summary = ""
    client = get_client()
    if client:
        try:
            from core.claude_client import call_structured as cs
            from prompts import _load_prompt  # not available — inline macro below
        except Exception:
            pass
        try:
            result = call_structured(
                client=client,
                system_prompt="You are a macro regime classifier. Classify market regime from index performance and headlines.",
                user_text=f"Index performance: {macro_raw.get('index_performance')}\nHeadlines:\n" + "\n".join(macro_raw.get('headlines', [])[:5]),
                tool_name="classify_macro",
                tool_description="Classify the current macro regime",
                tool_schema={
                    "type": "object",
                    "properties": {
                        "regime": {"type": "string", "enum": ["BULLISH", "NEUTRAL", "BEARISH"]},
                        "summary": {"type": "string"},
                    },
                    "required": ["regime", "summary"],
                },
                cache_system=True,
            )
            if result:
                macro_regime = result.get("regime", "NEUTRAL")
                macro_summary = result.get("summary", "")
        except Exception as e:
            print(f"[ExitMonitor] Macro classification failed: {e}")

    # AI exit analysis
    exit_decision = ExitDecision(action="HOLD", urgency="NORMAL", rationale="No AI analysis available.")
    if client:
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "exit_analyst.txt")
            with open(prompt_path, "r") as f:
                content = f.read()
            parts = content.split("---", 1)
            sys_prompt = parts[0].strip()
            user_template = parts[1].strip() if len(parts) > 1 else parts[0].strip()

            user_text = user_template.format(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                current_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                days_held=days_held,
                pnl_pct=pnl_pct,
                atr_14=technicals.get("atr_14", "N/A"),
                rsi_14=technicals.get("rsi_14", "N/A"),
                macd_histogram=technicals.get("macd_histogram", "N/A"),
                adx_14=technicals.get("adx_14", "N/A"),
                weekly_trend=technicals.get("weekly_trend", "N/A"),
                ema_20=technicals.get("ema_20", "N/A"),
                support_levels=technicals.get("support_levels", []),
                resistance_levels=technicals.get("resistance_levels", []),
                entry_rationale=entry_rationale[:400],
                macro_regime=macro_regime,
                macro_summary=macro_summary,
            )

            result = call_structured(
                client=client,
                system_prompt=sys_prompt,
                user_text=user_text,
                tool_name="submit_exit_decision",
                tool_description="Submit the exit/hold decision for this active position",
                tool_schema={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["HOLD", "TRAIL_SL", "EXIT_NOW"]},
                        "urgency": {"type": "string", "enum": ["NORMAL", "URGENT"]},
                        "new_stop_loss": {"type": "number"},
                        "rationale": {"type": "string"},
                        "exit_reason": {
                            "type": "string",
                            "enum": ["STOP_LOSS", "TAKE_PROFIT", "THESIS_INVALIDATED", "MACRO_REVERSAL", "FORCED_EXIT"],
                        },
                    },
                    "required": ["action", "urgency", "rationale"],
                },
            )

            if result:
                exit_decision = ExitDecision(
                    action=result.get("action", "HOLD"),
                    urgency=result.get("urgency", "NORMAL"),
                    new_stop_loss=result.get("new_stop_loss"),
                    rationale=result.get("rationale", ""),
                    exit_reason=result.get("exit_reason"),
                )
        except Exception as e:
            print(f"[ExitMonitor] AI exit analysis failed for {symbol}: {e}")
            exit_decision = ExitDecision(action="HOLD", urgency="NORMAL", rationale=f"AI analysis error: {e}")

    print(f"[ExitMonitor] {symbol}: P&L={pnl_pct}%, Days={days_held} → {exit_decision.action} ({exit_decision.urgency})")
    return exit_decision
