"""
LangGraph analysis nodes — all LLM calls use Claude via core.claude_client.
Nodes run in parallel: technical_analyst + fundamental_analyst → risk_manager.
"""

import os
import base64
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.state import AgentState, RiskDecision
from tools.market_data import market_data_tool
from tools.fundamental_news import fundamental_news_tool, MacroContext
from db.memory import retrieve_similar_experiences
from core.config import settings
from core.claude_client import get_client, call_structured, call_text

# ── Optional Langfuse tracing ─────────────────────────────────────────────────
langfuse_handler = None
try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    if settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
        if settings.LANGFUSE_HOST:
            os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST
        langfuse_handler = LangfuseCallbackHandler()
        print(f"[Langfuse] Tracing active → {os.environ.get('LANGFUSE_HOST', 'cloud.langfuse.com')}")
except Exception as e:
    print(f"[Langfuse] Disabled: {e}")


def _load_prompt(filename: str) -> tuple[str, str]:
    """Loads a prompt file; splits into system / user parts at '---'."""
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = content.split("---", 1)
    return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ("", parts[0].strip())


def _encode_image(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Node 1: Technical Analyst ─────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def technical_analyst_node(state: AgentState) -> dict:
    symbol = state['symbol']
    technicals = market_data_tool.fetch_advanced_technicals(symbol)

    if "error" in technicals:
        return {
            "technical_analysis": technicals,
            "technical_narrative": f"Data error: {technicals['error']}",
            "messages": [f"[Technical] ERROR: {technicals['error']}"],
        }

    narrative = "[Technical] No AI analysis — ANTHROPIC_API_KEY not configured."
    client = get_client()
    if client:
        try:
            sys_prompt, user_template = _load_prompt("technical_analyst.txt")
            user_text = user_template.format(technicals=technicals)
            image_b64 = _encode_image(technicals.get("chart_path"))
            narrative = call_text(client, sys_prompt, user_text, image_base64=image_b64)
        except Exception as e:
            narrative = f"[Technical] Vision analysis failed: {e}"

    return {
        "technical_analysis": technicals,
        "technical_narrative": narrative,
        "messages": [f"[Technical] Completed for {symbol}. RSI={technicals.get('rsi_14')}, ADX={technicals.get('adx_14')}."],
    }


# ── Node 2: Fundamental + Macro Analyst ──────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fundamental_analyst_node(state: AgentState) -> dict:
    symbol = state['symbol']
    fundamentals = fundamental_news_tool.get_comparative_fundamentals(symbol)
    sentiment = fundamental_news_tool.get_micro_sentiment_score(symbol)
    macro_raw = fundamental_news_tool.get_macro_context()
    sector_perf = fundamental_news_tool.get_sector_performance()

    # Macro regime classification via Claude structured output
    macro: MacroContext = MacroContext(sentiment_enum="NEUTRAL", risk_multiplier=1.0, summary="Macro data unavailable.")
    client = get_client()
    if client:
        try:
            sys_prompt, user_template = _load_prompt("macro_analyst.txt")
            user_text = user_template.format(
                index_performance=macro_raw.get('index_performance', {}),
                headlines="\n".join(macro_raw.get('headlines', [])),
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
                        "risk_multiplier": {"type": "number", "description": "1.2 for BULLISH, 1.0 for NEUTRAL, 0.7 for BEARISH"},
                        "summary": {"type": "string", "description": "One-sentence analytical summary of the current regime"},
                    },
                    "required": ["sentiment_enum", "risk_multiplier", "summary"],
                },
            )
            if result:
                macro = MacroContext(**result)
                print(f"[Macro] Regime: {macro.sentiment_enum} (×{macro.risk_multiplier})")
        except Exception as e:
            print(f"[Macro] Classification failed: {e}")
            macro = MacroContext(sentiment_enum="NEUTRAL", risk_multiplier=1.0, summary=f"MACRO FAILURE: {e}")

    return {
        "fundamental_analysis": fundamentals,
        "sentiment_analysis": sentiment,
        "macro_context": macro,
        "sector_performance": sector_perf,
        "messages": [
            f"[Fundamental] {symbol}: PE={fundamentals.get('pe_ratio')}, ROE={fundamentals.get('roe')}, "
            f"Sentiment={sentiment.get('label', 'N/A')}. Macro={macro.sentiment_enum}."
        ],
    }


# ── Node 3: Risk Manager (Synthesis + Guardrails) ─────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def risk_manager_node(state: AgentState) -> dict:
    symbol = state['symbol']
    tech = state.get("technical_analysis") or {}
    fund = state.get("fundamental_analysis") or {}
    sent = state.get("sentiment_analysis") or {}
    macro = state.get("macro_context")
    narrative = state.get("technical_narrative", "")
    sector_perf = state.get("sector_performance") or {}

    # RL memory context
    rl_context = ""
    try:
        similar = retrieve_similar_experiences(
            query_text=f"{symbol} {str(fund)} {str(sent)}", n_results=2
        )
        if similar and similar.get('documents') and similar['documents'][0]:
            rl_context = "\n".join(similar['documents'][0])
    except Exception:
        pass

    # Fallback decision object for failure cases
    def _fail_decision(reason: str) -> RiskDecision:
        return RiskDecision(
            chain_of_thought_1_technicals="FAILED",
            chain_of_thought_2_fundamentals="FAILED",
            chain_of_thought_3_risk="FAILED",
            proposed_action="ERROR",
            proposed_entry=tech.get('latest_price', 0.0),
            proposed_stop_loss=0.0,
            proposed_take_profit=0.0,
            conviction_tier="LOW",
            win_probability_score=0,
            risk_percentage=0.0,
            expected_holding_days=0,
            final_rationale=reason,
        )

    decision: RiskDecision | None = None
    client = get_client()

    if client:
        try:
            sys_prompt, user_template = _load_prompt("risk_manager.txt")
            user_text = user_template.format(
                symbol=symbol,
                current_price=tech.get('latest_price'),
                atr_14=tech.get('atr_14'),
                rsi_14=tech.get('rsi_14'),
                macd_histogram=tech.get('macd_histogram'),
                adx_14=tech.get('adx_14'),
                stoch_k=tech.get('stoch_k'),
                bb_pct_b=tech.get('bb_pct_b'),
                ema_20=tech.get('ema_20'),
                ema_50=tech.get('ema_50'),
                weekly_trend=tech.get('weekly_trend'),
                support_levels=tech.get('support_levels'),
                resistance_levels=tech.get('resistance_levels'),
                technical_narrative=narrative[:600],
                pe_ratio=fund.get('pe_ratio'),
                roe=fund.get('roe'),
                roce=fund.get('roce', 'N/A'),
                debt_to_equity=fund.get('debt_to_equity'),
                promoter_holding=fund.get('promoter_holding', 'N/A'),
                sentiment_score=sent.get('score', 0),
                sentiment_label=sent.get('label', 'NEUTRAL'),
                sentiment_summary=sent.get('summary', ''),
                macro_env=macro.sentiment_enum if macro else 'NEUTRAL',
                macro_risk=macro.risk_multiplier if macro else 1.0,
                macro_summary=macro.summary if macro else '',
                sector_performance=sector_perf,
                rl_context=rl_context or "No similar past trades found.",
            )

            result = call_structured(
                client=client,
                system_prompt=sys_prompt,
                user_text=user_text,
                tool_name="submit_trade_decision",
                tool_description="Submit the final structured trade decision",
                tool_schema={
                    "type": "object",
                    "properties": {
                        "chain_of_thought_1_technicals": {"type": "string"},
                        "chain_of_thought_2_fundamentals": {"type": "string"},
                        "chain_of_thought_3_risk": {"type": "string"},
                        "proposed_action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                        "proposed_entry": {"type": "number"},
                        "proposed_stop_loss": {"type": "number"},
                        "proposed_take_profit": {"type": "number"},
                        "conviction_tier": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                        "win_probability_score": {"type": "integer", "minimum": 1, "maximum": 100},
                        "risk_percentage": {"type": "number"},
                        "expected_holding_days": {"type": "integer"},
                        "final_rationale": {"type": "string"},
                    },
                    "required": [
                        "chain_of_thought_1_technicals", "chain_of_thought_2_fundamentals",
                        "chain_of_thought_3_risk", "proposed_action", "proposed_entry",
                        "proposed_stop_loss", "proposed_take_profit", "conviction_tier",
                        "win_probability_score", "risk_percentage", "expected_holding_days",
                        "final_rationale",
                    ],
                },
            )

            if result:
                decision = RiskDecision(**result)
        except Exception as e:
            print(f"[Risk] Claude decision failed for {symbol}: {e}")
            decision = _fail_decision(f"Claude reasoning error: {e}")

    if not decision:
        decision = _fail_decision("No LLM client available or empty response.")

    # ── Python Hard Guardrails ──────────────────────────────────────────────────
    is_safe = True
    warnings: list[str] = []
    strategy_risk = settings.strategy.get("risk", {})
    max_abs_risk = strategy_risk.get("max_absolute_risk_limit", 0.08)
    min_sl_atr = strategy_risk.get("min_sl_atr_multiplier", 0.5)
    atr = tech.get('atr_14', 0)

    if decision.proposed_action == "BUY":
        if decision.proposed_stop_loss >= decision.proposed_entry:
            is_safe = False
            warnings.append("CRITICAL: Stop loss >= entry price (hallucination guard).")

        if atr > 0:
            min_sl = decision.proposed_entry - (atr * min_sl_atr)
            if decision.proposed_stop_loss > min_sl:
                is_safe = False
                warnings.append(
                    f"CRITICAL: SL {decision.proposed_stop_loss:.2f} too tight. "
                    f"Min safe SL = {min_sl:.2f} ({min_sl_atr}×ATR)."
                )

        rr = 0.0
        if decision.proposed_entry > decision.proposed_stop_loss > 0:
            risk_pts = decision.proposed_entry - decision.proposed_stop_loss
            reward_pts = decision.proposed_take_profit - decision.proposed_entry
            rr = reward_pts / risk_pts if risk_pts > 0 else 0
        if rr < 1.5:
            is_safe = False
            warnings.append(f"CRITICAL: R:R ratio {rr:.1f} below minimum 1.5.")

    # Conviction-based risk override (Python enforces, not LLM)
    risk_per_tier = strategy_risk.get("risk_per_tier", {"HIGH": 0.015, "MEDIUM": 0.010, "LOW": 0.005})
    if decision.proposed_action in ("BUY", "SELL"):
        tier = (decision.conviction_tier or "LOW").upper()
        decision.risk_percentage = risk_per_tier.get(tier, 0.005)

    if decision.risk_percentage > max_abs_risk:
        is_safe = False
        warnings.append(f"CRITICAL: Risk {decision.risk_percentage:.1%} exceeds absolute max {max_abs_risk:.1%}.")

    # Macro override — halve risk footprint in bearish conditions
    if macro and macro.sentiment_enum == "BEARISH" and decision.proposed_action == "BUY":
        decision.risk_percentage = round(decision.risk_percentage * macro.risk_multiplier, 4)
        warnings.append(f"WARNING: Bearish macro — risk reduced to {decision.risk_percentage:.1%}.")

    if not is_safe:
        decision.proposed_action = "ABORT_UNSAFE"

    return {
        "decision": decision,
        "is_safe_to_execute": is_safe,
        "guardrail_warnings": " | ".join(warnings),
        "rl_context": rl_context,
        "messages": [
            f"[Risk] {symbol}: Action={decision.proposed_action}, "
            f"Conviction={decision.conviction_tier}, Safe={is_safe}."
        ],
    }
