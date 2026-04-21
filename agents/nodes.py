"""
LangGraph analysis nodes — all LLM calls use Claude via core.claude_client.
Nodes run in parallel: technical_analyst + fundamental_analyst → conviction_filter → risk_manager.
Supports strategy_type: "swing" | "positional" | "value"
"""

import os
import base64
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.state import AgentState, RiskDecision
from agents.llm_utils import classify_macro, load_prompt as _load_prompt
from tools.market_data import market_data_tool
from tools.fundamental_news import fundamental_news_tool, MacroContext
from db.memory import retrieve_similar_experiences
from core.config import settings
from core.claude_client import get_client, call_structured, call_text

langfuse_handler = None
try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    if settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
        if settings.LANGFUSE_HOST:
            os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST
        langfuse_handler = LangfuseCallbackHandler()
        print(f"[Langfuse] Tracing active -> {os.environ.get('LANGFUSE_HOST', 'cloud.langfuse.com')}")
except Exception as e:
    print(f"[Langfuse] Disabled: {e}")


def _encode_image(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _strategy_prompt_file(strategy_type: str) -> str:
    return "positional_analyst.txt" if strategy_type in ("positional", "value") else "technical_analyst.txt"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def technical_analyst_node(state: AgentState) -> dict:
    symbol = state['symbol']
    strategy_type = state.get('strategy_type', 'swing')

    technicals = market_data_tool.fetch_advanced_technicals(symbol)
    if "error" in technicals:
        return {
            "technical_analysis": technicals,
            "technical_narrative": f"Data error: {technicals['error']}",
            "weekly_data": {}, "monthly_data": {}, "timeframe_confluence": 0,
            "messages": [f"[Technical] ERROR: {technicals['error']}"],
        }

    weekly_data = market_data_tool.fetch_weekly_data(symbol)
    monthly_data = {}
    confluence = 0

    if strategy_type in ("positional", "value"):
        monthly_data = market_data_tool.fetch_monthly_data(symbol)
        if technicals.get("weekly_trend") == "UP": confluence += 1
        if weekly_data.get("weekly_structure") in ("UP", "STRONG_UP"): confluence += 1
        if monthly_data.get("monthly_trend") == "UP": confluence += 1
    else:
        if technicals.get("weekly_trend") == "UP": confluence += 1
        if weekly_data.get("weekly_structure") in ("UP", "STRONG_UP"): confluence += 1

    narrative = "[Technical] No AI analysis -- ANTHROPIC_API_KEY not configured."
    client = get_client()
    if client:
        try:
            sys_prompt, user_template = _load_prompt(_strategy_prompt_file(strategy_type))
            user_text = user_template.format(
                technicals=technicals,
                weekly_data=weekly_data,
                monthly_data=monthly_data,
                timeframe_confluence=confluence,
                strategy_type=strategy_type,
            )
            image_b64 = _encode_image(technicals.get("chart_path"))
            narrative = call_text(client, sys_prompt, user_text, image_base64=image_b64)
        except Exception as e:
            narrative = f"[Technical] Analysis failed: {e}"

    return {
        "technical_analysis": technicals,
        "technical_narrative": narrative,
        "weekly_data": weekly_data,
        "monthly_data": monthly_data,
        "timeframe_confluence": confluence,
        "messages": [
            f"[Technical] {symbol} ({strategy_type}): RSI={technicals.get('rsi_14')}, "
            f"ADX={technicals.get('adx_14')}, Confluence={confluence}."
        ],
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fundamental_analyst_node(state: AgentState) -> dict:
    symbol = state['symbol']
    strategy_type = state.get('strategy_type', 'swing')

    fundamentals = fundamental_news_tool.get_comparative_fundamentals(symbol)
    sentiment = fundamental_news_tool.get_micro_sentiment_score(symbol)
    macro_raw = fundamental_news_tool.get_macro_context()
    sector_perf = fundamental_news_tool.get_sector_performance()
    macro: MacroContext = classify_macro(macro_raw)

    research_dict = {}
    if strategy_type in ("positional", "value"):
        try:
            from agents.research_agent import run_deep_research
            report = run_deep_research(symbol, strategy_type)
            research_dict = {
                "research_score": report.research_score,
                "recommendation": report.recommendation,
                "business_summary": report.business_summary,
                "upcoming_catalysts": report.upcoming_catalysts,
                "key_risks": report.key_risks,
            }
        except Exception as e:
            print(f"[Fundamental] Research agent failed for {symbol}: {e}")

    return {
        "fundamental_analysis": fundamentals,
        "sentiment_analysis": sentiment,
        "macro_context": macro,
        "sector_performance": sector_perf,
        "research_report": research_dict,
        "messages": [
            f"[Fundamental] {symbol}: PE={fundamentals.get('pe_ratio')}, ROE={fundamentals.get('roe')}, "
            f"Sentiment={sentiment.get('label', 'N/A')}. Macro={macro.sentiment_enum}."
        ],
    }


def conviction_filter_node(state: AgentState) -> dict:
    symbol = state['symbol']
    strategy_type = state.get('strategy_type', 'swing')
    tech = state.get("technical_analysis") or {}
    fund = state.get("fundamental_analysis") or {}
    sent = state.get("sentiment_analysis") or {}
    macro = state.get("macro_context")
    research = state.get("research_report") or {}

    from agents.conviction_scorer import score_conviction
    score = score_conviction(
        symbol=symbol,
        technical_data=tech,
        fundamental_data=fund,
        sentiment_data=sent,
        macro_sentiment=macro.sentiment_enum if macro else "NEUTRAL",
        macro_risk_multiplier=macro.risk_multiplier if macro else 1.0,
        research_score=research.get("research_score", 50),
        strategy_type=strategy_type,
    )

    return {
        "conviction_score": score.total,
        "conviction_passes": score.passes_threshold,
        "messages": [
            f"[Conviction] {symbol}: {score.total}/100 ({score.tier}) | "
            f"{'PASSES' if score.passes_threshold else 'FILTERED'}"
        ],
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def risk_manager_node(state: AgentState) -> dict:
    symbol = state['symbol']
    strategy_type = state.get('strategy_type', 'swing')

    if not state.get("conviction_passes", True):
        decision = RiskDecision(
            chain_of_thought_1_technicals="Filtered by conviction scorer.",
            chain_of_thought_2_fundamentals="Filtered by conviction scorer.",
            chain_of_thought_3_risk="Filtered by conviction scorer.",
            proposed_action="HOLD",
            proposed_entry=0.0, proposed_stop_loss=0.0, proposed_take_profit=0.0,
            conviction_tier="LOW", win_probability_score=0, risk_percentage=0.0,
            expected_holding_days=0,
            final_rationale=f"Conviction score {state.get('conviction_score', 0)}/100 below threshold.",
        )
        return {
            "decision": decision, "is_safe_to_execute": False,
            "guardrail_warnings": "FILTERED: Below conviction threshold.",
            "messages": [f"[Risk] {symbol}: FILTERED -- conviction too low."],
        }

    tech = state.get("technical_analysis") or {}
    fund = state.get("fundamental_analysis") or {}
    sent = state.get("sentiment_analysis") or {}
    macro = state.get("macro_context")
    narrative = state.get("technical_narrative", "")
    sector_perf = state.get("sector_performance") or {}
    weekly_data = state.get("weekly_data") or {}
    monthly_data = state.get("monthly_data") or {}
    research = state.get("research_report") or {}
    confluence = state.get("timeframe_confluence", 0)

    rl_context = ""
    try:
        similar = retrieve_similar_experiences(
            query_text=f"{symbol} {str(fund)} {str(sent)}", n_results=2
        )
        if similar and similar.get('documents') and similar['documents'][0]:
            rl_context = "\n".join(similar['documents'][0])
    except Exception:
        pass

    def _fail_decision(reason: str) -> RiskDecision:
        return RiskDecision(
            chain_of_thought_1_technicals="FAILED",
            chain_of_thought_2_fundamentals="FAILED",
            chain_of_thought_3_risk="FAILED",
            proposed_action="ERROR",
            proposed_entry=tech.get('latest_price', 0.0),
            proposed_stop_loss=0.0, proposed_take_profit=0.0,
            conviction_tier="LOW", win_probability_score=0, risk_percentage=0.0,
            expected_holding_days=0, final_rationale=reason,
        )

    decision = None
    client = get_client()

    if client:
        try:
            sys_prompt, user_template = _load_prompt("risk_manager.txt")
            user_text = user_template.format(
                symbol=symbol, strategy_type=strategy_type,
                current_price=tech.get('latest_price'),
                atr_14=tech.get('atr_14'), rsi_14=tech.get('rsi_14'),
                macd_histogram=tech.get('macd_histogram'), adx_14=tech.get('adx_14'),
                stoch_k=tech.get('stoch_k'), bb_pct_b=tech.get('bb_pct_b'),
                ema_20=tech.get('ema_20'), ema_50=tech.get('ema_50'),
                weekly_trend=tech.get('weekly_trend'),
                weekly_structure=weekly_data.get('weekly_structure', 'N/A'),
                weekly_rsi=weekly_data.get('weekly_rsi', 'N/A'),
                monthly_trend=monthly_data.get('monthly_trend', 'N/A'),
                timeframe_confluence=confluence,
                support_levels=tech.get('support_levels'),
                resistance_levels=tech.get('resistance_levels'),
                technical_narrative=narrative[:600],
                pe_ratio=fund.get('pe_ratio'), roe=fund.get('roe'),
                roce=fund.get('roce', 'N/A'), debt_to_equity=fund.get('debt_to_equity'),
                promoter_holding=fund.get('promoter_holding', 'N/A'),
                revenue_growth=fund.get('revenue_growth', 'N/A'),
                sentiment_score=sent.get('score', 0),
                sentiment_label=sent.get('label', 'NEUTRAL'),
                sentiment_summary=sent.get('summary', ''),
                macro_env=macro.sentiment_enum if macro else 'NEUTRAL',
                macro_risk=macro.risk_multiplier if macro else 1.0,
                macro_summary=macro.summary if macro else '',
                sector_performance=sector_perf,
                research_score=research.get('research_score', 'N/A'),
                research_recommendation=research.get('recommendation', 'N/A'),
                upcoming_catalysts=research.get('upcoming_catalysts', ''),
                conviction_score=state.get('conviction_score', 0),
                rl_context=rl_context or "No similar past trades found.",
            )
            result = call_structured(
                client=client, system_prompt=sys_prompt, user_text=user_text,
                tool_name="submit_trade_decision",
                tool_description="Submit the final structured trade decision",
                tool_schema={
                    "type": "object",
                    "properties": {
                        "chain_of_thought_1_technicals":   {"type": "string"},
                        "chain_of_thought_2_fundamentals": {"type": "string"},
                        "chain_of_thought_3_risk":         {"type": "string"},
                        "proposed_action":      {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                        "proposed_entry":       {"type": "number"},
                        "proposed_stop_loss":   {"type": "number"},
                        "proposed_take_profit": {"type": "number"},
                        "conviction_tier":      {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                        "win_probability_score":{"type": "integer", "minimum": 1, "maximum": 100},
                        "risk_percentage":      {"type": "number"},
                        "expected_holding_days":{"type": "integer"},
                        "final_rationale":      {"type": "string"},
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
        decision = _fail_decision("No LLM client or empty response.")

    is_safe = True
    warnings = []
    strategy_cfg = settings.strategy.get("strategies", {}).get(strategy_type, {})
    risk_cfg = settings.strategy.get("risk", {})
    max_abs_risk = risk_cfg.get("max_absolute_risk_per_trade", 0.02)
    min_sl_atr = risk_cfg.get("min_sl_atr_multiplier", 0.5)
    min_rr = strategy_cfg.get("min_rr_ratio", 1.5)
    atr = tech.get('atr_14', 0)

    if decision.proposed_action == "BUY":
        if decision.proposed_stop_loss >= decision.proposed_entry:
            is_safe = False
            warnings.append("CRITICAL: Stop loss >= entry (hallucination guard).")
        if atr > 0:
            min_sl = decision.proposed_entry - (atr * min_sl_atr)
            if decision.proposed_stop_loss > min_sl:
                is_safe = False
                warnings.append(
                    f"CRITICAL: SL {decision.proposed_stop_loss:.2f} too tight. "
                    f"Min safe SL = {min_sl:.2f} ({min_sl_atr}xATR)."
                )
        rr = 0.0
        if decision.proposed_entry > decision.proposed_stop_loss > 0:
            risk_pts = decision.proposed_entry - decision.proposed_stop_loss
            reward_pts = decision.proposed_take_profit - decision.proposed_entry
            rr = reward_pts / risk_pts if risk_pts > 0 else 0
        if rr < min_rr:
            is_safe = False
            warnings.append(f"CRITICAL: R:R {rr:.1f} below minimum {min_rr}.")
        if strategy_type == "positional" and confluence < 2:
            is_safe = False
            warnings.append(f"CRITICAL: Only {confluence}/3 timeframes aligned for positional trade.")

    risk_per_tier = {
        "HIGH":   strategy_cfg.get("risk_per_trade", 0.01),
        "MEDIUM": strategy_cfg.get("risk_per_trade", 0.01) * 0.75,
        "LOW":    strategy_cfg.get("risk_per_trade", 0.01) * 0.50,
    }
    if decision.proposed_action in ("BUY", "SELL"):
        tier = (decision.conviction_tier or "LOW").upper()
        decision.risk_percentage = risk_per_tier.get(tier, 0.005)

    if decision.risk_percentage > max_abs_risk:
        is_safe = False
        warnings.append(f"CRITICAL: Risk {decision.risk_percentage:.1%} exceeds max {max_abs_risk:.1%}.")

    if macro and macro.sentiment_enum == "BEARISH" and decision.proposed_action == "BUY":
        decision.risk_percentage = round(decision.risk_percentage * macro.risk_multiplier, 4)
        warnings.append(f"WARNING: Bearish macro -- risk reduced to {decision.risk_percentage:.1%}.")

    if not is_safe:
        decision.proposed_action = "ABORT_UNSAFE"

    return {
        "decision": decision,
        "is_safe_to_execute": is_safe,
        "guardrail_warnings": " | ".join(warnings),
        "rl_context": rl_context,
        "messages": [
            f"[Risk] {symbol} ({strategy_type}): Action={decision.proposed_action}, "
            f"Conviction={decision.conviction_tier}, Safe={is_safe}."
        ],
    }
