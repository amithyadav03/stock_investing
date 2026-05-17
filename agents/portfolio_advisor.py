"""
Portfolio Advisor — analyses existing Kite demat holdings and gives HOLD/ADD_MORE/EXIT advice.
Runs daily post-market and on every morning brief.
"""

from datetime import datetime
from typing import Optional
from core.claude_client import get_client, call_structured
from core.cache import cache, TTL_FUNDAMENTALS
from tools.fundamental_news import fundamental_news_tool
from tools.market_data import market_data_tool
from agents.llm_utils import load_prompt as _load_prompt
from db.schema import SessionLocal, PortfolioHolding


def advise_holding(holding: dict) -> dict:
    """
    Analyses a single holding and returns advice.
    holding: {symbol, quantity, avg_price, current_price, pnl_pct, ...}
    Returns: {symbol, action, rationale, urgency}
    """
    symbol = holding["symbol"]
    cache_key = f"holding_advice_{symbol}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    client = get_client()
    fallback = {
        "symbol": symbol,
        "action": "HOLD",
        "rationale": "No AI analysis available — hold by default.",
        "urgency": "NORMAL",
    }

    if not client:
        return fallback

    technicals = _safe_fetch(lambda: market_data_tool.fetch_advanced_technicals(symbol), {})
    if "error" in technicals:
        technicals = {}

    fundamentals = _safe_fetch(lambda: fundamental_news_tool.get_comparative_fundamentals(symbol), {})
    news = _safe_fetch(lambda: fundamental_news_tool.fetch_live_news_snippets(target_keyword=symbol), [])
    news_text = "\n".join(f"• {h}" for h in news[:8]) if news else "No recent news."

    pnl_pct = holding.get("pnl_pct", 0.0)
    avg_price = holding.get("avg_price", 0.0)
    current_price = holding.get("current_price", avg_price)
    qty = holding.get("quantity", 0)

    try:
        sys_prompt, user_template = _load_prompt("portfolio_advisor.txt")
        user_text = user_template.format(
            symbol=symbol,
            quantity=qty,
            avg_price=avg_price,
            current_price=current_price,
            pnl_pct=round(pnl_pct, 2),
            unrealized_pnl=round((current_price - avg_price) * qty, 2),
            rsi_14=technicals.get("rsi_14", "N/A"),
            adx_14=technicals.get("adx_14", "N/A"),
            ema_20=technicals.get("ema_20", "N/A"),
            ema_50=technicals.get("ema_50", "N/A"),
            weekly_trend=technicals.get("weekly_trend", "N/A"),
            macd_histogram=technicals.get("macd_histogram", "N/A"),
            support_levels=technicals.get("support_levels", []),
            pe_ratio=fundamentals.get("pe_ratio", "N/A"),
            roe=fundamentals.get("roe", "N/A"),
            roce=fundamentals.get("roce", "N/A"),
            debt_to_equity=fundamentals.get("debt_to_equity", "N/A"),
            promoter_holding=fundamentals.get("promoter_holding", "N/A"),
            recent_news=news_text,
        )

        result = call_structured(
            client=client,
            system_prompt=sys_prompt,
            user_text=user_text,
            tool_name="submit_holding_advice",
            tool_description="Submit advice for this existing portfolio holding",
            tool_schema={
                "type": "object",
                "properties": {
                    "action":    {"type": "string", "enum": ["HOLD", "ADD_MORE", "EXIT"]},
                    "urgency":   {"type": "string", "enum": ["NORMAL", "URGENT"]},
                    "rationale": {"type": "string"},
                    "add_more_entry": {"type": "number"},
                    "exit_target":    {"type": "number"},
                },
                "required": ["action", "urgency", "rationale"],
            },
            use_haiku=False,
        )

        if result:
            advice = {
                "symbol": symbol,
                "action": result.get("action", "HOLD"),
                "urgency": result.get("urgency", "NORMAL"),
                "rationale": result.get("rationale", ""),
                "add_more_entry": result.get("add_more_entry"),
                "exit_target": result.get("exit_target"),
            }
            cache.set(cache_key, advice, ttl_seconds=14400)  # 4 hrs
            _persist_advice(symbol, advice)
            return advice

    except Exception as e:
        print(f"[PortfolioAdvisor] Failed for {symbol}: {e}")

    return fallback


def advise_all_holdings(holdings: list[dict]) -> list[dict]:
    """Runs advise_holding on every holding. Returns list of advice dicts."""
    results = []
    for h in holdings:
        advice = advise_holding(h)
        results.append({**h, **advice})
        print(f"[PortfolioAdvisor] {h['symbol']}: {advice['action']} ({advice['urgency']})")
    return results


def _persist_advice(symbol: str, advice: dict):
    """Writes advice back to PortfolioHolding table."""
    session = SessionLocal()
    try:
        record = session.query(PortfolioHolding).filter(
            PortfolioHolding.symbol == symbol
        ).first()
        if record:
            record.advisor_action = advice["action"]
            record.advisor_rationale = advice.get("rationale", "")[:1000]
            record.advisor_updated_at = datetime.utcnow()
            session.commit()
    except Exception as e:
        print(f"[PortfolioAdvisor] DB persist failed for {symbol}: {e}")
    finally:
        session.close()


def get_portfolio_advice_from_db() -> list[dict]:
    """Reads the latest stored advice for all holdings."""
    session = SessionLocal()
    try:
        records = session.query(PortfolioHolding).all()
        return [
            {
                "symbol": r.symbol,
                "quantity": r.quantity,
                "avg_price": r.avg_price,
                "current_price": r.current_price or r.avg_price,
                "pnl_pct": r.pnl_pct or 0.0,
                "pnl_amount": r.pnl_amount or 0.0,
                "action": r.advisor_action or "HOLD",
                "rationale": r.advisor_rationale or "",
                "urgency": "NORMAL",
            }
            for r in records
        ]
    finally:
        session.close()


def _safe_fetch(fn, default):
    try:
        return fn()
    except Exception:
        return default
