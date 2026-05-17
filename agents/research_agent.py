"""
Deep Research Agent — Claude-powered qualitative analysis for a single stock.
Goes beyond indicators: reads news, management commentary, fundamentals narrative.
Uses Sonnet for depth. Called post-market on top screener candidates.
"""

from dataclasses import dataclass, field
from typing import Optional
from core.claude_client import get_client, call_structured, call_text
from core.cache import cache, TTL_FUNDAMENTALS
from tools.fundamental_news import fundamental_news_tool
from agents.llm_utils import load_prompt as _load_prompt


@dataclass
class ResearchReport:
    symbol: str
    business_summary: str = ""
    recent_developments: str = ""
    fundamental_quality: str = ""
    management_quality: str = ""
    competitive_position: str = ""
    key_risks: str = ""
    upcoming_catalysts: str = ""
    research_score: int = 0          # 0-100: overall research quality score
    recommendation: str = "NEUTRAL"  # STRONG_BUY / BUY / NEUTRAL / AVOID
    error: Optional[str] = None


def run_deep_research(symbol: str, strategy_type: str = "swing") -> ResearchReport:
    """
    Runs full deep research pipeline on a symbol.
    Returns ResearchReport with qualitative insights.
    """
    cache_key = f"research_{symbol}_{strategy_type}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    client = get_client()
    if not client:
        return ResearchReport(symbol=symbol, error="No LLM client.")

    # Gather raw data
    fundamentals = _safe_fetch(lambda: fundamental_news_tool.get_comparative_fundamentals(symbol), {})
    news = _safe_fetch(lambda: fundamental_news_tool.fetch_live_news_snippets(target_keyword=symbol), [])
    sentiment = _safe_fetch(lambda: fundamental_news_tool.get_micro_sentiment_score(symbol), {})

    news_text = "\n".join(f"• {h}" for h in news[:15]) if news else "No recent news found."

    try:
        sys_prompt, user_template = _load_prompt("research_agent.txt")
        user_text = user_template.format(
            symbol=symbol,
            strategy_type=strategy_type,
            pe_ratio=fundamentals.get("pe_ratio", "N/A"),
            roe=fundamentals.get("roe", "N/A"),
            roce=fundamentals.get("roce", "N/A"),
            debt_to_equity=fundamentals.get("debt_to_equity", "N/A"),
            promoter_holding=fundamentals.get("promoter_holding", "N/A"),
            revenue_growth=fundamentals.get("revenue_growth", "N/A"),
            profit_growth=fundamentals.get("profit_growth", "N/A"),
            sector=fundamentals.get("sector", "N/A"),
            peer_comparison=fundamentals.get("peer_comparison", "N/A"),
            recent_news=news_text,
            sentiment_label=sentiment.get("label", "NEUTRAL"),
            sentiment_summary=sentiment.get("summary", ""),
        )

        result = call_structured(
            client=client,
            system_prompt=sys_prompt,
            user_text=user_text,
            tool_name="submit_research_report",
            tool_description="Submit the deep research report for this stock",
            tool_schema={
                "type": "object",
                "properties": {
                    "business_summary":      {"type": "string"},
                    "recent_developments":   {"type": "string"},
                    "fundamental_quality":   {"type": "string"},
                    "management_quality":    {"type": "string"},
                    "competitive_position":  {"type": "string"},
                    "key_risks":             {"type": "string"},
                    "upcoming_catalysts":    {"type": "string"},
                    "research_score":        {"type": "integer", "minimum": 0, "maximum": 100},
                    "recommendation":        {"type": "string", "enum": ["STRONG_BUY", "BUY", "NEUTRAL", "AVOID"]},
                },
                "required": ["business_summary", "fundamental_quality", "key_risks",
                             "research_score", "recommendation"],
            },
            use_haiku=False,
        )

        if result:
            report = ResearchReport(
                symbol=symbol,
                business_summary=result.get("business_summary", ""),
                recent_developments=result.get("recent_developments", ""),
                fundamental_quality=result.get("fundamental_quality", ""),
                management_quality=result.get("management_quality", ""),
                competitive_position=result.get("competitive_position", ""),
                key_risks=result.get("key_risks", ""),
                upcoming_catalysts=result.get("upcoming_catalysts", ""),
                research_score=int(result.get("research_score", 50)),
                recommendation=result.get("recommendation", "NEUTRAL"),
            )
            cache.set(cache_key, report, TTL_FUNDAMENTALS)
            print(f"[Research] {symbol}: Score={report.research_score}, Rec={report.recommendation}")
            return report

    except Exception as e:
        print(f"[Research] Failed for {symbol}: {e}")
        return ResearchReport(symbol=symbol, error=str(e))

    return ResearchReport(symbol=symbol, error="No response from Claude.")


def get_research_summary(symbol: str) -> str:
    """Returns a short research summary string for Telegram messages."""
    report = run_deep_research(symbol)
    if report.error:
        return f"Research unavailable: {report.error}"
    lines = []
    if report.business_summary:
        lines.append(f"*Business*: {report.business_summary[:200]}")
    if report.upcoming_catalysts:
        lines.append(f"*Catalysts*: {report.upcoming_catalysts[:200]}")
    if report.key_risks:
        lines.append(f"*Risks*: {report.key_risks[:150]}")
    lines.append(f"*Research Score*: {report.research_score}/100 | {report.recommendation}")
    return "\n".join(lines)


def _safe_fetch(fn, default):
    try:
        return fn()
    except Exception:
        return default
