"""
Conviction Scorer — synthesises all analysis into a 0-100 score.
Uses Haiku (fast, cheap) since this runs on every screener candidate.
Only proposals scoring >= conviction_threshold reach the user.
"""

from dataclasses import dataclass, field
from typing import Optional
from core.claude_client import get_client, call_structured
from core.config import settings


@dataclass
class ConvictionScore:
    total: int = 0                    # 0-100 overall
    technicals: int = 0               # 0-30
    fundamentals: int = 0             # 0-30
    macro_sentiment: int = 0          # 0-20
    research_quality: int = 0         # 0-20
    breakdown: str = ""
    tier: str = "LOW"                 # HIGH (80+) / MEDIUM (65-79) / LOW (<65)
    passes_threshold: bool = False


def score_conviction(
    symbol: str,
    technical_data: dict,
    fundamental_data: dict,
    sentiment_data: dict,
    macro_sentiment: str = "NEUTRAL",
    macro_risk_multiplier: float = 1.0,
    research_score: int = 50,
    strategy_type: str = "swing",
) -> ConvictionScore:
    """
    Runs conviction scoring using Claude Haiku.
    Falls back to rule-based scoring if LLM unavailable.
    """
    threshold = settings.strategy.get("strategies", {}).get(strategy_type, {}).get(
        "conviction_threshold", 65
    )

    # DEGRADED MODE: when screener.in is unavailable, cap max conviction to avoid
    # trading on incomplete governance/pledge data. 64 = just below any threshold.
    data_quality = fundamental_data.get("data_quality", "FULL")
    degraded_cap = 64 if data_quality == "DEGRADED" else 100

    client = get_client()
    if not client:
        result = _rule_based_score(technical_data, fundamental_data, sentiment_data,
                                   macro_sentiment, research_score, strategy_type, threshold)
        if degraded_cap < 100:
            result.total = min(result.total, degraded_cap)
            result.passes_threshold = result.total >= threshold
            result.breakdown += " | DATA DEGRADED: score capped at 64 (screener.in unavailable)"
        return result

    try:
        system_prompt = (
            "You are a quantitative conviction scorer for Indian equity trades. "
            "Score each factor objectively based on the data. Be precise and conservative."
        )

        user_text = f"""Score the conviction for a {strategy_type.upper()} trade on {symbol}.

TECHNICAL DATA:
- RSI: {technical_data.get('rsi_14', 'N/A')} (ideal: 45-65 for entry)
- ADX: {technical_data.get('adx_14', 'N/A')} (>25 = strong trend)
- MACD Histogram: {technical_data.get('macd_histogram', 'N/A')} (positive = bullish)
- Price vs EMA20: {_pct_from_ema(technical_data)}%
- Price vs EMA50: {_pct_from_ema50(technical_data)}%
- Weekly Trend: {technical_data.get('weekly_trend', 'N/A')}
- BB %B: {technical_data.get('bb_pct_b', 'N/A')} (0=oversold, 1=overbought)
- Stochastic K: {technical_data.get('stoch_k', 'N/A')}
- Relative Strength (30d): {technical_data.get('relative_strength_30d', 'N/A')}

FUNDAMENTAL DATA:
- P/E Ratio: {fundamental_data.get('pe_ratio', 'N/A')} (sector median: {fundamental_data.get('sector_pe_median', 'N/A')})
- PE vs Sector: {_pe_vs_sector_label(fundamental_data)}
- ROE: {fundamental_data.get('roe', 'N/A')}%
- ROCE: {fundamental_data.get('roce', 'N/A')}%
- Debt/Equity: {fundamental_data.get('debt_to_equity', 'N/A')}
- Promoter Holding: {fundamental_data.get('promoter_holding', 'N/A')}%
- Revenue Growth: {fundamental_data.get('revenue_growth', 'N/A')}%
- Profit Growth: {fundamental_data.get('profit_growth', 'N/A')}%
- Quality Score: {fundamental_data.get('quality_score', 'N/A')}/100
- Promoter Pledge: {fundamental_data.get('promoter_pledge', 'N/A')}
- Book Value: {fundamental_data.get('book_value', 'N/A')}
- EPS Growth: {fundamental_data.get('eps_growth', 'N/A')}
- Revenue Growth: {fundamental_data.get('revenue_growth', 'N/A')}%
- VWAP Deviation: {technical_data.get('vwap_deviation_pct', 'N/A')}%
- OBV Trend: {technical_data.get('obv_trend', 'N/A')}
- RSI Divergence: {technical_data.get('rsi_divergence', 'N/A')}
- MACD Divergence: {technical_data.get('macd_divergence', 'N/A')}
- 6M Momentum: {technical_data.get('momentum_6m', 'N/A')}
- 12M Momentum: {technical_data.get('momentum_12m', 'N/A')}

SENTIMENT:
- Label: {sentiment_data.get('label', 'NEUTRAL')}
- Score: {sentiment_data.get('score', 0)}

MACRO:
- Regime: {macro_sentiment}
- Risk Multiplier: {macro_risk_multiplier}

RESEARCH QUALITY SCORE (pre-computed): {research_score}/100

Strategy: {strategy_type} (swing=5-30d, positional=30-180d)
Conviction threshold to reach user: {threshold}/100

Score each dimension and total. Be precise — do not round to 50s and 70s reflexively."""

        result = call_structured(
            client=client,
            system_prompt=system_prompt,
            user_text=user_text,
            tool_name="submit_conviction_score",
            tool_description="Submit conviction score breakdown",
            tool_schema={
                "type": "object",
                "properties": {
                    "technicals_score":       {"type": "integer", "minimum": 0, "maximum": 30},
                    "fundamentals_score":     {"type": "integer", "minimum": 0, "maximum": 30},
                    "macro_sentiment_score":  {"type": "integer", "minimum": 0, "maximum": 20},
                    "research_quality_score": {"type": "integer", "minimum": 0, "maximum": 20},
                    "breakdown":              {"type": "string"},
                },
                "required": ["technicals_score", "fundamentals_score",
                             "macro_sentiment_score", "research_quality_score", "breakdown"],
            },
            use_haiku=True,
            cache_system=True,
        )

        if result:
            tech = int(result.get("technicals_score", 0))
            fund = int(result.get("fundamentals_score", 0))
            macro = int(result.get("macro_sentiment_score", 0))
            research = int(result.get("research_quality_score", 0))
            total = tech + fund + macro + research

            # Macro adjustment
            if macro_sentiment == "BEARISH":
                penalized = int(total * 0.88)  # Soft penalty: reduce ~12%, don't kill the trade
                total = penalized
            elif macro_sentiment == "BULLISH":
                total = min(100, int(total * 1.05))  # Small bullish boost, capped at 100

            # PE overvaluation penalty: if PE > 1.5× sector median, penalize
            pe_penalty = _compute_pe_penalty(fundamental_data)
            if pe_penalty > 0:
                total = max(0, total - pe_penalty)

            # Degraded data cap
            total = min(total, degraded_cap)
            degraded_note = " | DATA DEGRADED: capped at 64" if degraded_cap < 100 else ""

            tier = "HIGH" if total >= 80 else "MEDIUM" if total >= 65 else "LOW"
            return ConvictionScore(
                total=total,
                technicals=tech,
                fundamentals=fund,
                macro_sentiment=macro,
                research_quality=research,
                breakdown=result.get("breakdown", "") + (f" | PE penalty: -{pe_penalty}" if pe_penalty else "") + degraded_note,
                tier=tier,
                passes_threshold=total >= threshold,
            )

    except Exception as e:
        print(f"[ConvictionScorer] LLM failed for {symbol}: {e}")

    return _rule_based_score(technical_data, fundamental_data, sentiment_data,
                              macro_sentiment, research_score, strategy_type, threshold)


def _rule_based_score(
    technical_data: dict,
    fundamental_data: dict,
    sentiment_data: dict,
    macro_sentiment: str,
    research_score: int,
    strategy_type: str,
    threshold: int,
) -> ConvictionScore:
    """Deterministic fallback scorer when LLM is unavailable."""
    tech = 0
    rsi_raw = technical_data.get("rsi_14")
    rsi = rsi_raw if rsi_raw is not None else 50
    adx_raw = technical_data.get("adx_14")
    adx = adx_raw if adx_raw is not None else 20
    macd_raw = technical_data.get("macd_histogram")
    macd = macd_raw if macd_raw is not None else 0

    if 40 <= rsi <= 65: tech += 8
    elif 30 <= rsi < 40 or 65 < rsi <= 75: tech += 4
    if adx > 25: tech += 8
    elif adx > 18: tech += 4
    if macd > 0: tech += 7
    if technical_data.get("weekly_trend") == "UP": tech += 7

    # Pullback-in-uptrend verification: cleanest swing entry is a controlled pullback
    # to EMA20 within a broader uptrend — not a surge entry at the top
    latest_price = technical_data.get("latest_price", 0) or 0
    ema20 = technical_data.get("ema_20", 0) or 0
    weekly_structure = technical_data.get("weekly_structure", "")
    if weekly_structure == "PULLBACK_IN_UPTREND" and ema20 > 0:
        price_to_ema_pct = (latest_price - ema20) / ema20 * 100 if ema20 > 0 else 999
        if 0 <= price_to_ema_pct <= 3:
            tech += 5   # Clean pullback to EMA20 within uptrend — highest quality entry
        elif price_to_ema_pct > 8:
            tech -= 4   # Extended far above EMA20 in pullback zone — lower quality
    elif weekly_structure in ("STRONG_UP", "UP") and ema20 > 0:
        price_to_ema_pct = (latest_price - ema20) / ema20 * 100 if ema20 > 0 else 999
        if price_to_ema_pct > 10:
            tech -= 3   # Chasing a surge — reduce score

    tech = min(tech, 30)

    fund = 0
    roe = fundamental_data.get("roe", 0) or 0
    roce = fundamental_data.get("roce", 0) or 0
    de = fundamental_data.get("debt_to_equity", 99) or 99
    if roe > 15: fund += 8
    elif roe > 10: fund += 4
    if roce > 15: fund += 8
    elif roce > 10: fund += 4
    if de < 0.5: fund += 8
    elif de < 1.0: fund += 4
    promoter = fundamental_data.get("promoter_holding", 0) or 0
    if promoter > 50: fund += 6
    fund = min(fund, 30)

    # Bonus from quality score
    quality = fundamental_data.get("quality_score", 0) or 0
    if quality > 70: fund = min(fund + 4, 30)
    elif quality > 50: fund = min(fund + 2, 30)

    # Quality-at-value bonus: cheap stock (PE < 0.7× sector) with strong fundamentals
    # Only award if profitability is genuine — prevents value traps
    pe_ratio = fundamental_data.get("pe_ratio") or 0
    sector_pe = fundamental_data.get("sector_pe_median") or 0
    try:
        if pe_ratio > 0 and sector_pe > 0:
            pe_ratio_f, sector_pe_f = float(pe_ratio), float(sector_pe)
            if pe_ratio_f / sector_pe_f < 0.7 and roe > 12 and roce > 12:
                fund = min(fund + 5, 30)  # Quality-at-value: genuinely cheap + profitable
    except (ValueError, TypeError):
        pass

    # Promoter pledge penalty
    pledge_str = fundamental_data.get("promoter_pledge", "N/A")
    if pledge_str and pledge_str != "N/A":
        try:
            pledge_val = float(str(pledge_str).replace('%', '').strip())
            if pledge_val > 30: fund = max(0, fund - 5)
        except (ValueError, TypeError):
            pass

    # PE overvaluation penalty
    pe_penalty = _compute_pe_penalty(fundamental_data)
    fund = max(0, fund - pe_penalty)

    macro = {"BULLISH": 17, "NEUTRAL": 13, "BEARISH": 9}.get(macro_sentiment, 13)

    research = min(int(research_score * 0.2), 20)
    total = tech + fund + macro + research
    tier = "HIGH" if total >= 80 else "MEDIUM" if total >= 65 else "LOW"
    return ConvictionScore(
        total=total, technicals=tech, fundamentals=fund,
        macro_sentiment=macro, research_quality=research,
        breakdown="Rule-based (LLM unavailable)",
        tier=tier, passes_threshold=total >= threshold,
    )


def _pct_from_ema(data: dict) -> str:
    price = data.get("latest_price", 0)
    ema = data.get("ema_20", 0)
    if price and ema and ema > 0:
        return str(round((price - ema) / ema * 100, 1))
    return "N/A"


def _pct_from_ema50(data: dict) -> str:
    price = data.get("latest_price", 0)
    ema = data.get("ema_50", 0)
    if price and ema and ema > 0:
        return str(round((price - ema) / ema * 100, 1))
    return "N/A"


def _compute_pe_penalty(fundamental_data: dict) -> int:
    """
    PE overvaluation penalty:
    - PE > 2× sector median → -10 pts (expensive bubble territory)
    - PE > 1.5× sector median → -5 pts (elevated, reduces conviction)
    - PE ≤ sector median → 0 (fair/cheap, no penalty)
    Returns penalty points to subtract from total.
    """
    pe_raw = fundamental_data.get("pe_ratio")
    sector_pe_raw = fundamental_data.get("sector_pe_median")
    if pe_raw is None or sector_pe_raw is None:
        return 0
    try:
        pe = float(pe_raw)
        sector_pe = float(sector_pe_raw)
        if sector_pe <= 0 or pe <= 0:
            return 0
        ratio = pe / sector_pe
        if ratio > 2.0:
            return 10
        elif ratio > 1.5:
            return 5
        return 0
    except (ValueError, TypeError):
        return 0


def _pe_vs_sector_label(fundamental_data: dict) -> str:
    """Returns human-readable PE vs sector label for the LLM prompt."""
    pe_raw = fundamental_data.get("pe_ratio")
    sector_pe_raw = fundamental_data.get("sector_pe_median")
    if pe_raw is None or sector_pe_raw is None:
        return "N/A"
    try:
        pe = float(pe_raw)
        sector_pe = float(sector_pe_raw)
        if sector_pe <= 0:
            return "N/A"
        ratio = pe / sector_pe
        if ratio > 2.0:
            return f"EXPENSIVE ({ratio:.1f}x sector median)"
        elif ratio > 1.5:
            return f"ELEVATED ({ratio:.1f}x sector median)"
        elif ratio < 0.7:
            return f"CHEAP ({ratio:.1f}x sector median)"
        return f"FAIR ({ratio:.1f}x sector median)"
    except (ValueError, TypeError):
        return "N/A"
