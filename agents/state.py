import operator
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from tools.fundamental_news import MacroContext


class RiskDecision(BaseModel):
    """Structured trade decision produced by the Risk Manager agent."""
    chain_of_thought_1_technicals: str = Field(description="Reasoning on chart structure, indicators, and price action.")
    chain_of_thought_2_fundamentals: str = Field(description="Reasoning on P/E, ROE, ROCE, debt, promoter holding, and news sentiment.")
    chain_of_thought_3_risk: str = Field(description="Risk profile — stop loss placement, R:R ratio, and macro overlay.")

    proposed_action: str = Field(description="'BUY', 'SELL', or 'HOLD'.")
    proposed_entry: float = Field(description="Exact numerical entry price.")
    proposed_stop_loss: float = Field(description="ATR-based stop loss price.")
    proposed_take_profit: float = Field(description="Take profit price.")

    conviction_tier: str = Field(description="Exactly one of: 'HIGH', 'MEDIUM', 'LOW'.")
    win_probability_score: int = Field(description="Integer 1–100: confidence in setup quality.")
    risk_percentage: float = Field(description="Capital at risk as a decimal (e.g. 0.01 = 1%).")
    expected_holding_days: int = Field(description="Estimated calendar days to hold until target.")
    final_rationale: str = Field(description="Markdown rationale combining all chain-of-thought steps.")


class AgentState(TypedDict):
    symbol: str
    strategy_type: str                  # "swing" | "positional" | "value"
    messages: Annotated[List[str], operator.add]

    # Analysis contexts (populated by parallel nodes)
    technical_analysis: Optional[Dict[str, Any]]
    technical_narrative: Optional[str]
    weekly_data: Optional[Dict[str, Any]]
    monthly_data: Optional[Dict[str, Any]]
    timeframe_confluence: Optional[int]    # 0-3: how many timeframes agree

    fundamental_analysis: Optional[Dict[str, Any]]
    sentiment_analysis: Optional[Dict[str, Any]]
    macro_context: Optional[MacroContext]
    sector_performance: Optional[Dict[str, str]]
    research_report: Optional[Dict[str, Any]]  # ResearchReport as dict
    rl_context: Optional[str]

    # Conviction scoring
    conviction_score: Optional[int]       # 0-100
    conviction_passes: Optional[bool]

    # Final decision + guardrail outputs
    decision: Optional[RiskDecision]
    is_safe_to_execute: Optional[bool]
    guardrail_warnings: Optional[str]
