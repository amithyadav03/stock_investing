import operator
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from tools.fundamental_news import MacroContext


class RiskDecision(BaseModel):
    """Structured trade decision produced by the Risk Manager agent."""
    chain_of_thought_1_technicals: str = Field(description="Step 1: Reasoning on chart structure, indicators, and price action.")
    chain_of_thought_2_fundamentals: str = Field(description="Step 2: Reasoning on P/E, ROE, ROCE, debt, promoter holding, and news sentiment.")
    chain_of_thought_3_risk: str = Field(description="Step 3: Risk profile — stop loss placement (ATR-based), R:R ratio, and macro overlay.")

    proposed_action: str = Field(description="'BUY', 'SELL', or 'HOLD'.")
    proposed_entry: float = Field(description="Exact numerical entry price.")
    proposed_stop_loss: float = Field(description="ATR-based stop loss price.")
    proposed_take_profit: float = Field(description="Take profit price (minimum 1:2 R:R).")

    conviction_tier: str = Field(description="Exactly one of: 'HIGH', 'MEDIUM', 'LOW'.")
    win_probability_score: int = Field(description="Integer 1–100: confidence in setup quality.")
    risk_percentage: float = Field(description="Capital at risk as a decimal (e.g. 0.015 = 1.5%).")
    expected_holding_days: int = Field(description="Estimated calendar days to hold until target.")
    final_rationale: str = Field(description="Markdown rationale combining all chain-of-thought steps.")


class AgentState(TypedDict):
    symbol: str
    messages: Annotated[List[str], operator.add]

    # Analysis contexts (populated by parallel nodes)
    technical_analysis: Optional[Dict[str, Any]]
    technical_narrative: Optional[str]       # Claude's visual chart narrative
    fundamental_analysis: Optional[Dict[str, Any]]
    sentiment_analysis: Optional[Dict[str, Any]]
    macro_context: Optional[MacroContext]
    sector_performance: Optional[Dict[str, str]]
    rl_context: Optional[str]

    # Final decision + guardrail outputs
    decision: Optional[RiskDecision]
    is_safe_to_execute: Optional[bool]
    guardrail_warnings: Optional[str]
