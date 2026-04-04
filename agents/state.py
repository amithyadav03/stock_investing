import operator
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from tools.fundamental_news import MacroContext

class RiskDecision(BaseModel):
    """ The core schema that the LLM is FORCED to output perfectly. """
    chain_of_thought_1_technicals: str = Field(description="Step 1: Your explicit reasoning about the mathematical indicators and visual chart.")
    chain_of_thought_2_fundamentals: str = Field(description="Step 2: Your explicit reasoning on the P/E ratios and current sentiment.")
    chain_of_thought_3_risk: str = Field(description="Step 3: Justify the absolute risk and stop loss placements factoring in Macro Context.")
    proposed_action: str = Field(description="'BUY', 'SELL', or 'HOLD'")
    proposed_entry: float = Field(description="The exact numerical entry price.")
    proposed_stop_loss: float = Field(description="The exact stop loss price. Should use ATR logic.")
    proposed_take_profit: float = Field(description="The exact take profit price.")
    risk_percentage: float = Field(description="Max risk as a float (e.g. 0.05 for 5%)")
    expected_holding_days: int = Field(description="Estimated number of calendar days to hold the position until the take profit is reached. e.g. 7, 14, 30.")
    final_rationale: str = Field(description="Markdown rationale of why this trade was proposed combining your Chain of Thought.")

class AgentState(TypedDict):
    symbol: str
    messages: Annotated[List[str], operator.add]
    
    # Contexts
    technical_analysis: Optional[Dict[str, Any]]
    fundamental_analysis: Optional[Dict[str, Any]]
    sentiment_analysis: Optional[Dict[str, Any]]
    macro_context: Optional[MacroContext]
    rl_context: Optional[str]
    
    # Decisions (Parsed via Structured Output)
    decision: Optional[RiskDecision]
    
    # Post-guardrail checks
    is_safe_to_execute: Optional[bool]
    guardrail_warnings: Optional[str]
