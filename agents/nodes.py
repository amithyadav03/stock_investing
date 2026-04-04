import os
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential # NEW
from agents.state import AgentState, RiskDecision
from tools.market_data import market_data_tool
from tools.fundamental_news import fundamental_news_tool
from db.memory import retrieve_similar_experiences
from core.config import settings

def get_llm():
    if not settings.OPENAI_API_KEY:
        return None
    return ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_prompt(filename: str) -> tuple[str, str]:
    """Loads a prompt file and splits it into system and user parts by '---'"""
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    with open(path, "r") as f:
        content = f.read()
    parts = content.split("---")
    return parts[0].strip(), parts[1].strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def technical_analyst_node(state: AgentState) -> dict:
    symbol = state['symbol']
    technicals = market_data_tool.fetch_advanced_technicals(symbol)
    
    analysis = "[MOCK EXPERT] Baseline mock analysis."
    llm = get_llm()
    if llm and "chart_path" in technicals:
        try:
            base64_image = encode_image(technicals['chart_path'])
            sys_template, user_template = load_prompt("technical_analyst.txt")
            
            sys_msg = SystemMessage(content=sys_template)
            prompt = user_template.format(technicals=technicals)
            
            user_msg = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ])
            analysis = llm.invoke([sys_msg, user_msg]).content
        except Exception as e:
            analysis = f"Error processing vision: {e}"

    return {
        "technical_analysis": technicals,
        "messages": [f"Visual and Math Technicals processed. Findings: {analysis[:50]}..."]
    }

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fundamental_sentiment_node(state: AgentState) -> dict:
    symbol = state['symbol']
    fundamentals = fundamental_news_tool.get_comparative_fundamentals(symbol)
    sentiment = fundamental_news_tool.get_micro_sentiment_score(symbol)
    macro = fundamental_news_tool.get_macro_context()
    
    return {
        "fundamental_analysis": fundamentals,
        "sentiment_analysis": sentiment,
        "macro_context": macro,
        "messages": ["Fundamentals, Sentiment, and Macro conditions fetched."]
    }

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def risk_manager_node(state: AgentState) -> dict:
    symbol = state['symbol']
    tech = state.get("technical_analysis", {})
    fund = state.get("fundamental_analysis", {})
    sent = state.get("sentiment_analysis", {})
    macro = state.get("macro_context", None)
    
    rl_context = ""
    try:
        similar = retrieve_similar_experiences(query_text=str(fund) + str(sent), n_results=1)
        if similar and similar['documents'] and similar['documents'][0]:
            rl_context = similar['documents'][0][0]
    except: pass

    decision = None
    llm = get_llm()
    if llm:
        structured_llm = llm.with_structured_output(RiskDecision)
        
        try:
            sys_template, user_template = load_prompt("risk_manager.txt")
            prompt = user_template.format(
                symbol=symbol,
                current_price=tech.get('latest_price'),
                atr_14=tech.get('atr_14'),
                rsi_14=tech.get('rsi_14'),
                macd_histogram=tech.get('macd_histogram'),
                macro_env=macro.sentiment_enum if macro else 'UNKNOWN',
                macro_risk=macro.risk_multiplier if macro else 1.0,
                rl_context=rl_context if rl_context else 'No similar past trades found.'
            )
            decision = structured_llm.invoke([SystemMessage(content=sys_template), HumanMessage(content=prompt)])
        except Exception as e:
            print(f"LLM Parsing failed: {e}")
            
    if not decision: 
        decision = RiskDecision(
            chain_of_thought_1_technicals="Mock",
            chain_of_thought_2_fundamentals="Mock",
            chain_of_thought_3_risk="Mock",
            proposed_action="BUY",
            proposed_entry=tech.get('latest_price', 100.0),
            proposed_stop_loss=tech.get('latest_price', 100.0) - (tech.get('atr_14', 2.0) * 2),
            proposed_take_profit=tech.get('latest_price', 100.0) * 1.05,
            risk_percentage=0.05,
            expected_holding_days=14,
            final_rationale="Simulated safe trade driven by fallback."
        )

    # 3. PYTHON HARD GUARDRAILS (Protecting Real Money)
    is_safe = True
    warnings = []
    
    # Load strategy from config
    strategy_risk = settings.strategy.get("risk", {})
    max_abs_risk = strategy_risk.get("max_absolute_risk_limit", 0.08)
    min_sl_atr = strategy_risk.get("min_sl_atr_multiplier", 0.5)

    # 3A. Simple Math Sanity
    if decision.proposed_action == "BUY" and decision.proposed_stop_loss >= decision.proposed_entry:
        is_safe = False
        warnings.append("CRITICAL: LLM hallucinations caused Stop Loss to be higher than Entry.")
        
    # 3B. ATR SLIPPAGE GUARDRAIL (Volatility Check)
    atr = tech.get('atr_14', 0)
    if decision.proposed_action == "BUY" and atr > 0:
        min_allowed_sl = decision.proposed_entry - (atr * min_sl_atr)
        if decision.proposed_stop_loss > min_allowed_sl:
            is_safe = False
            warnings.append(f"CRITICAL: Proposed SL {decision.proposed_stop_loss} is too tight. Minimum safe SL given current volatility is {min_allowed_sl} ({min_sl_atr}x ATR).")

    # 3C. Max Risk per trade limit
    if decision.risk_percentage > max_abs_risk:
        is_safe = False
        warnings.append(f"CRITICAL: Proposed risk {decision.risk_percentage} exceeds absolute limit of {max_abs_risk}%.")

    # 3D. Macro overriding
    if macro and macro.sentiment_enum == "BEARISH" and decision.proposed_action == "BUY":
        warnings.append("WARNING: Buying in a Bearish Macro environment. Halving proposed risk footprint.")
        decision.risk_percentage *= macro.risk_multiplier

    if not is_safe:
        decision.proposed_action = "ABORT_UNSAFE"

    return {
        "decision": decision,
        "is_safe_to_execute": is_safe,
        "guardrail_warnings": " | ".join(warnings),
        "rl_context": rl_context,
        "messages": [f"Risk Evaluated. Outcome: {decision.proposed_action}. Safe: {is_safe}"]
    }
