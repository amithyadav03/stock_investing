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

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    
    # Langfuse v4 expects keys in os.environ. 
    # We set them explicitly from settings to ensure everything connects.
    if settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
        if settings.LANGFUSE_HOST:
            os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST
            
        langfuse_handler = LangfuseCallbackHandler()
        print(f"[Langfuse] ✅ Tracing active → {os.environ.get('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
    else:
        langfuse_handler = None
        print("[Langfuse] ⚠️  Keys not set in .env — tracing disabled.")
except Exception as e:
    langfuse_handler = None
    print(f"[Langfuse] ❌ Failed to initialise handler: {e}")

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
    with open(path, "r", encoding="utf-8") as f:
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
    macro_raw = fundamental_news_tool.get_macro_context()
    
    # DYNAMIC MACO SYNTHESIS (NEW)
    macro = None
    llm = get_llm()
    if llm:
        try:
            # We use a structured output to classify the current market heartbeat
            sys_template, user_template = load_prompt("macro_analyst.txt")
            structured_llm = llm.with_structured_output(MacroContext)
            
            prompt = user_template.format(
                index_performance=macro_raw.get('index_performance'),
                headlines=chr(10).join(macro_raw.get('headlines', []))
            )
            
            macro = structured_llm.invoke([
                SystemMessage(content=sys_template), 
                HumanMessage(content=prompt)
            ])
            print(f"[Macro Heartbeat] Determined Regime: {macro.sentiment_enum} ({macro.risk_multiplier}x)")
        except Exception as e:
            # Rule 3 Compliance: Log exactly what happened and notify potential state degradation
            print(f"[Macro ERROR] Global Synthesis failed: {e}")
            macro = MacroContext(
                sentiment_enum="NEUTRAL", 
                risk_multiplier=1.0, 
                summary=f"!! MACRO FAILURE !! {e}. Defaulting to Neutral for safety."
            )

    return {
        "fundamental_analysis": fundamentals,
        "sentiment_analysis": sentiment,
        "macro_context": macro,
        "messages": [f"Fundamentals fetched. Macro Sentiment: {macro.sentiment_enum if macro else 'UNKNOWN'}."]
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
            # Tracing is now handled globally by the Graph invoke call in main.py
            decision = structured_llm.invoke(
                [SystemMessage(content=sys_template), HumanMessage(content=prompt)]
            )
        except Exception as e:
            print(f"LLM Parsing failed: {e}")
            decision = RiskDecision(
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
                final_rationale=f"AI Reasoning Error: {e}"
            )
            
    if not decision: 
        decision = RiskDecision(
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
            final_rationale="Reasoning Engine Failure: AI returned no decision."
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

    risk_per_tier = strategy_risk.get("risk_per_tier", {"HIGH": 0.015, "MEDIUM": 0.010, "LOW": 0.005})
    
    # Force risk percentage based strictly on LLM Conviction
    if decision.proposed_action in ["BUY", "SELL"]:
        tier = decision.conviction_tier.upper() if decision.conviction_tier else "LOW"
        decision.risk_percentage = risk_per_tier.get(tier, 0.005)

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
