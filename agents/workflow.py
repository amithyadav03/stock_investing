from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.nodes import technical_analyst_node, fundamental_sentiment_node, risk_manager_node

def build_trading_graph():
    # Initialize State Graph
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("technical_analyst", technical_analyst_node)
    workflow.add_node("fundamental_analyst", fundamental_sentiment_node)
    workflow.add_node("risk_manager", risk_manager_node)
    
    # Define edges: Start -> Technical & Fundamental (Parallel execution possible, but let's do sequential for simplicity)
    workflow.set_entry_point("technical_analyst")
    workflow.add_edge("technical_analyst", "fundamental_analyst")
    workflow.add_edge("fundamental_analyst", "risk_manager")
    workflow.add_edge("risk_manager", END)
    
    # Compile
    app = workflow.compile()
    return app

trading_agent_app = build_trading_graph()
