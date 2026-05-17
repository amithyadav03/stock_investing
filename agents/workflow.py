"""
LangGraph workflow — strategy-aware routing.
Swing and positional both use the same nodes with different strategy_type in state.
Graph: START → parallel_analysis → conviction_filter → risk_manager → END
"""

import concurrent.futures
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.nodes import (
    technical_analyst_node,
    fundamental_analyst_node,
    conviction_filter_node,
    risk_manager_node,
)


def parallel_analysis_node(state: AgentState) -> dict:
    """Runs technical and fundamental analysis concurrently."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tech_future = pool.submit(technical_analyst_node, state)
        fund_future = pool.submit(fundamental_analyst_node, state)
        tech_result = tech_future.result()
        fund_result = fund_future.result()

    merged: dict = {}
    merged.update(tech_result)
    merged.update(fund_result)
    merged["messages"] = tech_result.get("messages", []) + fund_result.get("messages", [])
    return merged


def build_trading_graph():
    """
    Graph:
      START → parallel_analysis (technical + fundamental concurrently)
            → conviction_filter (0-100 scoring, gates low-quality candidates)
            → risk_manager (final decision + guardrails)
            → END
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("parallel_analysis", parallel_analysis_node)
    workflow.add_node("conviction_filter", conviction_filter_node)
    workflow.add_node("risk_manager", risk_manager_node)

    workflow.set_entry_point("parallel_analysis")
    workflow.add_edge("parallel_analysis", "conviction_filter")
    workflow.add_edge("conviction_filter", "risk_manager")
    workflow.add_edge("risk_manager", END)

    return workflow.compile()


trading_agent_app = build_trading_graph()
