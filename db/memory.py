"""
ChromaDB vector memory — stores trade experiences for RL-style context retrieval.
The RL loop is: execute trade → monitor → exit → write outcome → future risk_manager reads it.
"""

import chromadb
from core.config import settings

chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)

experience_collection = chroma_client.get_or_create_collection(
    name="trade_experiences",
    metadata={"hnsw:space": "cosine"},
)


def add_experience(trade_id: str, document: str, metadatas: dict):
    """
    Stores a completed trade outcome in vector memory.
    Called after a TradeExecution is closed (exit recorded).

    Args:
        trade_id: Unique ID string (e.g. "exec_42")
        document: Rich narrative: "RELIANCE BUY at 2400, exited at 2520 (+5%) after 12 days. RSI was 58, MACD bullish. Thesis: breakout above resistance. Macro was NEUTRAL."
        metadatas: Tags for filtering e.g. {"symbol": "RELIANCE", "direction": "BUY", "outcome": "WIN", "pnl_pct": 5.0}
    """
    try:
        experience_collection.add(
            documents=[document],
            metadatas=[metadatas],
            ids=[trade_id],
        )
    except Exception as e:
        print(f"[Memory] Failed to store experience {trade_id}: {e}")


def retrieve_similar_experiences(query_text: str, n_results: int = 3) -> dict:
    """
    Semantic search over past trade experiences.
    Used by risk_manager_node to inject RL context.
    """
    try:
        count = experience_collection.count()
        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        actual_n = min(n_results, count)
        return experience_collection.query(
            query_texts=[query_text],
            n_results=actual_n,
        )
    except Exception as e:
        print(f"[Memory] Retrieval failed: {e}")
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


def record_trade_outcome(execution_id: int, symbol: str, direction: str, entry_price: float,
                         exit_price: float, exit_reason: str, pnl_pct: float,
                         days_held: int, rationale: str, conviction: str, macro: str):
    """
    Convenience wrapper — builds the narrative and stores it after exit.
    """
    outcome = "WIN" if pnl_pct > 0 else "LOSS"
    doc = (
        f"{symbol} {direction} trade: entry ₹{entry_price}, exit ₹{exit_price} "
        f"({'+' if pnl_pct >= 0 else ''}{pnl_pct}%) after {days_held} days. "
        f"Exit reason: {exit_reason}. Conviction: {conviction}. Macro: {macro}. "
        f"Original thesis: {rationale[:200]}"
    )
    add_experience(
        trade_id=f"exec_{execution_id}",
        document=doc,
        metadatas={
            "symbol": symbol,
            "direction": direction,
            "outcome": outcome,
            "exit_reason": exit_reason,
            "pnl_pct": pnl_pct,
            "conviction": conviction,
            "macro": macro,
        },
    )
    print(f"[Memory] Stored experience for exec_{execution_id}: {symbol} {outcome} ({pnl_pct}%)")
