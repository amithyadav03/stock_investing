"""
Position tracker — queries active (OPEN) TradeExecution records and enriches with live prices.
"""

from datetime import datetime
from typing import List, Dict, Any

from db.schema import SessionLocal, TradeExecution
from tools.market_data import market_data_tool


def get_open_positions() -> List[Dict[str, Any]]:
    """Returns all currently OPEN positions with live P&L."""
    session = SessionLocal()
    try:
        positions = session.query(TradeExecution).filter(TradeExecution.status == "OPEN").all()
        result = []
        for p in positions:
            current_price = market_data_tool.get_current_price(p.symbol)
            if p.direction == "BUY":
                pnl_pct = round(((current_price - p.entry_price) / p.entry_price) * 100, 2) if p.entry_price else 0.0
            else:
                pnl_pct = round(((p.entry_price - current_price) / p.entry_price) * 100, 2) if p.entry_price else 0.0

            days_held = (datetime.now() - p.entry_time).days if p.entry_time else 0
            result.append({
                "execution_id": p.id,
                "proposal_id": p.proposal_id,
                "symbol": p.symbol,
                "direction": p.direction,
                "quantity": p.quantity,
                "entry_price": p.entry_price,
                "current_price": current_price,
                "stop_loss": None,   # fetched from proposal
                "take_profit": None,
                "pnl_pct": pnl_pct,
                "days_held": days_held,
                "entry_time": p.entry_time,
            })
        return result
    finally:
        session.close()


def get_position_with_proposal(execution_id: int) -> Dict[str, Any] | None:
    """Returns execution + proposal data merged, for exit evaluation."""
    from db.schema import TradeProposal
    session = SessionLocal()
    try:
        execution = session.query(TradeExecution).filter(TradeExecution.id == execution_id).first()
        if not execution:
            return None
        proposal = session.query(TradeProposal).filter(TradeProposal.id == execution.proposal_id).first()
        current_price = market_data_tool.get_current_price(execution.symbol)
        days_held = (datetime.now() - execution.entry_time).days if execution.entry_time else 0

        if execution.direction == "BUY":
            pnl_pct = round(((current_price - execution.entry_price) / execution.entry_price) * 100, 2)
        else:
            pnl_pct = round(((execution.entry_price - current_price) / execution.entry_price) * 100, 2)

        return {
            "execution_id": execution.id,
            "proposal_id": execution.proposal_id,
            "symbol": execution.symbol,
            "direction": execution.direction,
            "quantity": execution.quantity,
            "entry_price": execution.entry_price,
            "entry_time": execution.entry_time,
            "current_price": current_price,
            "stop_loss": proposal.stop_loss if proposal else 0.0,
            "take_profit": proposal.take_profit if proposal else 0.0,
            "pnl_pct": pnl_pct,
            "days_held": days_held,
            "entry_rationale": (proposal.rationale or "")[:400] if proposal else "",
            "conviction_tier": (proposal.conviction_tier or "LOW") if proposal else "LOW",
        }
    finally:
        session.close()


def mark_position_closed(execution_id: int, exit_price: float, exit_reason: str):
    """Closes a position and records P&L. Triggers RL memory write."""
    session = SessionLocal()
    try:
        execution = session.query(TradeExecution).filter(TradeExecution.id == execution_id).first()
        if not execution:
            return

        execution.exit_price = exit_price
        execution.exit_time = datetime.utcnow()
        execution.exit_reason = exit_reason
        execution.status = "CLOSED"

        if execution.entry_price and execution.entry_price > 0:
            if execution.direction == "BUY":
                pnl_pct = ((exit_price - execution.entry_price) / execution.entry_price) * 100
            else:
                pnl_pct = ((execution.entry_price - exit_price) / execution.entry_price) * 100
            execution.realized_pnl_pct = round(pnl_pct, 2)
            execution.realized_pnl = round((exit_price - execution.entry_price) * execution.quantity, 2)

        session.commit()

        # Write to RL memory
        try:
            from db.memory import record_trade_outcome
            from db.schema import TradeProposal
            proposal = session.query(TradeProposal).filter(TradeProposal.id == execution.proposal_id).first()
            days_held = (execution.exit_time - execution.entry_time).days if execution.entry_time else 0
            record_trade_outcome(
                execution_id=execution_id,
                symbol=execution.symbol,
                direction=execution.direction,
                entry_price=execution.entry_price,
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl_pct=execution.realized_pnl_pct or 0.0,
                days_held=days_held,
                rationale=(proposal.rationale or "") if proposal else "",
                conviction=(proposal.conviction_tier or "LOW") if proposal else "LOW",
                macro="UNKNOWN",
            )
        except Exception as e:
            print(f"[PositionTracker] RL memory write failed: {e}")
    finally:
        session.close()
