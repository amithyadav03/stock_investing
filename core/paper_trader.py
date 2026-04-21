"""
Paper trading simulator — full trade lifecycle without real Kite orders.
Tracks P&L vs Nifty 50 benchmark for performance validation.
"""

from datetime import datetime
from typing import Optional
from db.schema import SessionLocal, PaperTrade, PerformanceLog, TradeProposal


def paper_execute(
    proposal_id: Optional[int],
    symbol: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    quantity: int,
    conviction_tier: str = "MEDIUM",
    conviction_score: int = 0,
    strategy_type: str = "swing",
) -> PaperTrade:
    """Create a paper trade record (called instead of Kite order in paper mode)."""
    session = SessionLocal()
    try:
        trade = PaperTrade(
            proposal_id=proposal_id,
            symbol=symbol,
            direction=direction,
            strategy_type=strategy_type,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.utcnow(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=entry_price,
            conviction_tier=conviction_tier,
            conviction_score=conviction_score,
            status="OPEN",
        )
        session.add(trade)
        session.commit()
        session.refresh(trade)
        print(f"[PaperTrader] OPENED: {symbol} {direction} x{quantity} @ ₹{entry_price}")
        return trade
    finally:
        session.close()


def mark_paper_positions_to_market() -> list[dict]:
    """
    Fetch current prices for all open paper trades and update current_price.
    Checks SL/TP breaches and auto-closes if hit.
    Returns list of positions that were auto-closed.
    """
    from tools.market_data import market_data_tool

    session = SessionLocal()
    auto_closed = []
    try:
        open_trades = session.query(PaperTrade).filter(PaperTrade.status == "OPEN").all()
        for trade in open_trades:
            try:
                price = market_data_tool.get_current_price(trade.symbol)
                if price <= 0:
                    continue
                trade.current_price = price

                closed = False
                if trade.direction == "BUY":
                    if price <= trade.stop_loss:
                        _close_paper_trade(session, trade, price, "STOP_LOSS")
                        closed = True
                    elif trade.take_profit and price >= trade.take_profit:
                        _close_paper_trade(session, trade, price, "TAKE_PROFIT")
                        closed = True
                elif trade.direction == "SELL":
                    if price >= trade.stop_loss:
                        _close_paper_trade(session, trade, price, "STOP_LOSS")
                        closed = True
                    elif trade.take_profit and price <= trade.take_profit:
                        _close_paper_trade(session, trade, price, "TAKE_PROFIT")
                        closed = True

                if closed:
                    auto_closed.append({"symbol": trade.symbol, "direction": trade.direction,
                                        "exit_price": price, "reason": trade.exit_reason})
            except Exception as e:
                print(f"[PaperTrader] MTM failed for {trade.symbol}: {e}")

        session.commit()
    finally:
        session.close()

    return auto_closed


def _close_paper_trade(session, trade: PaperTrade, exit_price: float, reason: str):
    trade.exit_price = exit_price
    trade.exit_time = datetime.utcnow()
    trade.exit_reason = reason
    trade.status = "CLOSED"
    if trade.entry_price and trade.entry_price > 0:
        if trade.direction == "BUY":
            pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            pnl_pct = ((trade.entry_price - exit_price) / trade.entry_price) * 100
        trade.realized_pnl_pct = round(pnl_pct, 2)
        trade.realized_pnl = round((exit_price - trade.entry_price) * trade.quantity, 2)
    print(f"[PaperTrader] CLOSED: {trade.symbol} @ ₹{exit_price} | {reason} | P&L: {trade.realized_pnl_pct:.2f}%")


def close_paper_position(paper_trade_id: int, exit_price: float, reason: str = "MANUAL") -> bool:
    session = SessionLocal()
    try:
        trade = session.query(PaperTrade).filter(PaperTrade.id == paper_trade_id).first()
        if not trade or trade.status != "OPEN":
            return False
        _close_paper_trade(session, trade, exit_price, reason)
        session.commit()
        return True
    finally:
        session.close()


def get_paper_performance() -> dict:
    """Returns cumulative paper trading P&L, win rate, and benchmark comparison."""
    import yfinance as yf
    session = SessionLocal()
    try:
        closed = session.query(PaperTrade).filter(PaperTrade.status == "CLOSED").all()
        open_trades = session.query(PaperTrade).filter(PaperTrade.status == "OPEN").all()

        if not closed and not open_trades:
            return {"status": "no_trades", "message": "No paper trades yet."}

        total_realized_pnl = sum(t.realized_pnl or 0 for t in closed)
        wins = [t for t in closed if (t.realized_pnl or 0) > 0]
        losses = [t for t in closed if (t.realized_pnl or 0) <= 0]
        win_rate = round(len(wins) / len(closed) * 100, 1) if closed else 0.0
        avg_win = round(sum(t.realized_pnl_pct or 0 for t in wins) / len(wins), 2) if wins else 0.0
        avg_loss = round(sum(t.realized_pnl_pct or 0 for t in losses) / len(losses), 2) if losses else 0.0

        unrealized_pnl = 0.0
        for t in open_trades:
            if t.current_price and t.entry_price:
                if t.direction == "BUY":
                    unrealized_pnl += (t.current_price - t.entry_price) * t.quantity
                else:
                    unrealized_pnl += (t.entry_price - t.current_price) * t.quantity

        from core.capital_manager import capital_manager
        total_capital = capital_manager.get_total_capital()
        total_return_pct = round((total_realized_pnl + unrealized_pnl) / total_capital * 100, 2)

        # Nifty 50 benchmark over same period
        nifty_return_pct = 0.0
        if closed:
            try:
                oldest = min(t.entry_time for t in closed if t.entry_time)
                days = (datetime.utcnow() - oldest).days + 1
                nifty = yf.Ticker("^NSEI").history(period=f"{min(days, 365)}d")
                if len(nifty) >= 2:
                    nifty_return_pct = round(
                        (nifty['Close'].iloc[-1] - nifty['Close'].iloc[0]) / nifty['Close'].iloc[0] * 100, 2
                    )
            except Exception:
                pass

        return {
            "total_trades": len(closed),
            "open_trades": len(open_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": win_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "realized_pnl": round(total_realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_return_pct": total_return_pct,
            "nifty_return_pct": nifty_return_pct,
            "alpha_pct": round(total_return_pct - nifty_return_pct, 2),
        }
    finally:
        session.close()


def log_daily_performance():
    """Snapshot today's portfolio state into PerformanceLog."""
    import yfinance as yf
    from core.capital_manager import capital_manager

    session = SessionLocal()
    try:
        perf = get_paper_performance()
        total = capital_manager.get_total_capital()
        deployed = capital_manager.get_deployed_capital()

        nifty_change = 0.0
        try:
            nifty = yf.Ticker("^NSEI").history(period="2d")
            if len(nifty) >= 2:
                nifty_change = round(
                    (nifty['Close'].iloc[-1] - nifty['Close'].iloc[-2]) / nifty['Close'].iloc[-2] * 100, 2
                )
        except Exception:
            pass

        today_closed = session.query(PaperTrade).filter(
            PaperTrade.status == "CLOSED",
            PaperTrade.exit_time >= datetime.utcnow().replace(hour=0, minute=0, second=0)
        ).all()
        realized_today = sum(t.realized_pnl or 0 for t in today_closed)
        wins_today = len([t for t in today_closed if (t.realized_pnl or 0) > 0])
        losses_today = len(today_closed) - wins_today

        log = PerformanceLog(
            date=datetime.utcnow(),
            mode="PAPER" if capital_manager.get_summary()["mode"] == "PAPER" else "LIVE",
            total_capital=total,
            deployed_capital=deployed,
            unrealized_pnl=perf.get("unrealized_pnl", 0),
            unrealized_pnl_pct=round(perf.get("unrealized_pnl", 0) / total * 100, 2) if total else 0,
            realized_pnl_today=realized_today,
            open_positions=perf.get("open_trades", 0),
            closed_today=len(today_closed),
            wins_today=wins_today,
            losses_today=losses_today,
            nifty_change_pct=nifty_change,
        )
        session.add(log)
        session.commit()
        print(f"[PaperTrader] Daily performance logged. Unrealized P&L: ₹{perf.get('unrealized_pnl', 0):,.0f}")
    finally:
        session.close()
