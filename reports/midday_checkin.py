"""
Mid-day check-in — sent at 12:30 PM IST on trading days.
Quick pulse: open position P&L, any technical alerts, watchlist movers.
"""

from datetime import datetime
from core.config import settings
from core.telegram_bot import _send
from core.capital_manager import capital_manager
from tools.market_data import market_data_tool
from db.schema import SessionLocal, PaperTrade, TradeExecution, TradeProposal


def build_midday_checkin() -> str:
    now = datetime.now().strftime("%d %b %Y, %H:%M IST")
    lines = [f"📊 *MID-DAY CHECK-IN — {now}*\n"]

    session = SessionLocal()
    try:
        if settings.PAPER_MODE:
            open_trades = session.query(PaperTrade).filter(PaperTrade.status == "OPEN").all()
        else:
            open_trades = session.query(TradeExecution).filter(TradeExecution.status == "OPEN").all()

        if not open_trades:
            lines.append("No open positions. Awaiting setups.")
            return "\n".join(lines)

        lines.append(f"*Open Positions: {len(open_trades)}*\n")
        total_unrealized = 0.0
        alerts = []

        for t in open_trades:
            try:
                current = market_data_tool.get_current_price(t.symbol)
                if not current or current <= 0:
                    current = getattr(t, 'current_price', t.entry_price) or t.entry_price

                pnl_pct = round((current - t.entry_price) / t.entry_price * 100, 2)
                pnl_inr = round((current - t.entry_price) * t.quantity, 2)
                total_unrealized += pnl_inr
                days = (datetime.utcnow() - t.entry_time).days if t.entry_time else 0

                emoji = "✅" if pnl_pct >= 0 else "🔴"
                pnl_str = f"+{pnl_pct}%" if pnl_pct >= 0 else f"{pnl_pct}%"

                sl = getattr(t, 'stop_loss', None)
                tp = getattr(t, 'take_profit', None)
                sl_pct = round((current - sl) / sl * 100, 1) if sl and sl > 0 else None
                tp_pct = round((tp - current) / current * 100, 1) if tp and tp > 0 else None

                trade_line = (
                    f"{emoji} `{t.symbol}` | ₹{t.entry_price}→₹{current} | "
                    f"`{pnl_str}` (₹{pnl_inr:+.0f}) | Day {days}"
                )
                if sl_pct is not None:
                    trade_line += f" | SL {sl_pct:+.1f}%"
                if tp_pct is not None:
                    trade_line += f" | TP {tp_pct:+.1f}%"
                lines.append(trade_line)

                # Alert conditions
                if sl and current <= sl * 1.02:
                    alerts.append(f"⚠️ `{t.symbol}` approaching stop loss (within 2%)")
                if tp and current >= tp * 0.97:
                    alerts.append(f"🎯 `{t.symbol}` approaching take profit (within 3%)")
                if days >= settings.strategy.get("exit", {}).get("max_holding_days_swing", 30) - 3:
                    alerts.append(f"⏱ `{t.symbol}` near max holding period ({days}d)")

            except Exception as e:
                lines.append(f"⚠️ `{t.symbol}` — data error: {e}")

        cap = capital_manager.get_summary()
        pnl_str = f"+₹{total_unrealized:,.0f}" if total_unrealized >= 0 else f"-₹{abs(total_unrealized):,.0f}"
        lines.append(f"\n*Total Unrealized P&L*: `{pnl_str}`")
        lines.append(f"*Heat*: {cap['portfolio_heat_pct']}% of capital at risk")

        if alerts:
            lines.append("\n*Alerts*:")
            lines.extend(alerts)

        # Pending approvals reminder
        pending_count = session.query(TradeProposal).filter(
            TradeProposal.status == "PENDING"
        ).count()
        if pending_count:
            lines.append(f"\n📋 {pending_count} proposal(s) still pending your review.")

    finally:
        session.close()

    return "\n".join(lines)


def send_midday_checkin():
    try:
        msg = build_midday_checkin()
        _send({
            "chat_id": settings.TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "Markdown",
        })
        print("[MidDay] Check-in sent.")
    except Exception as e:
        print(f"[MidDay] Send failed: {e}")
