"""
EOD Summary — sent at 5:00 PM IST post-market.
Covers: day P&L, closed positions, new trade proposals for tomorrow (with AMO buttons).
"""

import json
import os
from datetime import datetime
from core.config import settings
from core.telegram_bot import _send, send_telegram_trade_proposal
from core.capital_manager import capital_manager
from core.paper_trader import get_paper_performance, log_daily_performance
from tools.nse_data import get_nifty_indices_performance
from db.schema import SessionLocal, PaperTrade, TradeExecution, TradeProposal, Watchlist


def build_eod_summary() -> str:
    today = datetime.now().strftime("%d %b %Y")
    lines = [f"📈 *EOD SUMMARY — {today}*\n"]

    session = SessionLocal()
    try:
        # ── Performance ────────────────────────────────────────────────────────
        perf = get_paper_performance()
        mode = "PAPER" if settings.PAPER_MODE else "LIVE"
        cap = capital_manager.get_summary()

        if isinstance(perf, dict) and perf.get("status") != "no_trades":
            total_return = perf.get("total_return_pct", 0)
            nifty = perf.get("nifty_return_pct", 0)
            alpha = perf.get("alpha_pct", 0)

            lines.append(f"*[{mode}] Portfolio Performance*")
            lines.append(f"Capital: ₹{cap['total_capital']:,.0f} | Deployed: ₹{cap['deployed_capital']:,.0f}")
            lines.append(
                f"Unrealized P&L: `{'+'  if perf.get('unrealized_pnl',0) >= 0 else ''}₹{perf.get('unrealized_pnl',0):,.0f}`"
            )

            # Nifty comparison
            nifty_perf = get_nifty_indices_performance()
            nifty_day = nifty_perf.get("Nifty 50", {}).get("day_change_pct", 0)
            lines.append(f"Nifty 50 today: `{'+'  if nifty_day >= 0 else ''}{nifty_day}%`")

        # ── Closed Today ───────────────────────────────────────────────────────
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        if settings.PAPER_MODE:
            closed_today = session.query(PaperTrade).filter(
                PaperTrade.status == "CLOSED",
                PaperTrade.exit_time >= today_start
            ).all()
        else:
            closed_today = session.query(TradeExecution).filter(
                TradeExecution.status == "CLOSED",
                TradeExecution.exit_time >= today_start
            ).all()

        if closed_today:
            lines.append(f"\n*Closed Today ({len(closed_today)})*")
            for t in closed_today:
                pnl = getattr(t, 'realized_pnl_pct', 0) or 0
                pnl_inr = getattr(t, 'realized_pnl', 0) or 0
                emoji = "✅" if pnl >= 0 else "❌"
                reason = getattr(t, 'exit_reason', 'MANUAL') or 'MANUAL'
                lines.append(
                    f"{emoji} `{t.symbol}` {t.direction} | "
                    f"₹{t.entry_price}→₹{t.exit_price} | "
                    f"`{'+' if pnl >= 0 else ''}{pnl:.2f}%` (₹{pnl_inr:+.0f}) | {reason}"
                )

        # ── Open Positions ─────────────────────────────────────────────────────
        if settings.PAPER_MODE:
            open_trades = session.query(PaperTrade).filter(PaperTrade.status == "OPEN").all()
        else:
            open_trades = session.query(TradeExecution).filter(TradeExecution.status == "OPEN").all()

        if open_trades:
            lines.append(f"\n*Open Positions ({len(open_trades)})*")
            for t in open_trades:
                cp = getattr(t, 'current_price', None) or t.entry_price
                pnl = round((cp - t.entry_price) / t.entry_price * 100, 2) if t.entry_price else 0
                days = (datetime.utcnow() - t.entry_time).days if t.entry_time else 0
                lines.append(
                    f"{'✅' if pnl >= 0 else '🔴'} `{t.symbol}` | ₹{cp} | "
                    f"`{'+' if pnl >= 0 else ''}{pnl:.1f}%` | Day {days}"
                )

        # ── Watchlist ──────────────────────────────────────────────────────────
        watchlist = session.query(Watchlist).filter(
            Watchlist.status == "ACTIVE"
        ).order_by(Watchlist.conviction_score.desc()).limit(3).all()
        if watchlist:
            lines.append(f"\n🔍 *Watchlist — waiting for slot*")
            for w in watchlist:
                lines.append(f"  • `{w.symbol}` [{w.strategy_type}] Score: {w.conviction_score}/100")

    finally:
        session.close()

    lines.append("\n_New proposals sent separately. Approve before 9:00 AM for tomorrow's AMO._")
    return "\n".join(lines)


def send_eod_summary_and_proposals():
    """Sends EOD summary, then individual proposal messages for any new PENDING proposals."""
    print("[EOD] Building EOD summary...")

    # Log daily performance
    try:
        log_daily_performance()
    except Exception as e:
        print(f"[EOD] Performance log failed: {e}")

    # Send summary
    try:
        msg = build_eod_summary()
        _send({
            "chat_id": settings.TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "Markdown",
        })
    except Exception as e:
        print(f"[EOD] Summary send failed: {e}")

    # Send individual proposal messages for new proposals created today
    session = SessionLocal()
    try:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        new_proposals = session.query(TradeProposal).filter(
            TradeProposal.status == "PENDING",
            TradeProposal.created_at >= today_start,
        ).order_by(TradeProposal.conviction_score.desc()).all()

        print(f"[EOD] Sending {len(new_proposals)} new proposal(s)...")
        for p in new_proposals:
            try:
                send_telegram_trade_proposal(
                    proposal_id=p.id,
                    symbol=p.symbol,
                    action=p.direction,
                    rationale=p.rationale or "",
                    entry=p.proposed_price,
                    sl=p.stop_loss,
                    tp=p.take_profit,
                    holding_days=p.expected_holding_days or 0,
                    conviction=p.conviction_tier or "MEDIUM",
                    win_prob=p.win_probability or 0,
                    technical_narrative=p.technical_narrative or "",
                    strategy_type=p.strategy_type or "swing",
                    conviction_score=p.conviction_score or 0,
                    research_summary=p.research_summary or "",
                )
            except Exception as e:
                print(f"[EOD] Proposal send failed for {p.symbol}: {e}")
    finally:
        session.close()

    print("[EOD] EOD summary and proposals sent.")
