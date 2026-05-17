"""
Morning Brief — sent at 8:30 AM IST on trading days.
Covers: overnight global cues, macro, portfolio snapshot, open trades, watchlist, pending AMOs.
"""

from datetime import datetime
from core.config import settings
from core.telegram_bot import _send
from tools.nse_data import get_global_cues, get_fii_dii_data, format_global_cues_text
from tools.kite_portfolio import get_holdings, sync_holdings_to_db
from agents.portfolio_advisor import advise_all_holdings, get_portfolio_advice_from_db
from core.capital_manager import capital_manager
from db.schema import SessionLocal, TradeProposal, PaperTrade, TradeExecution, Watchlist


def build_morning_brief() -> str:
    today = datetime.now().strftime("%d %b %Y, %a")
    lines = [f"🌅 *MORNING BRIEF — {today}*\n"]

    # ── Global Cues ────────────────────────────────────────────────────────────
    lines.append("🌍 *Overnight Global Cues*")
    cues = get_global_cues()
    lines.append(format_global_cues_text(cues) if cues else "_Data unavailable_")

    # ── FII/DII ────────────────────────────────────────────────────────────────
    fii_dii = get_fii_dii_data()
    if not fii_dii.get("error") and fii_dii.get("summary"):
        lines.append(f"\n🏦 *Institutional Flows (5d)*\n{fii_dii['summary']}")

    # ── Capital Summary ────────────────────────────────────────────────────────
    cap = capital_manager.get_summary()
    mode_badge = "📄 PAPER" if cap["mode"] == "PAPER" else "💰 LIVE"
    lines.append(
        f"\n{mode_badge} *Capital*: ₹{cap['total_capital']:,.0f} | "
        f"Deployed: ₹{cap['deployed_capital']:,.0f} ({cap['utilization_pct']}%) | "
        f"Available: ₹{cap['available_capital']:,.0f} | "
        f"Slots: {cap['slots_available']}/{cap['max_positions']}"
    )

    # ── Open Trades ────────────────────────────────────────────────────────────
    session = SessionLocal()
    try:
        if settings.PAPER_MODE:
            open_trades = session.query(PaperTrade).filter(PaperTrade.status == "OPEN").all()
        else:
            open_trades = session.query(TradeExecution).filter(TradeExecution.status == "OPEN").all()

        if open_trades:
            lines.append(f"\n📊 *Open Trades ({len(open_trades)})*")
            for t in open_trades:
                current = t.current_price if hasattr(t, 'current_price') and t.current_price else t.entry_price
                if current and t.entry_price:
                    pnl = round((current - t.entry_price) / t.entry_price * 100, 2)
                    pnl_str = f"+{pnl}%" if pnl >= 0 else f"{pnl}%"
                    emoji = "✅" if pnl >= 0 else "🔴"
                    days = (datetime.utcnow() - t.entry_time).days if t.entry_time else 0
                    strat = getattr(t, 'strategy_type', 'swing')
                    lines.append(
                        f"{emoji} `{t.symbol}` {t.direction} | "
                        f"₹{t.entry_price} → ₹{current} | `{pnl_str}` | Day {days} | {strat}"
                    )
        else:
            lines.append("\n📊 *Open Trades*: None")

        # ── Pending AMO Approvals ──────────────────────────────────────────────
        pending = session.query(TradeProposal).filter(
            TradeProposal.status == "PENDING"
        ).order_by(TradeProposal.created_at.desc()).limit(5).all()

        if pending:
            lines.append(f"\n⏰ *Pending Approvals ({len(pending)}) — AMO cutoff 9:00 AM*")
            for p in pending:
                lines.append(
                    f"  • `{p.symbol}` {p.direction} @ ₹{p.proposed_price} "
                    f"| SL ₹{p.stop_loss} | TP ₹{p.take_profit} "
                    f"| [{p.strategy_type}] [APPROVE: /approve_{p.id}]"
                )

        # ── Watchlist ──────────────────────────────────────────────────────────
        watchlist = session.query(Watchlist).filter(
            Watchlist.status == "ACTIVE"
        ).order_by(Watchlist.conviction_score.desc()).limit(5).all()

        if watchlist:
            lines.append(f"\n🔍 *Watchlist ({len(watchlist)} candidates)*")
            for w in watchlist:
                lines.append(
                    f"  • `{w.symbol}` [{w.strategy_type}] | Score: {w.conviction_score}/100 "
                    f"| {w.direction} @ ₹{w.proposed_entry}"
                )

    finally:
        session.close()

    # ── Existing Portfolio ─────────────────────────────────────────────────────
    holdings = get_holdings()
    if holdings:
        lines.append(f"\n💼 *Kite Portfolio ({len(holdings)} holdings)*")
        advice_list = get_portfolio_advice_from_db()
        advice_map = {a["symbol"]: a for a in advice_list}
        total_pnl = sum(h.get("pnl_pct", 0) for h in holdings)
        for h in holdings[:8]:  # Show max 8 to keep message readable
            pnl = h.get("pnl_pct", 0)
            emoji = "✅" if pnl >= 0 else "🔴"
            advice = advice_map.get(h["symbol"], {}).get("action", "HOLD")
            advice_emoji = {"HOLD": "⏸", "ADD_MORE": "➕", "EXIT": "🚨"}.get(advice, "⏸")
            lines.append(
                f"{emoji} `{h['symbol']}` {h['quantity']}sh | "
                f"₹{h['avg_price']} → ₹{h.get('current_price', h['avg_price'])} | "
                f"`{'+' if pnl >= 0 else ''}{pnl:.1f}%` {advice_emoji}{advice}"
            )
        if len(holdings) > 8:
            lines.append(f"  _...and {len(holdings)-8} more_")
        avg_portfolio_pnl = round(total_pnl / len(holdings), 1)
        lines.append(f"  Portfolio avg P&L: `{'+' if avg_portfolio_pnl >= 0 else ''}{avg_portfolio_pnl}%`")

    lines.append("\n_Market opens at 9:15 AM IST. Good luck today!_ 🎯")
    return "\n".join(lines)


def send_morning_brief():
    """Builds and sends the morning brief via Telegram."""
    print("[MorningBrief] Syncing holdings and building brief...")
    try:
        holdings = get_holdings()
        if holdings:
            sync_holdings_to_db(holdings)
            advise_all_holdings(holdings)
    except Exception as e:
        print(f"[MorningBrief] Holdings sync failed: {e}")

    try:
        msg = build_morning_brief()
        _send({
            "chat_id": settings.TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "Markdown",
        })
        print("[MorningBrief] Sent.")
    except Exception as e:
        print(f"[MorningBrief] Send failed: {e}")
