"""
Telegram notification hub — all user-facing messages route through here.
"""

import requests
from core.config import settings


def _send(payload: dict) -> bool:
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        print(f"[Telegram MOCK] {payload.get('text', '')[:120]}")
        return True
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[Telegram] Send failed: {e}")
        return False


def send_telegram_trade_proposal(
    proposal_id: int,
    symbol: str,
    action: str,
    rationale: str,
    entry: float,
    sl: float,
    tp: float,
    holding_days: int = 0,
    conviction: str = "MEDIUM",
    win_prob: int = 0,
    technical_narrative: str = "",
    strategy_type: str = "swing",
    conviction_score: int = 0,
    research_summary: str = "",
) -> bool:
    rr_ratio = "N/A"
    if action == "BUY" and entry > 0 and sl > 0:
        risk = round(entry - sl, 2)
        reward = round(tp - entry, 2)
        rr_ratio = f"1:{round(reward / risk, 1)}" if risk > 0 else "N/A"
    elif action == "SELL" and entry > 0 and sl > 0:
        risk = round(sl - entry, 2)
        reward = round(entry - tp, 2)
        rr_ratio = f"1:{round(reward / risk, 1)}" if risk > 0 else "N/A"

    conviction_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "⚠️"}.get(conviction.upper(), "📊")
    strategy_emoji = {"swing": "🔄", "positional": "📐", "value": "💎"}.get(strategy_type, "📊")
    mode_tag = "📄 PAPER" if settings.PAPER_MODE else "💰 LIVE"

    msg = (
        f"🚨 *New Trade Proposal* {strategy_emoji} [{strategy_type.upper()}] {mode_tag}\n\n"
        f"*Symbol*: `{symbol}`\n"
        f"*Action*: `{action}`\n"
        f"*Conviction*: {conviction_emoji} `{conviction}` | Score: `{conviction_score}/100` | Win prob: {win_prob}%\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"*Entry*:       ₹{entry}\n"
        f"*Stop Loss*:   ₹{sl}\n"
        f"*Take Profit*: ₹{tp}\n"
        f"*Hold*:        ~{holding_days} days\n"
        f"*Risk/Reward*: {rr_ratio}\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"*Rationale*:\n{rationale[:500]}\n"
    )
    if research_summary:
        msg += f"\n*Research*:\n{research_summary[:250]}\n"
    if technical_narrative:
        msg += f"\n*Chart*: {technical_narrative[:200]}\n"

    msg += "\n_Approve before 9:00 AM for AMO execution tomorrow._"

    reply_markup = {
        "inline_keyboard": [[
            {"text": "✅ Approve AMO",   "callback_data": f"APPROVE_{proposal_id}"},
            {"text": "❌ Reject",        "callback_data": f"REJECT_{proposal_id}"},
            {"text": "🔍 Research",      "callback_data": f"RESEARCH_{proposal_id}"},
        ]]
    }
    return _send({
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown",
        "reply_markup": reply_markup,
    })


def send_exit_alert(
    execution_id: int,
    symbol: str,
    action: str,
    current_price: float,
    pnl_pct: float,
    rationale: str,
    new_sl: float = None,
    urgency: str = "NORMAL",
) -> bool:
    urgency_prefix = "🚨 *URGENT* " if urgency == "URGENT" else "📊 "
    action_emoji = {"EXIT_NOW": "🔴", "TRAIL_SL": "🔵", "HOLD": "🟢"}.get(action, "⚪")
    pnl_str = f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%"
    pnl_emoji = "✅" if pnl_pct >= 0 else "❌"

    msg = (
        f"{urgency_prefix}*Position Alert: {symbol}* {action_emoji}\n\n"
        f"*Action*: `{action}`\n"
        f"*Price*: ₹{current_price}\n"
        f"*P&L*: {pnl_emoji} `{pnl_str}`\n"
    )
    if new_sl and action == "TRAIL_SL":
        msg += f"*New Stop Loss*: ₹{new_sl}\n"
    msg += f"\n*Rationale*: {rationale[:400]}"

    if action == "EXIT_NOW":
        reply_markup = {
            "inline_keyboard": [[
                {"text": "✅ Exit Now",      "callback_data": f"EXIT_{execution_id}"},
                {"text": "⏸ Hold for now",  "callback_data": f"HOLD_{execution_id}"},
            ]]
        }
    elif action == "TRAIL_SL" and new_sl:
        reply_markup = {
            "inline_keyboard": [[
                {"text": f"✅ Trail SL to ₹{new_sl}", "callback_data": f"TRAILSL_{execution_id}_{new_sl}"},
                {"text": "⏸ Keep current SL",         "callback_data": f"HOLD_{execution_id}"},
            ]]
        }
    else:
        reply_markup = None

    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown",
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    return _send(payload)


def send_portfolio_advice(advice_list: list) -> bool:
    """Sends AI advice on existing Kite holdings."""
    if not advice_list:
        return True

    urgent = [a for a in advice_list if a.get("urgency") == "URGENT" or a.get("action") in ("EXIT", "ADD_MORE")]
    normal = [a for a in advice_list if a not in urgent]

    lines = ["💼 *Portfolio Advice Update*\n"]
    action_emoji = {"HOLD": "⏸ HOLD", "ADD_MORE": "➕ ADD MORE", "EXIT": "🚨 EXIT"}

    for a in urgent:
        emoji = "🚨" if a.get("action") == "EXIT" else "➕"
        lines.append(
            f"{emoji} *{a['symbol']}* — {action_emoji.get(a['action'], a['action'])}\n"
            f"P&L: {'+' if a.get('pnl_pct',0) >= 0 else ''}{a.get('pnl_pct',0):.1f}%\n"
            f"_{a.get('rationale', '')[:200]}_\n"
        )

    if normal:
        lines.append("*Routine Holdings*:")
        for a in normal[:5]:
            lines.append(
                f"⏸ `{a['symbol']}` {'+' if a.get('pnl_pct',0) >= 0 else ''}{a.get('pnl_pct',0):.1f}% — HOLD"
            )

    return _send({
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": "\n".join(lines),
        "parse_mode": "Markdown",
    })


def send_watchlist_update(new_candidates: list) -> bool:
    """Notifies about new high-conviction candidates added to watchlist."""
    if not new_candidates:
        return True
    lines = ["🔍 *Watchlist Updated*\n"]
    for c in new_candidates[:5]:
        lines.append(
            f"• `{c['symbol']}` [{c.get('strategy_type','swing')}] | "
            f"Score: {c.get('conviction_score',0)}/100 | {c.get('direction','BUY')} @ ₹{c.get('proposed_entry',0)}"
        )
    return _send({
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": "\n".join(lines),
        "parse_mode": "Markdown",
    })


def send_portfolio_summary(positions: list) -> bool:
    """Daily portfolio snapshot."""
    if not positions:
        msg = "📋 *Portfolio Summary*\n\nNo open positions."
    else:
        lines = ["📋 *Portfolio Summary*\n"]
        total_pnl = 0.0
        for p in positions:
            pnl = p.get("pnl_pct", 0)
            total_pnl += pnl
            emoji = "✅" if pnl >= 0 else "❌"
            lines.append(
                f"{emoji} `{p['symbol']}` {p['direction']} | "
                f"₹{p['entry_price']} → ₹{p['current_price']} | "
                f"`{'+' if pnl >= 0 else ''}{pnl:.1f}%` | {p['days_held']}d"
            )
        avg_pnl = total_pnl / len(positions)
        lines.append(f"\n*Open Positions*: {len(positions)}")
        lines.append(f"*Avg P&L*: `{'+' if avg_pnl >= 0 else ''}{avg_pnl:.1f}%`")
        msg = "\n".join(lines)
    return _send({"chat_id": settings.TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})


def notify_error(symbol: str, error_msg: str) -> bool:
    msg = (
        f"⚠️ *Analysis Error*\n\n"
        f"*Symbol*: `{symbol}`\n"
        f"*Error*: {error_msg[:300]}\n\n"
        f"_{symbol} skipped for safety._"
    )
    return _send({"chat_id": settings.TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})


def notify_circuit_breaker(reason: str) -> bool:
    msg = (
        f"🛑 *CIRCUIT BREAKER TRIGGERED*\n\n"
        f"*Reason*: {reason}\n\n"
        f"_All new trade entries are paused. Review your positions._"
    )
    return _send({"chat_id": settings.TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})


def notify_paper_trade_executed(symbol: str, direction: str, entry: float, qty: int) -> bool:
    msg = (
        f"📄 *Paper Trade Opened*\n\n"
        f"`{symbol}` {direction} x{qty} @ ₹{entry}\n"
        f"_(Paper mode — no real order placed)_"
    )
    return _send({"chat_id": settings.TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
