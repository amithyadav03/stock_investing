import requests
from core.config import settings

def send_telegram_trade_proposal(
    proposal_id: int, symbol: str, action: str, rationale: str,
    entry: float, sl: float, tp: float, holding_days: int = 0
):
    """
    Sends a rich markdown message to the configured telegram chat
    with inline buttons to Approve or Reject the trade.
    """
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        print(f"[MOCK TELEGRAM] Would send to Telegram: {action} {symbol} at {entry}")
        return True

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"

    # Risk/reward calculation for informational display
    if action == "BUY" and entry > 0:
        risk = round(entry - sl, 2)
        reward = round(tp - entry, 2)
        rr_ratio = round(reward / risk, 1) if risk > 0 else "N/A"
    else:
        rr_ratio = "N/A"

    msg_text = (
        f"🚨 *New AI Trade Proposal* 🚨\n\n"
        f"*Symbol*: `{symbol}`\n"
        f"*Action*: `{action}`\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"*Entry*:       ₹{entry}\n"
        f"*Stop Loss*:   ₹{sl}\n"
        f"*Take Profit*: ₹{tp}\n"
        f"*Timeframe*:   ~{holding_days} days\n"
        f"*Risk/Reward*: 1:{rr_ratio}\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"*Rationale*: {rationale[:500]}"
    )

    reply_markup = {
        "inline_keyboard": [[
            {"text": "✅ Approve", "callback_data": f"APPROVE_{proposal_id}"},
            {"text": "❌ Reject",  "callback_data": f"REJECT_{proposal_id}"}
        ]]
    }

    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": msg_text,
        "parse_mode": "Markdown",
        "reply_markup": reply_markup
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("[Telegram] Proposal sent successfully.")
    except Exception as e:
        print(f"[Telegram] Failed to send: {e}")

def notify_error(symbol: str, error_msg: str):
    """Sends a technical error notification to Telegram."""
    text = (
        f"⚠️ **TECHNICAL ERROR IN ANALYSIS**\n\n"
        f"**Symbol**: {symbol}\n"
        f"**Status**: Failed to generate trade decision.\n\n"
        f"**Rationale**: {error_msg}\n\n"
        f"_{symbol} skipped for safety._"
    )
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"[Telegram] Error alert sent for {symbol}.")
    except Exception as e:
        print(f"[Telegram] Failed to send error alert: {e}")
