"""
Position Monitor — runs periodically during market hours to evaluate open positions.

Run manually:  python scripts/monitor_positions.py
Schedule with cron (9:30 AM to 3:30 PM IST, every 30 min):
  */30 9-15 * * 1-5 cd /path/to/stock_investing && python scripts/monitor_positions.py
Or use `launch.py` which auto-schedules this.
"""

import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.position_tracker import get_open_positions, get_position_with_proposal
from core.telegram_bot import send_portfolio_summary, notify_error, send_exit_alert
from agents.exit_monitor import evaluate_exit
from db.schema import init_db, SessionLocal, PositionMonitorLog
from core.config import settings


def is_market_hours() -> bool:
    """Returns True if current IST time is within NSE market hours (9:15–15:30)."""
    from datetime import timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    if now.weekday() >= 5:  # Weekend
        return False
    open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_time <= now <= close_time


def run_monitor(force: bool = False):
    """
    Evaluates all open positions.
    force=True bypasses market hours check (for testing).
    """
    init_db()

    if not force and not is_market_hours():
        print(f"[Monitor] Outside market hours ({datetime.now().strftime('%H:%M')}). Skipping.")
        return

    positions = get_open_positions()
    print(f"\n[Monitor] {datetime.now().strftime('%Y-%m-%d %H:%M')} — Checking {len(positions)} open position(s).")

    if not positions:
        print("[Monitor] No open positions. Nothing to monitor.")
        return

    # Send daily portfolio summary at market open (9:15–9:45)
    from datetime import timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist)
    if 9 <= now_ist.hour <= 9 and now_ist.minute <= 45:
        send_portfolio_summary(positions)

    for pos in positions:
        symbol = pos['symbol']
        try:
            data = get_position_with_proposal(pos['execution_id'])
            if not data:
                print(f"  [{symbol}] Could not fetch position data. Skipping.")
                continue

            exit_decision = evaluate_exit(
                proposal_id=data["proposal_id"],
                symbol=data["symbol"],
                direction=data["direction"],
                entry_price=data["entry_price"],
                stop_loss=data["stop_loss"],
                take_profit=data["take_profit"],
                quantity=data["quantity"],
                entry_rationale=data["entry_rationale"],
                entry_time=data["entry_time"],
            )

            # Log evaluation
            session = SessionLocal()
            try:
                log = PositionMonitorLog(
                    execution_id=data["execution_id"],
                    symbol=symbol,
                    action=exit_decision.action,
                    urgency=exit_decision.urgency,
                    current_price=data["current_price"],
                    pnl_pct=data["pnl_pct"],
                    new_stop_loss=exit_decision.new_stop_loss,
                    rationale=exit_decision.rationale,
                )
                session.add(log)
                session.commit()
            finally:
                session.close()

            pnl_str = f"{'+' if data['pnl_pct'] >= 0 else ''}{data['pnl_pct']:.2f}%"
            print(f"  [{symbol}] P&L={pnl_str}, Days={data['days_held']} → {exit_decision.action} ({exit_decision.urgency})")

            if exit_decision.action in ("EXIT_NOW", "TRAIL_SL"):
                send_exit_alert(
                    execution_id=data["execution_id"],
                    symbol=symbol,
                    action=exit_decision.action,
                    current_price=data["current_price"],
                    pnl_pct=data["pnl_pct"],
                    rationale=exit_decision.rationale,
                    new_sl=exit_decision.new_stop_loss,
                    urgency=exit_decision.urgency,
                )

        except Exception as e:
            print(f"  [{symbol}] Monitor error: {e}")
            notify_error(symbol, f"Position monitor failed: {e}")


if __name__ == "__main__":
    force_run = "--force" in sys.argv
    run_monitor(force=force_run)
