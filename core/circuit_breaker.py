"""
Circuit breaker — halts all new trade entries when equity protection thresholds are hit.
Checks: daily P&L loss limit, consecutive loss streak, max open positions, peak drawdown.

Emergency halt (exit_all_positions) is called when a circuit trips mid-session.
"""

from datetime import datetime, date
from db.schema import SessionLocal, TradeExecution, CircuitBreakerLog
from core.config import settings


class CircuitBreaker:
    def is_trading_allowed(self) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Checks conditions from strategy_config.yaml:
        - max_daily_loss_pct: halt if today's realized P&L < -X%
        - max_consecutive_losses: halt after N consecutive losses
        - max_open_positions: halt if too many active trades
        - max_drawdown_from_peak_pct: halt if equity dropped 15% from all-time high
        - intraday drawdown from live open positions
        """
        risk_cfg = settings.strategy.get("risk", {})
        max_daily_loss = risk_cfg.get("max_daily_loss_pct", 0.02)
        max_consec_losses = risk_cfg.get("max_consecutive_losses", 3)
        max_open_pos = risk_cfg.get("max_open_positions", 5)

        session = SessionLocal()
        try:
            # 1. Daily loss limit — use IST midnight boundary
            import pytz as _pytz
            _IST = _pytz.timezone("Asia/Kolkata")
            _now_ist = datetime.now(_IST)
            _today_ist_midnight_utc = _now_ist.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(_pytz.utc).replace(tzinfo=None)
            today_closed = session.query(TradeExecution).filter(
                TradeExecution.status == "CLOSED",
                TradeExecution.exit_time >= _today_ist_midnight_utc,
            ).all()
            if today_closed:
                from core.capital_manager import capital_manager
                total_capital = capital_manager.get_total_capital()
                total_pnl_abs = sum(t.realized_pnl or 0 for t in today_closed)
                total_pnl_pct = (total_pnl_abs / total_capital * 100) if total_capital > 0 else 0.0
                if total_pnl_pct < -(max_daily_loss * 100):
                    self._log(session, f"Daily loss limit hit: {total_pnl_pct:.2f}%", total_pnl_pct, None)
                    return False, f"CIRCUIT BREAKER: Daily loss {total_pnl_pct:.2f}% exceeds -{max_daily_loss*100:.0f}% limit."

            # 2. Consecutive loss streak
            recent = session.query(TradeExecution).filter(
                TradeExecution.status == "CLOSED"
            ).order_by(TradeExecution.exit_time.desc()).limit(max_consec_losses).all()
            if len(recent) == max_consec_losses and all((t.realized_pnl_pct or 0) < 0 for t in recent):
                self._log(session, f"{max_consec_losses} consecutive losses", None, max_consec_losses)
                return False, f"CIRCUIT BREAKER: {max_consec_losses} consecutive losses. System paused."

            # 3. Max open positions
            open_count = session.query(TradeExecution).filter(TradeExecution.status == "OPEN").count()
            if open_count >= max_open_pos:
                return False, f"CIRCUIT BREAKER: {open_count} positions open (max {max_open_pos}). No new entries."

            # 4. Peak-to-trough drawdown (from PerformanceLog)
            max_drawdown = risk_cfg.get("max_drawdown_from_peak_pct", 0.15)
            try:
                from db.schema import PerformanceLog
                if settings.PAPER_MODE:
                    logs = session.query(PerformanceLog).order_by(PerformanceLog.date).all()
                    if logs:
                        peak = max((l.total_capital + (l.unrealized_pnl or 0) for l in logs), default=0)
                        latest_log = logs[-1]
                        current_equity = latest_log.total_capital + (latest_log.unrealized_pnl or 0)
                        if peak > 0 and (peak - current_equity) / peak >= max_drawdown:
                            self._log(session, f"Peak drawdown breached: {(peak-current_equity)/peak:.1%}", None, None)
                            return False, f"CIRCUIT BREAKER: Drawdown {(peak-current_equity)/peak:.1%} from peak."
            except Exception as e:
                print(f"[CircuitBreaker] Drawdown check failed: {e}")

            # 5. Intraday drawdown — compute real-time unrealized P&L on open positions
            try:
                from tools.market_data import market_data_tool
                from core.capital_manager import capital_manager
                total_cap = capital_manager.get_total_capital()
                open_trades = session.query(TradeExecution).filter(TradeExecution.status == "OPEN").all()
                intraday_loss = 0.0
                for t in open_trades:
                    if t.entry_price and t.entry_price > 0:
                        current_px = market_data_tool.get_current_price(t.symbol)
                        if current_px > 0:
                            if t.direction == "BUY":
                                intraday_loss += (current_px - t.entry_price) * (t.quantity or 0)
                            else:
                                intraday_loss += (t.entry_price - current_px) * (t.quantity or 0)
                # If unrealized loss > 3% of total capital, soft warn (not halt)
                intraday_loss_pct = intraday_loss / total_cap * 100 if total_cap > 0 else 0
                if intraday_loss_pct < -3.0:
                    print(f"[CircuitBreaker] WARNING: Intraday unrealized loss {intraday_loss_pct:.2f}% — monitor closely.")
            except Exception as e:
                print(f"[CircuitBreaker] Intraday drawdown check failed: {e}")

            return True, "OK"
        finally:
            session.close()

    def exit_all_positions(self, reason: str = "CIRCUIT_BREAKER_EMERGENCY"):
        """
        Emergency: close all open paper trades at current market price.
        For live mode, sends urgent Telegram alert with manual exit instructions.
        """
        from core.config import settings as _s
        from core.telegram_bot import _send

        session = SessionLocal()
        try:
            if _s.PAPER_MODE:
                from db.schema import PaperTrade
                from tools.market_data import market_data_tool
                open_paper = session.query(PaperTrade).filter(PaperTrade.status == "OPEN").all()
                closed_count = 0
                for trade in open_paper:
                    try:
                        price = market_data_tool.get_current_price(trade.symbol)
                        if price > 0:
                            trade.exit_price = price
                            trade.exit_time = datetime.utcnow()
                            trade.exit_reason = reason
                            trade.status = "CLOSED"
                            if trade.entry_price and trade.direction == "BUY":
                                trade.realized_pnl = round((price - trade.entry_price) * (trade.quantity or 0), 2)
                                trade.realized_pnl_pct = round((price - trade.entry_price) / trade.entry_price * 100, 2)
                            elif trade.entry_price and trade.direction == "SELL":
                                trade.realized_pnl = round((trade.entry_price - price) * (trade.quantity or 0), 2)
                                trade.realized_pnl_pct = round((trade.entry_price - price) / trade.entry_price * 100, 2)
                            closed_count += 1
                    except Exception as e:
                        print(f"[CircuitBreaker] Emergency close failed for {trade.symbol}: {e}")
                session.commit()
                print(f"[CircuitBreaker] Emergency closed {closed_count} paper positions. Reason: {reason}")
                _send({
                    "chat_id": _s.TELEGRAM_CHAT_ID,
                    "text": f"🛑 *EMERGENCY EXIT TRIGGERED*\n\nAll {closed_count} paper positions closed.\n*Reason*: {reason}",
                    "parse_mode": "Markdown",
                })
            else:
                # Live mode: we cannot auto-exit without user confirmation
                # Send urgent Telegram alert for manual action
                open_live = session.query(TradeExecution).filter(TradeExecution.status == "OPEN").all()
                symbols = [t.symbol for t in open_live]
                _send({
                    "chat_id": _s.TELEGRAM_CHAT_ID,
                    "text": (
                        f"🚨 *CIRCUIT BREAKER — MANUAL EXIT REQUIRED*\n\n"
                        f"*Reason*: {reason}\n"
                        f"*Open positions*: {', '.join(symbols)}\n\n"
                        f"_Please exit these positions manually on Kite immediately._"
                    ),
                    "parse_mode": "Markdown",
                })
        finally:
            session.close()

    def _log(self, session, reason: str, daily_pnl: float | None, consec: int | None):
        log = CircuitBreakerLog(
            reason=reason,
            daily_pnl_pct=daily_pnl,
            consecutive_losses=consec,
        )
        session.add(log)
        session.commit()
        print(f"[CircuitBreaker] TRIPPED: {reason}")


circuit_breaker = CircuitBreaker()
