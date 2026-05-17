"""
Circuit breaker — halts all new trade entries when equity protection thresholds are hit.
Checks: daily P&L loss limit, consecutive loss streak, max open positions.
"""

from datetime import datetime, date
from db.schema import SessionLocal, TradeExecution, CircuitBreakerLog
from core.config import settings


class CircuitBreaker:
    def is_trading_allowed(self) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Checks three conditions from strategy_config.yaml:
        - max_daily_loss_pct: halt if today's realized P&L < -X%
        - max_consecutive_losses: halt after N consecutive losses
        - max_open_positions: halt if too many active trades
        """
        risk_cfg = settings.strategy.get("risk", {})
        max_daily_loss = risk_cfg.get("max_daily_loss_pct", 0.03)       # 3%
        max_consec_losses = risk_cfg.get("max_consecutive_losses", 3)   # 3 in a row
        max_open_pos = risk_cfg.get("max_open_positions", 5)            # 5 simultaneous

        session = SessionLocal()
        try:
            # 1. Daily loss limit
            today = date.today()
            today_closed = session.query(TradeExecution).filter(
                TradeExecution.status == "CLOSED",
                TradeExecution.exit_time >= datetime.combine(today, datetime.min.time()),
            ).all()
            if today_closed:
                total_pnl_pct = sum(t.realized_pnl_pct or 0 for t in today_closed)
                if total_pnl_pct < -(max_daily_loss * 100):
                    self._log(session, f"Daily loss limit hit: {total_pnl_pct:.1f}%", total_pnl_pct, None)
                    return False, f"CIRCUIT BREAKER: Daily loss {total_pnl_pct:.1f}% exceeds limit -{max_daily_loss*100:.0f}%."

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

            # 4. Peak-to-trough drawdown check
            max_drawdown = risk_cfg.get("max_drawdown_from_peak_pct", 0.15)  # 15%
            try:
                from db.schema import PerformanceLog, PaperTrade
                from core.config import settings as _s
                if _s.PAPER_MODE:
                    # Compute peak equity from PerformanceLog
                    logs = session.query(PerformanceLog).order_by(PerformanceLog.date).all()
                    if logs:
                        # Peak equity = total_capital + max cumulative unrealized + realized
                        peak = max((l.total_capital + (l.unrealized_pnl or 0) for l in logs), default=0)
                        latest_log = logs[-1]
                        current_equity = latest_log.total_capital + (latest_log.unrealized_pnl or 0)
                        if peak > 0 and (peak - current_equity) / peak >= max_drawdown:
                            self._log(session, f"Peak drawdown breached: {(peak - current_equity)/peak:.1%}", None, None)
                            return False, f"CIRCUIT BREAKER: Drawdown {(peak-current_equity)/peak:.1%} from peak. Review required."
            except Exception as e:
                print(f"[CircuitBreaker] Drawdown check failed: {e}")

            return True, "OK"
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
