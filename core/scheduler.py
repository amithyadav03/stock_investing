"""
Always-on APScheduler orchestrator.
Runs all jobs 24/7: research during off-hours, execution during market hours.
Market hours only gate order execution, not analysis.
"""

import pytz
from datetime import datetime, time as dtime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

IST = pytz.timezone("Asia/Kolkata")
scheduler = BackgroundScheduler(timezone=IST)


# ── Market Hours Helpers ───────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(9, 15) <= t <= dtime(15, 30)


def is_trading_day() -> bool:
    return datetime.now(IST).weekday() < 5


def is_within_window(start: dtime, end: dtime) -> bool:
    t = datetime.now(IST).time()
    return start <= t <= end


# ── Job Implementations ────────────────────────────────────────────────────────

def job_news_monitor():
    """24/7: Check RSS feeds, flag urgent news affecting open positions."""
    try:
        from tools.fundamental_news import fundamental_news_tool
        from core.cache import cache, TTL_NEWS

        cached = cache.get("latest_news")
        if cached:
            return

        headlines = fundamental_news_tool.fetch_live_news_snippets()
        if headlines:
            cache.set("latest_news", headlines, TTL_NEWS)

        # Check if any headlines contain held symbols
        from db.schema import SessionLocal, PaperTrade, TradeExecution
        from core.config import settings
        session = SessionLocal()
        try:
            if settings.PAPER_MODE:
                open_trades = session.query(PaperTrade).filter(PaperTrade.status == "OPEN").all()
            else:
                open_trades = session.query(TradeExecution).filter(TradeExecution.status == "OPEN").all()

            held_symbols = {t.symbol.lower() for t in open_trades}
            urgent_news = [h for h in headlines if any(s in h.lower() for s in held_symbols)]

            if urgent_news:
                from core.telegram_bot import _send
                lines = ["📰 *Urgent News Alert*\n"]
                lines.extend(f"• {h[:200]}" for h in urgent_news[:5])
                _send({
                    "chat_id": settings.TELEGRAM_CHAT_ID,
                    "text": "\n".join(lines),
                    "parse_mode": "Markdown",
                })
        finally:
            session.close()
    except Exception as e:
        print(f"[Scheduler] news_monitor error: {e}")


def job_macro_update():
    """Every 4 hours 24/7: Refresh macro regime and cache it."""
    try:
        from core.cache import cache, TTL_MACRO
        cache.invalidate("macro_regime")
        from tools.fundamental_news import fundamental_news_tool
        from agents.llm_utils import classify_macro
        macro_raw = fundamental_news_tool.get_macro_context()
        macro = classify_macro(macro_raw)
        cache.set("macro_regime", macro, TTL_MACRO)
        print(f"[Scheduler] Macro updated: {macro.sentiment_enum}")
    except Exception as e:
        print(f"[Scheduler] macro_update error: {e}")


def job_portfolio_sync():
    """9:00 AM: Sync Kite holdings, run portfolio advisor before market opens."""
    if not is_trading_day():
        return
    try:
        from tools.kite_portfolio import get_holdings, sync_holdings_to_db
        from agents.portfolio_advisor import advise_all_holdings
        print("[Scheduler] Syncing portfolio holdings...")
        holdings = get_holdings()
        if holdings:
            sync_holdings_to_db(holdings)
            advice = advise_all_holdings(holdings)
            urgent = [a for a in advice if a.get("urgency") == "URGENT" or a.get("action") in ("EXIT", "ADD_MORE")]
            if urgent:
                from core.telegram_bot import send_portfolio_advice
                send_portfolio_advice(urgent)
        print(f"[Scheduler] Portfolio sync done: {len(holdings)} holdings.")
    except Exception as e:
        print(f"[Scheduler] portfolio_sync error: {e}")


def job_morning_brief():
    """8:30 AM: Send morning brief."""
    if not is_trading_day():
        return
    try:
        from reports.morning_brief import send_morning_brief
        send_morning_brief()
    except Exception as e:
        print(f"[Scheduler] morning_brief error: {e}")


def job_position_monitor():
    """Every 10 min during market hours: monitor open positions for exit signals."""
    if not is_market_open():
        return
    try:
        from core.config import settings
        from db.schema import SessionLocal, PaperTrade, TradeExecution, PositionMonitorLog
        from agents.exit_monitor import evaluate_exit
        from core.telegram_bot import send_exit_alert
        from core.position_tracker import get_position_with_proposal
        from core.paper_trader import mark_paper_positions_to_market

        if settings.PAPER_MODE:
            # Auto-close paper positions that hit SL/TP
            auto_closed = mark_paper_positions_to_market()
            for c in auto_closed:
                print(f"[Scheduler] Paper position auto-closed: {c['symbol']} {c['reason']}")

        session = SessionLocal()
        try:
            if settings.PAPER_MODE:
                open_trades = session.query(PaperTrade).filter(PaperTrade.status == "OPEN").all()
            else:
                open_trades = session.query(TradeExecution).filter(TradeExecution.status == "OPEN").all()
        finally:
            session.close()

        if not open_trades:
            return

        for trade in open_trades:
            try:
                data = get_position_with_proposal(trade.id) if not settings.PAPER_MODE else _get_paper_trade_data(trade)
                if not data:
                    continue

                exit_dec = evaluate_exit(
                    proposal_id=data.get("proposal_id"),
                    symbol=data["symbol"],
                    direction=data["direction"],
                    entry_price=data["entry_price"],
                    stop_loss=data["stop_loss"],
                    take_profit=data["take_profit"],
                    quantity=data["quantity"],
                    entry_rationale=data.get("entry_rationale", ""),
                    entry_time=data.get("entry_time", datetime.utcnow()),
                )

                # Log
                session2 = SessionLocal()
                try:
                    log = PositionMonitorLog(
                        execution_id=trade.id,
                        symbol=data["symbol"],
                        action=exit_dec.action,
                        urgency=exit_dec.urgency,
                        current_price=data.get("current_price", 0),
                        pnl_pct=data.get("pnl_pct", 0),
                        new_stop_loss=exit_dec.new_stop_loss,
                        rationale=exit_dec.rationale[:500],
                    )
                    session2.add(log)
                    session2.commit()
                finally:
                    session2.close()

                if exit_dec.action in ("EXIT_NOW", "TRAIL_SL"):
                    send_exit_alert(
                        execution_id=trade.id,
                        symbol=data["symbol"],
                        action=exit_dec.action,
                        current_price=data.get("current_price", 0),
                        pnl_pct=data.get("pnl_pct", 0),
                        rationale=exit_dec.rationale,
                        new_sl=exit_dec.new_stop_loss,
                        urgency=exit_dec.urgency,
                    )
            except Exception as e:
                print(f"[Scheduler] position_monitor error for {getattr(trade, 'symbol', '?')}: {e}")

    except Exception as e:
        print(f"[Scheduler] position_monitor fatal: {e}")


def _get_paper_trade_data(trade) -> dict:
    """Build position data dict from a PaperTrade record."""
    from tools.market_data import market_data_tool
    from datetime import datetime as dt
    current = market_data_tool.get_current_price(trade.symbol)
    entry = trade.entry_price or 0
    pnl = round((current - entry) / entry * 100, 2) if entry > 0 else 0.0
    days = (dt.utcnow() - trade.entry_time).days if trade.entry_time else 0
    return {
        "execution_id": trade.id,
        "proposal_id": trade.proposal_id,
        "symbol": trade.symbol,
        "direction": trade.direction,
        "quantity": trade.quantity,
        "entry_price": entry,
        "entry_time": trade.entry_time,
        "current_price": current,
        "stop_loss": trade.stop_loss or 0,
        "take_profit": trade.take_profit or 0,
        "pnl_pct": pnl,
        "days_held": days,
        "entry_rationale": "",
    }


def job_midday_checkin():
    """12:30 PM: Send mid-day check-in."""
    if not is_trading_day():
        return
    try:
        from reports.midday_checkin import send_midday_checkin
        send_midday_checkin()
    except Exception as e:
        print(f"[Scheduler] midday_checkin error: {e}")


def job_post_market_analysis():
    """3:45 PM: Run post-market deep analysis pipeline on top candidates."""
    if not is_trading_day():
        return
    try:
        _run_analysis_pipeline()
    except Exception as e:
        print(f"[Scheduler] post_market_analysis error: {e}")


def _run_analysis_pipeline():
    """
    Full post-market pipeline:
    1. Run pre-screener (Nifty 500)
    2. For top N candidates: run full agent workflow
    3. Create trade proposals for high-conviction setups
    4. Add remaining good candidates to watchlist
    """
    from scripts.pre_screener import run_pre_screener
    from agents.workflow import trading_agent_app
    from agents.state import AgentState
    from core.capital_manager import capital_manager
    from core.circuit_breaker import circuit_breaker
    from db.schema import SessionLocal, TradeProposal, Watchlist
    from core.config import settings
    from datetime import datetime, timedelta

    allowed, reason = circuit_breaker.is_trading_allowed()
    if not allowed:
        print(f"[Scheduler] Circuit breaker active: {reason}")
        return

    print("[Scheduler] Running post-market analysis pipeline...")
    elite_symbols = run_pre_screener()
    if not elite_symbols:
        fallback = settings.strategy.get("scanning", {}).get("fallback_symbols", [])
        elite_symbols = fallback[:10]

    top_n = settings.strategy.get("scanning", {}).get("pre_screener_top_n", 20)
    candidates = elite_symbols[:top_n]

    # Determine which strategies are enabled
    strats = settings.strategy.get("strategies", {})
    strategies_to_run = [s for s in ("swing", "positional") if strats.get(s, {}).get("enabled", False)]

    new_proposals = []
    watchlist_candidates = []

    for symbol in candidates:
        for strategy_type in strategies_to_run:
            can_open, reason = capital_manager.can_open_new_position(strategy_type)

            initial_state = AgentState(
                symbol=symbol,
                strategy_type=strategy_type,
                messages=[],
                technical_analysis=None, technical_narrative=None,
                weekly_data=None, monthly_data=None, timeframe_confluence=None,
                fundamental_analysis=None, sentiment_analysis=None,
                macro_context=None, sector_performance=None,
                research_report=None, rl_context=None,
                conviction_score=None, conviction_passes=None,
                decision=None, is_safe_to_execute=None, guardrail_warnings=None,
            )

            try:
                print(f"[Scheduler] Analysing {symbol} [{strategy_type}]...")
                output = trading_agent_app.invoke(initial_state)
                decision = output.get("decision")
                if not decision:
                    continue

                if decision.proposed_action in ("BUY", "SELL") and output.get("is_safe_to_execute"):
                    # Size the position
                    sizing = capital_manager.calculate_position_size(
                        entry_price=decision.proposed_entry,
                        stop_loss=decision.proposed_stop_loss,
                        conviction_tier=decision.conviction_tier,
                        strategy_type=strategy_type,
                    )
                    qty = sizing.get("quantity", 1)

                    if can_open:
                        _create_proposal(output, symbol, strategy_type, decision, qty, settings)
                        new_proposals.append(symbol)
                    else:
                        # Slot full — add to watchlist
                        score = output.get("conviction_score", 0)
                        _add_to_watchlist(symbol, strategy_type, decision, score, settings)
                        watchlist_candidates.append({"symbol": symbol, "strategy_type": strategy_type,
                                                     "conviction_score": score, "direction": decision.proposed_action,
                                                     "proposed_entry": decision.proposed_entry})
            except Exception as e:
                print(f"[Scheduler] Analysis failed for {symbol} [{strategy_type}]: {e}")

    print(f"[Scheduler] Pipeline done. Proposals: {len(new_proposals)}, Watchlist: {len(watchlist_candidates)}")

    if watchlist_candidates:
        from core.telegram_bot import send_watchlist_update
        send_watchlist_update(watchlist_candidates[:5])


def _create_proposal(output, symbol, strategy_type, decision, qty, settings):
    from db.schema import SessionLocal, TradeProposal
    session = SessionLocal()
    try:
        tech = output.get("technical_analysis") or {}
        avg_vol = tech.get("average_volume_30d", 0)
        max_liq = settings.strategy.get("risk", {}).get("max_liquidity_percent_of_volume", 0.01)
        if avg_vol > 0:
            max_by_liq = int(avg_vol * max_liq)
            qty = min(qty, max_by_liq) if max_by_liq > 0 else qty

        proposal = TradeProposal(
            symbol=symbol,
            direction=decision.proposed_action,
            strategy_type=strategy_type,
            proposed_price=decision.proposed_entry,
            stop_loss=decision.proposed_stop_loss,
            take_profit=decision.proposed_take_profit,
            quantity=max(1, qty),
            risk_percentage=decision.risk_percentage,
            conviction_tier=decision.conviction_tier,
            conviction_score=output.get("conviction_score", 0),
            win_probability=decision.win_probability_score,
            expected_holding_days=decision.expected_holding_days,
            rationale=decision.final_rationale + f"\n\nWarnings: {output.get('guardrail_warnings', '')}",
            technical_narrative=(output.get("technical_narrative") or "")[:1000],
            research_summary=(output.get("research_report") or {}).get("business_summary", "")[:500],
            guardrail_warnings=output.get("guardrail_warnings", ""),
            status="PENDING",
        )
        session.add(proposal)
        session.commit()
        print(f"[Scheduler] Proposal created: {symbol} {strategy_type} {decision.proposed_action}")
    finally:
        session.close()


def _add_to_watchlist(symbol, strategy_type, decision, score, settings):
    from db.schema import SessionLocal, Watchlist
    from datetime import datetime, timedelta
    session = SessionLocal()
    try:
        existing = session.query(Watchlist).filter(
            Watchlist.symbol == symbol,
            Watchlist.strategy_type == strategy_type,
            Watchlist.status == "ACTIVE",
        ).first()
        if existing:
            existing.conviction_score = score
            existing.proposed_entry = decision.proposed_entry
        else:
            w = Watchlist(
                symbol=symbol,
                strategy_type=strategy_type,
                direction=decision.proposed_action,
                proposed_entry=decision.proposed_entry,
                proposed_stop_loss=decision.proposed_stop_loss,
                proposed_take_profit=decision.proposed_take_profit,
                conviction_score=score,
                conviction_tier=decision.conviction_tier,
                rationale=decision.final_rationale[:500],
                expires_at=datetime.utcnow() + timedelta(days=3),
            )
            session.add(w)
        session.commit()
    finally:
        session.close()


def job_eod_report():
    """5:00 PM: Send EOD summary and new proposal messages."""
    if not is_trading_day():
        return
    try:
        from reports.eod_summary import send_eod_summary_and_proposals
        send_eod_summary_and_proposals()
    except Exception as e:
        print(f"[Scheduler] eod_report error: {e}")


def job_expire_watchlist():
    """Daily: expire stale watchlist entries."""
    try:
        from db.schema import SessionLocal, Watchlist
        from datetime import datetime
        session = SessionLocal()
        try:
            expired = session.query(Watchlist).filter(
                Watchlist.status == "ACTIVE",
                Watchlist.expires_at <= datetime.utcnow(),
            ).all()
            for w in expired:
                w.status = "EXPIRED"
            session.commit()
            if expired:
                print(f"[Scheduler] Expired {len(expired)} watchlist entries.")
        finally:
            session.close()
    except Exception as e:
        print(f"[Scheduler] expire_watchlist error: {e}")


# ── Scheduler Setup ────────────────────────────────────────────────────────────

def setup_scheduler():
    """Register all jobs and start the scheduler."""
    # 24/7 jobs
    scheduler.add_job(
        job_news_monitor, IntervalTrigger(minutes=15),
        id="news_monitor", max_instances=1, replace_existing=True
    )
    scheduler.add_job(
        job_macro_update, IntervalTrigger(hours=4),
        id="macro_update", max_instances=1, replace_existing=True
    )

    # Pre-market daily (weekdays)
    scheduler.add_job(
        job_portfolio_sync,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone=IST),
        id="portfolio_sync", max_instances=1, replace_existing=True
    )
    scheduler.add_job(
        job_morning_brief,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=30, timezone=IST),
        id="morning_brief", max_instances=1, replace_existing=True
    )

    # Market hours — position monitor every 10 min (check internally)
    scheduler.add_job(
        job_position_monitor,
        IntervalTrigger(minutes=10),
        id="position_monitor", max_instances=1, replace_existing=True
    )

    # Mid-day
    scheduler.add_job(
        job_midday_checkin,
        CronTrigger(day_of_week="mon-fri", hour=12, minute=30, timezone=IST),
        id="midday_checkin", max_instances=1, replace_existing=True
    )

    # Post-market analysis
    scheduler.add_job(
        job_post_market_analysis,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=45, timezone=IST),
        id="post_market_analysis", max_instances=1, replace_existing=True
    )

    # EOD report
    scheduler.add_job(
        job_eod_report,
        CronTrigger(day_of_week="mon-fri", hour=17, minute=0, timezone=IST),
        id="eod_report", max_instances=1, replace_existing=True
    )

    # Daily housekeeping
    scheduler.add_job(
        job_expire_watchlist,
        CronTrigger(hour=6, minute=0, timezone=IST),
        id="expire_watchlist", max_instances=1, replace_existing=True
    )

    scheduler.start()
    print("[Scheduler] Started. Jobs:")
    for job in scheduler.get_jobs():
        print(f"  - {job.id}: next run {job.next_run_time}")


def shutdown_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)
        print("[Scheduler] Stopped.")
