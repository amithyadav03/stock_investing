"""
FastAPI server — entry points for the AI trading agent system.

Endpoints:
  POST /scan              — Trigger AI analysis for a symbol (background)
  POST /telegram-webhook  — Handle Telegram approve/reject/exit callbacks
  POST /monitor-positions — Trigger exit evaluation on all open positions (background)
  GET  /positions         — List open positions with live P&L
  GET  /watchlist         — Active watchlist candidates
  GET  /paper-performance — Paper trading P&L and stats
  GET  /portfolio-advice  — Current advisor actions on Kite holdings
  GET  /health            — System health check
"""

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
import requests

from db.schema import init_db, SessionLocal, TradeProposal, TradeExecution, PositionMonitorLog, Watchlist
from agents.workflow import trading_agent_app
from agents.exit_monitor import evaluate_exit
from agents.state import AgentState
from core.telegram_bot import (
    send_telegram_trade_proposal, send_exit_alert,
    notify_error, notify_circuit_breaker, send_portfolio_summary, _send,
)
from core.validator import pre_execution_validation
from core.circuit_breaker import circuit_breaker
from core.position_tracker import get_open_positions, get_position_with_proposal, mark_position_closed
from core.paper_trader import paper_execute, close_paper_position, get_paper_performance
from core.scheduler import setup_scheduler, shutdown_scheduler
from agents.portfolio_advisor import get_portfolio_advice_from_db
from core.config import settings
from kiteconnect import KiteConnect
from datetime import datetime


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    settings.validate_critical_keys()
    setup_scheduler()
    yield
    shutdown_scheduler()


app = FastAPI(title="AI Swing Trade Agent", lifespan=lifespan)


class ScanRequest(BaseModel):
    symbol: str
    strategy_type: str = "swing"


# ─── Agent Workflow ────────────────────────────────────────────────────────────

def run_agent_workflow(symbol: str, strategy_type: str = "swing"):
    """Full analysis pipeline → Telegram proposal (if actionable)."""

    # Circuit breaker check before analysis
    allowed, cb_reason = circuit_breaker.is_trading_allowed()
    if not allowed:
        print(f"[Orchestrator] Circuit breaker active for {symbol}: {cb_reason}")
        notify_circuit_breaker(cb_reason)
        return

    initial_state = AgentState(
        symbol=symbol,
        strategy_type=strategy_type,
        messages=[],
        technical_analysis=None,
        technical_narrative=None,
        weekly_data=None,
        monthly_data=None,
        timeframe_confluence=None,
        fundamental_analysis=None,
        sentiment_analysis=None,
        macro_context=None,
        sector_performance=None,
        research_report=None,
        rl_context=None,
        conviction_score=None,
        conviction_passes=None,
        decision=None,
        is_safe_to_execute=None,
        guardrail_warnings=None,
    )

    print(f"\n[Orchestrator] Starting analysis for {symbol}...")
    final_output = trading_agent_app.invoke(initial_state)

    decision = final_output.get("decision")
    warnings = final_output.get("guardrail_warnings", "")
    tech = final_output.get("technical_analysis") or {}
    narrative = final_output.get("technical_narrative", "")

    if not decision:
        print(f"[Orchestrator] No decision produced for {symbol}.")
        return

    # ── Quantity sizing ─────────────────────────────────────────────────────────
    portfolio_capital = settings.strategy.get("trading", {}).get("portfolio_capital", 1_000_000)
    risk_amount = portfolio_capital * decision.risk_percentage
    risk_per_share = abs(decision.proposed_entry - decision.proposed_stop_loss)
    calculated_qty = max(1, int(risk_amount / risk_per_share)) if risk_per_share > 0 else 1

    max_liq_pct = settings.strategy.get("risk", {}).get("max_liquidity_percent_of_volume", 0.01)
    avg_vol = tech.get("average_volume_30d", 0)
    if avg_vol > 0:
        max_by_liq = int(avg_vol * max_liq_pct)
        calculated_qty = min(calculated_qty, max_by_liq) if max_by_liq > 0 else calculated_qty
    else:
        warnings += " | WARNING: Volume data missing, defaulting qty to 1."
        calculated_qty = 1

    # ── Persist proposal ────────────────────────────────────────────────────────
    session = SessionLocal()
    try:
        proposal = TradeProposal(
            symbol=symbol,
            direction=decision.proposed_action,
            proposed_price=decision.proposed_entry,
            stop_loss=decision.proposed_stop_loss,
            take_profit=decision.proposed_take_profit,
            quantity=calculated_qty,
            risk_percentage=decision.risk_percentage,
            conviction_tier=decision.conviction_tier,
            win_probability=decision.win_probability_score,
            expected_holding_days=decision.expected_holding_days,
            rationale=decision.final_rationale + f"\n\nGuardrails: {warnings}",
            technical_narrative=narrative[:1000],
            guardrail_warnings=warnings,
            status="PENDING" if final_output.get("is_safe_to_execute") else "ABORTED_BY_GUARDRAIL",
        )
        session.add(proposal)
        session.commit()
        session.refresh(proposal)

        if decision.proposed_action == "ERROR":
            notify_error(symbol, decision.final_rationale)
        elif decision.proposed_action in ("BUY", "SELL") and proposal.status == "PENDING":
            print(f"[Orchestrator] {symbol} → {decision.proposed_action}. Sending Telegram proposal.")
            send_telegram_trade_proposal(
                proposal_id=proposal.id,
                symbol=symbol,
                action=proposal.direction,
                rationale=proposal.rationale,
                entry=proposal.proposed_price,
                sl=proposal.stop_loss,
                tp=proposal.take_profit,
                holding_days=decision.expected_holding_days,
                conviction=proposal.conviction_tier,
                win_prob=proposal.win_probability,
                technical_narrative=narrative,
                strategy_type=strategy_type,
                conviction_score=final_output.get("conviction_score") or 0,
                research_summary=str(final_output.get("research_report") or "")[:250],
            )
        else:
            print(f"[Orchestrator] {symbol} → {decision.proposed_action} (no Telegram needed).")
    finally:
        session.close()


# ─── Position Exit Monitoring ──────────────────────────────────────────────────

def run_exit_monitor():
    """Evaluates all open positions and sends exit alerts via Telegram."""
    positions = get_open_positions()
    print(f"[ExitMonitor] Checking {len(positions)} open positions...")

    for pos in positions:
        data = get_position_with_proposal(pos["execution_id"])
        if not data:
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

        # Log every evaluation
        session = SessionLocal()
        try:
            log = PositionMonitorLog(
                execution_id=data["execution_id"],
                symbol=data["symbol"],
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

        if exit_decision.action in ("EXIT_NOW", "TRAIL_SL"):
            send_exit_alert(
                execution_id=data["execution_id"],
                symbol=data["symbol"],
                action=exit_decision.action,
                current_price=data["current_price"],
                pnl_pct=data["pnl_pct"],
                rationale=exit_decision.rationale,
                new_sl=exit_decision.new_stop_loss,
                urgency=exit_decision.urgency,
            )


# ─── API Endpoints ─────────────────────────────────────────────────────────────

@app.post("/scan")
async def trigger_scan(req: ScanRequest, bg: BackgroundTasks):
    bg.add_task(run_agent_workflow, req.symbol, req.strategy_type)
    return {"message": f"Agent dispatched for {req.symbol} [{req.strategy_type}]."}


@app.post("/monitor-positions")
async def trigger_monitor(bg: BackgroundTasks):
    bg.add_task(run_exit_monitor)
    return {"message": "Exit monitor triggered for all open positions."}


@app.get("/positions")
async def list_positions():
    return {"positions": get_open_positions()}


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    """Handles Approve/Reject/Exit/TrailSL callbacks from Telegram inline buttons."""
    # Validate Telegram webhook secret if configured
    if settings.TELEGRAM_WEBHOOK_SECRET:
        token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if token != settings.TELEGRAM_WEBHOOK_SECRET:
            return JSONResponse(status_code=403, content={"error": "Forbidden"})

    payload = await request.json()

    if "callback_query" not in payload:
        return {"status": "ok"}

    cq = payload["callback_query"]
    callback_data: str = cq["data"]
    callback_id: str = cq["id"]

    # Acknowledge the callback immediately (stops spinner)
    requests.post(
        f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/answerCallbackQuery",
        json={"callback_query_id": callback_id, "text": f"Processing: {callback_data.split('_')[0]}..."},
        timeout=5,
    )

    parts = callback_data.split("_")
    action = parts[0]

    # ── Trade Proposal: APPROVE ────────────────────────────────────────────────
    if action == "APPROVE":
        proposal_id = int(parts[1])
        session = SessionLocal()
        try:
            # SELECT FOR UPDATE prevents duplicate webhook execution
            proposal = session.query(TradeProposal).filter(
                TradeProposal.id == proposal_id,
                TradeProposal.status == "PENDING",
            ).with_for_update().first()
            if not proposal:
                return {"status": "ok", "msg": "Already processed or not found."}

            is_valid = pre_execution_validation(proposal.symbol, proposal.proposed_price)
            if not is_valid:
                proposal.status = "ABORTED"
                session.commit()
                return {"status": "ok", "msg": "Aborted — price drifted beyond tolerance."}

            # Price band check: abort if stock moved > 15% from yesterday (approaching circuit)
            try:
                from tools.market_data import market_data_tool
                current_px = market_data_tool.get_current_price(proposal.symbol)
                if current_px > 0 and proposal.proposed_price > 0:
                    move_pct = abs(current_px - proposal.proposed_price) / proposal.proposed_price * 100
                    if move_pct > 15:
                        proposal.status = "ABORTED"
                        session.commit()
                        print(f"[Webhook] {proposal.symbol} moved {move_pct:.1f}% — possible circuit limit. Aborted.")
                        return {"status": "ok", "msg": f"Aborted — {move_pct:.1f}% price move suggests circuit limit."}
            except Exception:
                pass

            # Recalculate position size at approval time using current price and available capital
            qty = max(1, getattr(proposal, 'quantity', 1) or 1)
            try:
                from tools.market_data import market_data_tool as _mdt
                from core.capital_manager import capital_manager
                live_price = _mdt.get_current_price(proposal.symbol)
                if live_price > 0 and proposal.stop_loss and abs(live_price - proposal.stop_loss) > 0:
                    sizing = capital_manager.calculate_position_size(
                        entry_price=live_price,
                        stop_loss=proposal.stop_loss,
                        conviction_tier=proposal.conviction_tier or "MEDIUM",
                        strategy_type=proposal.strategy_type or "swing",
                    )
                    recalc_qty = sizing.get("quantity", 0)
                    if recalc_qty > 0:
                        qty = recalc_qty
                        print(f"[Webhook] Qty recalculated at approval: {qty} (live price ₹{live_price:.2f})")
            except Exception as e:
                print(f"[Webhook] Qty recalc failed, using original: {e}")

            if settings.PAPER_MODE:
                try:
                    paper_trade = paper_execute(
                        proposal_id=proposal.id,
                        symbol=proposal.symbol,
                        direction=proposal.direction,
                        entry_price=proposal.proposed_price,
                        stop_loss=proposal.stop_loss,
                        take_profit=proposal.take_profit,
                        quantity=qty,
                        conviction_tier=proposal.conviction_tier or "MEDIUM",
                        conviction_score=proposal.conviction_score or 0,
                        strategy_type=proposal.strategy_type or "swing",
                    )
                    proposal.status = "EXECUTED"
                    session.commit()
                    from core.telegram_bot import notify_paper_trade_executed
                    notify_paper_trade_executed(proposal.symbol, proposal.direction,
                                                proposal.proposed_price, qty)
                    print(f"[Webhook] Paper trade #{paper_trade.id} opened for {proposal.symbol}.")
                except Exception as e:
                    proposal.status = "KITE_FAILED"
                    session.commit()
                    print(f"[Webhook] Paper execute failed: {e}")
            else:
                try:
                    kite = KiteConnect(api_key=settings.KITE_API_KEY)
                    kite.set_access_token(settings.KITE_ACCESS_TOKEN)

                    strat = settings.strategy.get("trading", {})
                    order_variety_str = strat.get("order_variety", "AMO").lower()
                    variety = kite.VARIETY_AMO if order_variety_str == "amo" else kite.VARIETY_REGULAR
                    product = getattr(kite, f"PRODUCT_{strat.get('product_type', 'CNC').upper()}")

                    order_id = kite.place_order(
                        tradingsymbol=proposal.symbol,
                        exchange=kite.EXCHANGE_NSE,
                        transaction_type=kite.TRANSACTION_TYPE_BUY if proposal.direction == "BUY" else kite.TRANSACTION_TYPE_SELL,
                        quantity=qty,
                        variety=variety,
                        order_type=kite.ORDER_TYPE_LIMIT,
                        product=product,
                        price=proposal.proposed_price,
                    )
                    print(f"[Webhook] Order placed on Kite. Order ID: {order_id}")

                    # Poll order status to confirm fill (AMO orders don't fill immediately)
                    import time as _time
                    order_status = "PENDING"
                    max_polls = 6  # poll up to 6 times, 5 sec apart
                    for _ in range(max_polls):
                        try:
                            orders = kite.orders()
                            for o in orders:
                                if str(o.get("order_id")) == str(order_id):
                                    order_status = o.get("status", "UNKNOWN")
                                    break
                        except Exception as poll_err:
                            print(f"[Webhook] Order status poll failed: {poll_err}")
                            break
                        if order_status in ("COMPLETE", "OPEN"):
                            break
                        if order_status in ("CANCELLED", "REJECTED"):
                            break
                        _time.sleep(5)

                    if order_status in ("CANCELLED", "REJECTED"):
                        proposal.status = "KITE_FAILED"
                        session.commit()
                        from core.telegram_bot import notify_error
                        notify_error(proposal.symbol, f"Kite order {order_id} was {order_status}. Please re-review.")
                        print(f"[Webhook] Order {order_id} {order_status}. Proposal marked KITE_FAILED.")
                        return {"status": "ok", "msg": f"Order {order_status} by Kite."}

                    proposal.status = "EXECUTED"
                    print(f"[Webhook] EXECUTED on Kite. Order ID: {order_id} | Status: {order_status}")

                    execution = TradeExecution(
                        proposal_id=proposal.id,
                        symbol=proposal.symbol,
                        direction=proposal.direction,
                        quantity=qty,
                        entry_price=proposal.proposed_price,
                        entry_time=datetime.utcnow(),
                        kite_order_id=str(order_id),
                        strategy_type=proposal.strategy_type or "swing",
                        status="OPEN",
                    )
                    session.add(execution)
                    session.commit()
                    print(f"[Webhook] TradeExecution #{execution.id} created for {proposal.symbol}.")

                except Exception as e:
                    proposal.status = "KITE_FAILED"
                    session.commit()
                    print(f"[Webhook] Kite execution failed: {e}")

        finally:
            session.close()

    # ── Trade Proposal: REJECT ─────────────────────────────────────────────────
    elif action == "REJECT":
        proposal_id = int(parts[1])
        session = SessionLocal()
        try:
            proposal = session.query(TradeProposal).filter(TradeProposal.id == proposal_id).first()
            if proposal and proposal.status == "PENDING":
                proposal.status = "REJECTED"
                session.commit()
                print(f"[Webhook] Proposal #{proposal_id} REJECTED.")
        finally:
            session.close()

    # ── Research: RESEARCH ────────────────────────────────────────────────────
    elif action == "RESEARCH":
        proposal_id = int(parts[1])
        session = SessionLocal()
        try:
            proposal = session.query(TradeProposal).filter(TradeProposal.id == proposal_id).first()
            if proposal:
                lines = [f"🔍 *Research: {proposal.symbol}*\n"]
                if proposal.research_summary:
                    lines.append(proposal.research_summary[:800])
                else:
                    lines.append("_No deep research available for this proposal._")
                lines.append(f"\n*Conviction Score*: {proposal.conviction_score or 'N/A'}/100")
                lines.append(f"*Strategy*: {proposal.strategy_type or 'swing'}")
                _send({
                    "chat_id": settings.TELEGRAM_CHAT_ID,
                    "text": "\n".join(lines),
                    "parse_mode": "Markdown",
                })
        finally:
            session.close()

    # ── Active Position: EXIT ──────────────────────────────────────────────────
    elif action == "EXIT":
        execution_id = int(parts[1])
        data = get_position_with_proposal(execution_id)
        if data:
            current_price = data["current_price"]
            if settings.PAPER_MODE:
                # For paper mode, find the PaperTrade matching this execution_id
                session = SessionLocal()
                try:
                    from db.schema import PaperTrade
                    paper = session.query(PaperTrade).filter(
                        PaperTrade.proposal_id == data.get("proposal_id"),
                        PaperTrade.symbol == data["symbol"],
                        PaperTrade.status == "OPEN",
                    ).first()
                    if paper:
                        close_paper_position(paper.id, current_price, "MANUAL_EXIT_APPROVED")
                        print(f"[Webhook] Paper position #{paper.id} closed at ₹{current_price}.")
                finally:
                    session.close()
            else:
                try:
                    kite = KiteConnect(api_key=settings.KITE_API_KEY)
                    kite.set_access_token(settings.KITE_ACCESS_TOKEN)
                    strat = settings.strategy.get("trading", {})
                    order_variety_str = strat.get("order_variety", "AMO").lower()
                    variety = kite.VARIETY_AMO if order_variety_str == "amo" else kite.VARIETY_REGULAR
                    product = getattr(kite, f"PRODUCT_{strat.get('product_type', 'CNC').upper()}")
                    exit_tx = kite.TRANSACTION_TYPE_SELL if data["direction"] == "BUY" else kite.TRANSACTION_TYPE_BUY
                    exit_order_id = kite.place_order(
                        tradingsymbol=data["symbol"],
                        exchange=kite.EXCHANGE_NSE,
                        transaction_type=exit_tx,
                        quantity=data["quantity"],
                        variety=kite.VARIETY_REGULAR,  # Exit always REGULAR (immediate)
                        order_type=kite.ORDER_TYPE_MARKET,
                        product=product,
                    )
                    # Poll exit order status before marking position closed
                    import time as _time
                    exit_status = "PENDING"
                    for _ in range(4):
                        try:
                            orders = kite.orders()
                            for o in orders:
                                if str(o.get("order_id")) == str(exit_order_id):
                                    exit_status = o.get("status", "UNKNOWN")
                                    break
                        except Exception:
                            break
                        if exit_status in ("COMPLETE", "CANCELLED", "REJECTED"):
                            break
                        _time.sleep(3)

                    if exit_status == "COMPLETE":
                        mark_position_closed(execution_id, current_price, "MANUAL_EXIT_APPROVED")
                        print(f"[Webhook] Position #{execution_id} EXITED. Order {exit_order_id} COMPLETE.")
                    else:
                        print(f"[Webhook] Exit order {exit_order_id} status: {exit_status}. Position NOT marked closed.")
                        from core.telegram_bot import notify_error
                        notify_error(data["symbol"], f"Exit order {exit_order_id} status={exit_status}. Verify manually on Kite.")
                except Exception as e:
                    print(f"[Webhook] Exit order failed: {e}")

    # ── Active Position: TRAIL_SL ──────────────────────────────────────────────
    elif action == "TRAILSL":
        execution_id = int(parts[1])
        new_sl = float(parts[2])
        session = SessionLocal()
        try:
            execution = session.query(TradeExecution).filter(TradeExecution.id == execution_id).first()
            if execution:
                proposal = session.query(TradeProposal).filter(TradeProposal.id == execution.proposal_id).first()
                if proposal:
                    proposal.stop_loss = new_sl
                    session.commit()
                    print(f"[Webhook] SL for {execution.symbol} trailed to ₹{new_sl}.")
        finally:
            session.close()

    # ── Active Position: HOLD (dismiss alert) ─────────────────────────────────
    elif action == "HOLD":
        print(f"[Webhook] User chose to HOLD position {parts[1]}.")

    return {"status": "ok"}


# ─── New Endpoints ─────────────────────────────────────────────────────────────

@app.get("/watchlist")
async def get_watchlist():
    """Active watchlist candidates ordered by conviction score."""
    session = SessionLocal()
    try:
        items = session.query(Watchlist).filter(
            Watchlist.status == "ACTIVE"
        ).order_by(Watchlist.conviction_score.desc()).all()
        return {
            "count": len(items),
            "candidates": [
                {
                    "id": w.id,
                    "symbol": w.symbol,
                    "strategy_type": w.strategy_type,
                    "direction": w.direction,
                    "proposed_entry": w.proposed_entry,
                    "conviction_score": w.conviction_score,
                    "added_at": str(w.added_at),
                    "expires_at": str(w.expires_at) if w.expires_at else None,
                }
                for w in items
            ],
        }
    finally:
        session.close()


@app.get("/paper-performance")
async def paper_performance():
    """Paper trading P&L summary vs Nifty 50 benchmark."""
    return get_paper_performance()


@app.get("/portfolio-advice")
async def portfolio_advice():
    """Last known advisor actions on Kite demat holdings."""
    advice = get_portfolio_advice_from_db()
    return {"count": len(advice), "holdings": advice}


@app.get("/health")
async def health():
    from core.claude_client import get_client
    from core.scheduler import scheduler
    return {
        "status": "ok",
        "mode": "PAPER" if settings.PAPER_MODE else "LIVE",
        "claude_configured": bool(settings.ANTHROPIC_API_KEY),
        "kite_configured": bool(settings.KITE_API_KEY and settings.KITE_ACCESS_TOKEN),
        "telegram_configured": bool(settings.TELEGRAM_BOT_TOKEN),
        "scheduler_running": scheduler.running if scheduler else False,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
