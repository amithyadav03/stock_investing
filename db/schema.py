from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import uuid
from core.config import settings

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TradeProposal(Base):
    """AI-generated trade hypotheses awaiting Telegram approval."""
    __tablename__ = "trade_proposals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    direction = Column(String)
    strategy_type = Column(String, default="swing")  # swing / positional / value

    proposed_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    quantity = Column(Integer, default=1)
    risk_percentage = Column(Float, default=0.0)
    conviction_tier = Column(String, default="LOW")
    conviction_score = Column(Integer, default=0)   # 0-100
    win_probability = Column(Integer, default=0)
    expected_holding_days = Column(Integer, default=0)

    rationale = Column(Text)
    technical_narrative = Column(Text, default="")
    research_summary = Column(Text, default="")
    guardrail_warnings = Column(Text, default="")

    # UUID prevents cross-restart duplicate processing of the same proposal
    idempotency_token = Column(String, unique=True, nullable=True)

    # PENDING → APPROVED → EXECUTED | REJECTED | ABORTED | KITE_FAILED | ABORTED_BY_GUARDRAIL
    status = Column(String, default="PENDING")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TradeExecution(Base):
    """Live trades executed on Kite. Closing this record triggers RL loop."""
    __tablename__ = "trade_executions"

    id = Column(Integer, primary_key=True, index=True)
    proposal_id = Column(Integer, index=True)
    symbol = Column(String, index=True)
    direction = Column(String)
    strategy_type = Column(String, default="swing")
    quantity = Column(Integer, default=1)

    entry_price = Column(Float)
    entry_time = Column(DateTime, default=datetime.utcnow)
    kite_order_id = Column(String, default="")

    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String, nullable=True)

    realized_pnl = Column(Float, nullable=True)
    realized_pnl_pct = Column(Float, nullable=True)

    status = Column(String, default="OPEN")
    created_at = Column(DateTime, default=datetime.utcnow)


class PaperTrade(Base):
    """Paper trading simulation — same lifecycle as TradeExecution but no Kite orders."""
    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True, index=True)
    proposal_id = Column(Integer, index=True, nullable=True)
    symbol = Column(String, index=True)
    direction = Column(String)
    strategy_type = Column(String, default="swing")
    quantity = Column(Integer, default=1)

    entry_price = Column(Float)
    entry_time = Column(DateTime, default=datetime.utcnow)

    stop_loss = Column(Float)
    take_profit = Column(Float)

    current_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String, nullable=True)

    realized_pnl = Column(Float, nullable=True)
    realized_pnl_pct = Column(Float, nullable=True)

    conviction_tier = Column(String, default="MEDIUM")
    conviction_score = Column(Integer, default=0)

    status = Column(String, default="OPEN")  # OPEN / CLOSED
    created_at = Column(DateTime, default=datetime.utcnow)


class Watchlist(Base):
    """High-conviction setups queued while position slots are full."""
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    strategy_type = Column(String, default="swing")
    direction = Column(String)

    proposed_entry = Column(Float)
    proposed_stop_loss = Column(Float)
    proposed_take_profit = Column(Float)

    conviction_score = Column(Integer, default=0)
    conviction_tier = Column(String, default="MEDIUM")
    rationale = Column(Text)

    # ACTIVE / PROMOTED (moved to proposal) / EXPIRED (stale)
    status = Column(String, default="ACTIVE")

    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)


class PortfolioHolding(Base):
    """Daily snapshot of existing Kite demat holdings."""
    __tablename__ = "portfolio_holdings"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    isin = Column(String, nullable=True)
    quantity = Column(Integer)
    avg_price = Column(Float)
    current_price = Column(Float, nullable=True)
    pnl_amount = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)

    advisor_action = Column(String, nullable=True)   # HOLD / ADD_MORE / EXIT
    advisor_rationale = Column(Text, nullable=True)
    advisor_updated_at = Column(DateTime, nullable=True)

    last_synced = Column(DateTime, default=datetime.utcnow)


class PositionMonitorLog(Base):
    """Audit trail for every exit-monitor evaluation."""
    __tablename__ = "position_monitor_logs"

    id = Column(Integer, primary_key=True, index=True)
    execution_id = Column(Integer, index=True)
    symbol = Column(String)
    action = Column(String)
    urgency = Column(String)
    current_price = Column(Float)
    pnl_pct = Column(Float)
    new_stop_loss = Column(Float, nullable=True)
    rationale = Column(Text)
    evaluated_at = Column(DateTime, default=datetime.utcnow)


class PerformanceLog(Base):
    """Daily portfolio performance snapshot for P&L tracking and benchmark comparison."""
    __tablename__ = "performance_log"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, index=True)
    mode = Column(String, default="PAPER")

    total_capital = Column(Float)
    deployed_capital = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    unrealized_pnl_pct = Column(Float, default=0.0)
    realized_pnl_today = Column(Float, default=0.0)
    cumulative_realized_pnl = Column(Float, default=0.0)

    open_positions = Column(Integer, default=0)
    closed_today = Column(Integer, default=0)
    wins_today = Column(Integer, default=0)
    losses_today = Column(Integer, default=0)

    nifty_change_pct = Column(Float, nullable=True)
    alpha = Column(Float, nullable=True)   # portfolio_return - nifty_return

    created_at = Column(DateTime, default=datetime.utcnow)


class CircuitBreakerLog(Base):
    """Records when the circuit breaker trips."""
    __tablename__ = "circuit_breaker_logs"

    id = Column(Integer, primary_key=True, index=True)
    reason = Column(String)
    daily_pnl_pct = Column(Float, nullable=True)
    consecutive_losses = Column(Integer, nullable=True)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    reset_at = Column(DateTime, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)
