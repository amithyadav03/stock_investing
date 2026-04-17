from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from core.config import settings

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TradeProposal(Base):
    """Stores AI-generated trade hypotheses awaiting Telegram approval."""
    __tablename__ = "trade_proposals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    direction = Column(String)          # BUY / SELL
    timeframe = Column(String, default="swing")

    proposed_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    quantity = Column(Integer, default=1)
    risk_percentage = Column(Float, default=0.0)
    conviction_tier = Column(String, default="LOW")
    win_probability = Column(Integer, default=0)
    expected_holding_days = Column(Integer, default=0)

    rationale = Column(Text)
    technical_narrative = Column(Text, default="")
    guardrail_warnings = Column(Text, default="")

    # PENDING → APPROVED → EXECUTED | REJECTED | ABORTED | KITE_FAILED | ABORTED_BY_GUARDRAIL
    status = Column(String, default="PENDING")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TradeExecution(Base):
    """
    Records live trades after Kite order placement.
    Outcome written here closes the RL feedback loop.
    """
    __tablename__ = "trade_executions"

    id = Column(Integer, primary_key=True, index=True)
    proposal_id = Column(Integer, index=True)
    symbol = Column(String, index=True)
    direction = Column(String)
    quantity = Column(Integer, default=1)

    entry_price = Column(Float)
    entry_time = Column(DateTime, default=datetime.utcnow)
    kite_order_id = Column(String, default="")

    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String, nullable=True)   # STOP_LOSS | TAKE_PROFIT | THESIS_INVALIDATED | MACRO_REVERSAL | MANUAL

    realized_pnl = Column(Float, nullable=True)   # absolute INR
    realized_pnl_pct = Column(Float, nullable=True)

    # Status: OPEN → CLOSED
    status = Column(String, default="OPEN")
    created_at = Column(DateTime, default=datetime.utcnow)


class PositionMonitorLog(Base):
    """Audit trail for every exit-monitor evaluation on an active position."""
    __tablename__ = "position_monitor_logs"

    id = Column(Integer, primary_key=True, index=True)
    execution_id = Column(Integer, index=True)
    symbol = Column(String)
    action = Column(String)          # HOLD | TRAIL_SL | EXIT_NOW
    urgency = Column(String)         # NORMAL | URGENT
    current_price = Column(Float)
    pnl_pct = Column(Float)
    new_stop_loss = Column(Float, nullable=True)
    rationale = Column(Text)
    evaluated_at = Column(DateTime, default=datetime.utcnow)


class CircuitBreakerLog(Base):
    """Records when the circuit breaker trips to protect equity."""
    __tablename__ = "circuit_breaker_logs"

    id = Column(Integer, primary_key=True, index=True)
    reason = Column(String)
    daily_pnl_pct = Column(Float, nullable=True)
    consecutive_losses = Column(Integer, nullable=True)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    reset_at = Column(DateTime, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)
