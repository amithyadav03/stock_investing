from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from core.config import settings

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class TradeProposal(Base):
    """
    Stores trade hypotheses, statuses, and validation details.
    """
    __tablename__ = "trade_proposals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    direction = Column(String) # BUY / SELL
    timeframe = Column(String) # e.g. "swing_1w"
    
    # The prices at the time of proposal
    proposed_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    
    # Rationale and validation
    rationale = Column(String)         # The markdown string sent to telegram
    status = Column(String, default="PENDING") # PENDING, APPROVED, REJECTED, EXECUTED, ABORTED
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TradeExecution(Base):
    """
    Stores executed trades and their actual final outcomes for the RL reflection loop.
    """
    __tablename__ = "trade_executions"

    id = Column(Integer, primary_key=True, index=True)
    proposal_id = Column(Integer)  # Link to TradeProposal
    symbol = Column(String)
    direction = Column(String)
    
    entry_price = Column(Float)
    entry_time = Column(DateTime)
    
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String, nullable=True) # e.g. "STOP_LOSS", "TAKE_PROFIT", "EMERGENCY_EXIT"
    
    realized_pnl = Column(Float, nullable=True)
    
def init_db():
    Base.metadata.create_all(bind=engine)
