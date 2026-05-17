"""
Capital manager — tracks available capital, position sizing, and exposure limits.
Works in both paper mode (₹5L virtual) and live mode (Kite margin API).
"""

from core.config import settings
from db.schema import SessionLocal, TradeExecution, PaperTrade


SECTOR_MAP = {
    # Banking & Finance
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "SBIN": "BANKING",
    "AXISBANK": "BANKING", "KOTAKBANK": "BANKING", "BAJFINANCE": "BANKING",
    "BAJAJFINSV": "BANKING", "HDFCLIFE": "BANKING", "SBILIFE": "BANKING",
    # IT
    "INFY": "IT", "TCS": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "MPHASIS": "IT", "LTIM": "IT", "PERSISTENT": "IT",
    # Pharma
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA",
    "DIVISLAB": "PHARMA", "AUROPHARMA": "PHARMA", "TORNTPHARM": "PHARMA",
    # Auto
    "TATAMOTORS": "AUTO", "MARUTI": "AUTO", "BAJAJ-AUTO": "AUTO",
    "HEROMOTOCO": "AUTO", "EICHERMOT": "AUTO", "M&M": "AUTO",
    # Energy & Oil
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "BPCL": "ENERGY",
    "IOC": "ENERGY", "NTPC": "ENERGY", "POWERGRID": "ENERGY",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",
    # Metals
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS", "HINDALCO": "METALS",
    "VEDL": "METALS", "NMDC": "METALS", "SAIL": "METALS",
    # Telecom
    "BHARTIARTL": "TELECOM", "IDEA": "TELECOM",
    # Infrastructure / Capital Goods
    "LT": "INFRA", "SIEMENS": "INFRA", "ABB": "INFRA",
}


class CapitalManager:

    def get_total_capital(self) -> float:
        if settings.PAPER_MODE:
            return settings.PAPER_CAPITAL
        # Live mode: read from config until Kite margin sync is wired
        return float(settings.strategy.get("capital", {}).get("live_capital", settings.PAPER_CAPITAL))

    def get_deployed_capital(self) -> float:
        """Sum of (entry_price × quantity) across all open positions."""
        session = SessionLocal()
        try:
            if settings.PAPER_MODE:
                trades = session.query(PaperTrade).filter(PaperTrade.status == "OPEN").all()
            else:
                trades = session.query(TradeExecution).filter(TradeExecution.status == "OPEN").all()
            return sum((t.entry_price or 0) * (t.quantity or 0) for t in trades)
        finally:
            session.close()

    def get_available_capital(self) -> float:
        return max(0.0, self.get_total_capital() - self.get_deployed_capital())

    def get_open_position_count(self) -> int:
        session = SessionLocal()
        try:
            if settings.PAPER_MODE:
                return session.query(PaperTrade).filter(PaperTrade.status == "OPEN").count()
            else:
                return session.query(TradeExecution).filter(TradeExecution.status == "OPEN").count()
        finally:
            session.close()

    def get_total_risk_deployed(self) -> float:
        """Sum of (entry - stop_loss) × quantity for all open positions."""
        session = SessionLocal()
        try:
            from db.schema import TradeProposal
            total_risk = 0.0
            if settings.PAPER_MODE:
                trades = session.query(PaperTrade).filter(PaperTrade.status == "OPEN").all()
                for t in trades:
                    risk_per_share = abs((t.entry_price or 0) - (t.stop_loss or 0))
                    total_risk += risk_per_share * (t.quantity or 0)
            else:
                execs = session.query(TradeExecution).filter(TradeExecution.status == "OPEN").all()
                for e in execs:
                    proposal = session.query(TradeProposal).filter(
                        TradeProposal.id == e.proposal_id
                    ).first()
                    if proposal:
                        risk_per_share = abs((e.entry_price or 0) - (proposal.stop_loss or 0))
                        total_risk += risk_per_share * (e.quantity or 0)
            return total_risk
        finally:
            session.close()

    def get_sector(self, symbol: str) -> str:
        return SECTOR_MAP.get(symbol.upper(), "OTHER")

    def can_open_new_position(self, strategy_type: str = "swing", symbol: str = "") -> tuple[bool, str]:
        """Returns (allowed, reason). Checks position count, capital, portfolio heat, and sector concentration."""
        max_pos = settings.strategy.get("risk", {}).get("max_open_positions", 5)
        count = self.get_open_position_count()
        if count >= max_pos:
            return False, f"Max {max_pos} positions reached ({count} open)."

        strategy_max = settings.strategy.get("strategies", {}).get(strategy_type, {}).get("max_positions", 3)
        # Count positions of this specific strategy
        session = SessionLocal()
        try:
            if settings.PAPER_MODE:
                strategy_count = session.query(PaperTrade).filter(
                    PaperTrade.status == "OPEN", PaperTrade.strategy_type == strategy_type
                ).count()
            else:
                strategy_count = session.query(TradeExecution).filter(
                    TradeExecution.status == "OPEN", TradeExecution.strategy_type == strategy_type
                ).count()
        finally:
            session.close()

        if strategy_count >= strategy_max:
            return False, f"Max {strategy_max} {strategy_type} positions reached."

        # Per-sector concentration cap
        sector_max = settings.strategy.get("risk", {}).get("max_positions_per_sector", 2)
        symbol_sector = self.get_sector(symbol)
        if symbol_sector != "OTHER":
            session2 = SessionLocal()
            try:
                sector_symbols = [s for s, sec in SECTOR_MAP.items() if sec == symbol_sector]
                if settings.PAPER_MODE:
                    sector_count = session2.query(PaperTrade).filter(
                        PaperTrade.status == "OPEN",
                        PaperTrade.symbol.in_(sector_symbols)
                    ).count()
                else:
                    sector_count = session2.query(TradeExecution).filter(
                        TradeExecution.status == "OPEN",
                        TradeExecution.symbol.in_(sector_symbols)
                    ).count()
                if sector_count >= sector_max:
                    return False, f"Sector concentration limit: already {sector_count} {symbol_sector} positions."
            finally:
                session2.close()

        available = self.get_available_capital()
        if available < settings.strategy.get("capital", {}).get("min_available_capital", 10000):
            return False, f"Insufficient capital: ₹{available:,.0f} available."

        # Portfolio heat check
        total = self.get_total_capital()
        risk_deployed = self.get_total_risk_deployed()
        max_heat = settings.strategy.get("risk", {}).get("max_portfolio_heat", 0.05)
        if total > 0 and (risk_deployed / total) >= max_heat:
            return False, f"Portfolio heat {risk_deployed/total:.1%} at max {max_heat:.1%}."

        return True, ""

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        conviction_tier: str = "MEDIUM",
        strategy_type: str = "swing",
    ) -> dict:
        """
        ATR/risk-based position sizing.
        Returns: {quantity, capital_deployed, risk_amount, risk_pct}
        """
        total = self.get_total_capital()
        strategy_cfg = settings.strategy.get("strategies", {}).get(strategy_type, {})
        base_risk_pct = strategy_cfg.get("risk_per_trade", 0.01)

        tier_mult = settings.strategy.get("risk", {}).get("conviction_multipliers", {})
        mult = tier_mult.get(conviction_tier.upper(), 0.75)
        risk_amount = total * base_risk_pct * mult

        # Hard cap
        max_abs = settings.strategy.get("risk", {}).get("max_absolute_risk_per_trade", 0.02)
        risk_amount = min(risk_amount, total * max_abs)

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0 or entry_price <= 0:
            return {"quantity": 0, "capital_deployed": 0.0, "risk_amount": 0.0, "risk_pct": 0.0}

        quantity = max(1, int(risk_amount / risk_per_share))

        # Single position cap: ≤ 20% of portfolio
        max_pos_pct = settings.strategy.get("risk", {}).get("max_position_size_pct", 0.20)
        max_by_capital = max(1, int((total * max_pos_pct) / entry_price))
        quantity = min(quantity, max_by_capital)

        return {
            "quantity": quantity,
            "capital_deployed": round(quantity * entry_price, 2),
            "risk_amount": round(quantity * risk_per_share, 2),
            "risk_pct": round((quantity * risk_per_share) / total * 100, 2) if total > 0 else 0.0,
        }

    def get_summary(self) -> dict:
        total = self.get_total_capital()
        deployed = self.get_deployed_capital()
        available = max(0.0, total - deployed)
        count = self.get_open_position_count()
        max_pos = settings.strategy.get("risk", {}).get("max_open_positions", 5)
        risk_deployed = self.get_total_risk_deployed()

        return {
            "mode": "PAPER" if settings.PAPER_MODE else "LIVE",
            "total_capital": total,
            "deployed_capital": round(deployed, 2),
            "available_capital": round(available, 2),
            "utilization_pct": round(deployed / total * 100, 1) if total > 0 else 0.0,
            "risk_deployed": round(risk_deployed, 2),
            "portfolio_heat_pct": round(risk_deployed / total * 100, 2) if total > 0 else 0.0,
            "open_positions": count,
            "max_positions": max_pos,
            "slots_available": max(0, max_pos - count),
        }


capital_manager = CapitalManager()
