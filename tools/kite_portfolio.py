"""
Zerodha Kite portfolio sync — fetches existing holdings and positions.
Used by the portfolio advisor and morning brief.
"""

from datetime import datetime
from typing import Optional
from core.config import settings


def _get_kite():
    if not (settings.KITE_API_KEY and settings.KITE_ACCESS_TOKEN):
        return None
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=settings.KITE_API_KEY)
        kite.set_access_token(settings.KITE_ACCESS_TOKEN)
        return kite
    except Exception as e:
        print(f"[KitePortfolio] Kite init failed: {e}")
        return None


def get_holdings() -> list[dict]:
    """
    Returns all demat holdings from Kite.
    Each entry: {symbol, isin, quantity, avg_price, last_price, pnl, pnl_pct}
    """
    kite = _get_kite()
    if not kite:
        return _get_holdings_from_db()

    try:
        raw = kite.holdings()
        holdings = []
        for h in raw:
            qty = h.get("quantity", 0)
            avg = h.get("average_price", 0.0)
            last = h.get("last_price", avg)
            pnl = round((last - avg) * qty, 2) if avg > 0 else 0.0
            pnl_pct = round((last - avg) / avg * 100, 2) if avg > 0 else 0.0
            holdings.append({
                "symbol": h.get("tradingsymbol", ""),
                "isin": h.get("isin", ""),
                "quantity": qty,
                "avg_price": round(avg, 2),
                "current_price": round(last, 2),
                "pnl_amount": pnl,
                "pnl_pct": pnl_pct,
                "exchange": h.get("exchange", "NSE"),
            })
        return holdings
    except Exception as e:
        print(f"[KitePortfolio] holdings() failed: {e}. Using DB snapshot.")
        return _get_holdings_from_db()


def get_positions() -> list[dict]:
    """
    Returns open intraday and overnight positions from Kite.
    Each entry: {symbol, quantity, avg_price, last_price, pnl, product}
    """
    kite = _get_kite()
    if not kite:
        return []

    try:
        raw = kite.positions()
        positions = []
        for p in raw.get("net", []):
            qty = p.get("quantity", 0)
            if qty == 0:
                continue
            avg = p.get("average_price", 0.0)
            last = p.get("last_price", avg)
            pnl = round(p.get("pnl", 0.0), 2)
            pnl_pct = round((last - avg) / avg * 100, 2) if avg > 0 else 0.0
            positions.append({
                "symbol": p.get("tradingsymbol", ""),
                "quantity": qty,
                "avg_price": round(avg, 2),
                "current_price": round(last, 2),
                "pnl_amount": pnl,
                "pnl_pct": pnl_pct,
                "product": p.get("product", "CNC"),
            })
        return positions
    except Exception as e:
        print(f"[KitePortfolio] positions() failed: {e}")
        return []


def sync_holdings_to_db(holdings: Optional[list] = None):
    """Upsert holdings into PortfolioHolding table for offline access."""
    from db.schema import SessionLocal, PortfolioHolding

    if holdings is None:
        holdings = get_holdings()
    if not holdings:
        return

    session = SessionLocal()
    try:
        existing_symbols = {
            h.symbol for h in session.query(PortfolioHolding).all()
        }
        for h in holdings:
            symbol = h["symbol"]
            if symbol in existing_symbols:
                record = session.query(PortfolioHolding).filter(
                    PortfolioHolding.symbol == symbol
                ).first()
                record.quantity = h["quantity"]
                record.avg_price = h["avg_price"]
                record.current_price = h["current_price"]
                record.pnl_amount = h["pnl_amount"]
                record.pnl_pct = h["pnl_pct"]
                record.last_synced = datetime.utcnow()
            else:
                record = PortfolioHolding(
                    symbol=symbol,
                    isin=h.get("isin", ""),
                    quantity=h["quantity"],
                    avg_price=h["avg_price"],
                    current_price=h["current_price"],
                    pnl_amount=h["pnl_amount"],
                    pnl_pct=h["pnl_pct"],
                )
                session.add(record)
        session.commit()
        print(f"[KitePortfolio] Synced {len(holdings)} holdings to DB.")
    finally:
        session.close()


def _get_holdings_from_db() -> list[dict]:
    """Fallback — read last synced holdings from DB."""
    from db.schema import SessionLocal, PortfolioHolding
    session = SessionLocal()
    try:
        records = session.query(PortfolioHolding).all()
        return [
            {
                "symbol": r.symbol,
                "isin": r.isin or "",
                "quantity": r.quantity,
                "avg_price": r.avg_price,
                "current_price": r.current_price or r.avg_price,
                "pnl_amount": r.pnl_amount or 0.0,
                "pnl_pct": r.pnl_pct or 0.0,
            }
            for r in records
        ]
    finally:
        session.close()


def get_portfolio_value() -> dict:
    """Total portfolio value including holdings and open positions."""
    holdings = get_holdings()
    invested = sum(h["avg_price"] * h["quantity"] for h in holdings)
    current = sum((h["current_price"] or h["avg_price"]) * h["quantity"] for h in holdings)
    pnl = current - invested
    pnl_pct = round(pnl / invested * 100, 2) if invested > 0 else 0.0
    return {
        "invested_value": round(invested, 2),
        "current_value": round(current, 2),
        "total_pnl": round(pnl, 2),
        "total_pnl_pct": pnl_pct,
        "holdings_count": len(holdings),
    }
