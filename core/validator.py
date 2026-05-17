from tools.market_data import market_data_tool
from core.config import settings


def pre_execution_validation(symbol: str, proposed_entry: float, max_slippage_pct: float = None) -> bool:
    """
    Validates current market price is within tolerance of proposed entry.
    Also sanity-checks that the proposed price isn't obviously wrong.
    Returns True if safe to execute.
    """
    if proposed_entry <= 0:
        print(f"[Validator] ABORTING: {symbol} proposed entry {proposed_entry} is invalid.")
        return False

    if max_slippage_pct is None:
        max_slippage_pct = settings.strategy.get("trading", {}).get(
            "slippage_tolerance_pct", 0.5
        )

    current_price = market_data_tool.get_current_price(symbol)

    if current_price <= 0:
        print("[Validator] Could not fetch current price. Aborting for safety.")
        return False

    # Sanity check: proposed price must be within 40% of current price
    # (catches AI hallucinations or stale data proposing absurd prices)
    sanity_ratio = abs(current_price - proposed_entry) / current_price * 100
    if sanity_ratio > 40:
        print(f"[Validator] ABORTING: {symbol} proposed entry {proposed_entry:.2f} is {sanity_ratio:.1f}% "
              f"away from current {current_price:.2f}. Likely stale or hallucinated.")
        return False

    slippage = abs(current_price - proposed_entry) / proposed_entry * 100
    if slippage > max_slippage_pct:
        print(f"[Validator] ABORTING: {symbol}. Slippage {slippage:.2f}% > {max_slippage_pct:.1f}% tolerance. "
              f"Proposed: {proposed_entry:.2f}, Current: {current_price:.2f}")
        return False

    print(f"[Validator] VALIDATED: {symbol} at {current_price:.2f} vs proposed {proposed_entry:.2f} ({slippage:.2f}% slippage).")
    return True
