from tools.market_data import market_data_tool

def pre_execution_validation(symbol: str, proposed_entry: float, max_slippage_pct: float = 0.5) -> bool:
    """
    Validates if the current market price is still within a safe zone
    compared to the original proposed entry price. Solves the issue of delayed approvals.
    """
    current_price = market_data_tool.get_current_price(symbol)
    
    if current_price == 0.0:
        print("[Validator] Could not fetch current price. Aborting for safety.")
        return False
        
    slippage = abs(current_price - proposed_entry) / proposed_entry * 100
    
    if slippage > max_slippage_pct:
        print(f"[Validator] ABORTING TRADE: {symbol}. Market moved too far. Proposed: {proposed_entry}, Current: {current_price}, Slippage: {slippage:.2f}%")
        return False
        
    print(f"[Validator] VALIDATED: {symbol} is safe to execute. Current: {current_price}, Proposed: {proposed_entry}")
    return True
