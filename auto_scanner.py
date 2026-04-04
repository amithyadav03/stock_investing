import requests
import time
from core.config import settings

# Pull strategy from YAML via settings
strategy = settings.strategy
STOCKS_TO_SCAN = strategy.get("scanning", {}).get("symbols", [])
SCAN_INTERVAL = strategy.get("scanning", {}).get("interval_seconds", 2)

def auto_scan_market():
    if not STOCKS_TO_SCAN:
        print("⚠️ No stocks found in strategy_config.yaml to scan.")
        return

    print(f"Starting Automatic Market Scan for {len(STOCKS_TO_SCAN)} equities...\n")
    
    for symbol in STOCKS_TO_SCAN:
        try:
            print(f"Dispatching Agent for: {symbol}")
            response = requests.post(
                "http://127.0.0.1:8000/scan",
                json={"symbol": symbol},
                timeout=5
            )
            print(response.json())
        except Exception as e:
            print(f"Failed to dispatch {symbol}: {e}")
        
        # Stagger the triggers so we don't completely overload the Kite API rate limits
        time.sleep(SCAN_INTERVAL)
        
    print("\nAll scans dispatched! Your AI is now grinding through the charts in the background.")

if __name__ == "__main__":
    auto_scan_market()
