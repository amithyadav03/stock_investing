import os
import sys
import json
import requests
import time

# Add parent directory to path to allow importing core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import settings

def auto_scan_market():
    # Attempt to load dynamic JSON list from Tier-1 screener
    download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db'))
    json_path = os.path.join(download_dir, 'daily_scan_list.json')
    
    symbols_to_scan = []
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            symbols_to_scan = json.load(f)
            print(f"Loaded {len(symbols_to_scan)} symbols dynamically from Tier-1 Screener.")
    else:
        # Fallback to YAML if the pre_screener hasn't run
        strategy = settings.strategy
        symbols_to_scan = strategy.get("scanning", {}).get("symbols", [])
        print("Tier-1 Screener JSON not found. Falling back to static YAML list.")

    if not symbols_to_scan:
        print("⚠️ No stocks found to scan.")
        return

    scan_interval = settings.strategy.get("scanning", {}).get("interval_seconds", 2)
    print(f"Starting Automatic Market Scan for {len(symbols_to_scan)} equities...\n")
    
    for symbol in symbols_to_scan:
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
        
        # Stagger the triggers
        time.sleep(scan_interval)
        
    print("\nAll scans dispatched! Your AI is now grinding through the charts in the background.")

if __name__ == "__main__":
    auto_scan_market()
