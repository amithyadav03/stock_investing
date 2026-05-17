"""
Thread-safe in-memory TTL cache.
Avoids hammering Kite/yfinance/Screener on every agent call.

TTLs used:
  prices      →  3 min  (live market data)
  technicals  →  30 min (indicator calculations)
  fundamentals→  24 hrs (screener.in data)
  news        →  15 min (RSS feeds)
  macro       →  4 hrs  (regime classification)
"""

from threading import Lock
from time import time


class TTLCache:
    def __init__(self):
        self._store: dict = {}
        self._lock = Lock()

    def get(self, key: str):
        with self._lock:
            entry = self._store.get(key)
            if entry and time() < entry["expires"]:
                return entry["value"]
            if entry:
                del self._store[key]
        return None

    def set(self, key: str, value, ttl_seconds: int):
        with self._lock:
            self._store[key] = {"value": value, "expires": time() + ttl_seconds}

    def invalidate(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def invalidate_prefix(self, prefix: str):
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]

    def size(self) -> int:
        with self._lock:
            return len(self._store)


# TTL constants (seconds)
TTL_PRICE = 180          # 3 min
TTL_TECHNICALS = 1800    # 30 min
TTL_FUNDAMENTALS = 86400 # 24 hrs
TTL_NEWS = 900           # 15 min
TTL_MACRO = 14400        # 4 hrs
TTL_GLOBAL_CUES = 3600   # 1 hr

cache = TTLCache()
